"""Five-seed held-out-fault reference-safety gate for FP-NAA."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from care_asd.evaluation.fp_naa_candidate import (
    _atomic_csv,
    _atomic_json,
    _cuda_device,
    _ensure_contract,
    _new_model,
    _sha256,
    _write_progress,
)
from care_asd.fp_naa_config import FPNAAConfig, load_fp_naa_config
from care_asd.fp_naa_reference_safety_config import (
    FROZEN_CONDITIONS,
    FPNAAReferenceSafetyConfig,
    ReferenceSafetyCondition,
    load_fp_naa_reference_safety_config,
)
from care_asd.models.fp_naa_adapter import BandwiseReferenceAdapter


@dataclass(frozen=True)
class FPNaaReferenceSafetyResult:
    output_directory: Path
    summary_path: Path
    gate_path: Path
    passed: bool
    c3_permitted: bool


@dataclass(frozen=True)
class _SafetyArrays:
    frame: pd.DataFrame
    noisy_clean: np.ndarray
    matched_reference: np.ndarray
    teacher_clean: np.ndarray
    fault_noisy: np.ndarray
    teacher_fault: np.ndarray
    leakage_clean: dict[str, np.ndarray]
    leakage_fault: dict[str, np.ndarray]
    silence_reference: np.ndarray


def run_fp_naa_reference_safety(
    *,
    base_cache_directory: str | Path,
    augmentation_cache_directory: str | Path,
    safety_cache_directory: str | Path,
    screening_checkpoint_directory: str | Path,
    confirmatory_checkpoint_directory: str | Path,
    confirmatory_directory: str | Path,
    confirmatory_lomo_directory: str | Path,
    output_directory: str | Path,
    config_path: str | Path,
    safety_config_path: str | Path,
    experiment_id: str,
    device: str = "cuda",
    preload_workers: int | None = None,
) -> FPNaaReferenceSafetyResult:
    """Evaluate five frozen C2 seeds under matched and corrupted reference conditions."""
    base_cache = Path(base_cache_directory).resolve()
    augmentation_cache = Path(augmentation_cache_directory).resolve()
    safety_cache = Path(safety_cache_directory).resolve()
    screening_checkpoints = Path(screening_checkpoint_directory).resolve()
    confirmatory_checkpoints = Path(confirmatory_checkpoint_directory).resolve()
    confirmatory = Path(confirmatory_directory).resolve()
    confirmatory_lomo = Path(confirmatory_lomo_directory).resolve()
    output = Path(output_directory).resolve()
    config = load_fp_naa_config(Path(config_path).resolve())
    safety = load_fp_naa_reference_safety_config(Path(safety_config_path).resolve())
    workers = config.training.workers if preload_workers is None else preload_workers
    if not 1 <= workers <= 16:
        raise ValueError("preload_workers must be in [1, 16]")
    _validate_gate_thresholds(config, safety)
    confirmatory_gate = confirmatory / "gate.json"
    lomo_gate = confirmatory_lomo / "gate.json"
    if not confirmatory_gate.is_file() or not lomo_gate.is_file():
        raise FileNotFoundError("Reference safety requires completed G3 core and LOMO gates")
    confirmatory_payload = json.loads(confirmatory_gate.read_text(encoding="utf-8"))
    lomo_payload = json.loads(lomo_gate.read_text(encoding="utf-8"))
    if not bool(confirmatory_payload.get("checks", {}).get("core_confirmatory")):
        raise ValueError("Reference safety is blocked because the G3 core gate failed")
    if not bool(lomo_payload.get("passed")):
        raise ValueError("Reference safety is blocked because confirmatory LOMO failed")
    for cache in (base_cache, augmentation_cache, safety_cache):
        if not (cache / "cache.json").is_file() or not (cache / "index.parquet").is_file():
            raise FileNotFoundError(f"Completed cache not found: {cache}")
    safety_metadata = json.loads((safety_cache / "cache.json").read_text(encoding="utf-8"))
    if safety_metadata.get("checkpoint_sha256") != config.provenance.checkpoint_sha256:
        raise ValueError("Reference-safety cache checkpoint does not match FP-NAA config")
    seeds = config.training.confirmatory_seeds
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("Reference safety requires exactly five unique confirmatory seeds")
    checkpoints = {
        seed: _checkpoint_path(
            seed=seed,
            screening_seeds=set(config.training.screening_seeds),
            screening_directory=screening_checkpoints,
            confirmatory_directory=confirmatory_checkpoints,
        )
        for seed in seeds
    }
    output.mkdir(parents=True, exist_ok=True)
    _ensure_contract(
        output,
        {
            "schema_version": 1,
            "kind": "fp_naa_reference_safety",
            "experiment_id": experiment_id,
            "config": config.model_dump(mode="json"),
            "safety_config": safety.model_dump(mode="json"),
            "base_cache_metadata_sha256": _sha256(base_cache / "cache.json"),
            "augmentation_cache_metadata_sha256": _sha256(augmentation_cache / "cache.json"),
            "safety_cache_metadata_sha256": _sha256(safety_cache / "cache.json"),
            "confirmatory_gate_sha256": _sha256(confirmatory_gate),
            "confirmatory_lomo_gate_sha256": _sha256(lomo_gate),
            "checkpoint_sha256": {
                str(seed): _sha256(checkpoint) for seed, checkpoint in checkpoints.items()
            },
        },
    )
    details_path = output / "reference_safety_retention.csv"
    summary_path = output / "reference_safety_summary.csv"
    gate_path = output / "gate.json"
    if details_path.is_file() and summary_path.is_file() and gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        return FPNaaReferenceSafetyResult(
            output,
            summary_path,
            gate_path,
            bool(gate["passed"]),
            bool(gate["c3_permitted"]),
        )
    _write_progress(output, stage="preload", completed=0, total=len(seeds))
    arrays = _load_safety_arrays(
        base_cache=base_cache,
        augmentation_cache=augmentation_cache,
        safety_cache=safety_cache,
        workers=workers,
    )
    unmatched_indices = _cross_machine_unmatched_indices(
        arrays.frame,
        seed=safety.unmatched_reference_seed,
    )
    device_value = _cuda_device(device)
    rows: list[dict[str, object]] = []
    for run_number, seed in enumerate(seeds, start=1):
        _write_progress(
            output,
            stage=f"reference_safety:seed{seed}",
            completed=run_number - 1,
            total=len(seeds),
        )
        model = _load_c2_model(
            checkpoint=checkpoints[seed],
            seed=seed,
            config=config,
            device=device_value,
        )
        metrics = _evaluate_seed(
            model=model,
            arrays=arrays,
            unmatched_indices=unmatched_indices,
            config=config,
            device=device_value,
        )
        for condition in FROZEN_CONDITIONS:
            retention, normal_shift = metrics[condition]
            for item, retention_value, shift_value in zip(
                arrays.frame.itertuples(index=False),
                retention,
                normal_shift,
                strict=True,
            ):
                rows.append(
                    {
                        "file_id": str(item.file_id),
                        "machine_type": str(item.machine_type),
                        "section": str(item.section),
                        "seed": seed,
                        "condition": condition,
                        "fault_family": str(item.heldout_fault_family),
                        "retention": float(retention_value),
                        "normal_shift_relative": float(shift_value),
                    }
                )
        _write_progress(
            output,
            stage=f"complete:seed{seed}",
            completed=run_number,
            total=len(seeds),
        )
    details = pd.DataFrame(rows)
    _validate_details(details, arrays.frame, seeds)
    _atomic_csv(details_path, details)
    summary = _summarize(details)
    _atomic_csv(summary_path, summary)
    gate = _reference_safety_gate(summary, safety)
    reference_safety_passed = bool(gate["passed"])
    gate["gate"] = "G3_final_contribution"
    gate["upstream_checks"] = {
        "confirmatory_core": True,
        "confirmatory_lomo": True,
    }
    gate["reference_safety_passed"] = reference_safety_passed
    gate["passed"] = reference_safety_passed
    _atomic_json(gate_path, gate)
    _atomic_json(
        output / "run.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "population": safety.population,
            "clips": len(arrays.frame),
            "seeds": seeds,
            "conditions": list(FROZEN_CONDITIONS),
            "device": str(device_value),
            "preload_workers": workers,
        },
    )
    _write_progress(output, stage="complete", completed=len(seeds), total=len(seeds))
    return FPNaaReferenceSafetyResult(
        output,
        summary_path,
        gate_path,
        bool(gate["passed"]),
        bool(gate["c3_permitted"]),
    )


def _load_safety_arrays(
    *,
    base_cache: Path,
    augmentation_cache: Path,
    safety_cache: Path,
    workers: int,
) -> _SafetyArrays:
    frame = pd.read_parquet(safety_cache / "index.parquet")
    frame = frame.sort_values("file_id", kind="stable").reset_index(drop=True)
    if (
        frame.empty
        or frame["file_id"].duplicated().any()
        or not frame["heldout"].astype(bool).all()
    ):
        raise ValueError("Reference-safety index must contain unique held-out rows")
    augmentation_index = pd.read_parquet(augmentation_cache / "index.parquet")
    augmentation_index = augmentation_index.set_index("file_id", verify_integrity=True)
    base_index = pd.read_parquet(base_cache / "index.parquet")
    base_index = base_index.set_index("file_id", verify_integrity=True)
    ids = frame["file_id"].astype(str).tolist()
    if not set(ids).issubset(augmentation_index.index.astype(str)) or not set(ids).issubset(
        base_index.index.astype(str)
    ):
        raise ValueError("Reference-safety IDs are not covered by the source caches")

    def load(file_id: str, row: pd.Series) -> tuple[np.ndarray, ...]:
        augmentation_row = augmentation_index.loc[file_id]
        base_row = base_index.loc[file_id]
        with np.load(
            augmentation_cache / str(augmentation_row["augmentation_file"]),
            allow_pickle=False,
        ) as payload:
            augmented = tuple(
                payload[name].copy()
                for name in (
                    "heldout_noisy_clean",
                    "heldout_reference",
                    "heldout_fault_noisy",
                    "heldout_fault_teacher",
                )
            )
        with np.load(base_cache / str(base_row["feature_file"]), allow_pickle=False) as payload:
            teacher_clean = payload["near"].copy()
        with np.load(safety_cache / str(row["safety_feature_file"]), allow_pickle=False) as payload:
            leakage = tuple(
                payload[f"leakage_{level}_{state}_reference"].copy()
                for level in ("low", "medium", "high")
                for state in ("clean", "fault")
            )
        values = (*augmented[:2], teacher_clean, *augmented[2:], *leakage)
        for value in values:
            _validate_grid(value)
        return values

    items = list(zip(ids, (row for _, row in frame.iterrows()), strict=True))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = list(executor.map(lambda item: load(item[0], item[1]), items))
    stacked = tuple(np.stack([item[index] for item in loaded]) for index in range(11))
    silence = np.load(safety_cache / "silence_reference.npy", allow_pickle=False)
    _validate_grid(silence)
    if silence.shape != stacked[0].shape[1:]:
        raise ValueError("Silence and held-out token grids have different shapes")
    return _SafetyArrays(
        frame=frame,
        noisy_clean=stacked[0],
        matched_reference=stacked[1],
        teacher_clean=stacked[2],
        fault_noisy=stacked[3],
        teacher_fault=stacked[4],
        leakage_clean={
            "low": stacked[5],
            "medium": stacked[7],
            "high": stacked[9],
        },
        leakage_fault={
            "low": stacked[6],
            "medium": stacked[8],
            "high": stacked[10],
        },
        silence_reference=silence,
    )


def _cross_machine_unmatched_indices(frame: pd.DataFrame, *, seed: int) -> np.ndarray:
    machines = frame["machine_type"].astype(str).to_numpy()
    if len(set(machines)) < 2:
        raise ValueError("Unmatched-reference stress requires at least two machine types")
    file_ids = frame["file_id"].astype(str).to_numpy()
    result = np.empty(len(frame), dtype=np.int64)
    for index, (machine, file_id) in enumerate(zip(machines, file_ids, strict=True)):
        candidates = np.flatnonzero(machines != machine)
        digest = _stable_integer(f"{seed}|{file_id}|unmatched")
        result[index] = candidates[digest % len(candidates)]
    if np.any(machines[result] == machines):
        raise AssertionError("Cross-machine reference construction failed")
    return result


def _evaluate_seed(
    *,
    model: BandwiseReferenceAdapter,
    arrays: _SafetyArrays,
    unmatched_indices: np.ndarray,
    config: FPNAAConfig,
    device: torch.device,
) -> dict[ReferenceSafetyCondition, tuple[np.ndarray, np.ndarray]]:
    retention_values: dict[ReferenceSafetyCondition, list[np.ndarray]] = {
        condition: [] for condition in FROZEN_CONDITIONS
    }
    shift_values: dict[ReferenceSafetyCondition, list[np.ndarray]] = {
        condition: [] for condition in FROZEN_CONDITIONS
    }
    batch_size = min(config.training.batch_size, 64)
    use_amp = config.training.mixed_precision and device.type == "cuda"
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(arrays.frame), batch_size):
            stop = min(start + batch_size, len(arrays.frame))
            noisy_clean = torch.as_tensor(arrays.noisy_clean[start:stop], device=device)
            fault_noisy = torch.as_tensor(arrays.fault_noisy[start:stop], device=device)
            teacher_clean = torch.as_tensor(arrays.teacher_clean[start:stop], device=device)
            teacher_fault = torch.as_tensor(arrays.teacher_fault[start:stop], device=device)
            matched_reference = torch.as_tensor(arrays.matched_reference[start:stop], device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                matched_clean = model(noisy_clean, matched_reference)
            for condition in FROZEN_CONDITIONS:
                clean_reference, fault_reference = _condition_references(
                    condition=condition,
                    arrays=arrays,
                    start=start,
                    stop=stop,
                    unmatched_indices=unmatched_indices,
                )
                clean_reference_tensor = torch.as_tensor(clean_reference, device=device)
                fault_reference_tensor = torch.as_tensor(fault_reference, device=device)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    if condition == "matched":
                        student_clean = matched_clean
                    else:
                        student_clean = model(noisy_clean, clean_reference_tensor)
                    student_fault = model(fault_noisy, fault_reference_tensor)
                retention = _retention(
                    student_clean.float(),
                    student_fault.float(),
                    teacher_clean.float(),
                    teacher_fault.float(),
                )
                shift = _relative_shift(student_clean.float(), matched_clean.float())
                retention_values[condition].append(retention.cpu().numpy())
                shift_values[condition].append(shift.cpu().numpy())
    return {
        condition: (
            np.concatenate(retention_values[condition]).astype(np.float64),
            np.concatenate(shift_values[condition]).astype(np.float64),
        )
        for condition in FROZEN_CONDITIONS
    }


def _condition_references(
    *,
    condition: ReferenceSafetyCondition,
    arrays: _SafetyArrays,
    start: int,
    stop: int,
    unmatched_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if condition == "matched":
        reference = arrays.matched_reference[start:stop]
        return reference, reference
    if condition == "unmatched":
        reference = arrays.matched_reference[unmatched_indices[start:stop]]
        return reference, reference
    if condition == "dropout":
        shape = (stop - start, *arrays.silence_reference.shape)
        reference = np.broadcast_to(arrays.silence_reference, shape).copy()
        return reference, reference
    if condition == "channel_swap":
        return arrays.noisy_clean[start:stop], arrays.fault_noisy[start:stop]
    level = condition.removeprefix("leakage_")
    return arrays.leakage_clean[level][start:stop], arrays.leakage_fault[level][start:stop]


def _retention(
    student_clean: torch.Tensor,
    student_fault: torch.Tensor,
    teacher_clean: torch.Tensor,
    teacher_fault: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    student_norm = torch.linalg.vector_norm(
        (student_fault - student_clean).reshape(len(student_clean), -1), dim=1
    )
    teacher_norm = torch.linalg.vector_norm(
        (teacher_fault - teacher_clean).reshape(len(teacher_clean), -1), dim=1
    )
    return torch.exp(-torch.log((student_norm + eps) / (teacher_norm + eps)).abs())


def _relative_shift(
    stressed: torch.Tensor,
    matched: torch.Tensor,
    eps: float = 1.0e-8,
) -> torch.Tensor:
    numerator = torch.linalg.vector_norm((stressed - matched).reshape(len(stressed), -1), dim=1)
    denominator = torch.linalg.vector_norm(matched.reshape(len(matched), -1), dim=1)
    return numerator / (denominator + eps)


def _summarize(details: pd.DataFrame) -> pd.DataFrame:
    seed_rows = (
        details.groupby(["condition", "seed"], sort=True)
        .agg(
            retention_median=("retention", "median"),
            retention_q05=("retention", lambda values: values.quantile(0.05)),
            normal_shift_relative_median=("normal_shift_relative", "median"),
            clips=("file_id", "count"),
        )
        .reset_index()
    )
    rows: list[dict[str, object]] = []
    for condition, group in seed_rows.groupby("condition", sort=False):
        all_values = details.loc[details["condition"] == condition, "retention"]
        rows.append(
            {
                "condition": condition,
                "retention_median": float(all_values.median()),
                "retention_worst_seed_q05": float(group["retention_q05"].min()),
                "normal_shift_relative_median": float(
                    details.loc[
                        details["condition"] == condition,
                        "normal_shift_relative",
                    ].median()
                ),
                "seeds": int(group["seed"].nunique()),
                "clips_per_seed": int(group["clips"].min()),
            }
        )
    summary = pd.DataFrame(rows)
    order = {condition: index for index, condition in enumerate(FROZEN_CONDITIONS)}
    summary["_order"] = summary["condition"].map(order)
    return summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def _reference_safety_gate(
    summary: pd.DataFrame,
    safety: FPNAAReferenceSafetyConfig,
) -> dict[str, object]:
    if set(summary["condition"].astype(str)) != set(FROZEN_CONDITIONS):
        raise ValueError("Reference-safety summary does not cover all frozen conditions")
    checks: dict[str, dict[str, bool]] = {}
    for row in summary.itertuples(index=False):
        checks[str(row.condition)] = {
            "retention_median": (
                float(row.retention_median) >= safety.gate.retention_median_minimum
            ),
            "retention_worst_seed_q05": (
                float(row.retention_worst_seed_q05) >= safety.gate.retention_worst_seed_q05_minimum
            ),
        }
    condition_pass = {condition: all(values.values()) for condition, values in checks.items()}
    matched_passed = condition_pass["matched"]
    stress_passed = all(
        passed for condition, passed in condition_pass.items() if condition != "matched"
    )
    passed = matched_passed and stress_passed
    return {
        "schema_version": 1,
        "gate": "G3_reference_safety",
        "population": safety.population,
        "thresholds": safety.gate.model_dump(mode="json"),
        "condition_checks": checks,
        "condition_passed": condition_pass,
        "matched_passed": matched_passed,
        "stress_passed": stress_passed,
        "c3_permitted": matched_passed and not stress_passed,
        "passed": passed,
        "decision": (
            "FP-NAA C2 passes the final reference-safety gate"
            if passed
            else (
                "Reference reliability is the active failure mode; one preregistered C3 revision is permitted"
                if matched_passed
                else "Matched-reference fault preservation failed; close C2 without a C3 rescue"
            )
        ),
    }


def _validate_details(details: pd.DataFrame, frame: pd.DataFrame, seeds: list[int]) -> None:
    expected = len(frame) * len(seeds) * len(FROZEN_CONDITIONS)
    if len(details) != expected:
        raise ValueError(f"Reference-safety details have {len(details)} rows, expected {expected}")
    if details.duplicated(["file_id", "seed", "condition"]).any():
        raise ValueError("Reference-safety details contain duplicate rows")
    if set(details["seed"].astype(int)) != set(seeds):
        raise ValueError("Reference-safety details do not cover the five frozen seeds")
    if set(details["condition"].astype(str)) != set(FROZEN_CONDITIONS):
        raise ValueError("Reference-safety details do not cover the frozen stress conditions")
    if not np.isfinite(details[["retention", "normal_shift_relative"]].to_numpy()).all():
        raise ValueError("Reference-safety metrics must be finite")
    if not details["retention"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("Reference-safety retention must lie in [0, 1]")


def _validate_gate_thresholds(
    config: FPNAAConfig,
    safety: FPNAAReferenceSafetyConfig,
) -> None:
    if (
        safety.gate.retention_median_minimum
        != config.gates.heldout_fault_delta_retention_median_minimum
        or safety.gate.retention_worst_seed_q05_minimum
        != config.gates.heldout_fault_delta_retention_q05_minimum
    ):
        raise ValueError("Safety thresholds must exactly match the frozen held-out fault gates")


def _checkpoint_path(
    *,
    seed: int,
    screening_seeds: set[int],
    screening_directory: Path,
    confirmatory_directory: Path,
) -> Path:
    root = screening_directory if seed in screening_seeds else confirmatory_directory
    path = root / f"seed{seed}" / "c2_fault_preserving.pt"
    if not path.is_file():
        raise FileNotFoundError(f"C2 checkpoint not found: {path}")
    return path


def _load_c2_model(
    *,
    checkpoint: Path,
    seed: int,
    config: FPNAAConfig,
    device: torch.device,
) -> BandwiseReferenceAdapter:
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload.get("candidate") != "c2_fault_preserving" or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"C2 checkpoint identity mismatch: {checkpoint}")
    if payload.get("config") != config.model_dump(mode="json"):
        raise ValueError(f"C2 checkpoint config mismatch: {checkpoint}")
    model = _new_model(config).to(device)
    model.load_state_dict(payload["model_state"])
    return model.eval()


def _validate_grid(value: np.ndarray) -> None:
    if value.ndim != 3 or value.dtype != np.float16 or not np.isfinite(value).all():
        raise ValueError("Reference-safety cache contains an invalid token grid")


def _stable_integer(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")
