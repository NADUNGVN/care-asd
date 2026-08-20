"""Real-cache, anomaly-label-free mechanism preflight for FP-NAA v10."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch

from care_asd.evaluation.fp_naa_backend import (
    accelerated_beam_scores,
    accelerated_global_knn_scores,
    accelerated_rdp_pool,
)
from care_asd.evaluation.fp_naa_candidate import (
    _adapt_rows,
    _BaseTokenStore,
    _c1_reuse_signature,
    _new_model,
    _preload_base_store,
)
from care_asd.evaluation.fp_naa_evidence_union import (
    ExpertCertificate,
    certify_evidence_expert,
    empirical_tail_evidence,
    monotone_evidence_union,
)
from care_asd.fp_naa_config import FPEvidenceUnionConfig, FPNAAConfig, load_fp_naa_config
from care_asd.models.fp_naa_adapter import BandwiseReferenceAdapter

ExpertName = Literal["tap0_rdp8_beam", "final_rdp4_beam", "final_global_ap"]


@dataclass(frozen=True)
class EvidencePreflightResult:
    output_directory: Path
    summary_path: Path
    certificates_path: Path
    policy_path: Path
    gate_path: Path
    gate_passed: bool


@dataclass(frozen=True)
class _ScoreSet:
    normal: np.ndarray
    in_clean: np.ndarray
    in_fault: np.ndarray
    heldout_clean: np.ndarray
    heldout_fault: np.ndarray


def run_evidence_union_preflight(
    *,
    base_cache_directory: str | Path,
    augmentation_cache_directory: str | Path,
    tap_cache_directory: str | Path,
    tap_contract_path: str | Path,
    c1_checkpoint_directory: str | Path,
    output_directory: str | Path,
    config_path: str | Path,
    workers: int = 12,
    device: str = "cuda",
) -> EvidencePreflightResult:
    """Certify fixed score experts from training normals and held-out pseudo faults."""
    if not 1 <= workers <= 16:
        raise ValueError("workers must be in [1, 16]")
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("V10 evidence preflight requires CUDA")
    base_cache = Path(base_cache_directory).resolve()
    augmentation_cache = Path(augmentation_cache_directory).resolve()
    tap_cache = Path(tap_cache_directory).resolve()
    tap_contract_source = Path(tap_contract_path).resolve()
    c1_checkpoints = Path(c1_checkpoint_directory).resolve()
    output = Path(output_directory).resolve()
    config_source = Path(config_path).resolve()
    config = load_fp_naa_config(config_source)
    union = config.evidence_union
    if union is None:
        raise ValueError("FP-NAA v10 config must define evidence_union")
    _validate_inputs(
        base_cache=base_cache,
        augmentation_cache=augmentation_cache,
        tap_cache=tap_cache,
        tap_contract=tap_contract_source,
        c1_checkpoints=c1_checkpoints,
        config=config,
    )
    output.mkdir(parents=True, exist_ok=True)
    gate_path = output / "gate.json"
    summary_path = output / "summary.csv"
    certificates_path = output / "expert_certificates.csv"
    policy_path = output / "policy.json"
    calibration_path = output / "normal_calibration.csv"
    contract_path = output / "contract.json"
    base_index = pd.read_parquet(base_cache / "index.parquet")
    tap_contract = _read_json(tap_contract_source)
    split = _selection_frame(tap_contract, base_index)
    contract = {
        "schema_version": 1,
        "kind": "fp_naa_v10_counterfactual_certified_monotone_evidence_preflight",
        "config_sha256": _sha256(config_source),
        "base_cache_metadata_sha256": _sha256(base_cache / "cache.json"),
        "base_cache_index_sha256": _sha256(base_cache / "index.parquet"),
        "augmentation_cache_metadata_sha256": _sha256(augmentation_cache / "cache.json"),
        "augmentation_cache_index_sha256": _sha256(augmentation_cache / "index.parquet"),
        "tap_contract_sha256": _sha256(tap_contract_source),
        "preflight_source_sha256": _sha256(Path(__file__)),
        "union_source_sha256": _sha256(Path(__file__).with_name("fp_naa_evidence_union.py")),
        "c1_checkpoints": {
            str(seed): _sha256(c1_checkpoints / f"seed{seed}" / "c1_mse.pt")
            for seed in union.c1_seeds
        },
        "selection_sha256": _canonical_sha(split.to_dict(orient="records")),
        "development_anomaly_labels_read": False,
        "heldout_family_used_for_selection": False,
    }
    _ensure_contract(contract_path, contract)
    required_outputs = (gate_path, summary_path, certificates_path, policy_path, calibration_path)
    if all(path.is_file() for path in required_outputs):
        _validate_completed_outputs(
            contract_path=contract_path,
            gate_path=gate_path,
            summary_path=summary_path,
            certificates_path=certificates_path,
            policy_path=policy_path,
            calibration_path=calibration_path,
        )
        gate = _read_json(gate_path)
        return EvidencePreflightResult(
            output,
            summary_path,
            certificates_path,
            policy_path,
            gate_path,
            bool(gate["passed"]),
        )

    augmentation_index = pd.read_parquet(augmentation_cache / "index.parquet")
    models = _load_c1_models(c1_checkpoints, config=config, union=union, device=device)
    base_store = _preload_base_store(base_cache, workers=workers)
    augmentation_lookup = augmentation_index.set_index("file_id", drop=False)
    if set(split["file_id"]).difference(augmentation_lookup.index):
        raise ValueError("V10 selection is not covered by the augmentation cache")

    certificate_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    machine_payloads: dict[str, dict[str, object]] = {}
    machine_scores: dict[
        str,
        tuple[_ScoreSet, dict[str, _ScoreSet], tuple[ExpertCertificate, ...]],
    ] = {}
    machines = sorted(split["machine_type"].astype(str).unique())
    for machine_number, machine in enumerate(machines, start=1):
        machine_split = split.loc[split["machine_type"].astype(str) == machine]
        train_rows = machine_split.loc[machine_split["role"] == "train"]
        in_rows = machine_split.loc[
            (machine_split["role"] == "validation") & ~machine_split["heldout"]
        ]
        heldout_rows = machine_split.loc[
            (machine_split["role"] == "validation") & machine_split["heldout"]
        ]
        if min(len(train_rows), len(in_rows), len(heldout_rows)) < union.crossfit_folds:
            raise ValueError(f"Insufficient V10 split rows for {machine}")
        final_arrays = _load_final_arrays(
            augmentation_cache,
            augmentation_lookup,
            train_rows,
            in_rows,
            heldout_rows,
            workers=workers,
        )
        tap_arrays = _load_tap_arrays(
            tap_cache,
            train_rows,
            in_rows,
            heldout_rows,
            workers=workers,
        )
        base_scores, seed_calibration = _score_c1_ensemble(
            models=models,
            base_store=base_store,
            train_rows=train_rows,
            final_arrays=final_arrays,
            config=config,
            union=union,
            device=device,
        )
        expert_scores = _score_supplementary_experts(
            final_arrays=final_arrays,
            tap_arrays=tap_arrays,
            train_ids=tuple(train_rows["file_id"].astype(str)),
            config=config,
            union=union,
            device=device,
        )
        base_evidence = _calibrate_score_set(
            base_scores,
            seed_calibration,
            ensemble=True,
            epsilon=union.calibration_epsilon,
        )
        expert_evidence: dict[str, _ScoreSet] = {}
        for name, scores in expert_scores.items():
            expert_evidence[name] = _calibrate_score_set(
                scores,
                {name: scores.normal},
                ensemble=False,
                epsilon=union.calibration_epsilon,
            )
        certificates: list[ExpertCertificate] = []
        for name in union.supplementary_experts:
            values = expert_evidence[name]
            certificate = certify_evidence_expert(
                name=name,
                normal_base=base_evidence.normal,
                normal_expert=values.normal,
                in_support_clean_base=base_evidence.in_clean,
                in_support_fault_base=base_evidence.in_fault,
                in_support_clean_expert=values.in_clean,
                in_support_fault_expert=values.in_fault,
                heldout_clean_base=base_evidence.heldout_clean,
                heldout_fault_base=base_evidence.heldout_fault,
                heldout_clean_expert=values.heldout_clean,
                heldout_fault_expert=values.heldout_fault,
                tail_probability=union.calibration_tail_probability,
                minimum_in_support_gain_median=union.minimum_in_support_evidence_gain_median,
                minimum_in_support_gain_q05=union.minimum_in_support_evidence_gain_q05,
                minimum_heldout_gain_median=union.minimum_heldout_evidence_gain_median,
                minimum_heldout_gain_q05=union.minimum_heldout_evidence_gain_q05,
                maximum_clean_activation_fraction=union.maximum_clean_activation_fraction,
            )
            certificates.append(certificate)
            certificate_rows.append({"machine_type": machine, **asdict(certificate)})
        machine_scores[machine] = (base_evidence, expert_evidence, tuple(certificates))
        machine_payloads[machine] = {
            "training_rows": len(train_rows),
            "in_support_rows": len(in_rows),
            "heldout_rows": len(heldout_rows),
        }
        for branch, seed_scores in seed_calibration.items():
            calibration_rows.extend(
                {
                    "machine_type": machine,
                    "branch": branch,
                    "normal_score": float(value),
                }
                for value in seed_scores
            )
        for name, expert_score_set in expert_scores.items():
            calibration_rows.extend(
                {
                    "machine_type": machine,
                    "branch": name,
                    "normal_score": float(value),
                }
                for value in expert_score_set.normal
            )
        _write_progress(
            output,
            stage="score",
            completed=machine_number,
            total=len(machines),
            machine=machine,
        )

    certificates_frame = pd.DataFrame(certificate_rows)
    globally_authorized = {
        str(name)
        for name, group in certificates_frame.groupby("name", sort=True)
        if float(group["eligible"].mean()) >= union.minimum_machine_pass_fraction
    }
    machine_results: dict[str, dict[str, object]] = {}
    for machine, (base_evidence, expert_evidence, machine_certificates) in machine_scores.items():
        active = tuple(
            item.name
            for item in machine_certificates
            if item.eligible and item.name in globally_authorized
        )
        penalties = {item.name: item.penalty for item in machine_certificates}
        candidate = _fuse_score_set(base_evidence, expert_evidence, penalties, active)
        in_gain, heldout_gain = _score_set_counterfactual_gain(candidate, base_evidence)
        machine_results[machine] = {
            "normal_activation_fraction": float(
                np.mean(candidate.normal > base_evidence.normal + 1.0e-12)
            ),
            "in_support_gain": in_gain,
            "heldout_gain": heldout_gain,
            "active_experts": active,
            "monotone": bool(
                all(
                    np.all(getattr(candidate, field) >= getattr(base_evidence, field))
                    for field in (
                        "normal",
                        "in_clean",
                        "in_fault",
                        "heldout_clean",
                        "heldout_fault",
                    )
                )
            ),
        }
        machine_payloads[machine].update(
            {
                "active_experts": list(active),
                "expert_penalties": penalties,
            }
        )
    summary = _summarize_certificates(certificates_frame, machine_results, union)
    _atomic_csv(certificates_path, certificates_frame)
    _atomic_csv(summary_path, summary)
    _atomic_csv(calibration_path, pd.DataFrame(calibration_rows))
    policy = {
        "schema_version": 1,
        "kind": "fp_naa_v10_ccmeu_policy",
        "config_sha256": _sha256(config_source),
        "contract_sha256": _sha256(contract_path),
        "artifacts": {
            "expert_certificates_sha256": _sha256(certificates_path),
            "normal_calibration_sha256": _sha256(calibration_path),
            "summary_sha256": _sha256(summary_path),
        },
        "tail_probability": union.calibration_tail_probability,
        "calibration_epsilon": union.calibration_epsilon,
        "crossfit_folds": union.crossfit_folds,
        "c1_seeds": union.c1_seeds,
        "globally_authorized_experts": sorted(globally_authorized),
        "machines": machine_payloads,
        "selection_uses": ["train_normal", "in_support_pseudo_fault"],
        "heldout_family_role": "gate_only",
        "development_anomaly_labels_read": False,
    }
    _atomic_json(policy_path, policy)
    gate = _make_gate(summary, machine_results, union, policy_path)
    _atomic_json(gate_path, gate)
    _write_progress(
        output, stage="complete", completed=len(machines), total=len(machines), machine="complete"
    )
    print(f"FP-NAA V10 evidence preflight complete. gate={gate['passed']}", flush=True)
    return EvidencePreflightResult(
        output,
        summary_path,
        certificates_path,
        policy_path,
        gate_path,
        bool(gate["passed"]),
    )


def _validate_inputs(
    *,
    base_cache: Path,
    augmentation_cache: Path,
    tap_cache: Path,
    tap_contract: Path,
    c1_checkpoints: Path,
    config: FPNAAConfig,
) -> None:
    for path in (
        base_cache / "cache.json",
        base_cache / "index.parquet",
        augmentation_cache / "cache.json",
        augmentation_cache / "index.parquet",
        tap_contract,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"V10 input is missing: {path}")
    union = cast(FPEvidenceUnionConfig, config.evidence_union)
    if tap_contract.parent.parent.name != union.tap_source_run_id:
        raise ValueError("Frozen V9 tap contract directory does not match V10 source run")
    if c1_checkpoints.name != union.c1_source_run_id:
        raise ValueError("Frozen C1 checkpoint directory does not match V10 source run")
    for seed in union.c1_seeds:
        if not (c1_checkpoints / f"seed{seed}" / "c1_mse.pt").is_file():
            raise FileNotFoundError(f"Frozen C1 checkpoint is missing for seed {seed}")
    tap = _read_json(tap_contract)
    if tap.get("kind") != "fp_naa_v9_preencoder_tangent_repair_preflight":
        raise ValueError("V10 requires the immutable V9 tap-0 cache contract")
    base_metadata = _read_json(base_cache / "cache.json")
    augmentation_metadata = _read_json(augmentation_cache / "cache.json")
    checkpoint_sha = config.provenance.checkpoint_sha256
    if tap.get("base_cache_metadata_sha256") != _sha256(base_cache / "cache.json"):
        raise ValueError("V9 tap contract base-cache metadata does not match V10")
    if tap.get("base_cache_index_sha256") != _sha256(base_cache / "index.parquet"):
        raise ValueError("V9 tap contract base-cache index does not match V10")
    if tap.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("V9 tap contract checkpoint does not match V10")
    if base_metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Base cache checkpoint does not match V10")
    if augmentation_metadata.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Augmentation cache checkpoint does not match V10")


def _selection_frame(contract: dict[str, object], base_index: pd.DataFrame) -> pd.DataFrame:
    raw = contract.get("selection")
    if not isinstance(raw, list) or not raw:
        raise ValueError("V9 tap contract selection is missing")
    selected = pd.DataFrame(raw)
    required = {"file_id", "role", "heldout"}
    if not required.issubset(selected.columns) or selected["file_id"].duplicated().any():
        raise ValueError("V9 selection contract is invalid")
    if set(selected["role"]) != {"train", "validation"}:
        raise ValueError("V9 selection roles are incomplete")
    metadata = base_index[["file_id", "machine_type", "section", "dataset_split", "condition"]]
    frame = selected.merge(metadata, on="file_id", how="left", validate="one_to_one")
    if frame["machine_type"].isna().any():
        raise ValueError("V9 selection is not covered by the base cache")
    if set(frame["dataset_split"]) != {"dev_train"} or set(frame["condition"]) != {"normal"}:
        raise ValueError("V10 preflight selection must contain training normals only")
    return frame.sort_values("file_id", kind="stable").reset_index(drop=True)


def _load_c1_models(
    checkpoints: Path,
    *,
    config: FPNAAConfig,
    union: FPEvidenceUnionConfig,
    device: str,
) -> dict[int, BandwiseReferenceAdapter]:
    result: dict[int, BandwiseReferenceAdapter] = {}
    for seed in union.c1_seeds:
        path = checkpoints / f"seed{seed}" / "c1_mse.pt"
        payload = torch.load(path, map_location=device, weights_only=True)
        if payload.get("candidate") != "c1_mse" or int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Frozen C1 checkpoint contract mismatch: {path}")
        source_config = payload.get("config")
        if not isinstance(source_config, dict) or _c1_reuse_signature(
            source_config
        ) != _c1_reuse_signature(config.model_dump(mode="json")):
            raise ValueError(f"Frozen C1 scoring/training signature mismatch: {path}")
        model = _new_model(config, candidate="c1_mse").to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.requires_grad_(False).eval()
        result[seed] = model
    return result


def _load_final_arrays(
    cache: Path,
    lookup: pd.DataFrame,
    train_rows: pd.DataFrame,
    in_rows: pd.DataFrame,
    heldout_rows: pd.DataFrame,
    *,
    workers: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    def load(file_id: str, heldout: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = lookup.loc[file_id]
        path = cache / str(row["augmentation_file"])
        prefix = "heldout_" if heldout else ""
        with np.load(path, allow_pickle=False) as payload:
            clean = np.asarray(payload[f"{prefix}noisy_clean"], dtype=np.float16)
            reference = np.asarray(payload[f"{prefix}reference"], dtype=np.float16)
            fault = np.asarray(payload[f"{prefix}fault_noisy"], dtype=np.float16)
        _validate_grids((clean, reference, fault), path)
        return clean, reference, fault

    def many(rows: pd.DataFrame, heldout: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ids = tuple(rows["file_id"].astype(str))
        with ThreadPoolExecutor(max_workers=min(workers, len(ids))) as executor:
            values = list(executor.map(lambda value: load(value, heldout), ids))
        columns = list(zip(*values, strict=True))
        return cast(
            tuple[np.ndarray, np.ndarray, np.ndarray], tuple(np.stack(item) for item in columns)
        )

    train = many(train_rows, False)
    in_support = many(in_rows, False)
    heldout = many(heldout_rows, True)
    return {
        "train": (train[0], train[1]),
        "in_clean": (in_support[0], in_support[1]),
        "in_fault": (in_support[2], in_support[1]),
        "heldout_clean": (heldout[0], heldout[1]),
        "heldout_fault": (heldout[2], heldout[1]),
    }


def _load_tap_arrays(
    cache: Path,
    train_rows: pd.DataFrame,
    in_rows: pd.DataFrame,
    heldout_rows: pd.DataFrame,
    *,
    workers: int,
) -> dict[str, np.ndarray]:
    def load(file_id: str, field: str) -> np.ndarray:
        path = cache / "features" / f"{hashlib.sha256(file_id.encode()).hexdigest()}.npz"
        with np.load(path, allow_pickle=False) as payload:
            value = np.asarray(payload[field], dtype=np.float16)
        _validate_grids((value,), path)
        return value

    def many(rows: pd.DataFrame, field: str) -> np.ndarray:
        ids = tuple(rows["file_id"].astype(str))
        with ThreadPoolExecutor(max_workers=min(workers, len(ids))) as executor:
            return np.stack(list(executor.map(lambda value: load(value, field), ids)))

    return {
        "train": many(train_rows, "tap0_noisy_clean"),
        "in_clean": many(in_rows, "tap0_noisy_clean"),
        "in_fault": many(in_rows, "tap0_fault_noisy"),
        "heldout_clean": many(heldout_rows, "heldout_tap0_noisy_clean"),
        "heldout_fault": many(heldout_rows, "heldout_tap0_fault_noisy"),
    }


def _score_c1_ensemble(
    *,
    models: dict[int, BandwiseReferenceAdapter],
    base_store: _BaseTokenStore,
    train_rows: pd.DataFrame,
    final_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    config: FPNAAConfig,
    union: FPEvidenceUnionConfig,
    device: str,
) -> tuple[dict[str, _ScoreSet], dict[str, np.ndarray]]:
    score_sets: dict[str, _ScoreSet] = {}
    calibration: dict[str, np.ndarray] = {}
    for seed in union.c1_seeds:
        model = models[seed]
        train_tokens = _adapt_rows(model, base_store, train_rows, config, torch.device(device))
        descriptors = _rdp(
            train_tokens, gamma=config.backend.rdp_gamma, config=config, device=device
        )
        normal = _crossfit_scores(
            descriptors,
            tuple(train_rows["file_id"].astype(str)),
            folds=union.crossfit_folds,
            kind="band",
            config=config,
            device=device,
        )
        queries: dict[str, np.ndarray] = {}
        for name in ("in_clean", "in_fault", "heldout_clean", "heldout_fault"):
            target, reference = final_arrays[name]
            adapted = _adapt_token_arrays(model, target, reference, config=config, device=device)
            query = _rdp(adapted, gamma=config.backend.rdp_gamma, config=config, device=device)
            queries[name] = _score_descriptors(
                query, descriptors, kind="band", config=config, device=device
            )
        branch = f"c1_seed{seed}"
        score_sets[branch] = _ScoreSet(normal=normal, **queries)
        calibration[branch] = normal
    return score_sets, calibration


def _score_supplementary_experts(
    *,
    final_arrays: dict[str, tuple[np.ndarray, np.ndarray]],
    tap_arrays: dict[str, np.ndarray],
    train_ids: tuple[str, ...],
    config: FPNAAConfig,
    union: FPEvidenceUnionConfig,
    device: str,
) -> dict[ExpertName, _ScoreSet]:
    result: dict[ExpertName, _ScoreSet] = {}
    for raw_name in union.supplementary_experts:
        name: ExpertName = raw_name
        if name == "tap0_rdp8_beam":
            arrays = tap_arrays
            descriptors = {
                key: _rdp(value, gamma=config.backend.rdp_gamma, config=config, device=device)
                for key, value in arrays.items()
            }
            kind: Literal["band", "global"] = "band"
        elif name == "final_rdp4_beam":
            arrays = {key: value[0] for key, value in final_arrays.items()}
            descriptors = {
                key: _rdp(value, gamma=4.0, config=config, device=device)
                for key, value in arrays.items()
            }
            kind = "band"
        elif name == "final_global_ap":
            arrays = {key: value[0] for key, value in final_arrays.items()}
            descriptors = {
                key: value.mean(axis=(1, 2), dtype=np.float32) for key, value in arrays.items()
            }
            kind = "global"
        else:  # pragma: no cover - strict config validation owns this boundary
            raise ValueError(f"Unsupported V10 expert: {name}")
        reference = descriptors["train"]
        normal = _crossfit_scores(
            reference,
            train_ids,
            folds=union.crossfit_folds,
            kind=kind,
            config=config,
            device=device,
        )
        result[name] = _ScoreSet(
            normal=normal,
            in_clean=_score_descriptors(
                descriptors["in_clean"], reference, kind=kind, config=config, device=device
            ),
            in_fault=_score_descriptors(
                descriptors["in_fault"], reference, kind=kind, config=config, device=device
            ),
            heldout_clean=_score_descriptors(
                descriptors["heldout_clean"], reference, kind=kind, config=config, device=device
            ),
            heldout_fault=_score_descriptors(
                descriptors["heldout_fault"], reference, kind=kind, config=config, device=device
            ),
        )
    return result


def _adapt_token_arrays(
    model: BandwiseReferenceAdapter,
    target: np.ndarray,
    reference: np.ndarray,
    *,
    config: FPNAAConfig,
    device: str,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    resolved_device = torch.device(device)
    use_amp = config.training.mixed_precision and resolved_device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, len(target), config.training.batch_size):
            left = torch.as_tensor(
                target[start : start + config.training.batch_size], device=resolved_device
            )
            right = torch.as_tensor(
                reference[start : start + config.training.batch_size], device=resolved_device
            )
            with torch.autocast(device_type=resolved_device.type, enabled=use_amp):
                adapted = model(left, right)
            outputs.append(adapted.float().cpu().numpy())
    return np.concatenate(outputs)


def _rdp(values: np.ndarray, *, gamma: float, config: FPNAAConfig, device: str) -> np.ndarray:
    return accelerated_rdp_pool(
        values,
        gamma=gamma,
        device=device,
        batch_size=config.frontend.inference_batch_size * 4,
        eps=config.backend.eps,
    )


def _crossfit_scores(
    descriptors: np.ndarray,
    file_ids: tuple[str, ...],
    *,
    folds: int,
    kind: Literal["band", "global"],
    config: FPNAAConfig,
    device: str,
) -> np.ndarray:
    if len(descriptors) != len(file_ids):
        raise ValueError("Cross-fit descriptors and file IDs differ in length")
    order = np.argsort(
        np.asarray([hashlib.sha256(value.encode()).hexdigest() for value in file_ids]),
        kind="stable",
    )
    assignment = np.empty(len(file_ids), dtype=np.int64)
    assignment[order] = np.arange(len(file_ids)) % folds
    result = np.full(len(file_ids), np.nan, dtype=np.float64)
    for fold in range(folds):
        query = assignment == fold
        reference = ~query
        if query.sum() == 0 or reference.sum() <= config.backend.local_density_neighbors:
            raise ValueError("Cross-fit fold has insufficient query/reference rows")
        result[query] = _score_descriptors(
            descriptors[query], descriptors[reference], kind=kind, config=config, device=device
        )
    if not np.isfinite(result).all():
        raise RuntimeError("Cross-fit scoring produced non-finite values")
    return result


def _score_descriptors(
    queries: np.ndarray,
    references: np.ndarray,
    *,
    kind: Literal["band", "global"],
    config: FPNAAConfig,
    device: str,
) -> np.ndarray:
    if kind == "band":
        scores, _ = accelerated_beam_scores(
            queries,
            references,
            neighbors=config.backend.local_density_neighbors,
            device=device,
            eps=config.backend.eps,
        )
    else:
        scores, _ = accelerated_global_knn_scores(
            queries,
            references,
            neighbors=config.backend.local_density_neighbors,
            device=device,
            eps=config.backend.eps,
        )
    if not np.isfinite(scores).all():
        raise RuntimeError("V10 expert scoring produced non-finite values")
    return scores


def _calibrate_score_set(
    scores: dict[str, _ScoreSet] | _ScoreSet,
    normal_references: dict[str, np.ndarray],
    *,
    ensemble: bool,
    epsilon: float = 1.0e-6,
) -> _ScoreSet:
    branches = scores if isinstance(scores, dict) else {next(iter(normal_references)): scores}
    fields = ("normal", "in_clean", "in_fault", "heldout_clean", "heldout_fault")
    calibrated: dict[str, np.ndarray] = {}
    for field in fields:
        values = []
        for name, score_set in branches.items():
            values.append(
                empirical_tail_evidence(
                    getattr(score_set, field), normal_references[name], epsilon=epsilon
                )
            )
        calibrated[field] = np.mean(values, axis=0) if ensemble else values[0]
    return _ScoreSet(**calibrated)


def _fuse_score_set(
    base: _ScoreSet,
    experts: dict[str, _ScoreSet],
    penalties: dict[str, float],
    active: tuple[str, ...],
) -> _ScoreSet:
    fields = ("normal", "in_clean", "in_fault", "heldout_clean", "heldout_fault")
    return _ScoreSet(
        **{
            field: monotone_evidence_union(
                getattr(base, field),
                {name: getattr(values, field) for name, values in experts.items()},
                penalties,
                active_experts=active,
            )
            for field in fields
        }
    )


def _score_set_counterfactual_gain(
    candidate: _ScoreSet, base: _ScoreSet
) -> tuple[np.ndarray, np.ndarray]:
    in_gain = (candidate.in_fault - candidate.in_clean) - (base.in_fault - base.in_clean)
    heldout_gain = (candidate.heldout_fault - candidate.heldout_clean) - (
        base.heldout_fault - base.heldout_clean
    )
    return in_gain, heldout_gain


def _summarize_certificates(
    certificates: pd.DataFrame,
    machine_results: dict[str, dict[str, object]],
    union: FPEvidenceUnionConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, group in certificates.groupby("name", sort=True):
        rows.append(
            {
                "expert": name,
                "machines": len(group),
                "machine_pass_fraction": float(group["eligible"].mean()),
                "globally_authorized": bool(
                    float(group["eligible"].mean()) >= union.minimum_machine_pass_fraction
                ),
                "clean_activation_fraction_mean": float(group["clean_activation_fraction"].mean()),
                "in_support_gain_median": float(group["in_support_gain_median"].median()),
                "in_support_gain_q05_worst_machine": float(group["in_support_gain_q05"].min()),
                "heldout_gain_median": float(group["heldout_gain_median"].median()),
                "heldout_gain_q05_worst_machine": float(group["heldout_gain_q05"].min()),
            }
        )
    all_in = np.concatenate(
        [cast(np.ndarray, value["in_support_gain"]) for value in machine_results.values()]
    )
    all_heldout = np.concatenate(
        [cast(np.ndarray, value["heldout_gain"]) for value in machine_results.values()]
    )
    rows.append(
        {
            "expert": "candidate_union",
            "machines": len(machine_results),
            "machine_pass_fraction": float(
                np.mean([bool(value["active_experts"]) for value in machine_results.values()])
            ),
            "globally_authorized": False,
            "clean_activation_fraction_mean": float(
                np.mean(
                    [
                        float(cast(float, value["normal_activation_fraction"]))
                        for value in machine_results.values()
                    ]
                )
            ),
            "in_support_gain_median": float(np.median(all_in)),
            "in_support_gain_q05_worst_machine": float(np.quantile(all_in, 0.05)),
            "heldout_gain_median": float(np.median(all_heldout)),
            "heldout_gain_q05_worst_machine": float(np.quantile(all_heldout, 0.05)),
        }
    )
    return pd.DataFrame(rows)


def _make_gate(
    summary: pd.DataFrame,
    machine_results: dict[str, dict[str, object]],
    union: FPEvidenceUnionConfig,
    policy_path: Path,
) -> dict[str, object]:
    experts = summary.loc[summary["expert"] != "candidate_union"]
    candidate = summary.loc[summary["expert"] == "candidate_union"].iloc[0]
    authorized = int(experts["globally_authorized"].sum())
    machine_coverage = float(candidate["machine_pass_fraction"])
    monotone = all(bool(value["monotone"]) for value in machine_results.values())
    policy = _read_json(policy_path)
    checks = {
        "active_expert_count": authorized >= union.minimum_active_experts,
        "machine_coverage": machine_coverage >= union.minimum_machine_pass_fraction,
        "clean_tail_budget": float(candidate["clean_activation_fraction_mean"])
        <= union.maximum_clean_activation_fraction,
        "in_support_gain_median": float(candidate["in_support_gain_median"])
        >= union.minimum_in_support_evidence_gain_median,
        "in_support_gain_q05": float(candidate["in_support_gain_q05_worst_machine"])
        >= union.minimum_in_support_evidence_gain_q05,
        "heldout_gain_median": float(candidate["heldout_gain_median"])
        >= union.minimum_heldout_evidence_gain_median,
        "heldout_gain_q05": float(candidate["heldout_gain_q05_worst_machine"])
        >= union.minimum_heldout_evidence_gain_q05,
        "per_clip_base_monotonicity": monotone,
        "development_anomaly_labels_absent": policy.get("development_anomaly_labels_read") is False,
        "selection_sources_frozen": policy.get("selection_uses")
        == ["train_normal", "in_support_pseudo_fault"],
        "heldout_family_gate_only": policy.get("heldout_family_role") == "gate_only",
    }
    return {
        "schema_version": 1,
        "gate": "V10_M_counterfactual_certified_monotone_evidence_union",
        "passed": all(checks.values()),
        "checks": checks,
        "criteria": {
            "minimum_active_experts": union.minimum_active_experts,
            "minimum_machine_pass_fraction": union.minimum_machine_pass_fraction,
            "maximum_clean_activation_fraction": union.maximum_clean_activation_fraction,
            "minimum_in_support_evidence_gain_median": union.minimum_in_support_evidence_gain_median,
            "minimum_in_support_evidence_gain_q05": union.minimum_in_support_evidence_gain_q05,
            "minimum_heldout_evidence_gain_median": union.minimum_heldout_evidence_gain_median,
            "minimum_heldout_evidence_gain_q05": union.minimum_heldout_evidence_gain_q05,
        },
        "metrics": {
            "authorized_experts": authorized,
            "machine_coverage": machine_coverage,
            "clean_activation_fraction": float(candidate["clean_activation_fraction_mean"]),
            "in_support_gain_median": float(candidate["in_support_gain_median"]),
            "in_support_gain_q05": float(candidate["in_support_gain_q05_worst_machine"]),
            "heldout_gain_median": float(candidate["heldout_gain_median"]),
            "heldout_gain_q05": float(candidate["heldout_gain_q05_worst_machine"]),
        },
        "contract_sha256": policy.get("contract_sha256"),
        "artifacts": policy.get("artifacts"),
        "policy_sha256": _sha256(policy_path),
        "authorization": (
            "A pass authorizes one frozen V10 development screening. It is not a performance "
            "result and does not authorize LOMO or evaluation-set labels."
        ),
    }


def _validate_grids(values: tuple[np.ndarray, ...], path: Path) -> None:
    if any(value.ndim != 3 or not np.isfinite(value).all() for value in values):
        raise ValueError(f"Invalid V10 token grid: {path}")
    if len({value.shape for value in values}) != 1:
        raise ValueError(f"Inconsistent V10 token grids: {path}")


def _ensure_contract(path: Path, payload: dict[str, object]) -> None:
    if path.is_file():
        if _canonical_sha(_read_json(path)) != _canonical_sha(payload):
            raise ValueError(f"V10 preflight contract mismatch: {path}")
        return
    _atomic_json(path, payload)


def _validate_completed_outputs(
    *,
    contract_path: Path,
    gate_path: Path,
    summary_path: Path,
    certificates_path: Path,
    policy_path: Path,
    calibration_path: Path,
) -> None:
    """Reject stale or modified artifacts before returning a cached result."""
    policy = _read_json(policy_path)
    if policy.get("contract_sha256") != _sha256(contract_path):
        raise ValueError("Completed V10 policy does not match its contract")
    artifacts = policy.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Completed V10 policy is missing artifact hashes")
    expected = {
        "summary_sha256": summary_path,
        "expert_certificates_sha256": certificates_path,
        "normal_calibration_sha256": calibration_path,
    }
    for key, path in expected.items():
        if artifacts.get(key) != _sha256(path):
            raise ValueError(f"Completed V10 artifact does not match policy: {path.name}")
    gate = _read_json(gate_path)
    if not isinstance(gate.get("passed"), bool):
        raise ValueError("Completed V10 gate has no boolean decision")
    if gate.get("policy_sha256") != _sha256(policy_path):
        raise ValueError("Completed V10 gate does not match its policy")


def _write_progress(output: Path, *, stage: str, completed: int, total: int, machine: str) -> None:
    _atomic_text(
        output / "progress.env",
        f"stage={stage}\ncompleted_machines={completed}\ntotal_machines={total}\n"
        f"current_machine={machine}\nupdated_utc={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
    )


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: object) -> str:
    value = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(value).hexdigest()
