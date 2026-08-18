"""Normal-only BEATs depth observability audit for the FP-NAA successor."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import soundfile as sf

from care_asd.data.fp_naa_augmentation_cache import _make_plans, _prepare
from care_asd.fp_naa_config import FPNAAConfig, load_fp_naa_config
from care_asd.models.beats_frontend import OfficialBEATsFrontend, fixed_duration_waveform


class TappedFrontend(Protocol):
    def extract_encoder_taps(
        self, waveforms: np.ndarray, *, taps: tuple[int, ...]
    ) -> dict[int, np.ndarray]: ...


TappedFrontendFactory = Callable[[FPNAAConfig, Path, Path, str], TappedFrontend]


@dataclass(frozen=True)
class FPNaaObservabilityResult:
    output_directory: Path
    summary_path: Path
    family_summary_path: Path
    gate_path: Path
    gate_passed: bool


@dataclass(frozen=True)
class _PreparedProbe:
    shard: Path
    file_id: str
    family: str
    waveforms: np.ndarray
    comparisons: tuple[tuple[str, str, int, int, int, int], ...]


def run_fp_naa_observability_probe(
    *,
    base_cache_directory: str | Path,
    audio_root: str | Path,
    cache_directory: str | Path,
    output_directory: str | Path,
    config_path: str | Path,
    beats_source_directory: str | Path,
    checkpoint_path: str | Path,
    workers: int = 12,
    device: str = "cuda",
    frontend_factory: TappedFrontendFactory | None = None,
) -> FPNaaObservabilityResult:
    """Measure counterfactual retention at frozen BEATs depths without anomaly labels."""
    if not 0 <= workers <= 16:
        raise ValueError("workers must be in [0, 16]")
    base_cache = Path(base_cache_directory).resolve()
    audio_root_path = Path(audio_root).resolve()
    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    config_source = Path(config_path).resolve()
    beats_source = Path(beats_source_directory).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    config = load_fp_naa_config(config_source)
    if config.observability is None:
        raise ValueError("FP-NAA config does not define an observability probe")
    taps = tuple(config.observability.encoder_taps)
    if _sha256(checkpoint) != config.provenance.checkpoint_sha256:
        raise ValueError("BEATs checkpoint SHA-256 does not match the observability config")
    index_path = base_cache / "index.parquet"
    metadata_path = base_cache / "cache.json"
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Completed base BEATs cache not found: {base_cache}")

    output.mkdir(parents=True, exist_ok=True)
    completed_gate = output / "gate.json"
    completed_summary = output / "tap_summary.csv"
    completed_family = output / "family_summary.csv"
    if completed_gate.is_file() and completed_summary.is_file() and completed_family.is_file():
        gate = json.loads(completed_gate.read_text(encoding="utf-8"))
        return FPNaaObservabilityResult(
            output, completed_summary, completed_family, completed_gate, bool(gate["passed"])
        )

    frame = pd.read_parquet(index_path)
    train = frame.loc[frame["dataset_split"] == "dev_train"].copy()
    if train.empty or train["file_id"].duplicated().any() or set(train["condition"]) != {"normal"}:
        raise ValueError("Observability probing requires unique normal development-training rows")
    train = train.sort_values("file_id", kind="stable").reset_index(drop=True)

    contract = {
        "schema_version": 1,
        "kind": "fp_naa_encoder_observability",
        "config_sha256": _sha256(config_source),
        "base_cache_metadata_sha256": _sha256(metadata_path),
        "base_cache_index_sha256": _sha256(index_path),
        "checkpoint_sha256": config.provenance.checkpoint_sha256,
        "beats_commit": config.provenance.beats_commit,
        "encoder_taps": list(taps),
        "selection_rule": config.observability.selection_rule,
        "selection_uses": "in_support_pseudo_faults_only",
        "heldout_family_use": "diagnostic_only",
    }
    contract_sha = _canonical_sha(contract)
    cache.mkdir(parents=True, exist_ok=True)
    shards = cache / "shards"
    shards.mkdir(exist_ok=True)
    cache_contract = cache / "contract.json"
    if cache_contract.is_file():
        existing = json.loads(cache_contract.read_text(encoding="utf-8"))
        if _canonical_sha(existing) != contract_sha:
            raise ValueError("Observability cache contract mismatch")
    else:
        _atomic_json(cache_contract, contract)
    _atomic_json(output / "contract.json", contract)

    plans = _make_plans(train, audio_root_path, shards, config)
    pending = [plan for plan in plans if not plan.feature_path.with_suffix(".json").is_file()]
    completed = len(plans) - len(pending)
    _write_progress(output, completed=completed, total=len(plans), stage="extract")
    frontend: TappedFrontend | None = None
    batch_size = config.frontend.inference_batch_size
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        if workers == 0:
            prepared = [_prepare_probe(plan, config) for plan in batch]
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                prepared = list(executor.map(lambda plan: _prepare_probe(plan, config), batch))
        if frontend is None:
            factory = frontend_factory or _default_frontend_factory
            frontend = factory(config, beats_source, checkpoint, device)
        offsets: list[int] = []
        cursor = 0
        for item in prepared:
            offsets.append(cursor)
            cursor += len(item.waveforms)
        waveforms = np.concatenate([item.waveforms for item in prepared], axis=0)
        extracted = frontend.extract_encoder_taps(waveforms, taps=taps)
        for item, offset in zip(prepared, offsets, strict=True):
            item_rows = _diagnose_item(item, offset=offset, extracted=extracted, taps=taps)
            _atomic_json(item.shard, {"schema_version": 1, "rows": item_rows})
        completed += len(prepared)
        _write_progress(output, completed=completed, total=len(plans), stage="extract")

    all_rows: list[dict[str, object]] = []
    for plan in plans:
        shard = plan.feature_path.with_suffix(".json")
        payload = json.loads(shard.read_text(encoding="utf-8"))
        all_rows.extend(payload["rows"])
    diagnostics = pd.DataFrame(all_rows)
    if len(diagnostics) < len(plans) * len(taps):
        raise RuntimeError("Observability diagnostics are incomplete")

    summary = _summarize_taps(diagnostics, taps=taps, config=config)
    family_summary = _summarize_families(diagnostics)
    selected = summary.loc[summary["eligible_in_support"], "tap"]
    selected_tap = int(selected.max()) if not selected.empty else None
    checks = {
        f"tap_{int(row.tap)}": bool(row.eligible_in_support)
        for row in summary.itertuples(index=False)
    }
    gate = {
        "schema_version": 1,
        "gate": "V6_frontend_observability_preflight",
        "passed": selected_tap is not None,
        "selected_tap": selected_tap,
        "selection_rule": "deepest eligible tap using in-support pseudo-faults only",
        "criteria": {
            "in_support_retention_median_minimum": config.gates.fault_delta_retention_median_minimum,
            "in_support_retention_q05_minimum": config.gates.fault_delta_retention_q05_minimum,
        },
        "checks": checks,
        "heldout_family": {
            "name": config.augmentation.heldout_fault_family,
            "role": "diagnostic only; never used for tap selection",
        },
        "note": (
            "A pass only authorizes implementation of a pre-encoder candidate. It is not a G2 "
            "performance result and does not authorize LOMO."
        ),
    }
    _atomic_csv(completed_summary, summary)
    _atomic_csv(completed_family, family_summary)
    _atomic_json(completed_gate, gate)
    _atomic_json(
        cache / "cache.json",
        {
            **contract,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "clips": len(plans),
            "diagnostic_rows": len(diagnostics),
            "contract_sha256": contract_sha,
        },
    )
    _write_progress(output, completed=len(plans), total=len(plans), stage="complete")
    return FPNaaObservabilityResult(
        output, completed_summary, completed_family, completed_gate, selected_tap is not None
    )


def _prepare_probe(plan: object, config: FPNAAConfig) -> _PreparedProbe:
    prepared = _prepare(plan, config)  # type: ignore[arg-type]
    target, sample_rate = sf.read(prepared.plan.target_audio, dtype="float32", always_2d=True)
    if sample_rate != config.frontend.sample_rate or target.shape[1] < 2:
        raise ValueError("Observability source audio must be stereo at the configured sample rate")
    teacher_clean = fixed_duration_waveform(
        target[:, 0],
        sample_rate=sample_rate,
        duration_seconds=config.frontend.duration_seconds,
    )
    named = {name: value for name, value in zip(prepared.names, prepared.waveforms, strict=True)}
    waveforms = [
        teacher_clean,
        named["noisy_clean"],
        named["fault_teacher"],
        named["fault_noisy"],
    ]
    comparisons: list[tuple[str, str, int, int, int, int]] = [
        ("in_support", prepared.plan.family, 0, 1, 2, 3)
    ]
    if prepared.plan.heldout:
        start = len(waveforms)
        waveforms.extend(
            [
                teacher_clean,
                named["heldout_noisy_clean"],
                named["heldout_fault_teacher"],
                named["heldout_fault_noisy"],
            ]
        )
        comparisons.append(
            (
                "heldout",
                config.augmentation.heldout_fault_family,
                start,
                start + 1,
                start + 2,
                start + 3,
            )
        )
    return _PreparedProbe(
        shard=prepared.plan.feature_path.with_suffix(".json"),
        file_id=prepared.plan.file_id,
        family=prepared.plan.family,
        waveforms=np.stack(waveforms),
        comparisons=tuple(comparisons),
    )


def _diagnose_item(
    item: _PreparedProbe,
    *,
    offset: int,
    extracted: dict[int, np.ndarray],
    taps: tuple[int, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    eps = 1.0e-12
    for tap in taps:
        values = extracted[tap]
        for fault_set, family, clean_i, noisy_i, teacher_fault_i, noisy_fault_i in item.comparisons:
            teacher_delta = values[offset + teacher_fault_i].astype(np.float64) - values[
                offset + clean_i
            ].astype(np.float64)
            noisy_delta = values[offset + noisy_fault_i].astype(np.float64) - values[
                offset + noisy_i
            ].astype(np.float64)
            teacher_norm = float(np.linalg.norm(teacher_delta.ravel()))
            noisy_norm = float(np.linalg.norm(noisy_delta.ravel()))
            ratio = noisy_norm / max(teacher_norm, eps)
            retention = float(np.exp(-abs(np.log(max(ratio, eps)))))
            denominator = max(teacher_norm * noisy_norm, eps)
            direction = float(np.sum(teacher_delta * noisy_delta) / denominator)
            transport_error = float(
                np.linalg.norm((noisy_delta - teacher_delta).ravel()) / max(teacher_norm, eps)
            )
            rows.append(
                {
                    "file_id": item.file_id,
                    "tap": tap,
                    "fault_set": fault_set,
                    "fault_family": family,
                    "retention": retention,
                    "direction_cosine": direction,
                    "teacher_delta_norm": teacher_norm,
                    "noisy_delta_norm": noisy_norm,
                    "transport_relative_error": transport_error,
                }
            )
    return rows


def _summarize_taps(
    diagnostics: pd.DataFrame, *, taps: tuple[int, ...], config: FPNAAConfig
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tap in taps:
        selected = diagnostics.loc[diagnostics["tap"] == tap]
        in_support = selected.loc[selected["fault_set"] == "in_support"]
        heldout = selected.loc[selected["fault_set"] == "heldout"]
        median = float(in_support["retention"].median())
        q05 = float(in_support["retention"].quantile(0.05))
        rows.append(
            {
                "tap": tap,
                "in_support_clips": len(in_support),
                "in_support_retention_median": median,
                "in_support_retention_q05": q05,
                "in_support_direction_median": float(in_support["direction_cosine"].median()),
                "in_support_transport_error_median": float(
                    in_support["transport_relative_error"].median()
                ),
                "heldout_clips": len(heldout),
                "heldout_retention_median": float(heldout["retention"].median()),
                "heldout_retention_q05": float(heldout["retention"].quantile(0.05)),
                "eligible_in_support": (
                    median >= config.gates.fault_delta_retention_median_minimum
                    and q05 >= config.gates.fault_delta_retention_q05_minimum
                ),
            }
        )
    return pd.DataFrame(rows)


def _summarize_families(diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = diagnostics.groupby(["tap", "fault_set", "fault_family"], sort=True)
    for (tap, fault_set, family), group in grouped:
        rows.append(
            {
                "tap": int(tap),
                "fault_set": str(fault_set),
                "fault_family": str(family),
                "clips": len(group),
                "retention_median": float(group["retention"].median()),
                "retention_q05": float(group["retention"].quantile(0.05)),
                "direction_median": float(group["direction_cosine"].median()),
                "transport_error_median": float(group["transport_relative_error"].median()),
                "transport_error_q90": float(group["transport_relative_error"].quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def _default_frontend_factory(
    config: FPNAAConfig, source: Path, checkpoint: Path, device: str
) -> TappedFrontend:
    return OfficialBEATsFrontend(
        source_directory=source,
        checkpoint_path=checkpoint,
        device=device,
        frequency_patches=config.frontend.frequency_patches,
        mixed_precision=config.frontend.inference_mixed_precision,
    )


def _write_progress(output: Path, *, completed: int, total: int, stage: str) -> None:
    text = (
        f"stage={stage}\ncompleted_clips={completed}\ntotal_clips={total}\n"
        f"updated_utc={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
    )
    _atomic_text(output / "progress.env", text)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _canonical_sha(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
