"""Waveform-grounded reference-leakage cache for FP-NAA safety evaluation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import soundfile as sf

from care_asd.data.fp_naa_augmentation_cache import _stable_seed
from care_asd.fp_naa_config import FPNAAConfig, load_fp_naa_config
from care_asd.fp_naa_reference_safety_config import (
    FPNAAReferenceSafetyConfig,
    load_fp_naa_reference_safety_config,
)
from care_asd.models.beats_frontend import OfficialBEATsFrontend, fixed_duration_waveform
from care_asd.signal.pseudo_faults import FaultFamily, inject_pseudo_fault, mix_paired_noise

LEAKAGE_NAMES = ("low", "medium", "high")


class ReferenceSafetyFrontend(Protocol):
    def extract(self, waveforms: np.ndarray) -> np.ndarray: ...


ReferenceSafetyFrontendFactory = Callable[[FPNAAConfig, Path, Path, str], ReferenceSafetyFrontend]


@dataclass(frozen=True)
class FPNaaReferenceSafetyCache:
    root: Path
    index_path: Path
    metadata_path: Path
    silence_reference_path: Path
    clips: int


@dataclass(frozen=True)
class _SafetyPlan:
    file_id: str
    target_audio: Path
    donor_audio: Path
    feature_path: Path
    fault_seed: int
    noise_snr_db: float
    fault_delta_level_db: float


@dataclass(frozen=True)
class _PreparedSafetyReferences:
    plan: _SafetyPlan
    waveforms: np.ndarray
    metadata: dict[str, object]


def build_fp_naa_reference_safety_cache(
    *,
    base_cache_directory: str | Path,
    augmentation_cache_directory: str | Path,
    audio_root: str | Path,
    output_directory: str | Path,
    config_path: str | Path,
    safety_config_path: str | Path,
    beats_source_directory: str | Path,
    checkpoint_path: str | Path,
    workers: int = 12,
    device: str = "cuda",
    frontend_factory: ReferenceSafetyFrontendFactory | None = None,
) -> FPNaaReferenceSafetyCache:
    """Cache physically mixed low/medium/high target leakage in the reference channel."""
    if not 0 <= workers <= 16:
        raise ValueError("workers must be in [0, 16]")
    base_cache = Path(base_cache_directory).resolve()
    augmentation_cache = Path(augmentation_cache_directory).resolve()
    root = Path(audio_root).resolve()
    output = Path(output_directory).resolve()
    config_source = Path(config_path).resolve()
    safety_source = Path(safety_config_path).resolve()
    beats_source = Path(beats_source_directory).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    config = load_fp_naa_config(config_source)
    safety = load_fp_naa_reference_safety_config(safety_source)
    if _sha256(checkpoint) != config.provenance.checkpoint_sha256:
        raise ValueError("BEATs checkpoint SHA-256 does not match FP-NAA config")
    base_metadata = base_cache / "cache.json"
    base_index = base_cache / "index.parquet"
    augmentation_metadata = augmentation_cache / "cache.json"
    augmentation_index = augmentation_cache / "index.parquet"
    for path in (base_metadata, base_index, augmentation_metadata, augmentation_index):
        if not path.is_file():
            raise FileNotFoundError(f"Required completed FP-NAA cache artifact not found: {path}")
    base_metadata_payload = json.loads(base_metadata.read_text(encoding="utf-8"))
    augmentation_metadata_payload = json.loads(augmentation_metadata.read_text(encoding="utf-8"))
    if base_metadata_payload.get("checkpoint_sha256") != config.provenance.checkpoint_sha256:
        raise ValueError("Base cache checkpoint does not match the frozen FP-NAA config")
    if (
        augmentation_metadata_payload.get("checkpoint_sha256")
        != config.provenance.checkpoint_sha256
    ):
        raise ValueError("Augmentation cache checkpoint does not match the frozen FP-NAA config")
    contract = _contract(
        config=config,
        safety=safety,
        base_metadata=base_metadata,
        augmentation_metadata=augmentation_metadata,
    )
    metadata_path = output / "cache.json"
    index_path = output / "index.parquet"
    silence_path = output / "silence_reference.npy"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("reference_safety_contract_sha256") != contract:
            raise ValueError("Completed reference-safety cache contract mismatch")
        if not index_path.is_file() or not silence_path.is_file():
            raise FileNotFoundError("Completed reference-safety cache is incomplete")
        return FPNaaReferenceSafetyCache(
            output,
            index_path,
            metadata_path,
            silence_path,
            int(metadata["clips"]),
        )

    base = pd.read_parquet(base_index)
    augmentation = pd.read_parquet(augmentation_index)
    required = {
        "file_id",
        "relative_path",
        "donor_file_id",
        "fault_seed",
        "requested_noise_snr_db",
        "requested_fault_delta_level_db",
        "heldout",
        "heldout_fault_family",
    }
    missing = sorted(required.difference(augmentation.columns))
    if missing:
        raise ValueError(f"Augmentation index is missing columns: {', '.join(missing)}")
    heldout = augmentation.loc[augmentation["heldout"].astype(bool)].copy()
    heldout = heldout.sort_values("file_id", kind="stable").reset_index(drop=True)
    if heldout.empty or heldout["file_id"].duplicated().any():
        raise ValueError("Reference-safety cache requires unique held-out fault rows")
    if set(heldout["heldout_fault_family"].astype(str)) != {
        config.augmentation.heldout_fault_family
    }:
        raise ValueError("Reference-safety population is not the frozen held-out fault family")
    donor_paths = {
        str(row.file_id): str(row.relative_path)
        for row in base.loc[base["dataset_split"] == "dev_train"].itertuples(index=False)
    }
    output.mkdir(parents=True, exist_ok=True)
    features = output / "features"
    features.mkdir(exist_ok=True)
    plans = _make_plans(heldout, donor_paths, root, features)
    pending = [plan for plan in plans if not plan.feature_path.is_file()]
    _write_progress(output, completed=len(plans) - len(pending), total=len(plans), stage="prepare")
    if workers == 0:
        prepared = [_prepare(plan, config, safety) for plan in pending]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared = list(executor.map(lambda plan: _prepare(plan, config, safety), pending))
    factory = frontend_factory or _default_frontend_factory
    frontend: ReferenceSafetyFrontend | None = None
    completed = len(plans) - len(pending)
    for item in prepared:
        if frontend is None:
            frontend = factory(config, beats_source, checkpoint, device)
        grids = frontend.extract(item.waveforms)
        if grids.ndim != 4 or len(grids) != 2 * len(LEAKAGE_NAMES):
            raise RuntimeError(f"Unexpected reference-safety BEATs shape: {grids.shape}")
        payload: dict[str, np.ndarray] = {}
        cursor = 0
        for level in LEAKAGE_NAMES:
            payload[f"leakage_{level}_clean_reference"] = grids[cursor].astype(np.float16)
            payload[f"leakage_{level}_fault_reference"] = grids[cursor + 1].astype(np.float16)
            cursor += 2
        payload["metadata_json"] = np.asarray(
            json.dumps(item.metadata, sort_keys=True, separators=(",", ":"))
        )
        _write_feature(item.plan.feature_path, payload)
        completed += 1
        _write_progress(output, completed=completed, total=len(plans), stage="extract")
    if frontend is None:
        frontend = factory(config, beats_source, checkpoint, device)
    if not silence_path.is_file():
        samples = round(config.frontend.sample_rate * config.frontend.duration_seconds)
        silence = frontend.extract(np.zeros((1, samples), dtype=np.float32))
        if silence.ndim != 4 or len(silence) != 1:
            raise RuntimeError(f"Unexpected silence BEATs shape: {silence.shape}")
        _atomic_npy(silence_path, silence[0].astype(np.float16))

    for plan in plans:
        _validate_feature(plan.feature_path, expected_file_id=plan.file_id)
    silence_grid = np.load(silence_path, allow_pickle=False)
    if silence_grid.ndim != 3 or silence_grid.dtype != np.float16:
        raise ValueError("Reference-safety silence grid is invalid")

    indexed = heldout.copy()
    indexed["safety_feature_file"] = [f"features/{plan.feature_path.name}" for plan in plans]
    temporary_index = output / "index.parquet.tmp"
    indexed.to_parquet(temporary_index, index=False)
    os.replace(temporary_index, index_path)
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clips": len(indexed),
        "fault_family": config.augmentation.heldout_fault_family,
        "leakage_machine_to_noise_db": safety.leakage_machine_to_noise_db,
        "reference_safety_contract_sha256": contract,
        "base_cache_metadata_sha256": _sha256(base_metadata),
        "augmentation_cache_metadata_sha256": _sha256(augmentation_metadata),
        "checkpoint_sha256": config.provenance.checkpoint_sha256,
    }
    _atomic_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _write_progress(output, completed=len(plans), total=len(plans), stage="complete")
    return FPNaaReferenceSafetyCache(
        output,
        index_path,
        metadata_path,
        silence_path,
        len(indexed),
    )


def _make_plans(
    heldout: pd.DataFrame,
    donor_paths: dict[str, str],
    audio_root: Path,
    features: Path,
) -> list[_SafetyPlan]:
    plans: list[_SafetyPlan] = []
    for row in heldout.itertuples(index=False):
        donor_id = str(row.donor_file_id)
        if donor_id not in donor_paths:
            raise ValueError(f"Reference donor is absent from base train index: {donor_id}")
        file_id = str(row.file_id)
        plans.append(
            _SafetyPlan(
                file_id=file_id,
                target_audio=_safe_audio_path(audio_root, str(row.relative_path)),
                donor_audio=_safe_audio_path(audio_root, donor_paths[donor_id]),
                feature_path=features / f"{hashlib.sha256(file_id.encode()).hexdigest()}.npz",
                fault_seed=int(row.fault_seed),
                noise_snr_db=float(row.requested_noise_snr_db),
                fault_delta_level_db=float(row.requested_fault_delta_level_db),
            )
        )
    return plans


def _prepare(
    plan: _SafetyPlan,
    config: FPNAAConfig,
    safety: FPNAAReferenceSafetyConfig,
) -> _PreparedSafetyReferences:
    target, target_rate = sf.read(plan.target_audio, dtype="float32", always_2d=True)
    donor, donor_rate = sf.read(plan.donor_audio, dtype="float32", always_2d=True)
    sample_rate = config.frontend.sample_rate
    if target_rate != sample_rate or donor_rate != sample_rate:
        raise ValueError("Reference-safety audio must use the frozen sample rate")
    if min(target.shape[1], donor.shape[1]) < 2:
        raise ValueError("Reference-safety audio must be stereo")
    clean = fixed_duration_waveform(
        target[:, 0],
        sample_rate=sample_rate,
        duration_seconds=config.frontend.duration_seconds,
    )
    donor_far = fixed_duration_waveform(
        donor[:, 1],
        sample_rate=sample_rate,
        duration_seconds=config.frontend.duration_seconds,
    )
    fault = inject_pseudo_fault(
        clean,
        sample_rate=sample_rate,
        family=cast(FaultFamily, config.augmentation.heldout_fault_family),
        seed=_stable_seed(plan.fault_seed, plan.file_id, "heldout"),
        delta_level_db=plan.fault_delta_level_db,
        peak_limit=config.augmentation.peak_limit,
    )
    paired = mix_paired_noise(
        clean,
        fault.waveform,
        donor_far,
        snr_db=plan.noise_snr_db,
        peak_limit=config.augmentation.peak_limit,
    )
    waveforms: list[np.ndarray] = []
    actual: dict[str, float] = {}
    for level in LEAKAGE_NAMES:
        clean_reference, fault_reference, actual_db = _reference_leakage_pair(
            noise=paired.reference_noise,
            clean=clean,
            faulty=fault.waveform,
            machine_to_noise_db=safety.leakage_machine_to_noise_db[level],
            peak_limit=config.augmentation.peak_limit,
        )
        waveforms.extend((clean_reference, fault_reference))
        actual[level] = actual_db
    return _PreparedSafetyReferences(
        plan=plan,
        waveforms=np.stack(waveforms),
        metadata={
            "file_id": plan.file_id,
            "fault_family": config.augmentation.heldout_fault_family,
            "fault_seed": _stable_seed(plan.fault_seed, plan.file_id, "heldout"),
            "actual_machine_to_noise_db": actual,
        },
    )


def _reference_leakage_pair(
    *,
    noise: np.ndarray,
    clean: np.ndarray,
    faulty: np.ndarray,
    machine_to_noise_db: float,
    peak_limit: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    noise_rms = _rms(noise)
    clean_rms = _rms(clean)
    if min(noise_rms, clean_rms) <= 1.0e-10:
        raise ValueError("Reference leakage requires non-silent noise and machine signals")
    gain = noise_rms * 10.0 ** (machine_to_noise_db / 20.0) / clean_rms
    clean_reference = np.asarray(noise, dtype=np.float64) + gain * np.asarray(
        clean, dtype=np.float64
    )
    fault_reference = np.asarray(noise, dtype=np.float64) + gain * np.asarray(
        faulty, dtype=np.float64
    )
    peak = max(float(np.max(np.abs(clean_reference))), float(np.max(np.abs(fault_reference))))
    shared_scale = min(1.0, peak_limit / max(peak, 1.0e-12))
    clean_reference *= shared_scale
    fault_reference *= shared_scale
    machine_component_rms = _rms(gain * np.asarray(clean) * shared_scale)
    noise_component_rms = _rms(np.asarray(noise) * shared_scale)
    actual = 20.0 * np.log10(machine_component_rms / max(noise_component_rms, 1.0e-12))
    return (
        clean_reference.astype(np.float32),
        fault_reference.astype(np.float32),
        float(actual),
    )


def _contract(
    *,
    config: FPNAAConfig,
    safety: FPNAAReferenceSafetyConfig,
    base_metadata: Path,
    augmentation_metadata: Path,
) -> str:
    payload = {
        "schema_version": 1,
        "config": config.model_dump(mode="json"),
        "safety": safety.model_dump(mode="json"),
        "base_cache_metadata_sha256": _sha256(base_metadata),
        "augmentation_cache_metadata_sha256": _sha256(augmentation_metadata),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _default_frontend_factory(
    config: FPNAAConfig,
    source: Path,
    checkpoint: Path,
    device: str,
) -> ReferenceSafetyFrontend:
    return OfficialBEATsFrontend(
        source_directory=source,
        checkpoint_path=checkpoint,
        device=device,
        frequency_patches=config.frontend.frequency_patches,
        mixed_precision=config.training.mixed_precision,
    )


def _safe_audio_path(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe audio relative path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Audio path escapes root: {relative}")
    return path


def _write_feature(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)


def _validate_feature(path: Path, *, expected_file_id: str) -> None:
    names = {
        f"leakage_{level}_{state}_reference"
        for level in LEAKAGE_NAMES
        for state in ("clean", "fault")
    }
    with np.load(path, allow_pickle=False) as payload:
        if not names.issubset(payload.files) or "metadata_json" not in payload.files:
            raise ValueError(f"Reference-safety feature is incomplete: {path}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        if str(metadata.get("file_id")) != expected_file_id:
            raise ValueError(f"Reference-safety feature identity mismatch: {path}")
        shape: tuple[int, ...] | None = None
        for name in names:
            value = payload[name]
            if value.ndim != 3 or value.dtype != np.float16 or not np.isfinite(value).all():
                raise ValueError(f"Reference-safety feature grid is invalid: {path}")
            if shape is None:
                shape = value.shape
            elif value.shape != shape:
                raise ValueError(f"Reference-safety feature shapes disagree: {path}")


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_progress(output: Path, *, completed: int, total: int, stage: str) -> None:
    _atomic_text(
        output / "progress.env",
        f"stage={stage}\ncompleted_clips={completed}\ntotal_clips={total}\n"
        f"updated_utc={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
    )


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value), dtype=np.float64)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
