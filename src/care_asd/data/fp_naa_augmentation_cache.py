"""Counterfactual noise/fault BEATs cache for FP-NAA C1/C2 training."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import pandas as pd
import soundfile as sf

from care_asd.fp_naa_config import FPNAAConfig, load_fp_naa_config
from care_asd.models.beats_frontend import OfficialBEATsFrontend, fixed_duration_waveform
from care_asd.signal.pseudo_faults import FaultFamily, inject_pseudo_fault, mix_paired_noise


class AugmentationFrontend(Protocol):
    def extract(self, waveforms: np.ndarray) -> np.ndarray: ...


AugmentationFrontendFactory = Callable[[FPNAAConfig, Path, Path, str], AugmentationFrontend]


@dataclass(frozen=True)
class FPNaaAugmentationCache:
    root: Path
    index_path: Path
    metadata_path: Path
    clips: int
    heldout_clips: int


@dataclass(frozen=True)
class _AugmentationPlan:
    file_id: str
    target_audio: Path
    donor_file_id: str
    donor_audio: Path
    feature_path: Path
    family: FaultFamily
    seed: int
    noise_snr_db: float
    fault_delta_level_db: float
    heldout: bool


@dataclass(frozen=True)
class _PreparedAugmentation:
    plan: _AugmentationPlan
    names: tuple[str, ...]
    waveforms: np.ndarray
    metadata: dict[str, object]


def build_fp_naa_augmentation_cache(
    *,
    base_cache_directory: str | Path,
    audio_root: str | Path,
    output_directory: str | Path,
    config_path: str | Path,
    beats_source_directory: str | Path,
    checkpoint_path: str | Path,
    workers: int = 12,
    device: str = "cuda",
    frontend_factory: AugmentationFrontendFactory | None = None,
) -> FPNaaAugmentationCache:
    """Build resumable normal-only counterfactual pairs for C1/C2."""
    if not 0 <= workers <= 16:
        raise ValueError("workers must be in [0, 16]")
    base_cache = Path(base_cache_directory).resolve()
    root = Path(audio_root).resolve()
    output = Path(output_directory).resolve()
    config_source = Path(config_path).resolve()
    beats_source = Path(beats_source_directory).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    config = load_fp_naa_config(config_source)
    if _sha256(checkpoint) != config.provenance.checkpoint_sha256:
        raise ValueError("BEATs checkpoint SHA-256 does not match FP-NAA config")
    base_metadata_path = base_cache / "cache.json"
    base_index_path = base_cache / "index.parquet"
    if not base_metadata_path.is_file() or not base_index_path.is_file():
        raise FileNotFoundError(f"Completed base BEATs cache not found: {base_cache}")
    base_metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    if base_metadata.get("checkpoint_sha256") != config.provenance.checkpoint_sha256:
        raise ValueError("Base cache checkpoint does not match FP-NAA config")
    metadata_path = output / "cache.json"
    index_path = output / "index.parquet"
    contract_sha = _augmentation_contract_sha(config, base_metadata_path)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("augmentation_contract_sha256") != contract_sha:
            raise ValueError("Completed augmentation cache contract mismatch")
        if not index_path.is_file():
            raise FileNotFoundError(f"Augmentation index missing: {index_path}")
        return FPNaaAugmentationCache(
            output,
            index_path,
            metadata_path,
            int(metadata["clips"]),
            int(metadata["heldout_clips"]),
        )

    frame = pd.read_parquet(base_index_path)
    train = frame.loc[frame["dataset_split"] == "dev_train"].copy()
    required = {
        "file_id",
        "relative_path",
        "feature_file",
        "machine_type",
        "section",
        "condition",
    }
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Base cache index is missing columns: {', '.join(missing)}")
    if train.empty or train["file_id"].duplicated().any() or set(train["condition"]) != {"normal"}:
        raise ValueError("Augmentation cache requires unique normal development training rows")
    train = train.sort_values("file_id", kind="stable").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    features = output / "features"
    features.mkdir(exist_ok=True)
    plans = _make_plans(train, root, features, config)
    pending = [plan for plan in plans if not plan.feature_path.exists()]
    completed = len(plans) - len(pending)
    _write_progress(output, completed=completed, total=len(plans), stage="extract")
    frontend: AugmentationFrontend | None = None
    for batch in _chunks(pending, config.frontend.inference_batch_size):
        if workers == 0:
            prepared = [_prepare(plan, config) for plan in batch]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                prepared = list(executor.map(lambda plan: _prepare(plan, config), batch))
        if frontend is None:
            factory = frontend_factory or _default_frontend_factory
            frontend = factory(config, beats_source, checkpoint, device)
        flattened = np.concatenate([item.waveforms for item in prepared], axis=0)
        grids = frontend.extract(flattened)
        if grids.ndim != 4 or len(grids) != len(flattened):
            raise RuntimeError(f"Unexpected augmented BEATs output shape: {grids.shape}")
        cursor = 0
        for item in prepared:
            count = len(item.names)
            payload = {
                name: _cache_grid(
                    grids[cursor + offset],
                    context=f"file_id={item.plan.file_id} field={name}",
                )
                for offset, name in enumerate(item.names)
            }
            payload["metadata_json"] = np.asarray(
                json.dumps(item.metadata, sort_keys=True, separators=(",", ":"))
            )
            _write_feature(item.plan.feature_path, payload)
            cursor += count
        completed += len(prepared)
        _write_progress(output, completed=completed, total=len(plans), stage="extract")

    metadata_rows = [_validate_and_read(plan.feature_path) for plan in plans]
    indexed = train.copy()
    indexed["augmentation_file"] = [f"features/{plan.feature_path.name}" for plan in plans]
    metadata_frame = pd.DataFrame(metadata_rows)
    for column in metadata_frame.columns:
        indexed[column] = metadata_frame[column]
    temporary_index = output / "index.parquet.tmp"
    indexed.to_parquet(temporary_index, index=False)
    os.replace(temporary_index, index_path)
    heldout_clips = int(indexed["heldout"].sum())
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clips": len(indexed),
        "heldout_clips": heldout_clips,
        "augmentation_contract_sha256": contract_sha,
        "base_cache_metadata_sha256": _sha256(base_metadata_path),
        "checkpoint_sha256": config.provenance.checkpoint_sha256,
        "train_fault_families": config.augmentation.train_fault_families,
        "heldout_fault_family": config.augmentation.heldout_fault_family,
    }
    _atomic_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _write_progress(output, completed=len(plans), total=len(plans), stage="complete")
    return FPNaaAugmentationCache(output, index_path, metadata_path, len(indexed), heldout_clips)


def _make_plans(
    train: pd.DataFrame,
    audio_root: Path,
    feature_root: Path,
    config: FPNAAConfig,
) -> list[_AugmentationPlan]:
    donors: dict[str, pd.Series] = {}
    for _, group in train.groupby(["machine_type", "section"], sort=True):
        ordered = group.sort_values("file_id", kind="stable")
        if len(ordered) < 2:
            raise ValueError("Each machine/section needs at least two clips for donor pairing")
        donor_rows = ordered.iloc[np.roll(np.arange(len(ordered)), 1)]
        for target_id, (_, donor) in zip(
            ordered["file_id"].astype(str), donor_rows.iterrows(), strict=True
        ):
            donors[target_id] = donor
    families = config.augmentation.train_fault_families
    plans: list[_AugmentationPlan] = []
    for row in train.itertuples(index=False):
        file_id = str(row.file_id)
        donor = donors[file_id]
        seed = _stable_seed(config.augmentation.seed, file_id, "augmentation")
        rng = np.random.default_rng(seed)
        family = cast(FaultFamily, families[int(rng.integers(0, len(families)))])
        noise_snr = float(
            rng.uniform(
                config.augmentation.noise_snr_db_min,
                config.augmentation.noise_snr_db_max,
            )
        )
        fault_level = float(
            rng.uniform(
                config.augmentation.fault_delta_level_db_min,
                config.augmentation.fault_delta_level_db_max,
            )
        )
        heldout = bool(rng.random() < config.augmentation.heldout_fraction)
        target_audio = _safe_audio_path(audio_root, str(row.relative_path))
        donor_audio = _safe_audio_path(audio_root, str(donor["relative_path"]))
        name = f"{hashlib.sha256(file_id.encode()).hexdigest()}.npz"
        plans.append(
            _AugmentationPlan(
                file_id=file_id,
                target_audio=target_audio,
                donor_file_id=str(donor["file_id"]),
                donor_audio=donor_audio,
                feature_path=feature_root / name,
                family=family,
                seed=seed,
                noise_snr_db=noise_snr,
                fault_delta_level_db=fault_level,
                heldout=heldout,
            )
        )
    return plans


def _prepare(plan: _AugmentationPlan, config: FPNAAConfig) -> _PreparedAugmentation:
    target, target_rate = sf.read(plan.target_audio, dtype="float32", always_2d=True)
    donor, donor_rate = sf.read(plan.donor_audio, dtype="float32", always_2d=True)
    sample_rate = config.frontend.sample_rate
    if (
        target_rate != sample_rate
        or donor_rate != sample_rate
        or min(target.shape[1], donor.shape[1]) < 2
    ):
        raise ValueError("Augmentation source audio must be stereo at the configured sample rate")
    clean = fixed_duration_waveform(
        target[:, 0], sample_rate=sample_rate, duration_seconds=config.frontend.duration_seconds
    )
    reference = fixed_duration_waveform(
        donor[:, 1], sample_rate=sample_rate, duration_seconds=config.frontend.duration_seconds
    )
    fault = inject_pseudo_fault(
        clean,
        sample_rate=sample_rate,
        family=plan.family,
        seed=plan.seed,
        delta_level_db=plan.fault_delta_level_db,
        peak_limit=config.augmentation.peak_limit,
    )
    pair = mix_paired_noise(
        clean,
        fault.waveform,
        reference,
        snr_db=plan.noise_snr_db,
        peak_limit=config.augmentation.peak_limit,
    )
    names = ["noisy_clean", "reference", "fault_teacher", "fault_noisy"]
    waveforms = [pair.clean_noisy, pair.reference_noise, fault.waveform, pair.fault_noisy]
    metadata: dict[str, object] = {
        "file_id": plan.file_id,
        "donor_file_id": plan.donor_file_id,
        "fault_family": plan.family,
        "fault_seed": plan.seed,
        "requested_noise_snr_db": plan.noise_snr_db,
        "actual_noise_snr_db": pair.actual_clean_snr_db,
        "requested_fault_delta_level_db": plan.fault_delta_level_db,
        "actual_fault_delta_level_db": fault.actual_delta_level_db,
        "fault_parameters_json": json.dumps(fault.parameters, sort_keys=True),
        "heldout": plan.heldout,
    }
    if plan.heldout:
        heldout_family = cast(FaultFamily, config.augmentation.heldout_fault_family)
        heldout_fault = inject_pseudo_fault(
            clean,
            sample_rate=sample_rate,
            family=heldout_family,
            seed=_stable_seed(plan.seed, plan.file_id, "heldout"),
            delta_level_db=plan.fault_delta_level_db,
            peak_limit=config.augmentation.peak_limit,
        )
        heldout_pair = mix_paired_noise(
            clean,
            heldout_fault.waveform,
            reference,
            snr_db=plan.noise_snr_db,
            peak_limit=config.augmentation.peak_limit,
        )
        names.extend(
            [
                "heldout_noisy_clean",
                "heldout_reference",
                "heldout_fault_teacher",
                "heldout_fault_noisy",
            ]
        )
        waveforms.extend(
            [
                heldout_pair.clean_noisy,
                heldout_pair.reference_noise,
                heldout_fault.waveform,
                heldout_pair.fault_noisy,
            ]
        )
        metadata["heldout_fault_family"] = heldout_family
        metadata["heldout_actual_fault_delta_level_db"] = heldout_fault.actual_delta_level_db
        metadata["heldout_fault_parameters_json"] = json.dumps(
            heldout_fault.parameters, sort_keys=True
        )
    return _PreparedAugmentation(
        plan=plan,
        names=tuple(names),
        waveforms=np.stack(waveforms),
        metadata=metadata,
    )


def _validate_and_read(path: Path) -> dict[str, object]:
    required = {"noisy_clean", "reference", "fault_teacher", "fault_noisy", "metadata_json"}
    with np.load(path, allow_pickle=False) as payload:
        if not required.issubset(payload.files):
            raise ValueError(f"Incomplete FP-NAA augmentation feature: {path}")
        metadata = json.loads(str(payload["metadata_json"].item()))
        shape: tuple[int, ...] | None = None
        for name in payload.files:
            if name == "metadata_json":
                continue
            value = payload[name]
            if value.ndim != 3 or value.dtype != np.float16 or not np.isfinite(value).all():
                raise ValueError(f"Invalid {name} token grid: {path}")
            if shape is None:
                shape = value.shape
            elif value.shape != shape:
                raise ValueError(f"Inconsistent augmentation token shape: {path}")
    if bool(metadata["heldout"]):
        expected = {
            "heldout_noisy_clean",
            "heldout_reference",
            "heldout_fault_teacher",
            "heldout_fault_noisy",
        }
        with np.load(path, allow_pickle=False) as payload:
            if not expected.issubset(payload.files):
                raise ValueError(f"Held-out fault fields missing: {path}")
    return cast(dict[str, object], metadata)


def _default_frontend_factory(
    config: FPNAAConfig, source: Path, checkpoint: Path, device: str
) -> AugmentationFrontend:
    return OfficialBEATsFrontend(
        source_directory=source,
        checkpoint_path=checkpoint,
        device=device,
        frequency_patches=config.frontend.frequency_patches,
        mixed_precision=config.frontend.inference_mixed_precision,
    )


def _write_feature(path: Path, payload: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        savez = cast(Callable[..., None], np.savez)
        savez(handle, **payload)
    os.replace(temporary, path)


def _cache_grid(grid: np.ndarray, *, context: str) -> np.ndarray:
    if grid.ndim != 3 or not np.isfinite(grid).all():
        raise RuntimeError(f"Non-finite BEATs tokens; cache write aborted: {context}")
    with np.errstate(over="ignore", invalid="ignore"):
        cached = np.asarray(grid, dtype=np.float16)
    if not np.isfinite(cached).all():
        raise RuntimeError(f"BEATs tokens overflow float16; cache write aborted: {context}")
    return cached


def _safe_audio_path(root: Path, relative_text: str) -> Path:
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe audio relative path: {relative}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Audio path escapes root: {relative}")
    return path


def _augmentation_contract_sha(config: FPNAAConfig, base_metadata: Path) -> str:
    payload = {
        "cache_schema_version": 2,
        "config_schema_version": config.schema_version,
        "base_cache_metadata_sha256": _sha256(base_metadata),
        "checkpoint_sha256": config.provenance.checkpoint_sha256,
        "frontend": config.frontend.model_dump(mode="json"),
        "augmentation": config.augmentation.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _stable_seed(seed: int, identity: str, purpose: str) -> int:
    payload = f"{seed}|{identity}|{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _write_progress(output: Path, *, completed: int, total: int, stage: str) -> None:
    _atomic_text(
        output / "progress.env",
        f"stage={stage}\ncompleted_clips={completed}\ntotal_clips={total}\n"
        f"updated_utc={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
    )


def _atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunks(items: list[_AugmentationPlan], size: int) -> Iterator[list[_AugmentationPlan]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
