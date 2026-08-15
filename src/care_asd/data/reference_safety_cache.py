"""Immutable near/RefSub vector cache and normal-only safety profiles."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

from care_asd.data.official_vector_cache import official_waveform_to_vectors
from care_asd.reference_safety_config import ReferenceSafetyExperimentConfig
from care_asd.signal.reference_safety import (
    ReferenceDiagnostics,
    aggregate_reference_profile,
    apply_reference_subtraction,
    diagnose_reference_pair,
    estimate_noise_transfer,
    noise_floor_spectrum,
)

_GROUP_COLUMNS = ["machine_type", "section"]


@dataclass(frozen=True)
class ReferenceSafetyVectorCache:
    """Paths and counts for one immutable SAFE-REF cache."""

    directory: Path
    index_path: Path
    profiles_path: Path
    metadata_path: Path
    clips: int


@dataclass(frozen=True)
class _CacheTask:
    audio_path: str
    output_path: str
    profile_path: str
    config: dict[str, object]


@dataclass(frozen=True)
class _FloorTask:
    audio_path: str
    config: dict[str, object]


def build_reference_safety_vector_cache(
    *,
    train_manifest_path: str | Path,
    train_audio_root: str | Path,
    test_manifest_path: str | Path,
    test_audio_root: str | Path,
    output_directory: str | Path,
    config: ReferenceSafetyExperimentConfig,
    workers: int = 1,
) -> ReferenceSafetyVectorCache:
    """Build paired official vectors and group profiles without reading test labels."""
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference-safety cache: {output}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    train_root = Path(train_audio_root)
    test_root = Path(test_audio_root)
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError("Train or test audio root does not exist")
    train_all = _read_manifest(train_manifest_path)
    test_all = _read_manifest(test_manifest_path)
    train = train_all.loc[
        train_all["dataset_split"].isin({"dev_train", "add_train"})
        & (train_all["condition"] == "normal")
    ].copy()
    test = test_all.loc[test_all["dataset_split"].isin({"dev_test", "eval_test"})].copy()
    if train.empty or test.empty:
        raise ValueError(
            "SAFE-REF cache requires normal training rows and dev/evaluation test rows"
        )
    if set(test["dataset_split"]) == {"eval_test"} and set(test["condition"]) != {"unknown"}:
        raise ValueError("Evaluation cache refuses manifests containing test normal/anomaly labels")
    _validate_paths(train["relative_path"], train_root)
    _validate_paths(test["relative_path"], test_root)
    train["audio_root"] = str(train_root.resolve())
    test["audio_root"] = str(test_root.resolve())
    combined = pd.concat([train, test], ignore_index=True)
    if combined["file_id"].duplicated().any():
        raise ValueError("Train/test SAFE-REF cache rows contain duplicate file_id values")
    train_groups = set(map(tuple, train[_GROUP_COLUMNS].itertuples(index=False, name=None)))
    test_groups = set(map(tuple, test[_GROUP_COLUMNS].itertuples(index=False, name=None)))
    missing_groups = sorted(test_groups.difference(train_groups))
    if missing_groups:
        raise ValueError(f"Test groups lack normal training profiles: {missing_groups}")

    output.mkdir(parents=True)
    features_directory = output / "features"
    profiles_directory = output / "profiles"
    features_directory.mkdir()
    profiles_directory.mkdir()
    floor_tasks = [
        _FloorTask(str(_row_audio_path(row)), config.model_dump())
        for row in train.itertuples(index=False)
    ]
    if workers == 1:
        floor_results = [_estimate_floor_worker(task) for task in floor_tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            floor_results = list(executor.map(_estimate_floor_worker, floor_tasks, chunksize=8))
    floors_by_group: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray]]] = defaultdict(
        lambda: ([], [])
    )
    for row, (near_floor, far_floor) in zip(
        train.itertuples(index=False), floor_results, strict=True
    ):
        group_key = (str(row.machine_type), str(row.section))
        floors_by_group[group_key][0].append(near_floor)
        floors_by_group[group_key][1].append(far_floor)

    profile_paths: dict[tuple[str, str], Path] = {}
    transfer_by_group: dict[tuple[str, str], np.ndarray] = {}
    for group_key, (near_floors, far_floors) in sorted(floors_by_group.items()):
        transfer = estimate_noise_transfer(
            np.stack(near_floors), np.stack(far_floors), config.stft, config.refsub
        )
        profile_path = profiles_directory / f"{_key('/'.join(group_key))}.npz"
        np.savez_compressed(profile_path, transfer_power=transfer)
        profile_paths[group_key] = profile_path
        transfer_by_group[group_key] = transfer

    tasks: list[_CacheTask] = []
    cache_files: list[str] = []
    for row in combined.itertuples(index=False):
        group_key = (str(row.machine_type), str(row.section))
        relative_cache = f"features/{_key(str(row.file_id))}.npz"
        cache_files.append(relative_cache)
        tasks.append(
            _CacheTask(
                audio_path=str(_row_audio_path(row)),
                output_path=str(output / relative_cache),
                profile_path=str(profile_paths[group_key]),
                config=config.model_dump(),
            )
        )
    if workers == 1:
        results = [_write_reference_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_write_reference_worker, tasks, chunksize=4))
    combined = combined.drop(columns=["audio_root"])
    combined["cache_file"] = cache_files
    result_columns = [
        "near_vector_count",
        "refsub_vector_count",
        "leakage_index",
        "transfer_instability",
        "spectral_drift",
        "estimated_noise_reduction_db",
        "risk_score",
    ]
    for column_index, column in enumerate(result_columns):
        combined[column] = [result[column_index] for result in results]

    profile_records: list[dict[str, object]] = []
    training_mask = combined["dataset_split"].isin({"dev_train", "add_train"})
    for group, rows in combined.loc[training_mask].groupby(_GROUP_COLUMNS, sort=True):
        group_key = (str(group[0]), str(group[1]))
        diagnostics = [
            ReferenceDiagnostics(
                leakage_index=float(row.leakage_index),
                transfer_instability=float(row.transfer_instability),
                spectral_drift=float(row.spectral_drift),
                noise_reduction_db=float(row.estimated_noise_reduction_db),
                risk_score=float(row.risk_score),
            )
            for row in rows.itertuples(index=False)
        ]
        profile = aggregate_reference_profile(
            transfer_by_group[group_key], diagnostics, config.profile
        )
        np.savez_compressed(
            profile_paths[group_key],
            transfer_power=profile.transfer_power,
            leakage_u95=profile.leakage_u95,
            transfer_instability_u95=profile.transfer_instability_u95,
            spectral_drift_u95=profile.spectral_drift_u95,
            noise_reduction_l05_db=profile.noise_reduction_l05_db,
            risk_score=profile.risk_score,
            training_clips=profile.training_clips,
        )
        profile_records.append(
            {
                "machine_type": group_key[0],
                "section": group_key[1],
                "profile_file": str(profile_paths[group_key].relative_to(output)).replace(
                    "\\", "/"
                ),
                "leakage_u95": profile.leakage_u95,
                "transfer_instability_u95": profile.transfer_instability_u95,
                "spectral_drift_u95": profile.spectral_drift_u95,
                "noise_reduction_l05_db": profile.noise_reduction_l05_db,
                "risk_score": profile.risk_score,
                "training_clips": profile.training_clips,
            }
        )
    index_path = output / "index.parquet"
    combined.to_parquet(index_path, index=False)
    profiles_path = output / "profiles.parquet"
    pd.DataFrame.from_records(profile_records).to_parquet(profiles_path, index=False)
    metadata_path = output / "cache.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clips": len(combined),
                "groups": len(profile_records),
                "feature_dim": 640,
                "views": ["near", "refsub"],
                "config": config.model_dump(),
                "train_manifest": str(Path(train_manifest_path)),
                "test_manifest": str(Path(test_manifest_path)),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ReferenceSafetyVectorCache(
        output, index_path, profiles_path, metadata_path, len(combined)
    )


def load_reference_vectors(path: str | Path, view: str) -> np.ndarray:
    """Load one cached official-vector view."""
    if view not in {"near", "refsub"}:
        raise ValueError("view must be near or refsub")
    with np.load(Path(path), allow_pickle=False) as source:
        values = np.asarray(source[f"{view}_vectors"], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 640:
        raise ValueError(f"Invalid {view} vector shape: {values.shape}")
    return values


def _write_reference_worker(
    task: _CacheTask,
) -> tuple[int, int, float, float, float, float, float]:
    config = ReferenceSafetyExperimentConfig.model_validate(task.config)
    near, far, sample_rate = _read_stereo(Path(task.audio_path))
    with np.load(task.profile_path, allow_pickle=False) as source:
        transfer = np.asarray(source["transfer_power"], dtype=np.float64)
    enhanced = apply_reference_subtraction(
        near, far, sample_rate, transfer, config.stft, config.refsub
    )
    diagnostics = diagnose_reference_pair(
        near, far, enhanced, sample_rate, transfer, config.stft, config.refsub
    )
    near_vectors = official_waveform_to_vectors(near, sample_rate)
    refsub_vectors = official_waveform_to_vectors(enhanced, sample_rate)
    np.savez_compressed(
        task.output_path,
        near_vectors=near_vectors,
        refsub_vectors=refsub_vectors,
    )
    return (
        len(near_vectors),
        len(refsub_vectors),
        diagnostics.leakage_index,
        diagnostics.transfer_instability,
        diagnostics.spectral_drift,
        diagnostics.noise_reduction_db,
        diagnostics.risk_score,
    )


def _estimate_floor_worker(task: _FloorTask) -> tuple[np.ndarray, np.ndarray]:
    config = ReferenceSafetyExperimentConfig.model_validate(task.config)
    near, far, sample_rate = _read_stereo(Path(task.audio_path))
    return (
        noise_floor_spectrum(near, sample_rate, config.stft, config.refsub),
        noise_floor_spectrum(far, sample_rate, config.stft, config.refsub),
    )


def _read_manifest(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Manifest not found: {source}")
    frame = pd.read_parquet(source)
    required = {
        "file_id",
        "relative_path",
        "machine_type",
        "section",
        "domain",
        "condition",
        "dataset_split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    return frame


def _read_stereo(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    values, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if values.ndim != 2 or values.shape[1] < 2 or not len(values):
        raise ValueError(f"Expected a non-empty stereo WAV: {path}")
    return values[:, 0], values[:, 1], int(sample_rate)


def _row_audio_path(row: Any) -> Path:
    root = Path(str(row.audio_root))
    return root / str(row.relative_path)


def _key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_paths(paths: pd.Series, root: Path) -> None:
    resolved_root = root.resolve()
    for value in paths:
        candidate = (resolved_root / str(value)).resolve()
        if resolved_root not in candidate.parents:
            raise ValueError(f"Unsafe relative path in manifest: {value}")
        if not candidate.is_file():
            raise FileNotFoundError(f"Audio file from manifest not found: {candidate}")
