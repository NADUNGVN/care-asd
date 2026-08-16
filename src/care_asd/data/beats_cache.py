"""Immutable stereo BEATs token cache for FP-NAA experiments."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import soundfile as sf

from care_asd.fp_naa_config import FPNAAConfig, load_fp_naa_config
from care_asd.models.beats_frontend import OfficialBEATsFrontend, fixed_duration_waveform


class TokenFrontend(Protocol):
    def extract(self, waveforms: np.ndarray) -> np.ndarray: ...


FrontendFactory = Callable[[FPNAAConfig, Path, Path, str], TokenFrontend]


@dataclass(frozen=True)
class BEATsCache:
    root: Path
    index_path: Path
    metadata_path: Path
    clips: int
    token_shape: tuple[int, int, int]


@dataclass(frozen=True)
class _AudioTask:
    file_id: str
    audio_path: Path
    feature_path: Path


@dataclass(frozen=True)
class _LoadedPair:
    file_id: str
    feature_path: Path
    waveforms: np.ndarray


def build_beats_token_cache(
    *,
    manifest_path: str | Path,
    audio_root: str | Path,
    output_directory: str | Path,
    config_path: str | Path,
    beats_source_directory: str | Path,
    checkpoint_path: str | Path,
    workers: int = 12,
    device: str = "cuda",
    frontend_factory: FrontendFactory | None = None,
) -> BEATsCache:
    """Extract fixed-duration near/far BEATs token grids with resumable clip writes."""
    if not 0 <= workers <= 16:
        raise ValueError("workers must be in [0, 16]")
    manifest_source = Path(manifest_path).resolve()
    root = Path(audio_root).resolve()
    output = Path(output_directory).resolve()
    config_source = Path(config_path).resolve()
    beats_source = Path(beats_source_directory).resolve()
    checkpoint = Path(checkpoint_path).resolve()
    config = load_fp_naa_config(config_source)
    if _sha256(checkpoint) != config.provenance.checkpoint_sha256:
        raise ValueError("BEATs checkpoint SHA-256 does not match the frozen FP-NAA config")

    metadata_path = output / "cache.json"
    index_path = output / "index.parquet"
    manifest_sha = _sha256(manifest_source)
    config_sha = _sha256(config_source)
    frontend_contract_sha = _frontend_contract_sha(config)
    if metadata_path.exists():
        return _load_completed_cache(
            output,
            manifest_sha=manifest_sha,
            config=config,
        )

    frame = pd.read_parquet(manifest_source)
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
    frame = frame.loc[frame["dataset_split"].isin({"dev_train", "dev_test"})].copy()
    if frame.empty or frame["file_id"].duplicated().any():
        raise ValueError("BEATs cache manifest must contain unique development clips")
    invalid_train = frame.loc[
        (frame["dataset_split"] == "dev_train") & (frame["condition"] != "normal")
    ]
    if not invalid_train.empty:
        raise ValueError("Development training rows must be normal")
    frame = frame.sort_values("file_id", kind="stable").reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    features = output / "features"
    features.mkdir(exist_ok=True)

    tasks: list[_AudioTask] = []
    feature_files: list[str] = []
    for row in frame.itertuples(index=False):
        relative = Path(str(row.relative_path))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe relative_path in manifest: {relative}")
        audio_path = (root / relative).resolve()
        if not audio_path.is_relative_to(root):
            raise ValueError(f"Audio path escapes audio root: {relative}")
        feature_name = f"{hashlib.sha256(str(row.file_id).encode()).hexdigest()}.npz"
        feature_path = features / feature_name
        feature_files.append(f"features/{feature_name}")
        if not feature_path.exists():
            tasks.append(
                _AudioTask(
                    file_id=str(row.file_id),
                    audio_path=audio_path,
                    feature_path=feature_path,
                )
            )

    frontend: TokenFrontend | None = None
    token_shape: tuple[int, int, int] | None = None
    completed = len(frame) - len(tasks)
    _write_progress(output, completed=completed, total=len(frame), stage="extract")
    batch_size = config.frontend.inference_batch_size
    for batch in _chunks(tasks, batch_size):
        loader = lambda task: _load_pair(task, config)  # noqa: E731
        if workers == 0:
            loaded = [loader(task) for task in batch]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                loaded = list(executor.map(loader, batch))
        if frontend is None:
            factory = frontend_factory or _default_frontend_factory
            frontend = factory(config, beats_source, checkpoint, device)
        waveforms = np.concatenate([item.waveforms for item in loaded], axis=0)
        grids = frontend.extract(waveforms)
        if grids.ndim != 4 or len(grids) != 2 * len(loaded):
            raise RuntimeError(f"Unexpected BEATs output shape: {grids.shape}")
        if grids.shape[2] != config.frontend.frequency_patches:
            raise RuntimeError("BEATs output frequency-patch count violates config")
        if grids.shape[3] != config.frontend.embedding_dim:
            raise RuntimeError("BEATs output embedding dimension violates config")
        token_shape = (int(grids.shape[1]), int(grids.shape[2]), int(grids.shape[3]))
        for index, item in enumerate(loaded):
            _write_feature(
                item.feature_path,
                near=grids[2 * index].astype(np.float16),
                far=grids[2 * index + 1].astype(np.float16),
            )
        completed += len(loaded)
        _write_progress(output, completed=completed, total=len(frame), stage="extract")

    if token_shape is None:
        token_shape = _validate_existing_features(output, feature_files)
    else:
        existing_shape = _validate_existing_features(output, feature_files)
        if existing_shape != token_shape:
            raise ValueError("Resumed BEATs feature files have inconsistent token shapes")
    indexed = frame.copy()
    indexed["feature_file"] = feature_files
    index_temporary = output / "index.parquet.tmp"
    indexed.to_parquet(index_temporary, index=False)
    os.replace(index_temporary, index_path)
    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clips": len(frame),
        "token_shape": list(token_shape),
        "channels": ["near", "far"],
        "dtype": "float16",
        "manifest_sha256": manifest_sha,
        "config_sha256": config_sha,
        "frontend_contract_sha256": frontend_contract_sha,
        "beats_repository": str(config.provenance.beats_repository),
        "beats_commit": config.provenance.beats_commit,
        "checkpoint_sha256": config.provenance.checkpoint_sha256,
        "duration_seconds": config.frontend.duration_seconds,
        "sample_rate": config.frontend.sample_rate,
    }
    _atomic_text(metadata_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    _write_progress(output, completed=len(frame), total=len(frame), stage="complete")
    return BEATsCache(output, index_path, metadata_path, len(frame), token_shape)


def _default_frontend_factory(
    config: FPNAAConfig,
    source: Path,
    checkpoint: Path,
    device: str,
) -> TokenFrontend:
    return OfficialBEATsFrontend(
        source_directory=source,
        checkpoint_path=checkpoint,
        device=device,
        frequency_patches=config.frontend.frequency_patches,
        mixed_precision=config.training.mixed_precision,
    )


def _load_pair(task: _AudioTask, config: FPNAAConfig) -> _LoadedPair:
    if not task.audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {task.audio_path}")
    waveform, sample_rate = sf.read(task.audio_path, dtype="float32", always_2d=True)
    if sample_rate != config.frontend.sample_rate:
        raise ValueError(f"Unexpected sample rate {sample_rate}: {task.audio_path}")
    if waveform.shape[1] < 2:
        raise ValueError(f"Expected stereo audio: {task.audio_path}")
    near = fixed_duration_waveform(
        waveform[:, 0],
        sample_rate=sample_rate,
        duration_seconds=config.frontend.duration_seconds,
    )
    far = fixed_duration_waveform(
        waveform[:, 1],
        sample_rate=sample_rate,
        duration_seconds=config.frontend.duration_seconds,
    )
    return _LoadedPair(task.file_id, task.feature_path, np.stack([near, far]))


def _write_feature(path: Path, *, near: np.ndarray, far: np.ndarray) -> None:
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, near=near, far=far)
    os.replace(temporary, path)


def _validate_existing_features(output: Path, feature_files: list[str]) -> tuple[int, int, int]:
    expected: tuple[int, int, int] | None = None
    for relative in feature_files:
        path = output / relative
        if not path.is_file():
            raise FileNotFoundError(f"BEATs cache feature is missing: {path}")
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {"near", "far"}:
                raise ValueError(f"Invalid BEATs feature payload: {path}")
            near = payload["near"]
            far = payload["far"]
        if near.shape != far.shape or near.ndim != 3 or near.dtype != np.float16:
            raise ValueError(f"Invalid BEATs token grids: {path}")
        shape = tuple(int(value) for value in near.shape)
        if expected is None:
            expected = shape
        elif shape != expected:
            raise ValueError(f"Inconsistent BEATs token grid: {path}")
    if expected is None:
        raise ValueError("No BEATs feature files were produced")
    return expected


def _load_completed_cache(
    output: Path,
    *,
    manifest_sha: str,
    config: FPNAAConfig,
) -> BEATsCache:
    metadata_path = output / "cache.json"
    index_path = output / "index.parquet"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "manifest_sha256": manifest_sha,
        "checkpoint_sha256": config.provenance.checkpoint_sha256,
        "beats_commit": config.provenance.beats_commit,
        "duration_seconds": config.frontend.duration_seconds,
        "sample_rate": config.frontend.sample_rate,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Completed BEATs cache provenance mismatch for {key}")
    if not index_path.is_file():
        raise FileNotFoundError(f"Completed BEATs cache index missing: {index_path}")
    token_shape = tuple(int(value) for value in metadata["token_shape"])
    if len(token_shape) != 3:
        raise ValueError("Invalid token_shape in completed BEATs cache")
    return BEATsCache(output, index_path, metadata_path, int(metadata["clips"]), token_shape)


def _frontend_contract_sha(config: FPNAAConfig) -> str:
    payload = {
        "schema_version": config.schema_version,
        "provenance": {
            "beats_repository": str(config.provenance.beats_repository),
            "beats_commit": config.provenance.beats_commit,
            "checkpoint_sha256": config.provenance.checkpoint_sha256,
        },
        "frontend": config.frontend.model_dump(mode="json"),
        "mixed_precision": config.training.mixed_precision,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _write_progress(output: Path, *, completed: int, total: int, stage: str) -> None:
    text = (
        f"stage={stage}\ncompleted_clips={completed}\ntotal_clips={total}\n"
        f"updated_utc={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n"
    )
    _atomic_text(output / "progress.env", text)


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


def _chunks(items: list[_AudioTask], size: int) -> Iterator[list[_AudioTask]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
