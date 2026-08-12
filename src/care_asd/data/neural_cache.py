"""Immutable base-feature cache shared by all Phase 5 MVP ablations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from care_asd.config import FeaturesConfig, FrontendConfig, SignalConfig
from care_asd.signal import CAREAudioFrontEnd

BASE_CHANNELS = (
    "near",
    "far",
    "residual",
    "coherence",
    "log_ratio",
    "phase_sin",
    "phase_cos",
    "path_confidence",
)


@dataclass(frozen=True)
class NeuralFeatureCache:
    """Paths and provenance emitted by a completed cache build."""

    directory: Path
    index_path: Path
    metadata_path: Path
    clips: int


def build_neural_feature_cache(
    *,
    manifest_path: str | Path,
    audio_root: str | Path,
    output_directory: str | Path,
    signal: SignalConfig,
    frontend: FrontendConfig,
    features: FeaturesConfig,
    workers: int = 1,
    limit: int | None = None,
) -> NeuralFeatureCache:
    """Create an immutable cache; no test labels are used for feature values."""
    manifest = Path(manifest_path)
    root = Path(audio_root)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite neural cache: {output}")
    if not manifest.is_file() or not root.is_dir():
        raise FileNotFoundError("Manifest or audio root does not exist")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    frame = pd.read_parquet(manifest)
    required = {
        "file_id",
        "relative_path",
        "machine_type",
        "dataset_split",
        "condition",
        "domain",
        "section",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        frame = frame.sort_values("file_id", kind="stable").head(limit).copy()
    _validate_relative_paths(frame["relative_path"], root)
    output.mkdir(parents=True)
    feature_directory = output / "features"
    feature_directory.mkdir()
    tasks = [
        (
            str(root / str(row.relative_path)),
            str(feature_directory / f"{_cache_key(str(row.file_id))}.npz"),
            signal.model_dump(),
            frontend.model_dump(),
            features.model_dump(),
        )
        for row in frame.itertuples(index=False)
    ]
    if workers == 1:
        for task in tasks:
            _write_feature_worker(task)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_write_feature_worker, tasks, chunksize=8))
    index = frame.copy()
    index["cache_file"] = [f"features/{_cache_key(str(value))}.npz" for value in index["file_id"]]
    index_path = output / "index.parquet"
    index.to_parquet(index_path, index=False)
    metadata_path = output / "cache.json"
    metadata_path.write_text(
        json.dumps(
            {
                "base_channels": list(BASE_CHANNELS),
                "clips": len(index),
                "features": features.model_dump(),
                "frontend": frontend.model_dump(),
                "manifest_sha256": _sha256(manifest),
                "signal": signal.model_dump(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return NeuralFeatureCache(output, index_path, metadata_path, len(index))


def load_cached_feature(path: str | Path, channels: Iterable[str]) -> np.ndarray:
    """Load selected channel maps in stable caller-supplied order."""
    requested = tuple(channels)
    invalid = sorted(set(requested).difference(BASE_CHANNELS))
    if invalid:
        raise ValueError(f"Unknown cached feature channel(s): {', '.join(invalid)}")
    with np.load(Path(path), allow_pickle=False) as source:
        return np.stack([np.asarray(source[name], dtype=np.float32) for name in requested])


def _write_feature_worker(
    task: tuple[str, str, dict[str, object], dict[str, object], dict[str, object]],
) -> None:
    audio_path, output_path, signal_data, frontend_data, feature_data = task
    waveform, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
    if waveform.shape[1] != 2:
        raise ValueError(f"Expected stereo WAV: {audio_path}")
    signal = SignalConfig.model_validate(signal_data)
    frontend = FrontendConfig.model_validate(frontend_data)
    features = FeaturesConfig.model_validate(feature_data)
    batch = CAREAudioFrontEnd(signal, frontend).transform(waveform.T, int(sample_rate))
    mel = _mel_filterbank(signal.n_fft, sample_rate, features)
    maps = {
        "near": _log_mel(batch.views["near"], mel),
        "far": _log_mel(batch.views["far"], mel),
        "residual": _log_mel(batch.views["residual"], mel),
        "coherence": _mel_project(batch.diagnostics["coherence"], mel),
        "log_ratio": _mel_project(batch.diagnostics["log_ratio"], mel),
        "phase_sin": _mel_project(batch.diagnostics["phase_sin"], mel),
        "phase_cos": _mel_project(batch.diagnostics["phase_cos"], mel),
        "path_confidence": _mel_project(batch.diagnostics["path_confidence"], mel),
    }
    np.savez_compressed(output_path, **maps)  # type: ignore[arg-type]


def _mel_filterbank(n_fft: int, sample_rate: int, features: FeaturesConfig) -> np.ndarray:
    maximum = sample_rate / 2.0 if features.fmax is None else min(features.fmax, sample_rate / 2.0)
    if maximum <= features.fmin:
        raise ValueError("fmax must exceed fmin")
    mel_low, mel_high = _hz_to_mel(features.fmin), _hz_to_mel(maximum)
    points = _mel_to_hz(np.linspace(mel_low, mel_high, features.n_mels + 2))
    bins = np.floor((n_fft + 1) * points / sample_rate).astype(int)
    filters = np.zeros((features.n_mels, n_fft // 2 + 1), dtype=np.float64)
    for index in range(features.n_mels):
        left, center, right = bins[index : index + 3]
        center = max(center, left + 1)
        right = max(right, center + 1)
        for bin_index in range(left, min(center, filters.shape[1])):
            filters[index, bin_index] = (bin_index - left) / (center - left)
        for bin_index in range(center, min(right, filters.shape[1])):
            filters[index, bin_index] = (right - bin_index) / (right - center)
    return filters


def _log_mel(stft: np.ndarray, mel: np.ndarray) -> np.ndarray:
    return np.asarray(np.log1p(mel @ (np.abs(stft) ** 2).T), dtype=np.float32)


def _mel_project(values: np.ndarray, mel: np.ndarray) -> np.ndarray:
    normalizer = np.maximum(mel.sum(axis=1, keepdims=True), 1.0e-12)
    return np.asarray((mel / normalizer) @ values.T, dtype=np.float32)


def _hz_to_mel(hz: float) -> float:
    return float(2595.0 * np.log10(1.0 + hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _cache_key(file_id: str) -> str:
    return hashlib.sha256(file_id.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_relative_paths(paths: pd.Series, root: Path) -> None:
    resolved_root = root.resolve()
    for value in paths:
        candidate = (resolved_root / str(value)).resolve()
        if resolved_root not in candidate.parents:
            raise ValueError(f"Unsafe relative path in manifest: {value}")
