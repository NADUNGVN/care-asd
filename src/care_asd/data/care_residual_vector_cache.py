"""CARE residual caches that retain the locked official vector/model contract."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from care_asd.config import FrontendConfig, SignalConfig
from care_asd.data.official_vector_cache import OFFICIAL_FEATURE_DIM, official_waveform_to_vectors
from care_asd.signal import SafeCAREFrontEnd


@dataclass(frozen=True)
class CareResidualVectorCache:
    """Immutable CARE-residual cache with official 640-dimensional vectors."""

    directory: Path
    index_path: Path
    metadata_path: Path
    clips: int


def build_care_residual_vector_cache(
    *,
    manifest_path: str | Path,
    audio_root: str | Path,
    output_directory: str | Path,
    signal: SignalConfig,
    frontend: FrontendConfig,
    workers: int = 1,
) -> CareResidualVectorCache:
    """Cache bounded CARE residuals after the official Mel/five-frame stack.

    The CARE STFT is inverted by normalized overlap-add before feature extraction.
    Thus the sole intentional change from Phase 6 is the causal CARE front-end;
    the official centred log-Mel and 640-value stacking contract remains intact.
    """
    manifest = Path(manifest_path)
    root = Path(audio_root)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite CARE residual vector cache: {output}")
    if not manifest.is_file() or not root.is_dir():
        raise FileNotFoundError("Manifest or audio root does not exist")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    index = pd.read_parquet(manifest)
    required = {
        "file_id",
        "relative_path",
        "machine_type",
        "dataset_split",
        "condition",
        "domain",
        "section",
    }
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    _validate_paths(index["relative_path"], root)
    output.mkdir(parents=True)
    features = output / "features"
    features.mkdir()
    tasks = [
        (
            str(root / str(row.relative_path)),
            str(features / f"{_key(str(row.file_id))}.npz"),
            signal.model_dump(),
            frontend.model_dump(),
        )
        for row in index.itertuples(index=False)
    ]
    if workers == 1:
        vector_counts = [_write_residual_vector_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            vector_counts = list(executor.map(_write_residual_vector_worker, tasks, chunksize=8))
    index = index.copy()
    index["cache_file"] = [f"features/{_key(str(value))}.npz" for value in index["file_id"]]
    index["vector_count"] = vector_counts
    index_path = output / "index.parquet"
    index.to_parquet(index_path, index=False)
    metadata_path = output / "cache.json"
    metadata_path.write_text(
        json.dumps(
            {
                "clips": len(index),
                "feature_dim": OFFICIAL_FEATURE_DIM,
                "frontend": frontend.model_dump(),
                "input": "channel_0_near_minus_bounded_causal_care_path_estimate",
                "official_feature_stack": "librosa.melspectrogram + 10*log10 + five-frame stack",
                "signal": signal.model_dump(),
                "waveform_reconstruction": "normalized_overlap_add",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return CareResidualVectorCache(output, index_path, metadata_path, len(index))


def care_residual_waveform(
    waveform: np.ndarray, signal: SignalConfig, frontend: FrontendConfig
) -> np.ndarray:
    """Return a time-domain CARE residual while preserving input length exactly."""
    values = np.asarray(waveform, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] < 1:
        raise ValueError("CARE residual requires finite waveform shape (2, samples)")
    if not np.isfinite(values).all():
        raise ValueError("CARE residual requires finite waveform values")
    residual_stft = SafeCAREFrontEnd(signal, frontend).transform(values).residual_stft
    return _normalized_overlap_add(
        residual_stft,
        length=values.shape[1],
        signal=signal,
        fallback=values[0],
    )


def _write_residual_vector_worker(
    task: tuple[str, str, dict[str, object], dict[str, object]],
) -> int:
    audio_path, output_path, signal_data, frontend_data = task
    waveform, sample_rate = sf.read(audio_path, dtype="float64", always_2d=True)
    if waveform.shape[1] != 2:
        raise ValueError(f"Expected stereo WAV: {audio_path}")
    residual = care_residual_waveform(
        waveform.T,
        SignalConfig.model_validate(signal_data),
        FrontendConfig.model_validate(frontend_data),
    )
    vectors = official_waveform_to_vectors(residual, int(sample_rate))
    np.savez_compressed(output_path, vectors=vectors)
    return len(vectors)


def _normalized_overlap_add(
    stft: np.ndarray,
    *,
    length: int,
    signal: SignalConfig,
    fallback: np.ndarray,
) -> np.ndarray:
    """Invert Safe CARE's deterministic frame geometry with overlap normalization."""
    if stft.ndim != 2 or length < 1:
        raise ValueError("Invalid STFT or output length for CARE residual reconstruction")
    window = np.hanning(signal.win_length).astype(np.float64)
    starts = _frame_starts(length, signal.win_length, signal.hop_length)
    if len(starts) != stft.shape[0]:
        raise ValueError("CARE residual STFT frame count does not match signal geometry")
    reconstructed = np.zeros(length, dtype=np.float64)
    normalization = np.zeros(length, dtype=np.float64)
    frames = np.fft.irfft(stft, n=signal.n_fft, axis=1)[:, : signal.win_length]
    for start, frame in zip(starts, frames, strict=True):
        end = min(start + signal.win_length, length)
        width = end - start
        reconstructed[start:end] += frame[:width] * window[:width]
        normalization[start:end] += window[:width] ** 2
    supported = normalization > np.finfo(np.float64).eps
    reconstructed[supported] /= normalization[supported]
    reconstructed[~supported] = fallback[~supported]
    return reconstructed


def _frame_starts(length: int, win_length: int, hop_length: int) -> list[int]:
    if length <= win_length:
        return [0]
    starts = list(range(0, length - win_length + 1, hop_length))
    final_start = length - win_length
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _key(file_id: str) -> str:
    return hashlib.sha256(file_id.encode("utf-8")).hexdigest()


def _validate_paths(paths: pd.Series, root: Path) -> None:
    resolved_root = root.resolve()
    for value in paths:
        candidate = (resolved_root / str(value)).resolve()
        if resolved_root not in candidate.parents:
            raise ValueError(f"Unsafe relative path in manifest: {value}")
