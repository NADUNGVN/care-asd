"""Exact-feature cache for the pinned official DCASE 2026 AE baseline."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OFFICIAL_FEATURE_DIM = 640
OFFICIAL_FRAMES = 5
OFFICIAL_MELS = 128


@dataclass(frozen=True)
class OfficialVectorCache:
    """Immutable cache whose vectors match the pinned baseline feature contract."""

    directory: Path
    index_path: Path
    metadata_path: Path
    clips: int


def build_official_vector_cache(
    *,
    manifest_path: str | Path,
    audio_root: str | Path,
    output_directory: str | Path,
    workers: int = 1,
) -> OfficialVectorCache:
    """Cache official channel-0 librosa five-frame log-Mel vectors per WAV."""
    manifest = Path(manifest_path)
    root = Path(audio_root)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite official vector cache: {output}")
    if not manifest.is_file() or not root.is_dir():
        raise FileNotFoundError("Manifest or audio root does not exist")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    _require_librosa()
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
        (str(root / str(row.relative_path)), str(features / f"{_key(str(row.file_id))}.npz"))
        for row in index.itertuples(index=False)
    ]
    if workers == 1:
        vector_counts = [_write_vector_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            vector_counts = list(executor.map(_write_vector_worker, tasks, chunksize=8))
    index = index.copy()
    index["cache_file"] = [f"features/{_key(str(value))}.npz" for value in index["file_id"]]
    index["vector_count"] = vector_counts
    index_path = output / "index.parquet"
    index.to_parquet(index_path, index=False)
    metadata_path = output / "cache.json"
    metadata_path.write_text(
        json.dumps(
            {
                "channel": 0,
                "clips": len(index),
                "feature_dim": OFFICIAL_FEATURE_DIM,
                "feature_spec": "librosa.melspectrogram + 20/power*log10 + five-frame stack",
                "frames": OFFICIAL_FRAMES,
                "librosa_version": _librosa_version(),
                "n_fft": 1024,
                "n_mels": OFFICIAL_MELS,
                "hop_length": 512,
                "official_baseline_commit": "f44242ec1f78f6cc34f53f43fb88be1ce5d13d47",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return OfficialVectorCache(output, index_path, metadata_path, len(index))


def load_official_vectors(path: str | Path) -> np.ndarray:
    """Read one immutable vector matrix as float32."""
    with np.load(Path(path), allow_pickle=False) as source:
        values = np.asarray(source["vectors"], dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != OFFICIAL_FEATURE_DIM:
        raise ValueError(f"Invalid official vector cache shape: {values.shape}")
    return values


def _write_vector_worker(task: tuple[str, str]) -> int:
    audio_path, output_path = task
    vectors = official_file_to_vectors(audio_path)
    np.savez_compressed(output_path, vectors=vectors)
    return len(vectors)


def official_file_to_vectors(audio_path: str | Path) -> np.ndarray:
    """Reproduce the pinned loader's channel-0 vectorisation exactly."""
    librosa = _require_librosa()
    signal, sample_rate = librosa.load(str(audio_path), sr=None, mono=False)
    if signal.ndim != 2 or signal.shape[0] < 2:
        raise ValueError(f"Expected stereo audio for official channel-0 feature: {audio_path}")
    return official_waveform_to_vectors(signal[0], int(sample_rate))


def official_waveform_to_vectors(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the locked official Mel/vector stack to one time-domain waveform."""
    if waveform.ndim != 1 or waveform.size == 0:
        raise ValueError("Official vectorisation requires one non-empty mono waveform")
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    librosa = _require_librosa()
    mel = librosa.feature.melspectrogram(
        y=np.asarray(waveform, dtype=np.float64),
        sr=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=OFFICIAL_MELS,
        power=2.0,
        fmax=None,
        fmin=0.0,
        win_length=None,
    )
    log_mel = 10.0 * np.log10(np.maximum(mel, np.finfo(float).eps))
    count = log_mel.shape[-1] - OFFICIAL_FRAMES + 1
    if count < 1:
        return np.empty((0, OFFICIAL_FEATURE_DIM), dtype=np.float32)
    vectors = np.empty((count, OFFICIAL_FEATURE_DIM), dtype=np.float32)
    for frame in range(OFFICIAL_FRAMES):
        vectors[:, OFFICIAL_MELS * frame : OFFICIAL_MELS * (frame + 1)] = log_mel[
            :, frame : frame + count
        ].T
    return vectors


def _require_librosa() -> Any:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "Official alignment requires librosa; install the project extra [official-alignment]"
        ) from exc
    return librosa


def _librosa_version() -> str:
    return str(_require_librosa().__version__)


def _key(file_id: str) -> str:
    return hashlib.sha256(file_id.encode("utf-8")).hexdigest()


def _validate_paths(paths: pd.Series, root: Path) -> None:
    resolved_root = root.resolve()
    for value in paths:
        candidate = (resolved_root / str(value)).resolve()
        if resolved_root not in candidate.parents:
            raise ValueError(f"Unsafe relative path in manifest: {value}")
