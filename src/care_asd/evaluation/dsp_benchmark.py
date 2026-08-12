"""Fair, deterministic development benchmark for the Phase 3 DSP controls.

This module intentionally uses a fixed normal-only spectral reference scorer.
It is not the proposed CARE-ASD neural model. Its sole purpose is to keep the
encoder/scorer constant while comparing the signal transforms themselves.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import soundfile as sf

from care_asd.config import SignalConfig
from care_asd.evaluation.official_baseline import (
    SCORE_COLUMNS,
    calculate_development_auc_metrics,
)
from care_asd.signal.dsp_baselines import (
    FeatureBatch,
    FrontEndName,
    available_dsp_frontends,
    create_dsp_frontend,
)

_REQUIRED_MANIFEST_COLUMNS = {
    "file_id",
    "relative_path",
    "machine_type",
    "section",
    "domain",
    "condition",
    "dataset_split",
}


@dataclass(frozen=True)
class DspBenchmarkResult:
    """Immutable paths emitted by one complete Phase 3 benchmark run."""

    output_directory: Path
    summary_path: Path
    overcancellation_path: Path
    run_metadata_path: Path


@dataclass(frozen=True)
class _ReferenceModel:
    view_names: tuple[str, ...]
    means: tuple[np.ndarray, ...]
    scales: tuple[np.ndarray, ...]
    score_fusion: str

    def score(self, features: _ReferenceFeatures) -> float:
        """Return a mean standardized-distance anomaly score."""
        if features.view_names != self.view_names:
            raise ValueError("Feature views differ from the fitted reference scorer")
        view_scores = []
        for vector, mean, scale in zip(features.vectors, self.means, self.scales, strict=True):
            if vector.shape != mean.shape:
                raise ValueError("Feature dimension differs from the fitted reference scorer")
            view_scores.append(float(np.mean(((vector - mean) / scale) ** 2)))
        if self.score_fusion == "mean":
            return float(np.mean(view_scores))
        return float(np.mean(view_scores))


@dataclass(frozen=True)
class _ReferenceFeatures:
    """Pickle-safe fixed-score inputs extracted from one stereo clip."""

    view_names: tuple[str, ...]
    vectors: tuple[np.ndarray, ...]
    score_fusion: str
    median_energy_ratio: float
    fraction_bins_below_0_2: float


def run_dsp_development_benchmark(
    *,
    manifest_path: str | Path,
    audio_root: str | Path,
    output_directory: str | Path,
    experiment_id: str,
    frontends: Iterable[str] | None = None,
    signal: SignalConfig | None = None,
    workers: int = 1,
) -> DspBenchmarkResult:
    """Benchmark pre-registered DSP controls on all DCASE development rows.

    Per machine type, the scorer is fit only to that machine's ``dev_train``
    normal clips. Every front-end therefore sees identical train/test rows and
    the same spectral summarizer and standardized-distance score. ``output``
    must not already exist so evidence cannot be silently replaced.
    """
    manifest = Path(manifest_path)
    root = Path(audio_root)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark directory: {output}")
    if not manifest.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    if not root.is_dir():
        raise FileNotFoundError(f"Audio root not found: {root}")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    requested_raw = tuple(frontends if frontends is not None else available_dsp_frontends())
    if not requested_raw:
        raise ValueError("At least one DSP front-end is required")
    invalid = sorted(set(requested_raw).difference(available_dsp_frontends()))
    if invalid:
        raise ValueError(f"Unknown DSP front-end(s): {', '.join(invalid)}")
    requested = tuple(cast(FrontEndName, name) for name in requested_raw)

    frame = pd.read_parquet(manifest)
    missing = sorted(_REQUIRED_MANIFEST_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    train = frame.loc[(frame["dataset_split"] == "dev_train") & (frame["condition"] == "normal")]
    test = frame.loc[frame["dataset_split"] == "dev_test"]
    if train.empty or test.empty:
        raise ValueError("Manifest must contain normal dev_train rows and dev_test rows")
    if not set(test["condition"]).issubset({"normal", "anomaly"}):
        raise ValueError("Development test rows must have normal/anomaly labels")
    _assert_relative_paths(frame["relative_path"], root)

    output.mkdir(parents=True)
    score_directory = output / "scores"
    metrics_directory = output / "metrics"
    score_directory.mkdir()
    metrics_directory.mkdir()
    signal_config = signal or SignalConfig()
    summary_rows: list[dict[str, str | float]] = []
    energy_rows: list[dict[str, str | float]] = []

    for frontend_name in requested:
        scores, energy = _score_frontend(
            frontend_name=frontend_name,
            signal=signal_config,
            audio_root=root,
            train=train,
            test=test,
            experiment_id=experiment_id,
            workers=workers,
        )
        score_path = score_directory / f"{frontend_name}.csv"
        scores.to_csv(score_path, index=False)
        metrics_path = metrics_directory / f"{frontend_name}.json"
        calculate_development_auc_metrics(score_path, metrics_path)
        summary_rows.extend(_summarize_metrics(frontend_name, metrics_path))
        energy_rows.extend(energy)

    summary_path = output / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    overcancellation_path = output / "overcancellation.csv"
    pd.DataFrame(energy_rows).to_csv(overcancellation_path, index=False)
    metadata_path = output / "benchmark.json"
    metadata_path.write_text(
        json.dumps(
            {
                "audio_root": str(root),
                "experiment_id": experiment_id,
                "frontends": list(requested),
                "manifest": str(manifest),
                "scorer": "fixed_log_spectral_mean_std_standardized_distance_v1",
                "signal_config": signal_config.model_dump(),
                "train_rows": len(train),
                "test_rows": len(test),
                "workers": workers,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return DspBenchmarkResult(output, summary_path, overcancellation_path, metadata_path)


def _score_frontend(
    *,
    frontend_name: FrontEndName,
    signal: SignalConfig,
    audio_root: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    experiment_id: str,
    workers: int,
) -> tuple[pd.DataFrame, list[dict[str, str | float]]]:
    fitted: dict[str, _ReferenceModel] = {}
    for machine_type, group in train.groupby("machine_type", sort=True):
        train_features = _extract_features(
            paths=[audio_root / str(relative) for relative in group["relative_path"]],
            frontend_name=frontend_name,
            signal=signal,
            workers=workers,
        )
        fitted[str(machine_type)] = _fit_reference(train_features)

    rows: list[dict[str, str | float]] = []
    energy_rows: list[dict[str, str | float]] = []
    ordered_test = test.sort_values("file_id", kind="stable").to_dict(orient="records")
    test_features = _extract_features(
        paths=[audio_root / str(item["relative_path"]) for item in ordered_test],
        frontend_name=frontend_name,
        signal=signal,
        workers=workers,
    )
    for item, feature in zip(ordered_test, test_features, strict=True):
        machine_type = str(item["machine_type"])
        if machine_type not in fitted:
            raise ValueError(f"No normal training clips for machine type: {machine_type}")
        rows.append(
            {
                "file_id": str(item["file_id"]),
                "machine_type": machine_type,
                "section": str(item["section"]),
                "domain": str(item["domain"]),
                "condition": str(item["condition"]),
                "anomaly_score": fitted[machine_type].score(feature),
                "model_id": "fixed_log_spectral_reference_v1",
                "experiment_id": experiment_id,
            }
        )
        energy_rows.append(
            {
                "file_id": str(item["file_id"]),
                "frontend": frontend_name,
                "machine_type": machine_type,
                "condition": str(item["condition"]),
                "median_view_to_near_energy_ratio": feature.median_energy_ratio,
                "fraction_bins_below_0_2": feature.fraction_bins_below_0_2,
            }
        )
    return pd.DataFrame(rows, columns=SCORE_COLUMNS), energy_rows


def _fit_reference(features: list[_ReferenceFeatures]) -> _ReferenceModel:
    if not features:
        raise ValueError("At least one normal training batch is required")
    first = features[0]
    view_names = first.view_names
    means: list[np.ndarray] = []
    scales: list[np.ndarray] = []
    for index, _ in enumerate(view_names):
        vectors = np.stack([item.vectors[index] for item in features])
        means.append(np.mean(vectors, axis=0))
        scales.append(np.maximum(np.std(vectors, axis=0), 1.0e-4))
    return _ReferenceModel(view_names, tuple(means), tuple(scales), first.score_fusion)


def _extract_features(
    *,
    paths: list[Path],
    frontend_name: FrontEndName,
    signal: SignalConfig,
    workers: int,
) -> list[_ReferenceFeatures]:
    if workers == 1:
        frontend = create_dsp_frontend(frontend_name, signal)
        return [_features_from_batch(frontend.transform(*_read_stereo(path))) for path in paths]
    tasks = [(str(path), frontend_name, signal.model_dump()) for path in paths]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(_extract_features_worker, tasks, chunksize=8))


def _extract_features_worker(
    task: tuple[str, FrontEndName, dict[str, object]],
) -> _ReferenceFeatures:
    path_string, frontend_name, signal_data = task
    frontend = create_dsp_frontend(frontend_name, SignalConfig.model_validate(signal_data))
    return _features_from_batch(frontend.transform(*_read_stereo(Path(path_string))))


def _features_from_batch(batch: FeatureBatch) -> _ReferenceFeatures:
    ratio = batch.diagnostics["view_to_near_energy_ratio"]
    return _ReferenceFeatures(
        view_names=tuple(batch.views),
        vectors=tuple(_spectral_summary(view) for view in batch.views.values()),
        score_fusion=batch.score_fusion,
        median_energy_ratio=float(np.median(ratio)),
        fraction_bins_below_0_2=float(np.mean(ratio < 0.2)),
    )


def _spectral_summary(stft: np.ndarray) -> np.ndarray:
    """Fixed encoder: concatenate per-frequency log-magnitude mean and std."""
    log_magnitude = np.log1p(np.abs(stft))
    return np.concatenate((np.mean(log_magnitude, axis=0), np.std(log_magnitude, axis=0)))


def _read_stereo(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest audio file not found: {path}")
    waveform, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if waveform.shape[1] != 2:
        raise ValueError(f"Expected exactly 2 channels in {path}, found {waveform.shape[1]}")
    return waveform.T, int(sample_rate)


def _assert_relative_paths(paths: pd.Series, audio_root: Path) -> None:
    root = audio_root.resolve()
    for value in paths:
        candidate = (root / str(value)).resolve()
        if root not in candidate.parents:
            raise ValueError(f"Manifest contains unsafe relative_path: {value}")


def _summarize_metrics(
    frontend_name: FrontEndName, metrics_path: Path
) -> list[dict[str, str | float]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    groups = payload["groups"]
    rows: list[dict[str, str | float]] = []
    for category, predicate in (
        ("all", lambda machine: True),
        ("real", lambda machine: not machine.endswith("Emu")),
        ("emulated", lambda machine: machine.endswith("Emu")),
    ):
        selected = [
            values for key, values in groups.items() if predicate(key.split("/", maxsplit=1)[0])
        ]
        if not selected:
            continue
        rows.append(
            {
                "frontend": frontend_name,
                "machine_category": category,
                "mean_auc_all": float(np.mean([item["auc_all"] for item in selected])),
                "mean_pauc_all_max_fpr_0_1": float(
                    np.mean([item["pauc_all_max_fpr_0_1"] for item in selected])
                ),
                "mean_auc_source": float(np.mean([item["auc_source"] for item in selected])),
                "mean_auc_target": float(np.mean([item["auc_target"] for item in selected])),
            }
        )
    return rows
