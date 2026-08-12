"""GPU MVP ablation runner with normal-only fitting and immutable evidence."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from care_asd.config import CareASDConfig
from care_asd.data.neural_cache import load_cached_feature
from care_asd.evaluation.official_baseline import SCORE_COLUMNS, calculate_development_auc_metrics
from care_asd.models.mvp_autoencoder import LightweightNearAutoencoder, approximate_parameter_count

MvpAblation = Literal["a00_near", "a01_near_far", "a02_care_multiview"]
_ABLATION_CHANNELS: dict[MvpAblation, tuple[str, ...]] = {
    "a00_near": ("near",),
    "a01_near_far": ("near", "far"),
    "a02_care_multiview": (
        "near",
        "far",
        "residual",
        "coherence",
        "log_ratio",
        "phase_sin",
        "phase_cos",
        "path_confidence",
    ),
}


@dataclass(frozen=True)
class MvpNeuralResult:
    """Paths emitted by a complete immutable MVP neural run."""

    output_directory: Path
    score_path: Path
    metrics_path: Path
    summary_path: Path
    model_card_path: Path


def available_mvp_ablations() -> tuple[MvpAblation, ...]:
    """Return the three pre-registered MVP comparisons in stable order."""
    return tuple(_ABLATION_CHANNELS)


def run_mvp_neural_development(
    *,
    cache_directory: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config: CareASDConfig,
    ablation: MvpAblation,
    epochs: int | None = None,
) -> MvpNeuralResult:
    """Train one ablation per machine from normal rows and score dev test only."""
    cache = Path(cache_directory)
    output = Path(output_directory)
    checkpoints = Path(checkpoint_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite neural report: {output}")
    index_path = cache / "index.parquet"
    metadata_path = cache / "cache.json"
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Cache requires index.parquet and cache.json")
    if ablation not in _ABLATION_CHANNELS:
        raise ValueError(f"Unknown MVP ablation: {ablation}")
    _set_seed(config.experiment.seed)
    device = _resolve_device(config.training.device)
    index = pd.read_parquet(index_path)
    required = set(SCORE_COLUMNS).difference({"anomaly_score", "model_id", "experiment_id"}) | {
        "cache_file",
        "dataset_split",
    }
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"Cache index missing required columns: {', '.join(missing)}")
    train = index.loc[
        (index["dataset_split"] == "dev_train") & (index["condition"] == "normal")
    ].copy()
    test = index.loc[index["dataset_split"] == "dev_test"].copy()
    if train.empty or test.empty or not set(test["condition"]).issubset({"normal", "anomaly"}):
        raise ValueError(
            "Cache lacks normal development training or labelled development test rows"
        )
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    chosen_epochs = config.training.epochs if epochs is None else epochs
    if chosen_epochs < 1:
        raise ValueError("epochs must be positive")
    channels = _ABLATION_CHANNELS[ablation]
    rows: list[dict[str, str | float]] = []
    model_cards: list[dict[str, object]] = []
    for machine_type, machine_train in train.groupby("machine_type", sort=True):
        model, normalizer, train_loss = _fit_machine(
            cache=cache,
            train=machine_train,
            channels=channels,
            config=config,
            epochs=chosen_epochs,
            device=device,
        )
        checkpoint = checkpoints / f"{ablation}_{machine_type}_seed{config.experiment.seed}.pt"
        torch.save(
            {
                "ablation": ablation,
                "channels": channels,
                "model_state": model.state_dict(),
                "normalizer": normalizer,
                "seed": config.experiment.seed,
            },
            checkpoint,
        )
        machine_test = test.loc[test["machine_type"] == machine_type].sort_values(
            "file_id", kind="stable"
        )
        if machine_test.empty:
            raise ValueError(f"No development test rows for machine: {machine_type}")
        scores = _score_machine(model, cache, machine_test, channels, normalizer, device)
        for item, score in zip(machine_test.to_dict(orient="records"), scores, strict=True):
            rows.append(
                {
                    "file_id": str(item["file_id"]),
                    "machine_type": str(item["machine_type"]),
                    "section": str(item["section"]),
                    "domain": str(item["domain"]),
                    "condition": str(item["condition"]),
                    "anomaly_score": score,
                    "model_id": f"mvp_{ablation}_near_reconstruction_mse",
                    "experiment_id": config.experiment.id,
                }
            )
        model_cards.append(
            {
                "ablation": ablation,
                "channels": list(channels),
                "checkpoint": str(checkpoint),
                "machine_type": str(machine_type),
                "parameters": approximate_parameter_count(model),
                "seed": config.experiment.seed,
                "train_clips": len(machine_train),
                "train_loss_final": train_loss,
            }
        )
    score_path = output / "scores.csv"
    score_frame = pd.DataFrame(rows, columns=SCORE_COLUMNS).sort_values("file_id", kind="stable")
    if len(score_frame) != len(test) or score_frame["file_id"].duplicated().any():
        raise ValueError("Neural score coverage does not match development test rows")
    score_frame.to_csv(score_path, index=False)
    metrics_path = output / "metrics.json"
    calculate_development_auc_metrics(score_path, metrics_path)
    summary_path = output / "summary.csv"
    _write_summary(metrics_path, ablation, summary_path)
    model_card_path = output / "model_card.json"
    model_card_path.write_text(
        json.dumps(model_cards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "ablation": ablation,
                "cache": str(cache),
                "cache_metadata": json.loads(metadata_path.read_text(encoding="utf-8")),
                "channels": list(channels),
                "config": config.model_dump(by_alias=True),
                "device": str(device),
                "epochs": chosen_epochs,
                "seed": config.experiment.seed,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return MvpNeuralResult(output, score_path, metrics_path, summary_path, model_card_path)


class _CachedClipDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        cache: Path,
        rows: pd.DataFrame,
        channels: tuple[str, ...],
        normalizer: dict[str, list[float]],
    ) -> None:
        self._cache = cache
        self._rows = rows.to_dict(orient="records")
        self._channels = channels
        self._mean = np.asarray(normalizer["mean"], dtype=np.float32)[:, None, None]
        self._std = np.asarray(normalizer["std"], dtype=np.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        row = self._rows[index]
        values = load_cached_feature(self._cache / str(row["cache_file"]), self._channels)
        values = (values - self._mean) / self._std
        target = values[:1]
        return torch.from_numpy(_crop_time(values)), torch.from_numpy(_crop_time(target))


def _fit_machine(
    *,
    cache: Path,
    train: pd.DataFrame,
    channels: tuple[str, ...],
    config: CareASDConfig,
    epochs: int,
    device: torch.device,
) -> tuple[LightweightNearAutoencoder, dict[str, list[float]], float]:
    normalizer = _fit_normalizer(cache, train, channels)
    dataset = _CachedClipDataset(cache, train, channels, normalizer)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = LightweightNearAutoencoder(len(channels), embedding_dim=config.model.embedding_dim).to(
        device
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    use_amp = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_value = float("nan")
    model.train()
    for _ in range(epochs):
        total, count = 0.0, 0
        for inputs, targets in loader:
            inputs, targets = (
                inputs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                prediction = model(inputs)
                loss = torch.mean((prediction - targets) ** 2)
            scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(inputs)
            count += len(inputs)
        loss_value = total / max(count, 1)
    return model.eval(), normalizer, loss_value


def _score_machine(
    model: nn.Module,
    cache: Path,
    rows: pd.DataFrame,
    channels: tuple[str, ...],
    normalizer: dict[str, list[float]],
    device: torch.device,
) -> list[float]:
    mean = np.asarray(normalizer["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(normalizer["std"], dtype=np.float32)[:, None, None]
    scores: list[float] = []
    with torch.no_grad():
        for row in rows.to_dict(orient="records"):
            values = load_cached_feature(cache / str(row["cache_file"]), channels)
            values = (values - mean) / std
            inputs = torch.from_numpy(values[None]).to(device)
            target = inputs[:, :1]
            prediction = model(inputs)
            scores.append(float(torch.mean((prediction - target) ** 2).cpu()))
    return scores


def _fit_normalizer(
    cache: Path, rows: pd.DataFrame, channels: tuple[str, ...]
) -> dict[str, list[float]]:
    total = np.zeros(len(channels), dtype=np.float64)
    squared = np.zeros(len(channels), dtype=np.float64)
    count = 0
    for row in rows.to_dict(orient="records"):
        values = load_cached_feature(cache / str(row["cache_file"]), channels).astype(np.float64)
        total += values.sum(axis=(1, 2))
        squared += (values**2).sum(axis=(1, 2))
        count += values.shape[1] * values.shape[2]
    mean = total / count
    variance = np.maximum(squared / count - mean**2, 1.0e-6)
    return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}


def _crop_time(values: np.ndarray, frames: int = 64) -> np.ndarray:
    if values.shape[-1] >= frames:
        start = (values.shape[-1] - frames) // 2
        return values[..., start : start + frames].copy()
    return np.pad(values, ((0, 0), (0, 0), (0, frames - values.shape[-1]))).astype(np.float32)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_summary(metrics_path: Path, ablation: MvpAblation, output: Path) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    groups = list(metrics["groups"].values())
    output.write_text(
        pd.DataFrame(
            [
                {
                    "ablation": ablation,
                    "mean_auc_all": float(np.mean([group["auc_all"] for group in groups])),
                    "mean_pauc_all_max_fpr_0_1": float(
                        np.mean([group["pauc_all_max_fpr_0_1"] for group in groups])
                    ),
                    "mean_auc_source": float(np.mean([group["auc_source"] for group in groups])),
                    "mean_auc_target": float(np.mean([group["auc_target"] for group in groups])),
                }
            ]
        ).to_csv(index=False),
        encoding="utf-8",
    )
