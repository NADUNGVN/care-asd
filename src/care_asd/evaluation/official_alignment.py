"""Internal, architecture-compatible reproduction of the official MSE baseline."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from care_asd.config import CareASDConfig
from care_asd.data.official_vector_cache import OFFICIAL_FEATURE_DIM, load_official_vectors
from care_asd.evaluation.official_baseline import SCORE_COLUMNS, calculate_development_auc_metrics
from care_asd.models.official_compatible import OfficialCompatibleAutoencoder


@dataclass(frozen=True)
class OfficialAlignmentResult:
    """Immutable evidence paths written by an internal official-alignment run."""

    output_directory: Path
    score_path: Path
    metrics_path: Path
    summary_path: Path


def run_official_alignment_development(
    *,
    cache_directory: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config: CareASDConfig,
) -> OfficialAlignmentResult:
    """Train the pinned AE contract on exact cache vectors and score clip MSE."""
    return _run_official_vector_development(
        cache_directory=cache_directory,
        output_directory=output_directory,
        checkpoint_directory=checkpoint_directory,
        config=config,
        frontend_name="official_compatible_near",
        model_id="official_compatible_dcase2026_ae_mse",
    )


def run_care_residual_alignment_development(
    *,
    cache_directory: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config: CareASDConfig,
) -> OfficialAlignmentResult:
    """Run the locked official AE protocol on a bounded CARE-residual cache."""
    return _run_official_vector_development(
        cache_directory=cache_directory,
        output_directory=output_directory,
        checkpoint_directory=checkpoint_directory,
        config=config,
        frontend_name="care_residual_official_stack",
        model_id="care_residual_official_compatible_dcase2026_ae_mse",
    )


def _run_official_vector_development(
    *,
    cache_directory: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config: CareASDConfig,
    frontend_name: str,
    model_id: str,
) -> OfficialAlignmentResult:
    """Internal common runner; public wrappers lock the permissible input contracts."""
    cache = Path(cache_directory)
    output = Path(output_directory)
    checkpoints = Path(checkpoint_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite alignment report: {output}")
    index_path = cache / "index.parquet"
    metadata_path = cache / "cache.json"
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("Official vector cache requires index.parquet and cache.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("feature_dim", -1)) != OFFICIAL_FEATURE_DIM:
        raise ValueError("Cache does not use the 640-dimensional official feature contract")
    index = pd.read_parquet(index_path)
    required = set(SCORE_COLUMNS).difference({"anomaly_score", "model_id", "experiment_id"}) | {
        "cache_file",
        "dataset_split",
        "vector_count",
    }
    missing = sorted(required.difference(index.columns))
    if missing:
        raise ValueError(f"Official vector cache index missing: {', '.join(missing)}")
    train = index.loc[(index["dataset_split"] == "dev_train") & (index["condition"] == "normal")]
    test = index.loc[index["dataset_split"] == "dev_test"]
    if train.empty or test.empty:
        raise ValueError("Cache lacks official normal development training or test rows")
    if config.training.epochs != 100 or config.training.batch_size != 256:
        raise ValueError("Official alignment requires exactly 100 epochs and batch size 256")
    device = _resolve_cuda(config.training.device)
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | float]] = []
    cards: list[dict[str, object]] = []
    for machine_type, machine_train in train.groupby("machine_type", sort=True):
        # Official shell scripts launch an independent seeded Python process per type.
        _set_official_seed(config.experiment.seed)
        vectors = _concatenate_vectors(cache, machine_train)
        model, final_loss, train_vectors = _fit_machine(vectors, config, device)
        checkpoint_path = checkpoints / f"official_alignment_{machine_type}_seed{config.experiment.seed}.pt"
        torch.save(
            {"model_state": model.state_dict(), "machine_type": machine_type, "seed": config.experiment.seed},
            checkpoint_path,
        )
        machine_test = test.loc[test["machine_type"] == machine_type].sort_values("file_id")
        for row in machine_test.itertuples(index=False):
            values = load_official_vectors(cache / str(row.cache_file))
            score = _score_clip(model, values, device, config.training.batch_size)
            records.append(
                {
                    "file_id": str(row.file_id),
                    "machine_type": str(row.machine_type),
                    "section": str(row.section),
                    "domain": str(row.domain),
                    "condition": str(row.condition),
                    "anomaly_score": score,
                    "model_id": model_id,
                    "experiment_id": config.experiment.id,
                }
            )
        cards.append(
            {
                "checkpoint": str(checkpoint_path),
                "final_train_loss": final_loss,
                "machine_type": str(machine_type),
                "parameters": sum(item.numel() for item in model.parameters()),
                "train_feature_vectors": train_vectors,
                "train_clips": len(machine_train),
            }
        )
    score_path = output / "scores.csv"
    scores = pd.DataFrame(records, columns=SCORE_COLUMNS).sort_values("file_id", kind="stable")
    if len(scores) != len(test) or scores["file_id"].duplicated().any():
        raise ValueError("Alignment score coverage does not match development test clips")
    scores.to_csv(score_path, index=False)
    metrics_path = output / "metrics.json"
    calculate_development_auc_metrics(score_path, metrics_path)
    summary_path = output / "summary.csv"
    _write_summary(metrics_path, summary_path, frontend_name)
    (output / "model_card.json").write_text(
        json.dumps(cards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "cache": str(cache),
                "cache_metadata": metadata,
                "config": config.model_dump(by_alias=True),
                "device": str(device),
                "official_contract": {
                    "architecture": "640-128-128-128-128-8-128-128-128-128-640",
                    "batch_norm": {"eps": 1.0e-3, "momentum": 0.01},
                    "score": "mean clip reconstruction MSE",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return OfficialAlignmentResult(output, score_path, metrics_path, summary_path)


def _concatenate_vectors(cache: Path, rows: pd.DataFrame) -> np.ndarray:
    values = [load_official_vectors(cache / str(row.cache_file)) for row in rows.itertuples(index=False)]
    if not values or any(item.size == 0 for item in values):
        raise ValueError("Official training cache includes an empty feature vector matrix")
    return np.concatenate(values, axis=0)


def _fit_machine(
    vectors: np.ndarray, config: CareASDConfig, device: torch.device
) -> tuple[OfficialCompatibleAutoencoder, float, int]:
    # Official code uses sklearn train_test_split(..., test_size=0.1), which only
    # removes a validation subset; it does not select epochs or hyperparameters.
    held_out = int(np.ceil(0.1 * len(vectors)))
    permutation = np.random.permutation(len(vectors))
    train_values = torch.from_numpy(vectors[permutation[held_out:]]).float()
    loader = DataLoader(
        TensorDataset(train_values),
        batch_size=config.training.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    model = OfficialCompatibleAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    final_loss = float("nan")
    model.train()
    for _ in range(config.training.epochs):
        total, count = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = torch.mean((reconstruction - batch) ** 2)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        final_loss = total / max(count, 1)
    return model.eval(), final_loss, len(train_values)


def _score_clip(
    model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int
) -> float:
    if not len(values):
        raise ValueError("Cannot score an empty official feature vector matrix")
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            reconstruction, _ = model(batch)
            total += float(torch.sum((reconstruction - batch) ** 2).cpu())
            count += reconstruction.numel()
    return total / count


def _resolve_cuda(requested: str) -> torch.device:
    if requested != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Official alignment requires an available CUDA GPU")
    return torch.device("cuda")


def _set_official_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _write_summary(metrics_path: Path, output: Path, frontend_name: str) -> None:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    groups = list(metrics["groups"].values())
    pd.DataFrame(
        [
            {
                "frontend": frontend_name,
                "mean_auc_all": float(np.mean([group["auc_all"] for group in groups])),
                "mean_pauc_all_max_fpr_0_1": float(
                    np.mean([group["pauc_all_max_fpr_0_1"] for group in groups])
                ),
                "mean_auc_source": float(np.mean([group["auc_source"] for group in groups])),
                "mean_auc_target": float(np.mean([group["auc_target"] for group in groups])),
            }
        ]
    ).to_csv(output, index=False)
