"""Prospectively locked B02 near-primary gated-residual comparison."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from care_asd.config import CareASDConfig
from care_asd.data.official_vector_cache import OFFICIAL_FEATURE_DIM, load_official_vectors
from care_asd.evaluation.official_baseline import SCORE_COLUMNS, calculate_development_auc_metrics
from care_asd.models.gated_fusion import GatedNearResidualAutoencoder
from care_asd.models.official_compatible import OfficialCompatibleAutoencoder


@dataclass(frozen=True)
class GatedFusionResult:
    output_directory: Path
    score_path: Path
    summary_path: Path


def run_gated_fusion_development(
    *,
    near_cache_directory: str | Path,
    residual_cache_directory: str | Path,
    reliability_index_path: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config: CareASDConfig,
) -> GatedFusionResult:
    """Train B02 with normal-only data; residual only assists near reconstruction."""
    if config.training.epochs != 100 or config.training.batch_size != 256:
        raise ValueError("B02 locks 100 epochs and batch size 256")
    output, checkpoints = Path(output_directory), Path(checkpoint_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite B02 report: {output}")
    device = _cuda_device(config.training.device)
    index = _join_inputs(
        Path(near_cache_directory), Path(residual_cache_directory), Path(reliability_index_path)
    )
    train = index.loc[(index["dataset_split"] == "dev_train") & (index["condition"] == "normal")]
    test = index.loc[index["dataset_split"] == "dev_test"].sort_values("file_id", kind="stable")
    if train.empty or test.empty:
        raise ValueError(
            "B02 requires normal development train and labelled development test clips"
        )
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str | float]] = []
    cards: list[dict[str, object]] = []
    for machine_type, machine_train in train.groupby("machine_type", sort=True):
        _set_seed(config.experiment.seed)
        near, residual, gate = _concatenate(
            machine_train, Path(near_cache_directory), Path(residual_cache_directory)
        )
        model, loss, vector_count = _fit(near, residual, gate, config, device)
        checkpoint = (
            checkpoints / f"b02_gated_fusion_{machine_type}_seed{config.experiment.seed}.pt"
        )
        torch.save({"model_state": model.state_dict(), "machine_type": machine_type}, checkpoint)
        for row in test.loc[test["machine_type"] == machine_type].itertuples(index=False):
            near_values = load_official_vectors(
                Path(near_cache_directory) / str(row.near_cache_file)
            )
            residual_values = load_official_vectors(
                Path(residual_cache_directory) / str(row.residual_cache_file)
            )
            records.append(
                _score_record(
                    row,
                    _score(model, near_values, residual_values, float(row.reliability), device),
                    config.experiment.id,
                )
            )
        cards.append(
            {
                "machine_type": str(machine_type),
                "checkpoint": str(checkpoint),
                "parameters": sum(p.numel() for p in model.parameters()),
                "b00_parameters": sum(
                    p.numel() for p in OfficialCompatibleAutoencoder().parameters()
                ),
                "reliability_mean_train": float(machine_train["reliability"].mean()),
                "final_train_loss": loss,
                "train_feature_vectors": vector_count,
            }
        )
    scores = pd.DataFrame(records, columns=SCORE_COLUMNS).sort_values("file_id", kind="stable")
    if len(scores) != len(test) or scores["file_id"].duplicated().any():
        raise ValueError("B02 scores do not cover exactly one row per development test clip")
    score_path = output / "scores.csv"
    scores.to_csv(score_path, index=False)
    metrics_path = output / "metrics.json"
    calculate_development_auc_metrics(score_path, metrics_path)
    summary_path = output / "summary.csv"
    _summary(metrics_path, summary_path)
    (output / "model_card.json").write_text(
        json.dumps(cards, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "near_cache": str(near_cache_directory),
                "residual_cache": str(residual_cache_directory),
                "reliability_index": str(reliability_index_path),
                "config": config.model_dump(by_alias=True),
                "contract": {
                    "near_is_target": True,
                    "reliability": "mean cached path_confidence per clip",
                    "capacity_matched_to_b00": True,
                    "no_labels_used_for_features_or_training": True,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return GatedFusionResult(output, score_path, summary_path)


def _score_record(row: Any, score: float, experiment_id: str) -> dict[str, str | float]:
    return {
        "file_id": str(row.file_id),
        "machine_type": str(row.machine_type),
        "section": str(row.section),
        "domain": str(row.domain),
        "condition": str(row.condition),
        "anomaly_score": score,
        "model_id": "b02_near_primary_gated_residual_ae",
        "experiment_id": experiment_id,
    }


def _join_inputs(near_dir: Path, residual_dir: Path, reliability_path: Path) -> pd.DataFrame:
    near = pd.read_parquet(near_dir / "index.parquet")
    residual = pd.read_parquet(residual_dir / "index.parquet")
    if not reliability_path.is_file():
        raise FileNotFoundError(f"Reliability index does not exist: {reliability_path}")
    required = {
        "file_id",
        "cache_file",
        "vector_count",
        "dataset_split",
        "machine_type",
        "section",
        "domain",
        "condition",
    }
    for label, frame in (("near", near), ("residual", residual)):
        if not required.issubset(frame.columns) or frame["file_id"].duplicated().any():
            raise ValueError(f"{label} cache lacks unique B02 required columns")
    keys = ["file_id", "dataset_split", "machine_type", "section", "domain", "condition"]
    merged = near.rename(
        columns={"cache_file": "near_cache_file", "vector_count": "near_vector_count"}
    ).merge(
        residual[[*keys, "cache_file", "vector_count"]].rename(
            columns={"cache_file": "residual_cache_file", "vector_count": "residual_vector_count"}
        ),
        on=keys,
        validate="one_to_one",
    )
    reliability = pd.read_parquet(reliability_path)
    if (
        not {"file_id", "reliability"}.issubset(reliability.columns)
        or reliability["file_id"].duplicated().any()
    ):
        raise ValueError("Reliability index requires unique file_id and reliability")
    merged = merged.merge(reliability, on="file_id", validate="one_to_one")
    if (
        len(merged) != len(near)
        or (merged["near_vector_count"] != merged["residual_vector_count"]).any()
    ):
        raise ValueError("B02 inputs must have identical file coverage and vector counts")
    if (~merged["reliability"].between(0.0, 1.0)).any():
        raise ValueError("Reliability values must lie in [0, 1]")
    return merged


def _concatenate(
    rows: pd.DataFrame, near_dir: Path, residual_dir: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = [
        (
            load_official_vectors(near_dir / str(row.near_cache_file)),
            load_official_vectors(residual_dir / str(row.residual_cache_file)),
            float(row.reliability),
        )
        for row in rows.itertuples(index=False)
    ]
    if not pairs or any(
        near.shape != residual.shape or not len(near) for near, residual, _ in pairs
    ):
        raise ValueError("B02 training has empty or mismatched vectors")
    return (
        np.concatenate([near for near, _, _ in pairs]),
        np.concatenate([residual for _, residual, _ in pairs]),
        np.concatenate([np.full(len(near), gate, dtype=np.float32) for near, _, gate in pairs]),
    )


def _fit(
    near: np.ndarray,
    residual: np.ndarray,
    gate: np.ndarray,
    config: CareASDConfig,
    device: torch.device,
) -> tuple[GatedNearResidualAutoencoder, float, int]:
    held_out = int(np.ceil(0.1 * len(near)))
    selected = np.random.permutation(len(near))[held_out:]
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(near[selected]).float(),
            torch.from_numpy(residual[selected]).float(),
            torch.from_numpy(gate[selected]).float(),
        ),
        batch_size=256,
        shuffle=True,
        pin_memory=True,
    )
    model = GatedNearResidualAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    final_loss = float("nan")
    model.train()
    for _ in range(100):
        total, count = 0.0, 0
        for near_batch, residual_batch, gate_batch in loader:
            near_batch = near_batch.to(device, non_blocking=True)
            residual_batch = residual_batch.to(device, non_blocking=True)
            gate_batch = gate_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(near_batch, residual_batch, gate_batch)
            loss = torch.mean((prediction - near_batch) ** 2)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.detach()) * len(near_batch)
            count += len(near_batch)
        final_loss = total / max(count, 1)
    return model.eval(), final_loss, len(selected)


def _score(
    model: nn.Module, near: np.ndarray, residual: np.ndarray, gate: float, device: torch.device
) -> float:
    if near.shape != residual.shape or near.ndim != 2 or near.shape[1] != OFFICIAL_FEATURE_DIM:
        raise ValueError("Invalid B02 test vectors")
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(near), 256):
            near_batch = torch.from_numpy(near[start : start + 256]).to(device)
            residual_batch = torch.from_numpy(residual[start : start + 256]).to(device)
            gates = torch.full((len(near_batch),), gate, device=device)
            prediction, _ = model(near_batch, residual_batch, gates)
            total += float(torch.sum((prediction - near_batch) ** 2).cpu())
            count += prediction.numel()
    return total / max(count, 1)


def _cuda_device(requested: str) -> torch.device:
    if requested != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B02 requires an available CUDA GPU")
    return torch.device("cuda")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _summary(metrics_path: Path, output: Path) -> None:
    groups = list(json.loads(metrics_path.read_text(encoding="utf-8"))["groups"].values())
    pd.DataFrame(
        [
            {
                "frontend": "b02_near_primary_gated_residual",
                "mean_auc_all": float(np.mean([group["auc_all"] for group in groups])),
                "mean_pauc_all_max_fpr_0_1": float(
                    np.mean([group["pauc_all_max_fpr_0_1"] for group in groups])
                ),
                "mean_auc_source": float(np.mean([group["auc_source"] for group in groups])),
                "mean_auc_target": float(np.mean([group["auc_target"] for group in groups])),
            }
        ]
    ).to_csv(output, index=False)
