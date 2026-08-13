"""Contracts for the cached Phase 5 MVP GPU model and normal-only runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from care_asd.config import CareASDConfig, TrainingConfig
from care_asd.evaluation.mvp_neural import (
    run_mvp_neural_development,
    run_mvp_neural_replication_development,
    run_mvp_neural_screening_development,
)
from care_asd.evaluation.paired_bootstrap import (
    write_paired_bootstrap_comparison,
    write_seed_ensemble_scores,
)
from care_asd.models import LightweightNearAutoencoder

torch = pytest.importorskip("torch")


def test_lightweight_autoencoder_reconstructs_near_shape() -> None:
    model = LightweightNearAutoencoder(input_channels=8, embedding_dim=64)
    values = torch.randn(2, 8, 128, 64)

    assert model(values).shape == (2, 1, 128, 64)
    assert model.encode(values).shape == (2, 64)


def test_mvp_runner_fits_only_train_normal_rows_and_scores_all_test_rows(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    features = cache / "features"
    features.mkdir(parents=True)
    rows: list[dict[str, str]] = []
    for index, (split, domain, condition) in enumerate(
        (
            ("dev_train", "source", "normal"),
            ("dev_train", "target", "normal"),
            ("dev_test", "source", "normal"),
            ("dev_test", "source", "anomaly"),
            ("dev_test", "target", "normal"),
            ("dev_test", "target", "anomaly"),
        )
    ):
        cache_file = f"features/{index}.npz"
        maps = {
            name: np.full((128, 64), index + channel, dtype=np.float32)
            for channel, name in enumerate(
                (
                    "near",
                    "far",
                    "residual",
                    "coherence",
                    "log_ratio",
                    "phase_sin",
                    "phase_cos",
                    "path_confidence",
                )
            )
        }
        np.savez_compressed(cache / cache_file, **maps)
        rows.append(
            {
                "file_id": f"ToyCar/test/{index}.wav",
                "relative_path": f"ToyCar/test/{index}.wav",
                "cache_file": cache_file,
                "machine_type": "ToyCar",
                "section": "section_00",
                "domain": domain,
                "condition": condition,
                "dataset_split": split,
            }
        )
    pd.DataFrame(rows).to_parquet(cache / "index.parquet", index=False)
    (cache / "cache.json").write_text(json.dumps({"test": True}), encoding="utf-8")
    config = CareASDConfig(
        experiment={"id": "unit_mvp", "seed": 7},
        training=TrainingConfig(
            epochs=1, batch_size=2, num_workers=0, device="cpu", mixed_precision=False
        ),
    )

    result = run_mvp_neural_development(
        cache_directory=cache,
        output_directory=tmp_path / "report",
        checkpoint_directory=tmp_path / "checkpoints",
        config=config,
        ablation="a01_near_far",
    )

    scores = pd.read_csv(result.score_path)
    assert len(scores) == 4
    assert set(scores["condition"]) == {"normal", "anomaly"}
    assert result.metrics_path.is_file() and result.model_card_path.is_file()
    bootstrap = write_paired_bootstrap_comparison(
        reference_scores=result.score_path,
        candidate_scores=result.score_path,
        output_path=tmp_path / "bootstrap.json",
        iterations=100,
    )
    assert (
        json.loads(bootstrap.read_text(encoding="utf-8"))["metric_delta_candidate_minus_reference"][
            "mean_auc"
        ]["mean"]
        == 0.0
    )

    screening = run_mvp_neural_screening_development(
        cache_directory=cache,
        output_directory=tmp_path / "screening",
        checkpoint_directory=tmp_path / "screening_checkpoints",
        config=config,
        preload_workers=1,
    )
    assert len(screening.results) == 3
    assert screening.summary_path.is_file()

    replication = run_mvp_neural_replication_development(
        cache_directory=cache,
        output_directory=tmp_path / "replication",
        checkpoint_directory=tmp_path / "replication_checkpoints",
        config=config,
        seeds=(42,),
        ablations=("a00_near",),
        preload_workers=1,
    )
    assert len(replication.results) == 1
    assert replication.summary_path.is_file()

    ensemble = write_seed_ensemble_scores(
        score_paths=[result.score_path, result.score_path],
        output_path=tmp_path / "ensemble.csv",
        model_id="unit_ensemble",
        experiment_id="unit_ensemble",
    )
    assert len(pd.read_csv(ensemble)) == 4
