from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from care_asd.evaluation.reference_safety import run_reference_safety_development
from care_asd.reference_safety_config import ReferenceSafetyExperimentConfig


def test_development_runner_writes_all_capacity_matched_systems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ZeroAutoencoder(torch.nn.Module):
        def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros_like(values), values[:, :8]

    monkeypatch.setattr(
        "care_asd.evaluation.reference_safety._fit_model",
        lambda vectors, config, device: (ZeroAutoencoder().to(device).eval(), 0.0),
    )

    def write_bootstrap(**kwargs: object) -> Path:
        output = Path(str(kwargs["output_path"]))
        output.write_text("{}\n", encoding="utf-8")
        return output

    monkeypatch.setattr(
        "care_asd.evaluation.reference_safety.write_paired_bootstrap_comparison",
        write_bootstrap,
    )
    cache = tmp_path / "cache"
    features = cache / "features"
    features.mkdir(parents=True)
    records: list[dict[str, object]] = []
    rng = np.random.default_rng(7)
    cases = [
        ("dev_train", "normal", "source"),
        ("dev_train", "normal", "source"),
        ("dev_train", "normal", "target"),
        ("dev_train", "normal", "target"),
        ("dev_test", "normal", "source"),
        ("dev_test", "anomaly", "source"),
        ("dev_test", "normal", "target"),
        ("dev_test", "anomaly", "target"),
    ]
    for index, (split, condition, domain) in enumerate(cases):
        relative = f"features/{index}.npz"
        near = rng.normal(loc=index * 0.02, scale=0.1, size=(6, 640)).astype(np.float32)
        refsub = (near * 0.9).astype(np.float32)
        np.savez_compressed(cache / relative, near_vectors=near, refsub_vectors=refsub)
        records.append(
            {
                "file_id": f"machine/{split}/clip_{index}.wav",
                "relative_path": f"machine/{split}/clip_{index}.wav",
                "machine_type": "machine",
                "section": "section_00",
                "domain": domain,
                "condition": condition,
                "dataset_split": split,
                "cache_file": relative,
            }
        )
    pd.DataFrame.from_records(records).to_parquet(cache / "index.parquet", index=False)
    pd.DataFrame(
        [
            {
                "machine_type": "machine",
                "section": "section_00",
                "risk_score": 0.1,
                "noise_reduction_l05_db": 2.0,
            }
        ]
    ).to_parquet(cache / "profiles.parquet", index=False)
    (cache / "cache.json").write_text(
        json.dumps({"views": ["near", "refsub"], "feature_dim": 640}), encoding="utf-8"
    )
    policy = tmp_path / "policy.yaml"
    policy.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "risk_max": 0.2,
                "benefit_min_db": 1.0,
                "calibration_cases": 32,
                "holdout_cases": 32,
                "calibration_false_safe_rate": 0.0,
                "calibration_coverage": 0.5,
                "source": "semi_synthetic_only",
            }
        ),
        encoding="utf-8",
    )
    config = ReferenceSafetyExperimentConfig(
        training={
            "epochs": 100,
            "batch_size": 256,
            "learning_rate": 1.0e-3,
            "device": "cpu",
            "screening_seeds": [1],
            "replication_seeds": [1],
        }
    )

    result = run_reference_safety_development(
        cache_directory=cache,
        policy_path=policy,
        output_directory=tmp_path / "output",
        checkpoint_directory=tmp_path / "checkpoints",
        config=config,
    )

    assert result.summary_path.is_file()
    assert set(pd.read_csv(result.summary_path)["system"]) == {
        "near",
        "unconditional_refsub",
        "safe_ref",
    }
    assert (result.output_directory / "safe_ref" / "scores.csv").is_file()
