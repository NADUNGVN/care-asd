"""Contract tests for official DCASE baseline score normalization."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from care_asd.evaluation.official_baseline import (
    SCORE_COLUMNS,
    calculate_development_auc_metrics,
    normalize_official_development_scores,
)


def _write_manifest(path: Path) -> Path:
    records = []
    for domain in ("source", "target"):
        for condition in ("normal", "anomaly"):
            basename = f"section_00_{domain}_test_{condition}_0000.wav"
            records.append(
                {
                    "file_id": f"ToyCar/test/{basename}",
                    "relative_path": f"ToyCar/test/{basename}",
                    "machine_type": "ToyCar",
                    "section": "section_00",
                    "domain": domain,
                    "condition": condition,
                    "dataset_split": "dev_test",
                }
            )
    pd.DataFrame(records).to_parquet(path, index=False)
    return path


def _write_scores(directory: Path) -> Path:
    directory.mkdir()
    path = directory / "anomaly_score_DCASE2026T2ToyCar_section_00_test_seed13711.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(
            [
                ["section_00_source_test_normal_0000.wav", 0.1],
                ["section_00_source_test_anomaly_0000.wav", 0.9],
                ["section_00_target_test_normal_0000.wav", 0.2],
                ["section_00_target_test_anomaly_0000.wav", 0.8],
            ]
        )
    return directory


def test_normalize_official_scores_and_compute_metrics(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.parquet")
    output = normalize_official_development_scores(
        official_score_directory=_write_scores(tmp_path / "official"),
        manifest_path=manifest,
        score_mode="mse",
        experiment_id="baseline_dev_seed13711",
        output_path=tmp_path / "scores.csv",
    )

    scores = pd.read_csv(output)
    assert list(scores.columns) == SCORE_COLUMNS
    assert len(scores) == 4
    assert set(scores["model_id"]) == {"official_dcase2026_ae_mse"}

    metrics_path = calculate_development_auc_metrics(output, tmp_path / "metrics.json")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    group = metrics["groups"]["ToyCar/section_00"]
    assert group["auc_all"] == pytest.approx(1.0)
    assert group["auc_source"] == pytest.approx(1.0)
    assert group["auc_target"] == pytest.approx(1.0)


def test_normalizer_accepts_official_dcase2026_machine_prefix(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.parquet")
    output = normalize_official_development_scores(
        official_score_directory=_write_scores(tmp_path / "official"),
        manifest_path=manifest,
        score_mode="mse",
        experiment_id="baseline_dev_seed13711",
        output_path=tmp_path / "scores.csv",
    )

    assert len(pd.read_csv(output)) == 4


def test_normalizer_rejects_incomplete_official_scores(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "manifest.parquet")
    directory = _write_scores(tmp_path / "official")
    score_file = next(directory.glob("*.csv"))
    score_file.write_text("section_00_source_test_normal_0000.wav,0.1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="coverage mismatch"):
        normalize_official_development_scores(
            official_score_directory=directory,
            manifest_path=manifest,
            score_mode="mahala",
            experiment_id="baseline_dev_seed13711",
            output_path=tmp_path / "scores.csv",
        )
