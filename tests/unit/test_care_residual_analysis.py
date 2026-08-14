"""Phase 8 frozen B00/B01 analysis contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from care_asd.evaluation.care_residual_analysis import analyze_care_residual_development
from care_asd.evaluation.official_baseline import SCORE_COLUMNS


def test_analysis_joins_identical_test_coverage_and_writes_evidence(tmp_path: Path) -> None:
    near_cache = _write_cache(tmp_path / "near", values=(2.0, 4.0))
    residual_cache = _write_cache(tmp_path / "residual", values=(1.0, 3.0))
    reference = _write_scores(tmp_path / "b00.csv", scores=(0.1, 0.2))
    candidate = _write_scores(tmp_path / "b01.csv", scores=(0.3, 0.1))

    result = analyze_care_residual_development(
        near_cache_directory=near_cache,
        residual_cache_directory=residual_cache,
        reference_scores=reference,
        candidate_scores=candidate,
        output_directory=tmp_path / "analysis",
    )

    per_clip = pd.read_csv(result.per_clip_path).sort_values("file_id")
    assert result.report_path.is_file()
    assert len(per_clip) == 2
    assert per_clip["residual_minus_near_logmel_db"].tolist() == [-1.0, -1.0]
    assert np.allclose(per_clip["score_delta_b01_minus_b00"], [0.2, -0.1])


def _write_cache(directory: Path, *, values: tuple[float, float]) -> Path:
    features = directory / "features"
    features.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index, value in enumerate(values):
        name = f"clip-{index}.npz"
        np.savez_compressed(features / name, vectors=np.full((2, 640), value, dtype=np.float32))
        rows.append(
            {
                "file_id": f"clip-{index}",
                "machine_type": "fan",
                "section": "section_00",
                "domain": "source",
                "condition": "normal" if index == 0 else "anomaly",
                "dataset_split": "dev_test",
                "cache_file": f"features/{name}",
                "vector_count": 2,
            }
        )
    pd.DataFrame(rows).to_parquet(directory / "index.parquet", index=False)
    (directory / "cache.json").write_text(json.dumps({"feature_dim": 640}), encoding="utf-8")
    return directory


def _write_scores(path: Path, *, scores: tuple[float, float]) -> Path:
    rows: list[dict[str, object]] = []
    for index, score in enumerate(scores):
        rows.append(
            {
                "file_id": f"clip-{index}",
                "machine_type": "fan",
                "section": "section_00",
                "domain": "source",
                "condition": "normal" if index == 0 else "anomaly",
                "anomaly_score": score,
                "model_id": "test",
                "experiment_id": "test",
            }
        )
    pd.DataFrame(rows, columns=SCORE_COLUMNS).to_csv(path, index=False)
    return path
