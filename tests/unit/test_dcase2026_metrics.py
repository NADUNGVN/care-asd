from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from care_asd.evaluation.dcase2026_metrics import calculate_dcase2026_official_metrics


def _row(
    index: int,
    *,
    domain: str,
    condition: str,
    score: float,
) -> dict[str, str | float]:
    return {
        "file_id": f"clip-{index}",
        "machine_type": "fan",
        "section": "section_00",
        "domain": domain,
        "condition": condition,
        "anomaly_score": score,
        "model_id": "test",
        "experiment_id": "test",
    }


def test_exact_metric_uses_all_anomalies_for_each_domain_auc(tmp_path: Path) -> None:
    # Source anomalies are intentionally below source normals. Target anomalies are high. A
    # same-domain-positive implementation would return source AUC=0; DCASE 2026 returns 0.5.
    rows = [
        _row(0, domain="source", condition="normal", score=0.4),
        _row(1, domain="source", condition="normal", score=0.5),
        _row(2, domain="target", condition="normal", score=0.1),
        _row(3, domain="target", condition="normal", score=0.2),
        _row(4, domain="source", condition="anomaly", score=0.3),
        _row(5, domain="source", condition="anomaly", score=0.35),
        _row(6, domain="target", condition="anomaly", score=0.8),
        _row(7, domain="target", condition="anomaly", score=0.9),
    ]
    scores = tmp_path / "scores.csv"
    pd.DataFrame(rows).to_csv(scores, index=False)

    metrics = calculate_dcase2026_official_metrics(scores, tmp_path / "metrics.json")
    payload = json.loads(metrics.read_text(encoding="utf-8"))

    assert payload["groups"]["fan/section_00"]["auc_source"] == pytest.approx(0.5)
    assert payload["groups"]["fan/section_00"]["auc_target"] == pytest.approx(1.0)
    assert payload["cell_count"] == 3
    assert 0.0 < payload["official_score"] <= 1.0


def test_exact_metric_rejects_missing_domain_normals(tmp_path: Path) -> None:
    scores = tmp_path / "scores.csv"
    pd.DataFrame(
        [
            _row(0, domain="source", condition="normal", score=0.1),
            _row(1, domain="source", condition="anomaly", score=0.9),
        ]
    ).to_csv(scores, index=False)

    with pytest.raises(ValueError, match="No target normal"):
        calculate_dcase2026_official_metrics(scores, tmp_path / "metrics.json")

