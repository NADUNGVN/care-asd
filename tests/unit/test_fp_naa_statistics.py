from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from care_asd.evaluation.dcase2026_metrics import calculate_dcase2026_official_metrics
from care_asd.evaluation.fp_naa_statistics import write_exact_official_paired_bootstrap


def _scores(path: Path, values: list[float]) -> Path:
    rows = []
    design = [
        ("source", "normal"),
        ("source", "normal"),
        ("target", "normal"),
        ("target", "normal"),
        ("source", "anomaly"),
        ("source", "anomaly"),
        ("target", "anomaly"),
        ("target", "anomaly"),
    ]
    for index, ((domain, condition), score) in enumerate(zip(design, values, strict=True)):
        rows.append(
            {
                "file_id": f"clip-{index}",
                "machine_type": "fan",
                "section": "section_00",
                "domain": domain,
                "condition": condition,
                "anomaly_score": score,
                "model_id": path.stem,
                "experiment_id": "unit",
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_exact_bootstrap_is_paired_and_uses_official_harmonic_score(tmp_path: Path) -> None:
    reference = _scores(tmp_path / "reference.csv", [0.4, 0.5, 0.1, 0.2, 0.3, 0.35, 0.8, 0.9])
    candidate = _scores(tmp_path / "candidate.csv", [0.2, 0.3, 0.0, 0.1, 0.7, 0.75, 0.8, 0.9])
    output = write_exact_official_paired_bootstrap(
        reference_scores=reference,
        candidate_scores=candidate,
        output_path=tmp_path / "bootstrap.json",
        iterations=200,
        seed=7,
    )
    payload = json.loads(output.read_text())
    official = calculate_dcase2026_official_metrics(candidate, tmp_path / "candidate_metrics.json")
    official_payload = json.loads(official.read_text())
    assert payload["observed"]["candidate_official_score"] == pytest.approx(
        official_payload["official_score"]
    )
    assert payload["observed"]["delta_candidate_minus_reference"] > 0.0
    assert payload["bootstrap"]["delta_candidate_minus_reference"]["ci95_low"] >= 0.0
    assert payload["stratification"] == "machine_type/section/domain/condition"


def test_identical_scores_have_exactly_zero_paired_delta(tmp_path: Path) -> None:
    scores = [0.1, 0.2, 0.15, 0.25, 0.7, 0.8, 0.75, 0.85]
    reference = _scores(tmp_path / "reference.csv", scores)
    candidate = _scores(tmp_path / "candidate.csv", scores)
    output = write_exact_official_paired_bootstrap(
        reference_scores=reference,
        candidate_scores=candidate,
        output_path=tmp_path / "bootstrap.json",
        iterations=100,
        seed=11,
    )
    interval = json.loads(output.read_text())["bootstrap"]["delta_candidate_minus_reference"]
    assert interval["mean"] == pytest.approx(0.0)
    assert interval["ci95_low"] == pytest.approx(0.0)
    assert interval["ci95_high"] == pytest.approx(0.0)
