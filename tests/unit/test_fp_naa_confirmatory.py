from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

pytest.importorskip("torch")

from care_asd.evaluation.fp_naa_confirmatory import _confirmatory_gate, _seed_score_path
from care_asd.fp_naa_config import FPNAAConfig


def _config_with_permissive_gates() -> FPNAAConfig:
    payload = yaml.safe_load(Path("configs/experiment/fp_naa_v1.yaml").read_text())
    payload["gates"].update(
        {
            "confirmatory_minimum_ensemble_official_score": 0.5,
            "confirmatory_minimum_gain_over_c1": 0.005,
            "confirmatory_maximum_machine_drop": 0.5,
        }
    )
    return FPNAAConfig.model_validate(payload)


def _metrics(path: Path, score: float, cell: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "official_score": score,
                "groups": {
                    "fan/section_00": {
                        "auc_source": cell,
                        "auc_target": cell,
                        "pauc_all_max_fpr_0_1": cell,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _c0_scores(path: Path) -> Path:
    design = [
        ("source", "normal", 0.1),
        ("source", "normal", 0.2),
        ("target", "normal", 0.15),
        ("target", "normal", 0.25),
        ("source", "anomaly", 0.8),
        ("target", "anomaly", 0.9),
    ]
    pd.DataFrame(
        [
            {
                "file_id": f"clip-{index}",
                "machine_type": "fan",
                "section": "section_00",
                "domain": domain,
                "condition": condition,
                "anomaly_score": score,
                "model_id": "c0",
                "experiment_id": "unit",
            }
            for index, (domain, condition, score) in enumerate(design)
        ]
    ).to_csv(path, index=False)
    return path


def test_confirmatory_gate_requires_exact_ci_and_all_retention_checks(tmp_path: Path) -> None:
    rows = []
    for seed in (13711, 42, 2026, 3407, 777):
        for candidate, score in (("c1_mse", 0.61), ("c2_fault_preserving", 0.62)):
            rows.append(
                {
                    "seed": seed,
                    "candidate": candidate,
                    "official_score": score,
                    "in_support_retention_median": 0.95,
                    "in_support_retention_q05": 0.80,
                    "heldout_retention_median": 0.90,
                    "heldout_retention_q05": 0.70,
                }
            )
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text(
        json.dumps(
            {
                "observed": {
                    "reference_official_score": 0.61,
                    "candidate_official_score": 0.62,
                },
                "bootstrap": {
                    "delta_candidate_minus_reference": {
                        "mean": 0.01,
                        "ci95_low": 0.001,
                        "ci95_high": 0.02,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    gate = _confirmatory_gate(
        summary=pd.DataFrame(rows),
        c0_scores=_c0_scores(tmp_path / "c0.csv"),
        c1_metrics=_metrics(tmp_path / "c1.json", 0.61, 0.61),
        c2_metrics=_metrics(tmp_path / "c2.json", 0.62, 0.62),
        bootstrap_path=bootstrap,
        config=_config_with_permissive_gates(),
    )
    assert gate["checks"]["core_confirmatory"] is True
    assert gate["checks"]["bootstrap_ci_low"] is True
    assert gate["passed"] is False


def test_seed_score_path_reuses_only_registered_screening_seeds(tmp_path: Path) -> None:
    screening = tmp_path / "screening"
    output = tmp_path / "confirmatory"
    old = screening / "seed42" / "c1_mse" / "scores.csv"
    new = output / "seed777" / "c1_mse" / "scores.csv"
    old.parent.mkdir(parents=True)
    new.parent.mkdir(parents=True)
    old.touch()
    new.touch()
    assert _seed_score_path(screening, output, 42, "c1_mse", {42}) == old
    assert _seed_score_path(screening, output, 777, "c1_mse", {42}) == new
