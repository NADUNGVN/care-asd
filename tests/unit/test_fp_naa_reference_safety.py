from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("torch")

from care_asd.evaluation.fp_naa_reference_safety import (
    _cross_machine_unmatched_indices,
    _reference_safety_gate,
)
from care_asd.fp_naa_reference_safety_config import (
    FROZEN_CONDITIONS,
    load_fp_naa_reference_safety_config,
)


def _safety_config():
    return load_fp_naa_reference_safety_config(
        Path("configs/experiment/fp_naa_reference_safety_v1.yaml")
    )


def test_unmatched_reference_is_deterministic_and_cross_machine() -> None:
    frame = pd.DataFrame(
        {
            "file_id": ["fan-0", "fan-1", "valve-0", "valve-1"],
            "machine_type": ["fan", "fan", "valve", "valve"],
        }
    )
    first = _cross_machine_unmatched_indices(frame, seed=8602)
    second = _cross_machine_unmatched_indices(frame, seed=8602)
    assert first.tolist() == second.tolist()
    machines = frame["machine_type"].to_numpy()
    assert (machines[first] != machines).all()


def test_reference_failure_permits_c3_only_when_matched_condition_passes() -> None:
    summary = pd.DataFrame(
        [
            {
                "condition": condition,
                "retention_median": 0.90 if condition != "leakage_high" else 0.80,
                "retention_worst_seed_q05": 0.70 if condition != "leakage_high" else 0.60,
            }
            for condition in FROZEN_CONDITIONS
        ]
    )
    gate = _reference_safety_gate(summary, _safety_config())
    assert gate["matched_passed"] is True
    assert gate["stress_passed"] is False
    assert gate["passed"] is False
    assert gate["c3_permitted"] is True

    summary.loc[summary["condition"] == "matched", "retention_median"] = 0.80
    gate = _reference_safety_gate(summary, _safety_config())
    assert gate["matched_passed"] is False
    assert gate["c3_permitted"] is False
