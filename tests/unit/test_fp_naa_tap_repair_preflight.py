from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from care_asd.data.fp_naa_augmentation_cache import _AugmentationPlan
from care_asd.evaluation.fp_naa_tap_repair_preflight import (
    _load_tap_arrays,
    _make_gate,
)
from care_asd.fp_naa_config import load_fp_naa_config


def _plan(path: Path) -> _AugmentationPlan:
    return _AugmentationPlan(
        file_id="clip-1",
        target_audio=path.parent / "target.wav",
        donor_file_id="clip-2",
        donor_audio=path.parent / "reference.wav",
        feature_path=path,
        family="periodic_resonance",
        heldout=True,
        seed=1,
        noise_snr_db=0.0,
        fault_delta_level_db=-18.0,
    )


def test_heldout_cache_reuses_clean_teacher_and_loads_heldout_fault(tmp_path: Path) -> None:
    shape = (2, 3, 4)
    values = {
        "heldout_tap0_noisy_clean": np.full(shape, 1.0, dtype=np.float16),
        "heldout_tap0_reference": np.full(shape, 2.0, dtype=np.float16),
        "heldout_tap0_fault_noisy": np.full(shape, 3.0, dtype=np.float16),
        "tap0_teacher_clean": np.full(shape, 4.0, dtype=np.float16),
        "heldout_tap0_teacher_fault": np.full(shape, 5.0, dtype=np.float16),
        "metadata_json": np.asarray(
            json.dumps({"file_id": "clip-1", "fault_family": "periodic_resonance"})
        ),
    }
    path = tmp_path / "clip.npz"
    np.savez_compressed(path, **values)

    arrays = _load_tap_arrays([_plan(path)], heldout=True, workers=0)

    assert arrays.families == ("friction_burst",)
    assert np.array_equal(arrays.teacher_clean[0], values["tap0_teacher_clean"])
    assert np.array_equal(arrays.teacher_fault[0], values["heldout_tap0_teacher_fault"])


def test_v9_gate_requires_tail_and_heldout_retention() -> None:
    repository = Path(__file__).resolve().parents[2]
    repair = load_fp_naa_config(repository / "configs" / "experiment" / "fp_naa_v9.yaml").tap_repair
    assert repair is not None
    summary = pd.DataFrame(
        [
            {
                "candidate": "p1_tap0_mse",
                "fault_set": "in_support",
                "retention_median": 0.82,
                "retention_q05": 0.49,
                "raw_retention_median": 0.70,
                "raw_retention_q05": 0.30,
                "normal_function_drift_median": 0.04,
            },
            {
                "candidate": "p2_tap0_actt",
                "fault_set": "in_support",
                "retention_median": 0.92,
                "retention_q05": 0.61,
                "raw_retention_median": 0.70,
                "raw_retention_q05": 0.30,
                "normal_function_drift_median": 0.08,
            },
            {
                "candidate": "p2_tap0_actt",
                "fault_set": "heldout",
                "retention_median": 0.87,
                "retention_q05": 0.66,
                "raw_retention_median": 0.73,
                "raw_retention_q05": 0.36,
                "normal_function_drift_median": 0.07,
            },
        ]
    )

    gate = _make_gate(
        summary,
        repair=repair,
        runtime_probe={"status": "passed"},
        update_norms={"common": 1.0, "p1": 1.0, "p2": 1.0},
        trainable_parameters=10,
    )

    assert gate["passed"] is True
    assert gate["checks"]["in_support_retention_q05"] is True
    assert gate["checks"]["heldout_retention_q05"] is True


def test_v9_gate_fails_a_nonfinite_optimizer_update() -> None:
    repository = Path(__file__).resolve().parents[2]
    repair = load_fp_naa_config(repository / "configs" / "experiment" / "fp_naa_v9.yaml").tap_repair
    assert repair is not None
    summary = pd.DataFrame(
        [
            {
                "candidate": "p1_tap0_mse",
                "fault_set": "in_support",
                "retention_median": 0.80,
                "retention_q05": 0.45,
                "raw_retention_median": 0.70,
                "raw_retention_q05": 0.30,
                "normal_function_drift_median": 0.04,
            },
            {
                "candidate": "p2_tap0_actt",
                "fault_set": "in_support",
                "retention_median": 0.92,
                "retention_q05": 0.62,
                "raw_retention_median": 0.70,
                "raw_retention_q05": 0.30,
                "normal_function_drift_median": 0.08,
            },
            {
                "candidate": "p2_tap0_actt",
                "fault_set": "heldout",
                "retention_median": 0.88,
                "retention_q05": 0.67,
                "raw_retention_median": 0.73,
                "raw_retention_q05": 0.36,
                "normal_function_drift_median": 0.07,
            },
        ]
    )

    gate = _make_gate(
        summary,
        repair=repair,
        runtime_probe={"status": "passed"},
        update_norms={"common": 1.0, "p1": 1.0, "p2": float("nan")},
        trainable_parameters=10,
    )

    assert gate["passed"] is False
    assert gate["checks"]["finite_real_updates"] is False
