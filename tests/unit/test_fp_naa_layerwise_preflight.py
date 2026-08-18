from __future__ import annotations

from pathlib import Path

import pandas as pd

from care_asd.data.fp_naa_augmentation_cache import _AugmentationPlan
from care_asd.evaluation.fp_naa_layerwise_preflight import _make_gate, _select_plans
from care_asd.fp_naa_config import load_fp_naa_config


def _plan(index: int, *, heldout: bool) -> _AugmentationPlan:
    return _AugmentationPlan(
        file_id=f"clip-{index:04d}",
        target_audio=Path(f"target-{index}.wav"),
        donor_file_id=f"donor-{index:04d}",
        donor_audio=Path(f"donor-{index}.wav"),
        feature_path=Path(f"feature-{index}.npz"),
        family="periodic_resonance",
        seed=index,
        noise_snr_db=0.0,
        fault_delta_level_db=-18.0,
        heldout=heldout,
    )


def test_v8_preflight_split_is_disjoint_and_frozen() -> None:
    config = load_fp_naa_config("configs/experiment/fp_naa_v8.yaml")
    assert config.layerwise is not None
    config.layerwise.preflight_train_clips = 10
    config.layerwise.preflight_validation_clips = 6
    config.layerwise.preflight_heldout_clips = 3
    plans = [_plan(index, heldout=index % 3 == 0) for index in range(60)]
    first_train, first_validation = _select_plans(plans, config.layerwise)
    second_train, second_validation = _select_plans(plans, config.layerwise)
    assert [plan.file_id for plan in first_train] == [plan.file_id for plan in second_train]
    assert [plan.file_id for plan in first_validation] == [
        plan.file_id for plan in second_validation
    ]
    assert len(first_train) == 10
    assert len(first_validation) == 6
    assert sum(plan.heldout for plan in first_validation) == 3
    assert {plan.file_id for plan in first_train}.isdisjoint(
        plan.file_id for plan in first_validation
    )


def test_v8_gate_requires_capacity_matched_retention_gain() -> None:
    config = load_fp_naa_config("configs/experiment/fp_naa_v8.yaml")
    assert config.layerwise is not None
    summary = pd.DataFrame(
        [
            {
                "candidate": "l1_layerwise_mse",
                "fault_set": "in_support",
                "retention_median": 0.80,
                "retention_q05": 0.40,
                "normal_function_drift_median": 0.01,
            },
            {
                "candidate": "l2_layerwise_fault_transport",
                "fault_set": "in_support",
                "retention_median": 0.92,
                "retention_q05": 0.62,
                "normal_function_drift_median": 0.05,
            },
            {
                "candidate": "l2_layerwise_fault_transport",
                "fault_set": "heldout",
                "retention_median": 0.87,
                "retention_q05": 0.61,
                "normal_function_drift_median": 0.04,
            },
        ]
    )
    gate = _make_gate(
        summary,
        layerwise=config.layerwise,
        update_norms={"common": 1.0, "l1": 0.5, "l2": 0.7},
        trainable_parameters=123,
        runtime_probe={"status": "passed", "frozen_path_relative_error": 0.0},
    )
    assert gate["passed"] is True
    assert gate["checks"]["median_gain_over_l1"] is True
    assert gate["checks"]["q05_gain_over_l1"] is True


def test_v8_config_freezes_all_layer_insertions() -> None:
    config = load_fp_naa_config("configs/experiment/fp_naa_v8.yaml")
    assert config.layerwise is not None
    assert config.layerwise.insertion_layers == list(range(1, 13))
    assert config.layerwise.preflight_seed == 2608
