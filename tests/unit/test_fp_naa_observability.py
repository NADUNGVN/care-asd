from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from care_asd.data.fp_naa_observability import (
    _diagnose_item,
    _PreparedProbe,
    _summarize_taps,
    _validate_tapped_output,
)
from care_asd.fp_naa_config import load_fp_naa_config


def test_observability_diagnostic_separates_equal_and_attenuated_deltas() -> None:
    item = _PreparedProbe(
        shard=Path("unused.json"),
        file_id="fan/train/normal.wav",
        family="amplitude_modulation",
        waveforms=np.zeros((4, 2), dtype=np.float32),
        comparisons=(("in_support", "amplitude_modulation", 0, 1, 2, 3),),
    )
    equal = np.asarray([0.0, 10.0, 2.0, 12.0], dtype=np.float32).reshape(4, 1, 1, 1)
    attenuated = np.asarray([0.0, 10.0, 2.0, 11.0], dtype=np.float32).reshape(4, 1, 1, 1)
    rows = _diagnose_item(item, offset=0, extracted={0: equal, 4: attenuated}, taps=(0, 4))
    assert rows[0]["retention"] == 1.0
    assert rows[0]["direction_cosine"] == 1.0
    assert rows[1]["retention"] == 0.5
    assert rows[1]["transport_relative_error"] == 0.5


def test_tap_summary_applies_unchanged_retention_gate() -> None:
    config = load_fp_naa_config("configs/experiment/fp_naa_v6.yaml")
    diagnostics = pd.DataFrame(
        [
            {
                "tap": tap,
                "fault_set": fault_set,
                "retention": retention,
                "direction_cosine": 1.0,
                "transport_relative_error": 0.0,
            }
            for tap, values in ((0, (0.95, 0.90)), (4, (0.95, 0.70)))
            for fault_set in ("in_support", "heldout")
            for retention in values
        ]
    )
    summary = _summarize_taps(diagnostics, taps=(0, 4), config=config)
    assert bool(summary.loc[summary["tap"] == 0, "eligible_in_support"].item())
    assert not bool(summary.loc[summary["tap"] == 4, "eligible_in_support"].item())


def test_v6_config_freezes_depth_selection_without_changing_g2() -> None:
    config = load_fp_naa_config("configs/experiment/fp_naa_v6.yaml")
    assert config.observability is not None
    assert config.observability.encoder_taps == [0, 4, 8, 12]
    assert config.gates.screening_minimum_gain_over_c1 == 0.005
    assert config.gates.fault_delta_retention_q05_minimum == 0.75


def test_tapped_output_contract_rejects_nonfinite_or_missing_depths() -> None:
    valid = np.ones((2, 1, 1, 3), dtype=np.float32)
    _validate_tapped_output({0: valid, 4: valid}, taps=(0, 4), expected_batch=2)
    with np.testing.assert_raises_regex(RuntimeError, "requested depths"):
        _validate_tapped_output({0: valid}, taps=(0, 4), expected_batch=2)
    invalid = valid.copy()
    invalid[0, 0, 0, 0] = np.nan
    with np.testing.assert_raises_regex(RuntimeError, "non-finite"):
        _validate_tapped_output({0: invalid}, taps=(0,), expected_batch=2)
