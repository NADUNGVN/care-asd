"""G0 invariants for the AP-CARE v2 controller."""

from __future__ import annotations

import numpy as np
import pytest

from care_asd.ap_care_config import APCAREExperimentConfig
from care_asd.signal.ap_care import APCAREController
from care_asd.signal.ap_care_simulation import APCARESyntheticCase, simulate_ap_care_case


def _config(
    *,
    max_gain: float = 1.0,
    risk_terms_bypass: bool = False,
    budget_enabled: bool = True,
) -> APCAREExperimentConfig:
    return APCAREExperimentConfig.model_validate(
        {
            "stft": {"n_fft": 128, "win_length": 128, "hop_length": 64},
            "transfer": {
                "frequency_smoothing_bins": 3,
                "machine_support_median_bins": 9,
            },
            "controller": {
                "max_gain": max_gain,
                "risk_terms_bypass": risk_terms_bypass,
                "budget_enabled": budget_enabled,
                "band_edges_hz": [0.0, 500.0, 2000.0],
                "warmup_frames": 0 if risk_terms_bypass else 1,
            },
            "simulation": {
                "sample_rate": 4000,
                "duration_seconds": 0.5,
                "cases": 32,
                "profile_clips": 2,
            },
            "gate": {"bootstrap_iterations": 100},
        }
    )


def _case() -> APCARESyntheticCase:
    return simulate_ap_care_case(
        sample_rate=4000,
        duration_seconds=0.5,
        seed=7,
        profile_clips=2,
        machine_leakage_gain=0.25,
        near_snr_db=0.0,
        far_environment_gain=1.5,
        path_delay_samples=8,
        path_gain_mismatch=1.0,
        path_delay_mismatch_samples=0,
        fault_amplitude=0.2,
        fault_support="in_support",
        fault_far_ratio=1.0,
        base_frequency_hz=220.0,
    )


def test_ap_care_preserves_near_view_and_enforces_each_band_budget() -> None:
    case = _case()
    config = _config()
    controller = APCAREController(config)
    profile = controller.fit(case.profile_waveforms, case.sample_rate)

    output = controller.transform(case.faulty_waveform, case.sample_rate, profile)

    assert output.near_stft.shape == output.residual_stft.shape
    assert np.all(output.actual_removed_energy <= output.permitted_removed_energy + 1.0e-9)
    assert np.all((output.controller_gain >= 0.0) & (output.controller_gain <= 1.0))
    assert np.all((output.leakage_risk >= 0.0) & (output.leakage_risk <= 1.0))
    assert np.all((output.transfer_uncertainty >= 0.0) & (output.transfer_uncertainty <= 1.0))


def test_zero_gain_is_exact_near_equivalence() -> None:
    case = _case()
    controller = APCAREController(_config(max_gain=0.0))
    profile = controller.fit(case.profile_waveforms, case.sample_rate)

    output = controller.transform(case.faulty_waveform, case.sample_rate, profile)

    assert np.array_equal(output.residual_stft, output.near_stft)
    assert np.count_nonzero(output.controller_gain) == 0
    assert np.count_nonzero(output.actual_removed_energy) == 0


def test_ap_care_is_causal_for_unchanged_prefix() -> None:
    case = _case()
    config = _config()
    controller = APCAREController(config)
    profile = controller.fit(case.profile_waveforms, case.sample_rate)
    changed = case.faulty_waveform.copy()
    changed[:, 700:] *= -2.0

    before = controller.transform(case.faulty_waveform, case.sample_rate, profile)
    after = controller.transform(changed, case.sample_rate, profile)

    assert np.allclose(before.residual_stft[:8], after.residual_stft[:8])
    assert np.allclose(before.controller_gain[:8], after.controller_gain[:8])


def test_frozen_component_application_uses_realized_filter() -> None:
    case = _case()
    controller = APCAREController(_config())
    profile = controller.fit(case.profile_waveforms, case.sample_rate)
    output = controller.transform(case.faulty_waveform, case.sample_rate, profile)

    transformed_fault = controller.apply_frozen(
        output,
        case.fault_component,
        case.sample_rate,
    )

    assert transformed_fault.shape == output.near_stft.shape
    assert np.all(np.isfinite(transformed_fault))


def test_bypassed_risk_and_budget_reproduce_declared_reference_proposal() -> None:
    case = _case()
    controller = APCAREController(_config(risk_terms_bypass=True, budget_enabled=False))
    profile = controller.fit(case.profile_waveforms, case.sample_rate)

    output = controller.transform(case.faulty_waveform, case.sample_rate, profile)
    expected = output.near_stft - profile.transfer_function[np.newaxis, :] * output.far_stft

    assert np.allclose(output.residual_stft, expected)
    assert np.all(output.controller_gain == 1.0)
    assert not np.any(output.bound_active)


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros((1, 128), dtype=np.float64),
        np.zeros((2, 0), dtype=np.float64),
        np.array([[0.0, np.nan], [0.0, 1.0]], dtype=np.float64),
    ],
)
def test_ap_care_rejects_invalid_waveform_boundaries(waveform: np.ndarray) -> None:
    case = _case()
    controller = APCAREController(_config())
    profile = controller.fit(case.profile_waveforms, case.sample_rate)
    with pytest.raises(ValueError):
        controller.transform(waveform, case.sample_rate, profile)
