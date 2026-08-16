from __future__ import annotations

import numpy as np
import pytest

from care_asd.signal.pseudo_faults import inject_pseudo_fault, mix_paired_noise


def _signal(sample_rate: int = 16_000) -> np.ndarray:
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    return 0.1 * np.sin(2.0 * np.pi * 440.0 * time)


@pytest.mark.parametrize(
    "family",
    [
        "periodic_resonance",
        "amplitude_modulation",
        "frequency_modulation",
        "friction_burst",
    ],
)
def test_pseudo_faults_are_deterministic_nontrivial_and_level_controlled(family: str) -> None:
    signal = _signal()
    first = inject_pseudo_fault(
        signal,
        sample_rate=16_000,
        family=family,  # type: ignore[arg-type]
        seed=42,
        delta_level_db=-18.0,
    )
    second = inject_pseudo_fault(
        signal,
        sample_rate=16_000,
        family=family,  # type: ignore[arg-type]
        seed=42,
        delta_level_db=-18.0,
    )
    np.testing.assert_array_equal(first.waveform, second.waveform)
    assert np.linalg.norm(first.perturbation) > 0.0
    assert first.actual_delta_level_db == pytest.approx(-18.0, abs=0.05)
    assert np.max(np.abs(first.waveform)) <= 0.999001


def test_paired_noise_preserves_the_counterfactual_difference() -> None:
    signal = _signal()
    fault = inject_pseudo_fault(
        signal,
        sample_rate=16_000,
        family="periodic_resonance",
        seed=7,
        delta_level_db=-24.0,
    )
    noise = 0.05 * np.random.default_rng(9).normal(size=len(signal))
    pair = mix_paired_noise(signal, fault.waveform, noise, snr_db=3.0)
    np.testing.assert_allclose(
        pair.fault_noisy - pair.clean_noisy,
        fault.waveform - signal,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        pair.clean_noisy - signal,
        pair.reference_noise,
        atol=1.0e-7,
    )
    assert pair.actual_clean_snr_db == pytest.approx(3.0, abs=0.05)
