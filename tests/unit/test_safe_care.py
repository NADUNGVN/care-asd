"""Tests for the safe, causal CARE residual view."""

from __future__ import annotations

import numpy as np
import pytest

from care_asd.config import FrontendConfig, GateConfig, ResidualConfig, SignalConfig
from care_asd.signal import SafeCAREFrontEnd, simulate_stereo_case


def _frontend(*, bypass: bool = False, cap: float = 0.25) -> SafeCAREFrontEnd:
    signal = SignalConfig(n_fft=16, win_length=16, hop_length=8)
    frontend = FrontendConfig(
        gate=GateConfig(min_value=0.0, max_value=0.9, bypass=bypass),
        residual=ResidualConfig(max_removed_energy_ratio=cap),
    )
    return SafeCAREFrontEnd(signal, frontend)


def test_near_view_is_preserved_and_removal_is_bounded() -> None:
    samples = np.arange(64, dtype=np.float64)
    waveform = np.stack((np.sin(samples / 5.0), np.sin(samples / 5.0) * 0.7))

    result = _frontend(cap=0.2).transform(waveform)

    expected_near = _frontend(cap=0.2)._stft(waveform[0])
    assert np.array_equal(result.near_stft, expected_near)
    assert np.all(result.removed_energy_ratio <= 0.2 + 1.0e-12)
    assert np.all((result.path_confidence >= 0.0) & (result.path_confidence <= 1.0))


def test_bypass_returns_original_near_as_residual() -> None:
    waveform = np.stack((np.linspace(-1.0, 1.0, 32), np.linspace(1.0, -1.0, 32)))

    result = _frontend(bypass=True).transform(waveform)

    assert np.array_equal(result.residual_stft, result.near_stft)
    assert np.count_nonzero(result.gate) == 0
    assert np.count_nonzero(result.removed_energy_ratio) == 0


def test_causal_ema_outputs_do_not_change_when_future_changes() -> None:
    samples = np.arange(64, dtype=np.float64)
    original = np.stack((np.sin(samples / 3.0), np.cos(samples / 4.0)))
    changed = original.copy()
    changed[:, 40:] *= -3.0

    before = _frontend().transform(original)
    after = _frontend().transform(changed)

    assert np.allclose(before.residual_stft[:4], after.residual_stft[:4])
    assert np.allclose(before.path_confidence[:4], after.path_confidence[:4])


def test_synthetic_fault_remains_available_in_the_preserved_near_view() -> None:
    case = simulate_stereo_case(sample_rate=160, duration_seconds=2.0, seed=7)
    frontend = _frontend()

    normal = frontend.transform(case.normal_waveform)
    faulty = frontend.transform(case.faulty_waveform)

    assert np.array_equal(faulty.near_stft, frontend._stft(case.faulty_waveform[0]))
    assert np.linalg.norm(faulty.near_stft - normal.near_stft) > 0.0
    assert np.linalg.norm(case.fault_component) > 0.0


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros((1, 16), dtype=np.float64),
        np.zeros((2, 0), dtype=np.float64),
        np.array([[0.0, np.nan], [0.0, 1.0]], dtype=np.float64),
    ],
)
def test_invalid_waveforms_are_rejected(waveform: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _frontend().transform(waveform)
