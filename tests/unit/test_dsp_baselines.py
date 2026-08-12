"""Numerical and causal contracts for Phase 3 DSP controls."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from care_asd.config import SignalConfig
from care_asd.signal.dsp_baselines import (
    FrontEndName,
    available_dsp_frontends,
    create_dsp_frontend,
)


def _signal() -> SignalConfig:
    return SignalConfig(n_fft=16, win_length=16, hop_length=8)


def _stereo() -> np.ndarray:
    samples = np.arange(96, dtype=np.float64)
    return np.stack((np.sin(samples / 4.0), 0.7 * np.cos(samples / 5.0)))


@pytest.mark.parametrize("name", available_dsp_frontends())
def test_all_dsp_frontends_are_deterministic_finite_and_shape_consistent(name: str) -> None:
    frontend = create_dsp_frontend(cast(FrontEndName, name), _signal())

    first = frontend.transform(_stereo(), 16000)
    second = frontend.transform(_stereo(), 16000)

    assert first.frontend_name == name
    assert first.sample_rate == 16000
    assert first.score_fusion == ("mean" if name == "late_score_fusion" else "none")
    assert first.views.keys() == second.views.keys()
    for view_name, view in first.views.items():
        assert view.shape == first.diagnostics["coherence"].shape
        assert np.isfinite(view).all()
        assert np.array_equal(view, second.views[view_name])
    assert np.all((first.diagnostics["coherence"] >= 0.0) & (first.diagnostics["coherence"] <= 1.0))
    assert np.isfinite(first.diagnostics["view_to_near_energy_ratio"]).all()


@pytest.mark.parametrize(
    "name,view_name",
    [
        ("difference", "difference"),
        ("spectral_subtraction", "residual"),
        ("wiener", "wiener"),
        ("coherence_mask", "coherence_mask"),
        ("adaptive_filter", "adaptive_residual"),
    ],
)
def test_zero_far_is_identity_for_cancellation_controls(name: str, view_name: str) -> None:
    waveform = _stereo()
    waveform[1] = 0.0
    near = create_dsp_frontend("near", _signal()).transform(waveform, 16000).views["near"]
    transformed = (
        create_dsp_frontend(cast(FrontEndName, name), _signal())
        .transform(waveform, 16000)
        .views[view_name]
    )

    assert np.allclose(transformed, near)


def test_adaptive_filter_does_not_use_future_frames() -> None:
    original = _stereo()
    changed = original.copy()
    changed[:, 48:] *= -4.0
    frontend = create_dsp_frontend("adaptive_filter", _signal())

    before = frontend.transform(original, 16000).views["adaptive_residual"]
    after = frontend.transform(changed, 16000).views["adaptive_residual"]

    assert np.allclose(before[:4], after[:4])


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros((1, 16), dtype=np.float64),
        np.zeros((2, 0), dtype=np.float64),
        np.array([[0.0, np.nan], [0.0, 1.0]], dtype=np.float64),
    ],
)
def test_dsp_frontends_reject_invalid_stereo_audio(waveform: np.ndarray) -> None:
    with pytest.raises(ValueError):
        create_dsp_frontend("near", _signal()).transform(waveform, 16000)
