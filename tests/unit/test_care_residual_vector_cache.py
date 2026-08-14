"""Phase 7 CARE residual waveform contract tests."""

from __future__ import annotations

import numpy as np

from care_asd.config import FrontendConfig, GateConfig, SignalConfig
from care_asd.data.care_residual_vector_cache import care_residual_waveform


def test_bypass_care_residual_reconstructs_near_waveform() -> None:
    sample_rate = 16_000
    samples = np.arange(sample_rate, dtype=np.float64) / sample_rate
    near = 0.4 * np.sin(2.0 * np.pi * 330.0 * samples)
    far = 0.2 * np.sin(2.0 * np.pi * 180.0 * samples)
    waveform = np.stack([near, far])

    residual = care_residual_waveform(
        waveform,
        SignalConfig(),
        FrontendConfig(gate=GateConfig(bypass=True)),
    )

    assert residual.shape == near.shape
    assert np.allclose(residual, near, atol=1.0e-10)


def test_care_residual_preserves_finite_length_for_non_hop_aligned_waveform() -> None:
    generator = np.random.default_rng(9)
    waveform = generator.normal(size=(2, 5_123)).astype(np.float64)

    residual = care_residual_waveform(waveform, SignalConfig(), FrontendConfig())

    assert residual.shape == (5_123,)
    assert np.isfinite(residual).all()
