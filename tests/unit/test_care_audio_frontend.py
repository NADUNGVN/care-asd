"""Public FeatureBatch contract tests for the Phase 4 Safe CARE adapter."""

from __future__ import annotations

import numpy as np

from care_asd.config import FrontendConfig, GateConfig, ResidualConfig, SignalConfig, TransferConfig
from care_asd.signal import CAREAudioFrontEnd


def _frontend(*, bypass: bool = False, smoothing: int = 1) -> CAREAudioFrontEnd:
    return CAREAudioFrontEnd(
        SignalConfig(n_fft=16, win_length=16, hop_length=8),
        FrontendConfig(
            transfer=TransferConfig(frequency_smoothing_bins=smoothing),
            gate=GateConfig(bypass=bypass),
            residual=ResidualConfig(max_removed_energy_ratio=0.3),
        ),
    )


def _waveform() -> np.ndarray:
    samples = np.arange(96, dtype=np.float64)
    return np.stack((np.sin(samples / 4.0), 0.6 * np.sin((samples - 2.0) / 4.0)))


def test_care_adapter_returns_all_required_multi_views_and_diagnostics() -> None:
    batch = _frontend().transform(_waveform(), 16000)

    assert batch.frontend_name == "care"
    assert tuple(batch.views) == ("near", "far", "residual")
    assert {
        "coherence",
        "gate",
        "snr_proxy",
        "path_confidence",
        "removed_energy_ratio",
        "log_ratio",
        "phase_sin",
        "phase_cos",
        "transfer_magnitude",
        "view_to_near_energy_ratio",
    }.issubset(batch.diagnostics)
    assert np.all((batch.diagnostics["gate"] >= 0.0) & (batch.diagnostics["gate"] <= 0.9))
    assert np.all(batch.diagnostics["removed_energy_ratio"] <= 0.3 + 1.0e-12)


def test_care_adapter_bypass_preserves_near_as_residual() -> None:
    batch = _frontend(bypass=True).transform(_waveform(), 16000)

    assert np.array_equal(batch.views["near"], batch.views["residual"])
    assert np.count_nonzero(batch.diagnostics["gate"]) == 0


def test_care_adapter_is_causal_and_frequency_smoothing_keeps_shapes() -> None:
    original = _waveform()
    changed = original.copy()
    changed[:, 48:] *= -4.0
    frontend = _frontend(smoothing=3)

    before = frontend.transform(original, 16000)
    after = frontend.transform(changed, 16000)

    assert np.allclose(before.views["residual"][:4], after.views["residual"][:4])
    assert before.views["near"].shape == before.diagnostics["transfer_magnitude"].shape
