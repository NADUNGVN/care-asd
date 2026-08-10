"""Deterministic stereo simulations with known machine and fault components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticStereoCase:
    """Normal/faulty stereo signals with ground-truth component access."""

    sample_rate: int
    normal_waveform: FloatArray
    faulty_waveform: FloatArray
    machine_component: FloatArray
    fault_component: FloatArray
    environmental_component: FloatArray


def simulate_stereo_case(
    *,
    sample_rate: int = 16_000,
    duration_seconds: float = 2.0,
    seed: int = 0,
    path_gain: float = 0.45,
    path_delay_samples: int = 12,
    fault_start_seconds: float = 1.0,
) -> SyntheticStereoCase:
    """Generate a controlled near/far scenario for CARE safety tests.

    The near channel contains a stronger machine source. The far channel
    contains an attenuated, delayed copy plus stronger environmental noise.
    The fault component propagates through the same physical path, which makes
    it a deliberately difficult test for residual cancellation methods.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be positive")
    if not 0.0 <= path_gain <= 1.0:
        raise ValueError("path_gain must be in [0, 1]")
    if path_delay_samples < 0:
        raise ValueError("path_delay_samples must be non-negative")

    samples = round(sample_rate * duration_seconds)
    if samples < 32:
        raise ValueError("duration_seconds must produce at least 32 samples")
    time = np.arange(samples, dtype=np.float64) / sample_rate
    machine = 0.50 * np.sin(2.0 * np.pi * 220.0 * time) + 0.15 * np.sin(
        2.0 * np.pi * 440.0 * time
    )

    fault = np.zeros(samples, dtype=np.float64)
    fault_start = round(fault_start_seconds * sample_rate)
    if 0 <= fault_start < samples:
        active = np.arange(samples - fault_start, dtype=np.float64) / sample_rate
        envelope = np.minimum(active / 0.03, 1.0)
        fault[fault_start:] = 0.20 * envelope * np.sin(2.0 * np.pi * 1_900.0 * active)

    rng = np.random.default_rng(seed)
    shared_noise = 0.05 * rng.standard_normal(samples)
    near_noise = shared_noise + 0.01 * rng.standard_normal(samples)
    far_noise = 1.80 * shared_noise + 0.03 * rng.standard_normal(samples)

    propagated_machine = _delay(machine, path_delay_samples) * path_gain
    propagated_fault = _delay(fault, path_delay_samples) * path_gain
    normal = np.stack((machine + near_noise, propagated_machine + far_noise))
    faulty = np.stack(
        (machine + fault + near_noise, propagated_machine + propagated_fault + far_noise)
    )
    return SyntheticStereoCase(
        sample_rate=sample_rate,
        normal_waveform=normal,
        faulty_waveform=faulty,
        machine_component=machine,
        fault_component=fault,
        environmental_component=shared_noise,
    )


def _delay(values: FloatArray, samples: int) -> FloatArray:
    if samples == 0:
        return values.copy()
    delayed = np.zeros_like(values)
    delayed[samples:] = values[:-samples]
    return delayed
