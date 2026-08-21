"""Normal-only, physically motivated pseudo-fault waveform transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from scipy.signal import butter, sosfilt

FaultFamily = Literal[
    "periodic_resonance",
    "amplitude_modulation",
    "frequency_modulation",
    "friction_burst",
]


@dataclass(frozen=True)
class PseudoFault:
    waveform: np.ndarray
    perturbation: np.ndarray
    family: FaultFamily
    requested_delta_level_db: float
    actual_delta_level_db: float
    parameters: dict[str, float]


@dataclass(frozen=True)
class PairedNoiseMixture:
    clean_noisy: np.ndarray
    fault_noisy: np.ndarray
    reference_noise: np.ndarray
    requested_snr_db: float
    actual_clean_snr_db: float
    noise_gain: float


def inject_pseudo_fault(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    family: FaultFamily,
    seed: int,
    delta_level_db: float,
    peak_limit: float = 0.999,
) -> PseudoFault:
    """Inject a reproducible fault signature without using anomalous recordings."""
    signal = _waveform(waveform, "waveform")
    if sample_rate <= 0 or not 0.0 < peak_limit <= 1.0:
        raise ValueError("sample_rate and peak_limit must be positive")
    rng = np.random.default_rng(seed)
    if family == "periodic_resonance":
        raw, parameters = _periodic_resonance(signal, sample_rate, rng)
    elif family == "amplitude_modulation":
        raw, parameters = _amplitude_modulation(signal, sample_rate, rng)
    elif family == "frequency_modulation":
        raw, parameters = _frequency_modulation(signal, sample_rate, rng)
    elif family == "friction_burst":
        raw, parameters = _friction_burst(signal, sample_rate, rng)
    else:
        raise ValueError(f"Unsupported pseudo-fault family: {family}")
    signal_rms = _rms(signal)
    raw_rms = _rms(raw)
    if signal_rms <= 1.0e-10 or raw_rms <= 1.0e-10:
        raise ValueError("Pseudo-fault generation requires non-silent signal and perturbation")
    target_delta_rms = signal_rms * 10.0 ** (delta_level_db / 20.0)
    perturbation = raw * (target_delta_rms / raw_rms)
    perturbation = _limit_shared_perturbation([signal], perturbation, peak_limit)
    faulty = signal + perturbation
    actual = 20.0 * np.log10(max(_rms(perturbation), 1.0e-12) / signal_rms)
    return PseudoFault(
        waveform=faulty.astype(np.float32),
        perturbation=perturbation.astype(np.float32),
        family=family,
        requested_delta_level_db=float(delta_level_db),
        actual_delta_level_db=float(actual),
        parameters=parameters,
    )


def mix_paired_noise(
    clean: np.ndarray,
    faulty: np.ndarray,
    reference_noise: np.ndarray,
    *,
    snr_db: float,
    peak_limit: float = 0.999,
) -> PairedNoiseMixture:
    """Add one identical scaled reference to clean/faulty counterfactual waveforms."""
    clean_signal = _waveform(clean, "clean")
    fault_signal = _waveform(faulty, "faulty")
    noise = _waveform(reference_noise, "reference_noise")
    if clean_signal.shape != fault_signal.shape or clean_signal.shape != noise.shape:
        raise ValueError("clean, faulty, and reference_noise must have equal lengths")
    clean_rms = _rms(clean_signal)
    noise_rms = _rms(noise)
    if clean_rms <= 1.0e-10 or noise_rms <= 1.0e-10:
        raise ValueError("Noise mixing requires non-silent clean and reference signals")
    requested_gain = clean_rms / (noise_rms * 10.0 ** (snr_db / 20.0))
    scaled_noise = noise * requested_gain
    limited_noise = _limit_shared_perturbation(
        [clean_signal, fault_signal],
        scaled_noise,
        peak_limit,
    )
    attenuation = _rms(limited_noise) / max(_rms(scaled_noise), 1.0e-12)
    actual = 20.0 * np.log10(clean_rms / max(_rms(limited_noise), 1.0e-12))
    return PairedNoiseMixture(
        clean_noisy=(clean_signal + limited_noise).astype(np.float32),
        fault_noisy=(fault_signal + limited_noise).astype(np.float32),
        reference_noise=limited_noise.astype(np.float32),
        requested_snr_db=float(snr_db),
        actual_clean_snr_db=float(actual),
        noise_gain=float(requested_gain * attenuation),
    )


def _periodic_resonance(
    signal: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    repetition_hz = float(rng.uniform(12.0, 90.0))
    resonance_hz = float(rng.uniform(700.0, min(5500.0, 0.42 * sample_rate)))
    decay_seconds = float(rng.uniform(0.006, 0.030))
    kernel_samples = max(8, round(5.0 * decay_seconds * sample_rate))
    time = np.arange(kernel_samples, dtype=np.float64) / sample_rate
    kernel = np.exp(-time / decay_seconds) * np.sin(2.0 * np.pi * resonance_hz * time)
    output = np.zeros_like(signal, dtype=np.float64)
    interval = sample_rate / repetition_hz
    position = float(rng.uniform(0.0, interval))
    while position < len(signal):
        start = round(position)
        stop = min(len(signal), start + len(kernel))
        amplitude = float(rng.uniform(0.7, 1.3))
        output[start:stop] += amplitude * kernel[: stop - start]
        position += interval * float(rng.uniform(0.96, 1.04))
    return output, {
        "repetition_hz": repetition_hz,
        "resonance_hz": resonance_hz,
        "decay_seconds": decay_seconds,
    }


def _amplitude_modulation(
    signal: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    modulation_hz = float(rng.uniform(4.0, 45.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    time = np.arange(len(signal), dtype=np.float64) / sample_rate
    perturbation = signal * np.sin(2.0 * np.pi * modulation_hz * time + phase)
    return perturbation, {"modulation_hz": modulation_hz, "phase": phase}


def _frequency_modulation(
    signal: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    modulation_hz = float(rng.uniform(2.0, 25.0))
    deviation_samples = float(rng.uniform(0.25, 2.0))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    positions = np.arange(len(signal), dtype=np.float64)
    warped = positions + deviation_samples * np.sin(
        2.0 * np.pi * modulation_hz * positions / sample_rate + phase
    )
    warped = np.clip(warped, 0.0, len(signal) - 1.0)
    shifted = np.interp(warped, positions, signal)
    return shifted - signal, {
        "modulation_hz": modulation_hz,
        "deviation_samples": deviation_samples,
        "phase": phase,
    }


def _friction_burst(
    signal: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, float]]:
    low_hz = float(rng.uniform(300.0, 1200.0))
    high_hz = float(rng.uniform(max(low_hz + 500.0, 2200.0), min(7000.0, 0.45 * sample_rate)))
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos")
    colored = sosfilt(sos, rng.normal(size=len(signal)))
    envelope = np.zeros(len(signal), dtype=np.float64)
    bursts = int(rng.integers(3, 9))
    for _ in range(bursts):
        duration = float(rng.uniform(0.025, 0.180))
        width = max(4, round(duration * sample_rate))
        start = int(rng.integers(0, max(1, len(signal) - width)))
        stop = min(len(signal), start + width)
        envelope[start:stop] += np.hanning(2 * (stop - start))[: stop - start]
    return colored * envelope, {
        "low_hz": low_hz,
        "high_hz": high_hz,
        "bursts": float(bursts),
    }


def _limit_shared_perturbation(
    bases: list[np.ndarray],
    perturbation: np.ndarray,
    peak_limit: float,
) -> np.ndarray:
    lower = np.full_like(perturbation, -np.inf, dtype=np.float64)
    upper = np.full_like(perturbation, np.inf, dtype=np.float64)
    for base in bases:
        lower = np.maximum(lower, -peak_limit - base)
        upper = np.minimum(upper, peak_limit - base)
    if np.any(lower > upper):
        raise ValueError("Base waveforms already violate the shared peak constraint")
    return cast(np.ndarray, np.clip(perturbation, lower, upper))


def _waveform(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return array


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value), dtype=np.float64)))
