"""Normal-only reference profiling and fixed far-reference subtraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

from care_asd.reference_safety_config import (
    ReferenceProfileConfig,
    ReferenceSafetyPolicy,
    ReferenceSTFTConfig,
    ReferenceSubtractionConfig,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
SafetyDecision = Literal["near", "refsub"]


@dataclass(frozen=True)
class ReferenceDiagnostics:
    """Clip-level normal-only diagnostics used by the safety controller."""

    leakage_index: float
    transfer_instability: float
    spectral_drift: float
    noise_reduction_db: float
    risk_score: float


@dataclass(frozen=True)
class ReferenceSafetyProfile:
    """One immutable profile for a machine/section training group."""

    transfer_power: FloatArray
    leakage_u95: float
    transfer_instability_u95: float
    spectral_drift_u95: float
    noise_reduction_l05_db: float
    risk_score: float
    training_clips: int


def stft_spectrum(
    waveform: NDArray[np.floating], sample_rate: int, config: ReferenceSTFTConfig
) -> ComplexArray:
    """Return a deterministic frequency-major complex STFT."""
    values = _validate_waveform(waveform)
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if len(values) < config.n_fft:
        values = np.pad(values, (0, config.n_fft - len(values)))
    noverlap = config.n_fft - config.hop_length
    _, _, spectrum = scipy_signal.stft(
        values,
        fs=sample_rate,
        window=config.window,
        nperseg=config.n_fft,
        noverlap=noverlap,
        nfft=config.n_fft,
        boundary="zeros",
        padded=True,
        return_onesided=True,
    )
    return np.asarray(spectrum, dtype=np.complex128)


def inverse_stft(
    spectrum: NDArray[np.complexfloating],
    sample_rate: int,
    length: int,
    config: ReferenceSTFTConfig,
) -> FloatArray:
    """Invert :func:`stft_spectrum` and restore the requested sample count."""
    values = np.asarray(spectrum, dtype=np.complex128)
    expected_bins = config.n_fft // 2 + 1
    if values.ndim != 2 or values.shape[0] != expected_bins:
        raise ValueError(f"Expected STFT shape ({expected_bins}, frames), got {values.shape}")
    if length < 1:
        raise ValueError("length must be positive")
    noverlap = config.n_fft - config.hop_length
    _, waveform = scipy_signal.istft(
        values,
        fs=sample_rate,
        window=config.window,
        nperseg=config.n_fft,
        noverlap=noverlap,
        nfft=config.n_fft,
        input_onesided=True,
        boundary=True,
    )
    if len(waveform) < length:
        waveform = np.pad(waveform, (0, length - len(waveform)))
    return np.asarray(waveform[:length], dtype=np.float64)


def noise_floor_spectrum(
    waveform: NDArray[np.floating],
    sample_rate: int,
    stft: ReferenceSTFTConfig,
    subtraction: ReferenceSubtractionConfig,
) -> FloatArray:
    """Estimate one per-frequency minimum-statistics power floor."""
    spectrum = stft_spectrum(waveform, sample_rate, stft)
    return np.asarray(
        np.quantile(np.abs(spectrum) ** 2, subtraction.noise_quantile, axis=1),
        dtype=np.float64,
    )


def estimate_noise_transfer(
    near_floors: NDArray[np.floating],
    far_floors: NDArray[np.floating],
    stft: ReferenceSTFTConfig,
    subtraction: ReferenceSubtractionConfig,
) -> FloatArray:
    """Estimate the robust far-to-near noise-power transfer from normal clips."""
    near = np.asarray(near_floors, dtype=np.float64)
    far = np.asarray(far_floors, dtype=np.float64)
    if near.ndim != 2 or near.shape != far.shape or not len(near):
        raise ValueError("near_floors and far_floors must be non-empty equal 2-D arrays")
    expected_bins = stft.n_fft // 2 + 1
    if near.shape[1] != expected_bins:
        raise ValueError(f"Expected {expected_bins} spectral bins, got {near.shape[1]}")
    ratio = near / np.maximum(far, stft.eps)
    transfer = np.median(ratio, axis=0)
    return np.asarray(
        np.clip(transfer, subtraction.transfer_min, subtraction.transfer_max),
        dtype=np.float64,
    )


def apply_reference_subtraction(
    near_waveform: NDArray[np.floating],
    far_waveform: NDArray[np.floating],
    sample_rate: int,
    transfer_power: NDArray[np.floating],
    stft: ReferenceSTFTConfig,
    subtraction: ReferenceSubtractionConfig,
) -> FloatArray:
    """Apply fixed floored spectral subtraction while preserving near phase."""
    near = _validate_waveform(near_waveform)
    far = _validate_waveform(far_waveform)
    if near.shape != far.shape:
        raise ValueError("near and far waveforms must have equal length")
    transfer = np.asarray(transfer_power, dtype=np.float64)
    expected_bins = stft.n_fft // 2 + 1
    if transfer.shape != (expected_bins,) or not np.all(np.isfinite(transfer)):
        raise ValueError(f"transfer_power must have shape ({expected_bins},) and be finite")
    near_stft = stft_spectrum(near, sample_rate, stft)
    far_stft = stft_spectrum(far, sample_rate, stft)
    near_power = np.abs(near_stft) ** 2
    removed = subtraction.alpha * transfer[:, None] * np.abs(far_stft) ** 2
    enhanced_power = np.maximum(near_power - removed, subtraction.beta * near_power)
    enhanced = np.sqrt(enhanced_power) * np.exp(1j * np.angle(near_stft))
    return inverse_stft(enhanced, sample_rate, len(near), stft)


def diagnose_reference_pair(
    near_waveform: NDArray[np.floating],
    far_waveform: NDArray[np.floating],
    enhanced_waveform: NDArray[np.floating],
    sample_rate: int,
    transfer_power: NDArray[np.floating],
    stft: ReferenceSTFTConfig,
    subtraction: ReferenceSubtractionConfig,
) -> ReferenceDiagnostics:
    """Compute the four bounded normal-only safety features for one clip."""
    near = _validate_waveform(near_waveform)
    far = _validate_waveform(far_waveform)
    enhanced = _validate_waveform(enhanced_waveform)
    if near.shape != far.shape or near.shape != enhanced.shape:
        raise ValueError("diagnostic waveforms must have equal length")
    near_stft = stft_spectrum(near, sample_rate, stft)
    far_stft = stft_spectrum(far, sample_rate, stft)
    enhanced_stft = stft_spectrum(enhanced, sample_rate, stft)
    near_power = np.abs(near_stft) ** 2
    far_power = np.abs(far_stft) ** 2
    enhanced_power = np.abs(enhanced_stft) ** 2
    cross = np.mean(near_stft * np.conj(far_stft), axis=1)
    mean_near = np.mean(near_power, axis=1)
    mean_far = np.mean(far_power, axis=1)
    coherence = np.clip(np.abs(cross) ** 2 / np.maximum(mean_near * mean_far, stft.eps), 0.0, 1.0)
    machine_floor = np.quantile(near_power, subtraction.noise_quantile, axis=1)
    machine_weight = np.maximum(mean_near - machine_floor, 0.0)
    leakage = float(
        np.sum(coherence * machine_weight) / np.maximum(np.sum(machine_weight), stft.eps)
    )

    near_floor = np.quantile(near_power, subtraction.noise_quantile, axis=1)
    far_floor = np.quantile(far_power, subtraction.noise_quantile, axis=1)
    clip_transfer = np.clip(
        near_floor / np.maximum(far_floor, stft.eps),
        subtraction.transfer_min,
        subtraction.transfer_max,
    )
    reference_transfer = np.asarray(transfer_power, dtype=np.float64)
    log_error = np.abs(
        np.log(np.maximum(clip_transfer, stft.eps))
        - np.log(np.maximum(reference_transfer, stft.eps))
    )
    instability = float(np.median(log_error))
    drift = float(
        np.sum(np.abs(np.sqrt(enhanced_power) - np.sqrt(near_power)))
        / np.maximum(np.sum(np.sqrt(near_power)), stft.eps)
    )
    enhanced_floor = np.quantile(enhanced_power, subtraction.noise_quantile, axis=1)
    noise_reduction = float(
        10.0
        * np.log10(
            np.maximum(np.sum(near_floor), stft.eps) / np.maximum(np.sum(enhanced_floor), stft.eps)
        )
    )
    risk = reference_risk_score(leakage, instability, drift)
    return ReferenceDiagnostics(leakage, instability, drift, noise_reduction, risk)


def aggregate_reference_profile(
    transfer_power: NDArray[np.floating],
    diagnostics: list[ReferenceDiagnostics],
    config: ReferenceProfileConfig,
) -> ReferenceSafetyProfile:
    """Aggregate clip diagnostics conservatively into one group profile."""
    if not diagnostics:
        raise ValueError("At least one diagnostic clip is required")
    leakage = float(
        np.quantile([item.leakage_index for item in diagnostics], config.upper_quantile)
    )
    instability = float(
        np.quantile([item.transfer_instability for item in diagnostics], config.upper_quantile)
    )
    drift = float(np.quantile([item.spectral_drift for item in diagnostics], config.upper_quantile))
    benefit = float(
        np.quantile([item.noise_reduction_db for item in diagnostics], config.lower_quantile)
    )
    return ReferenceSafetyProfile(
        transfer_power=np.asarray(transfer_power, dtype=np.float64),
        leakage_u95=leakage,
        transfer_instability_u95=instability,
        spectral_drift_u95=drift,
        noise_reduction_l05_db=benefit,
        risk_score=reference_risk_score(leakage, instability, drift),
        training_clips=len(diagnostics),
    )


def reference_risk_score(leakage: float, instability: float, drift: float) -> float:
    """Combine monotone risk components without learned weights."""
    values = (
        float(np.clip(leakage, 0.0, 1.0)),
        float(np.clip(instability / (1.0 + max(instability, 0.0)), 0.0, 1.0)),
        float(np.clip(drift / (1.0 + max(drift, 0.0)), 0.0, 1.0)),
    )
    return max(values)


def select_reference_view(
    profile: ReferenceSafetyProfile, policy: ReferenceSafetyPolicy
) -> SafetyDecision:
    """Return the conservative all-or-nothing view decision."""
    if (
        profile.risk_score <= policy.risk_max
        and profile.noise_reduction_l05_db >= policy.benefit_min_db
    ):
        return "refsub"
    return "near"


def _validate_waveform(values: NDArray[np.floating]) -> FloatArray:
    waveform = np.asarray(values, dtype=np.float64)
    if waveform.ndim != 1 or not len(waveform):
        raise ValueError("waveform must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(waveform)):
        raise ValueError("waveform must contain only finite values")
    return waveform
