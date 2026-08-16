"""Deterministic anomaly-preserving bounded reference cancellation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter

from care_asd.ap_care_config import APCAREExperimentConfig, APCARESTFTConfig

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class APCAREProfile:
    """Immutable training-normal statistics used by AP-CARE inference."""

    sample_rate: int
    training_clips: int
    near_power: FloatArray
    far_power: FloatArray
    cross_spectrum: ComplexArray
    transfer_function: ComplexArray
    reverse_transfer_magnitude: FloatArray
    machine_support: FloatArray
    transfer_dispersion: FloatArray


@dataclass(frozen=True)
class APCAREOutput:
    """Stable AP-CARE spectral output and auditable controller diagnostics."""

    near_stft: ComplexArray
    far_stft: ComplexArray
    residual_stft: ComplexArray
    cancellation_filter: ComplexArray
    controller_gain: FloatArray
    noise_utility: FloatArray
    leakage_risk: FloatArray
    transfer_uncertainty: FloatArray
    transfer_gain_mismatch: FloatArray
    transfer_delay_mismatch_seconds: FloatArray
    coherence: FloatArray
    proposed_removed_energy: FloatArray
    permitted_removed_energy: FloatArray
    actual_removed_energy: FloatArray
    bound_active: BoolArray
    band_edges_hz: tuple[float, ...]


class APCAREController:
    """Fit normal-only statistics and apply a causal bounded controller.

    ``fit`` accepts only normal stereo clips. ``transform`` keeps the original
    near STFT in the output and creates a separate residual candidate. The
    controller never consumes anomaly labels or future frames.
    """

    def __init__(self, config: APCAREExperimentConfig) -> None:
        self._config = config

    def fit(
        self,
        normal_waveforms: NDArray[np.floating],
        sample_rate: int,
    ) -> APCAREProfile:
        """Estimate an AP-CARE profile from ``(clips, 2, samples)`` normal audio."""
        values = np.asarray(normal_waveforms, dtype=np.float64)
        if values.ndim != 3 or values.shape[1] != 2 or values.shape[0] < 2:
            raise ValueError("normal_waveforms must have shape (clips>=2, 2, samples)")
        if values.shape[2] == 0 or not np.all(np.isfinite(values)):
            raise ValueError("normal_waveforms must be non-empty and finite")
        self._validate_sample_rate(sample_rate)
        expected_bins = self._config.stft.n_fft // 2 + 1
        if self._config.transfer.frequency_smoothing_bins > expected_bins:
            raise ValueError("frequency_smoothing_bins exceeds the STFT bin count")
        if self._config.transfer.machine_support_median_bins > expected_bins:
            raise ValueError("machine_support_median_bins exceeds the STFT bin count")

        clip_statistics: list[tuple[FloatArray, FloatArray, ComplexArray, ComplexArray]] = []
        for clip in values:
            near = causal_stft(clip[0], self._config.stft)
            far = causal_stft(clip[1], self._config.stft)
            near_power = self._smooth_real(np.mean(np.abs(near) ** 2, axis=0))
            far_power = self._smooth_real(np.mean(np.abs(far) ** 2, axis=0))
            cross = self._smooth_complex(np.mean(near * np.conj(far), axis=0))
            transfer = self._bound_transfer(
                cross / (far_power + self._config.transfer.regularization)
            )
            clip_statistics.append((near_power, far_power, cross, transfer))

        near_power = np.mean([item[0] for item in clip_statistics], axis=0)
        far_power = np.mean([item[1] for item in clip_statistics], axis=0)
        cross = np.mean([item[2] for item in clip_statistics], axis=0)
        transfer = self._bound_transfer(cross / (far_power + self._config.transfer.regularization))
        reverse = np.abs(cross) / (near_power + self._config.transfer.regularization)
        support = self._machine_support(near_power)
        dispersion_rows: list[FloatArray] = []
        for _, _, _, clip_transfer in clip_statistics:
            magnitude_error = np.abs(
                np.log(
                    (np.abs(clip_transfer) + self._config.stft.eps)
                    / (np.abs(transfer) + self._config.stft.eps)
                )
            )
            phase_delta = np.unwrap(np.angle(clip_transfer * np.conj(transfer)))
            phase_error = np.abs(phase_delta - phase_delta[0]) / np.pi
            dispersion_rows.append(magnitude_error + phase_error)
        dispersion = np.median(np.stack(dispersion_rows, axis=0), axis=0)
        return APCAREProfile(
            sample_rate=sample_rate,
            training_clips=values.shape[0],
            near_power=np.asarray(near_power, dtype=np.float64),
            far_power=np.asarray(far_power, dtype=np.float64),
            cross_spectrum=np.asarray(cross, dtype=np.complex128),
            transfer_function=np.asarray(transfer, dtype=np.complex128),
            reverse_transfer_magnitude=np.asarray(reverse, dtype=np.float64),
            machine_support=np.asarray(support, dtype=np.float64),
            transfer_dispersion=np.asarray(dispersion, dtype=np.float64),
        )

    def transform(
        self,
        waveform: NDArray[np.floating],
        sample_rate: int,
        profile: APCAREProfile,
    ) -> APCAREOutput:
        """Apply AP-CARE to one already ordered ``(near, far)`` waveform."""
        values = self._validate_stereo(waveform)
        self._validate_profile(profile, sample_rate)
        near = causal_stft(values[0], self._config.stft)
        far = causal_stft(values[1], self._config.stft)
        frames, bins = near.shape
        noise_utility = np.empty((frames, bins), dtype=np.float64)
        leakage_risk = np.empty((frames, bins), dtype=np.float64)
        uncertainty = np.empty((frames, bins), dtype=np.float64)
        gain_mismatch = np.empty(frames, dtype=np.float64)
        delay_mismatch = np.empty(frames, dtype=np.float64)
        coherence = np.empty((frames, bins), dtype=np.float64)

        cross_ema = profile.cross_spectrum.copy()
        near_power_ema = profile.near_power.copy()
        far_power_ema = profile.far_power.copy()
        alpha = self._config.transfer.alpha
        controller = self._config.controller
        for frame in range(frames):
            cross_now = self._smooth_complex(near[frame] * np.conj(far[frame]))
            near_now = self._smooth_real(np.abs(near[frame]) ** 2)
            far_now = self._smooth_real(np.abs(far[frame]) ** 2)
            cross_ema = alpha * cross_ema + (1.0 - alpha) * cross_now
            near_power_ema = alpha * near_power_ema + (1.0 - alpha) * near_now
            far_power_ema = alpha * far_power_ema + (1.0 - alpha) * far_now

            coherence_row = np.clip(
                np.abs(cross_ema) ** 2 / (near_power_ema * far_power_ema + self._config.stft.eps),
                0.0,
                1.0,
            )
            far_dominance_db = 10.0 * np.log10(
                (far_power_ema + self._config.stft.eps) / (near_power_ema + self._config.stft.eps)
            )
            noise_row = _sigmoid(
                (far_dominance_db - controller.far_dominance_midpoint_db)
                / controller.far_dominance_scale_db
            )
            leakage_row = profile.machine_support
            current_transfer = self._bound_transfer(
                cross_ema / (far_power_ema + self._config.transfer.regularization)
            )
            uncertainty_value, gain_error, delay_error = self._transfer_uncertainty(
                current_transfer=current_transfer,
                current_far_power=far_power_ema,
                coherence=coherence_row,
                profile=profile,
                sample_rate=sample_rate,
            )
            uncertainty_row = np.full(bins, uncertainty_value, dtype=np.float64)
            noise_utility[frame] = noise_row
            leakage_risk[frame] = leakage_row
            uncertainty[frame] = uncertainty_row
            gain_mismatch[frame] = gain_error
            delay_mismatch[frame] = delay_error
            coherence[frame] = coherence_row

        if controller.risk_terms_bypass:
            gain = np.full_like(noise_utility, controller.max_gain)
        else:
            gain = controller.max_gain * noise_utility * (1.0 - leakage_risk) * (1.0 - uncertainty)
        gain = np.clip(gain, 0.0, controller.max_gain)
        if controller.warmup_frames:
            gain[: min(controller.warmup_frames, frames)] = 0.0
        proposal = gain * profile.transfer_function[np.newaxis, :] * far
        (
            effective_gain,
            proposed_energy,
            permitted_energy,
            actual_energy,
            bound_active,
        ) = self._apply_band_budget(gain, proposal, near, sample_rate)
        cancellation_filter = effective_gain * profile.transfer_function[np.newaxis, :]
        residual = near - cancellation_filter * far
        return APCAREOutput(
            near_stft=near,
            far_stft=far,
            residual_stft=np.asarray(residual, dtype=np.complex128),
            cancellation_filter=np.asarray(cancellation_filter, dtype=np.complex128),
            controller_gain=np.asarray(effective_gain, dtype=np.float64),
            noise_utility=noise_utility,
            leakage_risk=leakage_risk,
            transfer_uncertainty=uncertainty,
            transfer_gain_mismatch=gain_mismatch,
            transfer_delay_mismatch_seconds=delay_mismatch,
            coherence=coherence,
            proposed_removed_energy=proposed_energy,
            permitted_removed_energy=permitted_energy,
            actual_removed_energy=actual_energy,
            bound_active=bound_active,
            band_edges_hz=controller.band_edges_hz,
        )

    def apply_frozen(
        self,
        output: APCAREOutput,
        component_waveform: NDArray[np.floating],
        sample_rate: int,
    ) -> ComplexArray:
        """Apply one realized AP-CARE filter to a known stereo component."""
        values = self._validate_stereo(component_waveform)
        self._validate_sample_rate(sample_rate)
        near = causal_stft(values[0], self._config.stft)
        far = causal_stft(values[1], self._config.stft)
        if near.shape != output.cancellation_filter.shape:
            raise ValueError("component waveform does not match the realized AP-CARE output")
        return np.asarray(near - output.cancellation_filter * far, dtype=np.complex128)

    def _apply_band_budget(
        self,
        gain: FloatArray,
        proposal: ComplexArray,
        near: ComplexArray,
        sample_rate: int,
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, BoolArray]:
        edges = self._config.controller.band_edges_hz
        frequencies = np.fft.rfftfreq(self._config.stft.n_fft, d=1.0 / sample_rate)
        frames = near.shape[0]
        bands = len(edges) - 1
        effective = gain.copy()
        proposed = np.zeros((frames, bands), dtype=np.float64)
        permitted = np.zeros((frames, bands), dtype=np.float64)
        actual = np.zeros((frames, bands), dtype=np.float64)
        active = np.zeros((frames, bands), dtype=np.bool_)
        cap = self._config.controller.max_removed_energy_ratio
        for band, (low, high) in enumerate(pairwise(edges)):
            if band == bands - 1:
                selected = (frequencies >= low) & (frequencies <= high)
            else:
                selected = (frequencies >= low) & (frequencies < high)
            if not np.any(selected):
                continue
            proposed[:, band] = np.sum(np.abs(proposal[:, selected]) ** 2, axis=1)
            near_energy = np.sum(np.abs(near[:, selected]) ** 2, axis=1)
            if not self._config.controller.budget_enabled:
                permitted[:, band] = proposed[:, band]
                actual[:, band] = proposed[:, band]
                continue
            permitted[:, band] = cap * near_energy
            scale = np.minimum(
                1.0,
                np.sqrt(permitted[:, band] / np.maximum(proposed[:, band], self._config.stft.eps)),
            )
            effective[:, selected] *= scale[:, np.newaxis]
            active[:, band] = proposed[:, band] > permitted[:, band] + self._config.stft.eps
            actual[:, band] = proposed[:, band] * scale**2
        return effective, proposed, permitted, actual, active

    def _validate_profile(self, profile: APCAREProfile, sample_rate: int) -> None:
        self._validate_sample_rate(sample_rate)
        if profile.sample_rate != sample_rate:
            raise ValueError("profile sample rate does not match waveform sample rate")
        expected = self._config.stft.n_fft // 2 + 1
        arrays = (
            profile.near_power,
            profile.far_power,
            profile.cross_spectrum,
            profile.transfer_function,
            profile.reverse_transfer_magnitude,
            profile.machine_support,
            profile.transfer_dispersion,
        )
        if any(item.shape != (expected,) for item in arrays):
            raise ValueError("profile arrays do not match the configured STFT")
        if any(not np.all(np.isfinite(item)) for item in arrays):
            raise ValueError("profile arrays must be finite")

    def _validate_sample_rate(self, sample_rate: int) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self._config.controller.band_edges_hz[-1] < sample_rate / 2.0:
            raise ValueError("band_edges_hz must cover the waveform Nyquist frequency")

    @staticmethod
    def _validate_stereo(waveform: NDArray[np.floating]) -> FloatArray:
        values = np.asarray(waveform, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != 2 or values.shape[1] == 0:
            raise ValueError("waveform must have shape (2, samples)")
        if not np.all(np.isfinite(values)):
            raise ValueError("waveform must contain only finite values")
        return values

    def _machine_support(self, near_power: FloatArray) -> FloatArray:
        db = 10.0 * np.log10(near_power + self._config.stft.eps)
        baseline = median_filter(
            db,
            size=self._config.transfer.machine_support_median_bins,
            mode="nearest",
        )
        prominence = db - baseline
        return _sigmoid(
            (prominence - self._config.transfer.machine_support_midpoint_db)
            / self._config.transfer.machine_support_scale_db
        )

    def _transfer_uncertainty(
        self,
        *,
        current_transfer: ComplexArray,
        current_far_power: FloatArray,
        coherence: FloatArray,
        profile: APCAREProfile,
        sample_rate: int,
    ) -> tuple[float, float, float]:
        """Estimate causal transfer gain and delay mismatch outside machine support."""
        eps = self._config.stft.eps
        controller = self._config.controller
        noise_support = np.asarray(profile.machine_support <= 0.10, dtype=np.float64)
        if np.count_nonzero(noise_support) < 3:
            noise_support = np.maximum(1.0 - profile.machine_support, eps)
        weights = noise_support * coherence * current_far_power
        weight_sum = float(np.sum(weights))
        if weight_sum <= eps:
            return (
                1.0,
                controller.uncertainty_log_magnitude_scale,
                controller.uncertainty_delay_max_seconds,
            )
        normalized = weights / weight_sum
        log_magnitude_ratio = np.log(
            (np.abs(current_transfer) + eps) / (np.abs(profile.transfer_function) + eps)
        )
        magnitude_error = abs(float(np.sum(normalized * log_magnitude_ratio)))
        magnitude_risk = magnitude_error / (
            magnitude_error + controller.uncertainty_log_magnitude_scale
        )
        phase_delta = np.unwrap(np.angle(current_transfer * np.conj(profile.transfer_function)))
        frequencies = np.fft.rfftfreq(self._config.stft.n_fft, d=1.0 / sample_rate)
        frequency_center = float(np.sum(normalized * frequencies))
        phase_center = float(np.sum(normalized * phase_delta))
        centered_frequency = frequencies - frequency_center
        denominator = float(np.sum(normalized * centered_frequency**2))
        slope = float(
            np.sum(normalized * centered_frequency * (phase_delta - phase_center))
            / max(denominator, eps)
        )
        delay_error_seconds = abs(slope) / (2.0 * np.pi)
        delay_risk = min(
            delay_error_seconds / controller.uncertainty_delay_max_seconds,
            1.0,
        )
        current_risk = 0.5 * (magnitude_risk + delay_risk)
        return (
            float(np.clip(current_risk, 0.0, 1.0)),
            magnitude_error,
            delay_error_seconds,
        )

    def _bound_transfer(self, transfer: ComplexArray) -> ComplexArray:
        magnitude = np.abs(transfer)
        scale = np.minimum(
            1.0,
            self._config.transfer.max_magnitude / np.maximum(magnitude, self._config.stft.eps),
        )
        return np.asarray(transfer * scale, dtype=np.complex128)

    def _smooth_real(self, values: NDArray[np.floating]) -> FloatArray:
        width = self._config.transfer.frequency_smoothing_bins
        array = np.asarray(values, dtype=np.float64)
        if width == 1:
            return array
        kernel = np.full(width, 1.0 / width, dtype=np.float64)
        return np.asarray(np.convolve(array, kernel, mode="same"), dtype=np.float64)

    def _smooth_complex(self, values: NDArray[np.complexfloating]) -> ComplexArray:
        width = self._config.transfer.frequency_smoothing_bins
        array = np.asarray(values, dtype=np.complex128)
        if width == 1:
            return array
        kernel = np.full(width, 1.0 / width, dtype=np.float64)
        return np.asarray(
            np.convolve(array.real, kernel, mode="same")
            + 1j * np.convolve(array.imag, kernel, mode="same"),
            dtype=np.complex128,
        )


def causal_stft(samples: NDArray[np.floating], config: APCARESTFTConfig) -> ComplexArray:
    """Return a deterministic frame-major STFT without future-frame centering."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("samples must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("samples must contain only finite values")
    window = np.hanning(config.win_length).astype(np.float64)
    if len(values) <= config.win_length:
        starts = [0]
    else:
        starts = list(range(0, len(values) - config.win_length + 1, config.hop_length))
        final_start = len(values) - config.win_length
        if starts[-1] != final_start:
            starts.append(final_start)
    frames: list[FloatArray] = []
    for start in starts:
        frame = values[start : start + config.win_length]
        if len(frame) < config.win_length:
            frame = np.pad(frame, (0, config.win_length - len(frame)))
        frames.append(frame * window)
    return np.asarray(np.fft.rfft(np.stack(frames), n=config.n_fft, axis=1), dtype=np.complex128)


def _sigmoid(values: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0))), dtype=np.float64)
