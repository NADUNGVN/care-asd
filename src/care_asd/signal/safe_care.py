"""Causal, bounded acoustic-path residual views for CARE-ASD.

The front-end intentionally preserves the original near-channel view. Its
residual is an auxiliary diagnostic/feature view, never a replacement for the
near microphone signal. This keeps downstream models able to retain machine
evidence even when the far microphone is correlated with it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from care_asd.config import FrontendConfig, SignalConfig
from care_asd.signal.dsp_baselines import FeatureBatch

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SafeCAREOutput:
    """Stable multi-view output of :class:`SafeCAREFrontEnd`.

    All spectral arrays use ``(frame, frequency_bin)`` ordering. ``near_stft``
    is the original near-channel STFT and is never attenuated by this module.
    """

    near_stft: ComplexArray
    far_stft: ComplexArray
    residual_stft: ComplexArray
    transfer_function: ComplexArray
    gate: FloatArray
    snr_proxy: FloatArray
    coherence: FloatArray
    path_confidence: FloatArray
    removed_energy_ratio: FloatArray


class SafeCAREFrontEnd:
    """Build a causal, energy-bounded acoustic-path residual view.

    ``causal_ema`` updates spectral statistics frame by frame, so changing a
    future input frame cannot alter an earlier output. ``static_per_clip`` is
    supplied only as a non-causal diagnostic baseline.
    """

    def __init__(self, signal: SignalConfig, frontend: FrontendConfig) -> None:
        if signal.win_length > signal.n_fft:
            raise ValueError("signal.win_length must not exceed signal.n_fft")
        self._signal = signal
        self._frontend = frontend

    def transform(self, waveform: NDArray[np.float64]) -> SafeCAREOutput:
        """Transform a two-channel waveform into safe CARE spectral views.

        ``waveform`` must use shape ``(2, samples)`` ordered as ``(near, far)``.
        Channel mapping belongs at the dataset boundary; this class accepts only
        already ordered channels.
        """
        if waveform.ndim != 2 or waveform.shape[0] != 2:
            raise ValueError("waveform must have shape (2, samples)")
        if waveform.shape[1] == 0:
            raise ValueError("waveform must contain at least one sample")
        if not np.isfinite(waveform).all():
            raise ValueError("waveform must contain only finite values")

        near = self._stft(np.asarray(waveform[0], dtype=np.float64))
        far = self._stft(np.asarray(waveform[1], dtype=np.float64))
        transfer, coherence = self._estimate_transfer(near, far)

        if self._frontend.gate.bypass:
            gate = np.zeros_like(coherence)
        else:
            snr_proxy = self._snr_proxy(near, far)
            gate = self._gate_from_statistics(coherence, snr_proxy)
        if self._frontend.gate.bypass:
            snr_proxy = self._snr_proxy(near, far)

        cancelled = gate * transfer * far
        cancelled, removed_ratio = self._bound_removed_energy(cancelled, near)
        residual = near - cancelled

        cap = self._frontend.residual.max_removed_energy_ratio
        normalized_removal = np.clip(removed_ratio / cap, 0.0, 1.0)
        confidence = np.clip(coherence * (1.0 - normalized_removal[:, np.newaxis]), 0.0, 1.0)

        return SafeCAREOutput(
            near_stft=near,
            far_stft=far,
            residual_stft=residual,
            transfer_function=transfer,
            gate=gate,
            snr_proxy=snr_proxy,
            coherence=coherence,
            path_confidence=confidence,
            removed_energy_ratio=removed_ratio,
        )

    def _stft(self, samples: FloatArray) -> ComplexArray:
        window = np.hanning(self._signal.win_length).astype(np.float64)
        n_samples = samples.shape[0]
        if n_samples <= self._signal.win_length:
            starts = [0]
        else:
            starts = list(
                range(0, n_samples - self._signal.win_length + 1, self._signal.hop_length)
            )
            final_start = n_samples - self._signal.win_length
            if starts[-1] != final_start:
                starts.append(final_start)

        frames: list[FloatArray] = []
        for start in starts:
            frame = samples[start : start + self._signal.win_length]
            if frame.shape[0] < self._signal.win_length:
                frame = np.pad(frame, (0, self._signal.win_length - frame.shape[0]))
            frames.append(frame * window)
        matrix = np.stack(frames, axis=0)
        return np.fft.rfft(matrix, n=self._signal.n_fft, axis=1).astype(np.complex128)

    def _estimate_transfer(
        self,
        near: ComplexArray,
        far: ComplexArray,
    ) -> tuple[ComplexArray, FloatArray]:
        if self._frontend.transfer.mode == "static_per_clip":
            return self._static_transfer(near, far)
        return self._causal_ema_transfer(near, far)

    def _static_transfer(
        self,
        near: ComplexArray,
        far: ComplexArray,
    ) -> tuple[ComplexArray, FloatArray]:
        cross = self._smooth_complex_frequency(np.mean(near * np.conj(far), axis=0))
        near_power = self._smooth_real_frequency(np.mean(np.abs(near) ** 2, axis=0))
        far_power = self._smooth_real_frequency(np.mean(np.abs(far) ** 2, axis=0))
        transfer_row = cross / (far_power + self._frontend.transfer.reg_floor)
        coherence_row = np.clip(
            (np.abs(cross) ** 2) / (near_power * far_power + self._signal.eps),
            0.0,
            1.0,
        )
        return (
            np.broadcast_to(transfer_row, near.shape).copy(),
            np.broadcast_to(coherence_row, near.shape).copy(),
        )

    def _causal_ema_transfer(
        self,
        near: ComplexArray,
        far: ComplexArray,
    ) -> tuple[ComplexArray, FloatArray]:
        alpha = self._frontend.transfer.alpha
        reg_floor = self._frontend.transfer.reg_floor
        frames, bins = near.shape
        transfer = np.empty((frames, bins), dtype=np.complex128)
        coherence = np.empty((frames, bins), dtype=np.float64)

        cross_ema: ComplexArray | None = None
        near_power_ema: FloatArray | None = None
        far_power_ema: FloatArray | None = None
        for frame in range(frames):
            cross = near[frame] * np.conj(far[frame])
            near_power = np.abs(near[frame]) ** 2
            far_power = np.abs(far[frame]) ** 2
            if cross_ema is None:
                cross_ema = cross
                near_power_ema = near_power
                far_power_ema = far_power
            else:
                assert near_power_ema is not None
                assert far_power_ema is not None
                cross_ema = alpha * cross_ema + (1.0 - alpha) * cross
                near_power_ema = alpha * near_power_ema + (1.0 - alpha) * near_power
                far_power_ema = alpha * far_power_ema + (1.0 - alpha) * far_power
            assert near_power_ema is not None
            assert far_power_ema is not None
            smoothed_cross = self._smooth_complex_frequency(cross_ema)
            smoothed_near_power = self._smooth_real_frequency(near_power_ema)
            smoothed_far_power = self._smooth_real_frequency(far_power_ema)
            transfer[frame] = smoothed_cross / (smoothed_far_power + reg_floor)
            coherence[frame] = np.clip(
                (np.abs(smoothed_cross) ** 2)
                / (smoothed_near_power * smoothed_far_power + self._signal.eps),
                0.0,
                1.0,
            )
        return transfer, coherence

    def _snr_proxy(self, near: ComplexArray, far: ComplexArray) -> FloatArray:
        ratio = (np.abs(near) ** 2 + self._signal.eps) / (np.abs(far) ** 2 + self._signal.eps)
        return np.clip(np.log(ratio), -12.0, 12.0)

    def _gate_from_statistics(self, coherence: FloatArray, snr_proxy: FloatArray) -> FloatArray:
        gate_config = self._frontend.gate
        logits = (
            gate_config.coherence_weight * (2.0 * coherence - 1.0)
            + gate_config.snr_weight * snr_proxy
            + gate_config.bias
        )
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        gate = gate_config.min_value + (gate_config.max_value - gate_config.min_value) * sigmoid
        return np.clip(gate, self._frontend.gate.min_value, self._frontend.gate.max_value)

    def _smooth_real_frequency(self, values: FloatArray) -> FloatArray:
        width = self._frontend.transfer.frequency_smoothing_bins
        if width == 1:
            return values
        kernel = np.full(width, 1.0 / width, dtype=np.float64)
        return np.convolve(values, kernel, mode="same")

    def _smooth_complex_frequency(self, values: ComplexArray) -> ComplexArray:
        width = self._frontend.transfer.frequency_smoothing_bins
        if width == 1:
            return values
        kernel = np.full(width, 1.0 / width, dtype=np.float64)
        return np.convolve(values.real, kernel, mode="same") + 1j * np.convolve(
            values.imag, kernel, mode="same"
        )

    def _bound_removed_energy(
        self,
        cancelled: ComplexArray,
        near: ComplexArray,
    ) -> tuple[ComplexArray, FloatArray]:
        removed = np.sum(np.abs(cancelled) ** 2, axis=1)
        near_energy = np.sum(np.abs(near) ** 2, axis=1)
        ratio = removed / np.maximum(near_energy, self._signal.eps)
        cap = self._frontend.residual.max_removed_energy_ratio
        scale = np.minimum(1.0, np.sqrt(cap / np.maximum(ratio, self._signal.eps)))
        bounded = cancelled * scale[:, np.newaxis]
        bounded_ratio = np.sum(np.abs(bounded) ** 2, axis=1) / np.maximum(
            near_energy,
            self._signal.eps,
        )
        return bounded, bounded_ratio


class CAREAudioFrontEnd:
    """Expose Safe CARE through the stable multi-view front-end contract.

    The class is deliberately an adapter instead of changing ``SafeCAREOutput``:
    existing callers retain the detailed acoustic-path result while Phase 5 can
    consume the common ``FeatureBatch`` interface shared with the DSP controls.
    """

    name = "care"

    def __init__(self, signal: SignalConfig, frontend: FrontendConfig) -> None:
        self._implementation = SafeCAREFrontEnd(signal, frontend)
        self._eps = signal.eps

    def transform(self, waveform: FloatArray, sample_rate: int) -> FeatureBatch:
        """Return near/far/residual views and auditable spatial diagnostics."""
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        output = self._implementation.transform(waveform)
        removed = np.broadcast_to(
            output.removed_energy_ratio[:, np.newaxis], output.near_stft.shape
        )
        phase = np.angle(output.near_stft * np.conj(output.far_stft))
        log_ratio = np.log(
            (np.abs(output.near_stft) + self._eps) / (np.abs(output.far_stft) + self._eps)
        )
        return FeatureBatch(
            frontend_name=self.name,
            sample_rate=sample_rate,
            views={
                "near": output.near_stft,
                "far": output.far_stft,
                "residual": output.residual_stft,
            },
            diagnostics={
                "coherence": output.coherence,
                "gate": output.gate,
                "snr_proxy": output.snr_proxy,
                "path_confidence": output.path_confidence,
                "removed_energy_ratio": removed,
                "log_ratio": log_ratio,
                "phase_sin": np.sin(phase),
                "phase_cos": np.cos(phase),
                "transfer_magnitude": np.abs(output.transfer_function),
                "view_to_near_energy_ratio": np.abs(output.residual_stft) ** 2
                / np.maximum(np.abs(output.near_stft) ** 2, self._eps),
            },
        )
