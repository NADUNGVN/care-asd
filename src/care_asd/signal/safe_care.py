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
        if frontend.gate.min_value > frontend.gate.max_value:
            raise ValueError("frontend.gate.min_value must not exceed max_value")
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
            gate = self._gate_from_coherence(coherence)

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
            starts = list(range(0, n_samples - self._signal.win_length + 1, self._signal.hop_length))
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
        cross = np.mean(near * np.conj(far), axis=0)
        near_power = np.mean(np.abs(near) ** 2, axis=0)
        far_power = np.mean(np.abs(far) ** 2, axis=0)
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
            transfer[frame] = cross_ema / (far_power_ema + reg_floor)
            coherence[frame] = np.clip(
                (np.abs(cross_ema) ** 2)
                / (near_power_ema * far_power_ema + self._signal.eps),
                0.0,
                1.0,
            )
        return transfer, coherence

    def _gate_from_coherence(self, coherence: FloatArray) -> FloatArray:
        gate = self._frontend.gate.min_value + (
            self._frontend.gate.max_value - self._frontend.gate.min_value
        ) * coherence
        return np.clip(gate, self._frontend.gate.min_value, self._frontend.gate.max_value)

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
