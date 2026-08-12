"""Deterministic, auditable DSP controls for the Safe CARE study.

These front-ends deliberately expose spectral views rather than a learned
embedding.  A later common encoder/scorer consumes the returned views; that
separation prevents a DSP comparison from silently changing the model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from care_asd.config import SignalConfig

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
FrontEndName = Literal[
    "near",
    "far",
    "average",
    "difference",
    "early_concat",
    "spectral_subtraction",
    "wiener",
    "coherence_mask",
    "adaptive_filter",
    "late_score_fusion",
]


@dataclass(frozen=True)
class FeatureBatch:
    """Public, validated output contract for every Phase 3 front-end.

    ``views`` contains one or more complex STFT arrays in ``(frames, bins)``
    order. ``diagnostics`` uses the same shape for frame/frequency quantities.
    No view overwrites the original near signal: callers choose the view(s)
    consumed by their common encoder.
    """

    frontend_name: FrontEndName
    sample_rate: int
    views: Mapping[str, ComplexArray]
    diagnostics: Mapping[str, FloatArray]
    score_fusion: Literal["none", "mean"] = "none"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not self.views:
            raise ValueError("FeatureBatch requires at least one spectral view")
        shape: tuple[int, int] | None = None
        for name, view in self.views.items():
            if view.ndim != 2 or view.shape[0] == 0 or view.shape[1] == 0:
                raise ValueError(f"view {name!r} must have non-empty (frames, bins) shape")
            if not np.isfinite(view).all():
                raise ValueError(f"view {name!r} contains NaN or Inf")
            if shape is None:
                shape = view.shape
            elif shape != view.shape:
                raise ValueError("all spectral views must use the same shape")
        assert shape is not None
        for name, value in self.diagnostics.items():
            if value.shape != shape:
                raise ValueError(f"diagnostic {name!r} must match spectral view shape")
            if not np.isfinite(value).all():
                raise ValueError(f"diagnostic {name!r} contains NaN or Inf")
        object.__setattr__(self, "views", MappingProxyType(dict(self.views)))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))


class AudioFrontEnd(Protocol):
    """Stable Phase 3 transform boundary used by every DSP control."""

    def transform(self, waveform: FloatArray, sample_rate: int) -> FeatureBatch:
        """Return deterministic spectral views from ordered ``(near, far)`` audio."""


class DSPFrontEnd:
    """One implementation for all named non-learned stereo controls.

    Static front-ends use a shared STFT implementation. The adaptive filter is
    causal: its coefficient at a frame is estimated only from that frame and
    preceding frames. ``subtraction_strength`` is fixed across all machines.
    """

    def __init__(
        self,
        name: FrontEndName,
        signal: SignalConfig,
        *,
        subtraction_strength: float = 0.5,
        coherence_alpha: float = 0.95,
        adaptive_step: float = 0.05,
        eps: float | None = None,
    ) -> None:
        if signal.win_length > signal.n_fft:
            raise ValueError("signal.win_length must not exceed signal.n_fft")
        if not 0.0 <= subtraction_strength <= 1.0:
            raise ValueError("subtraction_strength must be in [0, 1]")
        if not 0.0 <= coherence_alpha < 1.0:
            raise ValueError("coherence_alpha must be in [0, 1)")
        if adaptive_step <= 0.0:
            raise ValueError("adaptive_step must be positive")
        self.name = name
        self._signal = signal
        self._subtraction_strength = subtraction_strength
        self._coherence_alpha = coherence_alpha
        self._adaptive_step = adaptive_step
        self._eps = signal.eps if eps is None else eps
        if self._eps <= 0.0:
            raise ValueError("eps must be positive")

    def transform(self, waveform: FloatArray, sample_rate: int) -> FeatureBatch:
        """Transform a finite stereo waveform ordered as ``(near, far)``."""
        audio = _validate_audio(waveform, sample_rate)
        near = self._stft(audio[0])
        far = self._stft(audio[1])
        causal_coherence = self._causal_coherence(near, far)
        if self.name == "near":
            return self._batch(sample_rate, {"near": near}, near, causal_coherence)
        if self.name == "far":
            return self._batch(sample_rate, {"far": far}, near, causal_coherence)
        if self.name == "average":
            return self._batch(sample_rate, {"average": 0.5 * (near + far)}, near, causal_coherence)
        if self.name == "difference":
            return self._batch(sample_rate, {"difference": near - far}, near, causal_coherence)
        if self.name == "early_concat":
            return self._batch(sample_rate, {"near": near, "far": far}, near, causal_coherence)
        if self.name == "late_score_fusion":
            return self._batch(
                sample_rate,
                {"near": near, "far": far},
                near,
                causal_coherence,
                score_fusion="mean",
            )
        if self.name == "spectral_subtraction":
            residual = _magnitude_subtract(near, far, self._subtraction_strength, self._eps)
            return self._batch(sample_rate, {"residual": residual}, near, causal_coherence)
        if self.name == "wiener":
            near_power = np.abs(near) ** 2
            far_power = np.abs(far) ** 2
            gain = np.where(
                far_power == 0.0,
                1.0,
                near_power / (near_power + far_power + self._eps),
            )
            return self._batch(
                sample_rate, {"wiener": gain * near}, near, causal_coherence, gain=gain
            )
        if self.name == "coherence_mask":
            # This pre-registered control is intentionally static, unlike the
            # causal adaptive filter below. The causal estimate remains a
            # diagnostic for deployment-oriented methods in Phase 4.
            static_coherence = self._static_coherence(near, far)
            gain = 1.0 - static_coherence
            return self._batch(
                sample_rate,
                {"coherence_mask": gain * near},
                near,
                causal_coherence,
                gain=gain,
                static_coherence=static_coherence,
            )
        if self.name == "adaptive_filter":
            residual, coefficient = self._causal_adaptive_residual(near, far)
            return self._batch(
                sample_rate,
                {"adaptive_residual": residual},
                near,
                causal_coherence,
                adaptive_gain=coefficient,
            )
        raise AssertionError(f"Unhandled DSP front-end: {self.name}")

    def _batch(
        self,
        sample_rate: int,
        views: Mapping[str, ComplexArray],
        near: ComplexArray,
        coherence: FloatArray,
        *,
        score_fusion: Literal["none", "mean"] = "none",
        **diagnostics: FloatArray,
    ) -> FeatureBatch:
        primary = next(iter(views.values()))
        ratio = np.abs(primary) ** 2 / np.maximum(np.abs(near) ** 2, self._eps)
        all_diagnostics = {
            "coherence": coherence,
            "view_to_near_energy_ratio": ratio,
            **diagnostics,
        }
        return FeatureBatch(
            frontend_name=self.name,
            sample_rate=sample_rate,
            views=views,
            diagnostics=all_diagnostics,
            score_fusion=score_fusion,
        )

    def _stft(self, samples: FloatArray) -> ComplexArray:
        window = np.hanning(self._signal.win_length).astype(np.float64)
        if samples.shape[0] <= self._signal.win_length:
            starts = [0]
        else:
            starts = list(
                range(0, samples.shape[0] - self._signal.win_length + 1, self._signal.hop_length)
            )
            final_start = samples.shape[0] - self._signal.win_length
            if starts[-1] != final_start:
                starts.append(final_start)
        frames = []
        for start in starts:
            frame = samples[start : start + self._signal.win_length]
            if len(frame) < self._signal.win_length:
                frame = np.pad(frame, (0, self._signal.win_length - len(frame)))
            frames.append(frame * window)
        return np.fft.rfft(np.stack(frames), n=self._signal.n_fft, axis=1).astype(np.complex128)

    def _causal_coherence(self, near: ComplexArray, far: ComplexArray) -> FloatArray:
        cross_ema: ComplexArray | None = None
        near_ema: FloatArray | None = None
        far_ema: FloatArray | None = None
        result = np.empty(near.shape, dtype=np.float64)
        for index in range(near.shape[0]):
            cross = near[index] * np.conj(far[index])
            near_power = np.abs(near[index]) ** 2
            far_power = np.abs(far[index]) ** 2
            if cross_ema is None:
                cross_ema, near_ema, far_ema = cross, near_power, far_power
            else:
                alpha = self._coherence_alpha
                cross_ema = alpha * cross_ema + (1.0 - alpha) * cross
                assert near_ema is not None and far_ema is not None
                near_ema = alpha * near_ema + (1.0 - alpha) * near_power
                far_ema = alpha * far_ema + (1.0 - alpha) * far_power
            assert near_ema is not None and far_ema is not None
            result[index] = np.clip(
                np.abs(cross_ema) ** 2 / (near_ema * far_ema + self._eps), 0.0, 1.0
            )
        return result

    def _static_coherence(self, near: ComplexArray, far: ComplexArray) -> FloatArray:
        """Return a clip-level magnitude-squared coherence, broadcast by frame."""
        cross = np.mean(near * np.conj(far), axis=0)
        near_power = np.mean(np.abs(near) ** 2, axis=0)
        far_power = np.mean(np.abs(far) ** 2, axis=0)
        value = np.clip(np.abs(cross) ** 2 / (near_power * far_power + self._eps), 0.0, 1.0)
        return np.broadcast_to(value, near.shape).copy()

    def _causal_adaptive_residual(
        self, near: ComplexArray, far: ComplexArray
    ) -> tuple[ComplexArray, FloatArray]:
        coefficient = np.zeros(near.shape[1], dtype=np.complex128)
        output = np.empty_like(near)
        coefficient_energy = np.empty(near.shape, dtype=np.float64)
        for index in range(near.shape[0]):
            estimate = coefficient * far[index]
            error = near[index] - estimate
            normalizer = np.abs(far[index]) ** 2 + self._eps
            coefficient = (
                coefficient + self._adaptive_step * np.conj(far[index]) * error / normalizer
            )
            output[index] = error
            coefficient_energy[index] = np.abs(coefficient) ** 2
        return output, coefficient_energy


def create_dsp_frontend(name: FrontEndName, signal: SignalConfig) -> DSPFrontEnd:
    """Create a named deterministic Phase 3 front-end."""
    return DSPFrontEnd(name, signal)


def available_dsp_frontends() -> tuple[FrontEndName, ...]:
    """Return the complete pre-registered Phase 3 control set."""
    return (
        "near",
        "far",
        "average",
        "difference",
        "early_concat",
        "spectral_subtraction",
        "wiener",
        "coherence_mask",
        "adaptive_filter",
        "late_score_fusion",
    )


def _validate_audio(waveform: FloatArray, sample_rate: int) -> FloatArray:
    audio = np.asarray(waveform, dtype=np.float64)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if audio.ndim != 2 or audio.shape[0] != 2 or audio.shape[1] == 0:
        raise ValueError("waveform must have non-empty shape (2, samples)")
    if not np.isfinite(audio).all():
        raise ValueError("waveform must contain only finite values")
    return audio


def _magnitude_subtract(
    near: ComplexArray,
    far: ComplexArray,
    strength: float,
    eps: float,
) -> ComplexArray:
    magnitude = np.maximum(np.abs(near) - strength * np.abs(far), 0.0)
    phase = near / np.maximum(np.abs(near), eps)
    return magnitude * phase
