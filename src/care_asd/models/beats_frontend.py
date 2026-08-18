"""Pinned Microsoft BEATs Iteration-3 feature extraction adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


class OfficialBEATsFrontend:
    """Thin runtime wrapper around the pinned upstream ``BEATs.py`` implementation."""

    def __init__(
        self,
        *,
        source_directory: str | Path,
        checkpoint_path: str | Path,
        device: str = "cuda",
        frequency_patches: int = 8,
        mixed_precision: bool = True,
    ) -> None:
        import torch

        source = Path(source_directory).resolve()
        checkpoint = Path(checkpoint_path).resolve()
        if not (source / "BEATs.py").is_file():
            raise FileNotFoundError(f"Pinned BEATs.py not found: {source}")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"BEATs checkpoint not found: {checkpoint}")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the configured BEATs extraction job")
        module = _load_beats_module(source)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or "cfg" not in payload or "model" not in payload:
            raise ValueError("Unexpected BEATs checkpoint payload")
        config_type = module.BEATsConfig
        model_type = module.BEATs
        model = model_type(config_type(payload["cfg"]))
        model.load_state_dict(payload["model"])
        model.eval()
        model.to(device)
        self._torch = torch
        self._model: Any = model
        self.device = device
        self.frequency_patches = frequency_patches
        self.mixed_precision = mixed_precision

    def extract(self, waveforms: np.ndarray) -> np.ndarray:
        """Extract ``[batch, time_patch, frequency_patch, 768]`` tokens."""
        values = np.asarray(waveforms, dtype=np.float32)
        if values.ndim != 2 or min(values.shape) < 1:
            raise ValueError("waveforms must have shape [batch, samples]")
        if not np.isfinite(values).all():
            raise ValueError("waveforms must be finite")
        torch = self._torch
        source = torch.from_numpy(values).to(self.device, non_blocking=True)
        autocast_enabled = self.mixed_precision and self.device.startswith("cuda")
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=autocast_enabled,
            ),
        ):
            tokens, padding_mask = self._model.extract_features(source, padding_mask=None)
        if padding_mask is not None:
            raise RuntimeError(
                "Fixed-duration BEATs extraction unexpectedly returned a padding mask"
            )
        output = reconstruct_frequency_grid(
            tokens.float().cpu().numpy(),
            frequency_patches=self.frequency_patches,
        )
        return output

    def extract_encoder_taps(
        self,
        waveforms: np.ndarray,
        *,
        taps: tuple[int, ...],
    ) -> dict[int, np.ndarray]:
        """Extract the patch projection (tap 0) and frozen Transformer depths 1--12.

        Tap zero is the normalized patch projection before positional convolution. Tap ``k``
        is the output after the first ``k`` Transformer blocks. The implementation deliberately
        uses the pinned upstream modules without editing the external BEATs checkout.
        """
        values = np.asarray(waveforms, dtype=np.float32)
        if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all():
            raise ValueError("waveforms must be a finite [batch, samples] array")
        if not taps or tuple(sorted(set(taps))) != taps:
            raise ValueError("taps must be a non-empty sorted tuple of unique depths")
        encoder_layers = int(self._model.cfg.encoder_layers)
        if taps[0] < 0 or taps[-1] > encoder_layers:
            raise ValueError(f"taps must be in [0, {encoder_layers}]")

        torch = self._torch
        source = torch.from_numpy(values).to(self.device, non_blocking=True)
        autocast_enabled = self.mixed_precision and self.device.startswith("cuda")
        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=autocast_enabled,
            ),
        ):
            fbank = self._model.preprocess(source)
            features = self._model.patch_embedding(fbank.unsqueeze(1))
            features = features.reshape(features.shape[0], features.shape[1], -1).transpose(1, 2)
            features = self._model.layer_norm(features)
            if self._model.post_extract_proj is not None:
                features = self._model.post_extract_proj(features)
            selected: dict[int, Any] = {0: features}
            positive = tuple(tap for tap in taps if tap > 0)
            if positive:
                encoded_input = self._model.dropout_input(features)
                _, layer_results = self._model.encoder(
                    encoded_input,
                    padding_mask=None,
                    layer=max(positive) - 1,
                )
                for tap in positive:
                    layer_value = layer_results[tap][0]
                    if tap == encoder_layers and bool(self._model.encoder.layer_norm_first):
                        layer_value = self._model.encoder.layer_norm(layer_value)
                    selected[tap] = layer_value.transpose(0, 1)

        return {
            tap: reconstruct_frequency_grid(
                selected[tap].float().cpu().numpy(),
                frequency_patches=self.frequency_patches,
            )
            for tap in taps
        }


def reconstruct_frequency_grid(
    tokens: np.ndarray,
    *,
    frequency_patches: int,
) -> np.ndarray:
    """Restore BEATs' time-major, frequency-minor flattened patch sequence."""
    values = np.asarray(tokens)
    if frequency_patches < 1:
        raise ValueError("frequency_patches must be positive")
    if values.ndim != 3 or values.shape[1] % frequency_patches != 0:
        raise RuntimeError(
            f"Cannot reconstruct {frequency_patches} frequency patches from {values.shape}"
        )
    if not np.isfinite(values).all():
        raise RuntimeError(
            "BEATs produced non-finite tokens; disable frontend inference mixed precision"
        )
    time_patches = values.shape[1] // frequency_patches
    return values.reshape(
        values.shape[0],
        time_patches,
        frequency_patches,
        values.shape[2],
    )


def fixed_duration_waveform(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    duration_seconds: float,
) -> np.ndarray:
    """Leading-crop or right-zero-pad a waveform to the frozen BEATs duration."""
    if sample_rate <= 0 or duration_seconds <= 0.0:
        raise ValueError("sample_rate and duration_seconds must be positive")
    values = np.asarray(waveform, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("waveform must be a finite non-empty vector")
    samples = round(sample_rate * duration_seconds)
    if len(values) >= samples:
        return values[:samples].copy()
    return np.pad(values, (0, samples - len(values)))


def _load_beats_module(source: Path) -> ModuleType:
    module_name = "care_asd_pinned_official_beats"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    spec = importlib.util.spec_from_file_location(module_name, source / "BEATs.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned BEATs module from {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
