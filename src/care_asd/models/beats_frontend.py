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
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            tokens, padding_mask = self._model.extract_features(source, padding_mask=None)
        if padding_mask is not None:
            raise RuntimeError("Fixed-duration BEATs extraction unexpectedly returned a padding mask")
        if tokens.ndim != 3 or tokens.shape[1] % self.frequency_patches != 0:
            raise RuntimeError(
                f"Cannot reconstruct {self.frequency_patches} frequency patches from {tokens.shape}"
            )
        time_patches = tokens.shape[1] // self.frequency_patches
        grid = tokens.reshape(
            tokens.shape[0],
            time_patches,
            self.frequency_patches,
            tokens.shape[2],
        )
        return grid.float().cpu().numpy()


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
