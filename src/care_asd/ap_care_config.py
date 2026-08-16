"""Validated configuration contracts for AP-CARE v2 experiments."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class APCARESTFTConfig(BaseModel):
    """Causal spectral-analysis contract."""

    model_config = ConfigDict(extra="forbid")

    n_fft: int = Field(default=1024, gt=0)
    win_length: int = Field(default=1024, gt=0)
    hop_length: int = Field(default=512, gt=0)
    window: Literal["hann"] = "hann"
    eps: float = Field(default=1.0e-8, gt=0.0)

    @model_validator(mode="after")
    def validate_lengths(self) -> APCARESTFTConfig:
        if self.win_length > self.n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if self.hop_length > self.win_length:
            raise ValueError("hop_length must not exceed win_length")
        return self


class APCARETransferConfig(BaseModel):
    """Training-normal transfer estimation and causal monitoring."""

    model_config = ConfigDict(extra="forbid")

    alpha: float = Field(default=0.95, ge=0.0, lt=1.0)
    regularization: float = Field(default=1.0e-8, gt=0.0)
    frequency_smoothing_bins: int = Field(default=5, ge=1)
    max_magnitude: float = Field(default=10.0, gt=0.0)
    machine_support_midpoint_db: float = 3.0
    machine_support_scale_db: float = Field(default=1.5, gt=0.0)
    machine_support_median_bins: int = Field(default=31, ge=3)

    @model_validator(mode="after")
    def validate_support_filter(self) -> APCARETransferConfig:
        if self.machine_support_median_bins % 2 == 0:
            raise ValueError("machine_support_median_bins must be odd")
        return self


class APCAREControllerConfig(BaseModel):
    """Deterministic AP-CARE controller and intervention budget."""

    model_config = ConfigDict(extra="forbid")

    max_gain: float = Field(default=1.0, ge=0.0, le=2.0)
    far_dominance_midpoint_db: float = -3.0
    far_dominance_scale_db: float = Field(default=6.0, gt=0.0)
    risk_terms_bypass: bool = False
    budget_enabled: bool = True
    uncertainty_log_magnitude_scale: float = Field(default=0.70, gt=0.0)
    uncertainty_delay_max_seconds: float = Field(default=0.020, gt=0.0)
    max_removed_energy_ratio: float = Field(default=0.20, gt=0.0, le=1.0)
    band_edges_hz: tuple[float, ...] = (0.0, 500.0, 2000.0, 8000.0)
    warmup_frames: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_bands(self) -> APCAREControllerConfig:
        if len(self.band_edges_hz) < 2:
            raise ValueError("band_edges_hz must contain at least two edges")
        if self.band_edges_hz[0] != 0.0:
            raise ValueError("band_edges_hz must start at 0 Hz")
        if any(right <= left for left, right in pairwise(self.band_edges_hz)):
            raise ValueError("band_edges_hz must be strictly increasing")
        return self


class APCAREG1GateConfig(BaseModel):
    """Pre-registered controlled-mechanism thresholds."""

    model_config = ConfigDict(extra="forbid")

    leakage_spearman_min: float = Field(default=0.60, ge=-1.0, le=1.0)
    leakage_spearman_ci_low_min: float = Field(default=0.40, ge=-1.0, le=1.0)
    uncertainty_spearman_min: float = Field(default=0.60, ge=-1.0, le=1.0)
    uncertainty_spearman_ci_low_min: float = Field(default=0.40, ge=-1.0, le=1.0)
    matched_attenuation_tolerance_db: float = Field(default=0.25, gt=0.0)
    retention_improvement_min: float = Field(default=0.10, ge=0.0)
    fault_retention_median_min: float = Field(default=0.90, ge=0.0)
    fault_retention_q05_min: float = Field(default=0.75, ge=0.0)
    noise_attenuation_median_min_db: float = 1.0
    active_case_fraction_min: float = Field(default=0.20, ge=0.0, le=1.0)
    active_attenuation_median_min_db: float = 1.0
    medium_high_leakage_min: float = Field(default=0.20, ge=0.0, le=1.0)
    bootstrap_iterations: int = Field(default=2000, ge=100)


class APCARESimulationConfig(BaseModel):
    """Controlled AP-CARE synthetic sweep."""

    model_config = ConfigDict(extra="forbid")

    cases: int = Field(default=512, ge=32)
    calibration_fraction: float = Field(default=0.50, gt=0.0, lt=1.0)
    duration_seconds: float = Field(default=2.0, gt=0.1)
    sample_rate: int = Field(default=16000, gt=1000)
    seed: int = Field(default=2026, ge=0)
    profile_clips: int = Field(default=3, ge=2)
    d00_alpha_grid: tuple[float, ...] = (0.10, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
    d00_floor_beta: float = Field(default=0.10, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_grid(self) -> APCARESimulationConfig:
        if not self.d00_alpha_grid or any(value <= 0.0 for value in self.d00_alpha_grid):
            raise ValueError("d00_alpha_grid must contain positive values")
        if len(set(self.d00_alpha_grid)) != len(self.d00_alpha_grid):
            raise ValueError("d00_alpha_grid values must be unique")
        return self


class APCAREExperimentConfig(BaseModel):
    """Top-level immutable AP-CARE v2 configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    experiment_id: str = "ap_care_v2_g1"
    stft: APCARESTFTConfig = Field(default_factory=APCARESTFTConfig)
    transfer: APCARETransferConfig = Field(default_factory=APCARETransferConfig)
    controller: APCAREControllerConfig = Field(default_factory=APCAREControllerConfig)
    simulation: APCARESimulationConfig = Field(default_factory=APCARESimulationConfig)
    gate: APCAREG1GateConfig = Field(default_factory=APCAREG1GateConfig)

    @model_validator(mode="after")
    def validate_frequency_contract(self) -> APCAREExperimentConfig:
        nyquist = self.simulation.sample_rate / 2.0
        if self.controller.band_edges_hz[-1] < nyquist:
            raise ValueError("band_edges_hz must cover the simulation Nyquist frequency")
        return self


def load_ap_care_config(path: str | Path) -> APCAREExperimentConfig:
    """Load one AP-CARE YAML file and validate all external input at the boundary."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"AP-CARE config not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("AP-CARE config root must be a mapping")
    return APCAREExperimentConfig.model_validate(payload)


def ap_care_config_hash(config: APCAREExperimentConfig) -> str:
    """Return a stable SHA-256 digest of a validated AP-CARE config."""
    canonical = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
