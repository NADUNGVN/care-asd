"""Validated configuration contracts for the SAFE-REF experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ReferenceSTFTConfig(BaseModel):
    """Spectral-analysis contract shared by profiling and RefSub."""

    model_config = ConfigDict(extra="forbid")

    n_fft: int = Field(default=1024, gt=0)
    hop_length: int = Field(default=512, gt=0)
    window: Literal["hann"] = "hann"
    eps: float = Field(default=1.0e-8, gt=0.0)

    @model_validator(mode="after")
    def validate_hop(self) -> ReferenceSTFTConfig:
        if self.hop_length > self.n_fft:
            raise ValueError("hop_length must not exceed n_fft")
        return self


class ReferenceSubtractionConfig(BaseModel):
    """Fixed minimum-statistics far-reference subtraction."""

    model_config = ConfigDict(extra="forbid")

    noise_quantile: float = Field(default=0.10, gt=0.0, lt=0.5)
    alpha: float = Field(default=1.5, gt=0.0)
    beta: float = Field(default=0.10, gt=0.0, le=1.0)
    transfer_min: float = Field(default=0.10, gt=0.0)
    transfer_max: float = Field(default=10.0, gt=0.0)

    @model_validator(mode="after")
    def validate_transfer_bounds(self) -> ReferenceSubtractionConfig:
        if self.transfer_min >= self.transfer_max:
            raise ValueError("transfer_min must be smaller than transfer_max")
        return self


class ReferenceProfileConfig(BaseModel):
    """Normal-only group-profile aggregation."""

    model_config = ConfigDict(extra="forbid")

    upper_quantile: float = Field(default=0.95, gt=0.5, lt=1.0)
    lower_quantile: float = Field(default=0.05, gt=0.0, lt=0.5)
    decision_scope: Literal["machine_section"] = "machine_section"


class ReferenceSimulationConfig(BaseModel):
    """Deterministic semi-synthetic safety calibration."""

    model_config = ConfigDict(extra="forbid")

    cases: int = Field(default=4096, ge=32)
    calibration_fraction: float = Field(default=0.5, gt=0.0, lt=1.0)
    duration_seconds: float = Field(default=2.0, gt=0.25)
    sample_rate: int = Field(default=16000, gt=1000)
    seed: int = Field(default=2026, ge=0)
    safe_retention_min: float = Field(default=0.90, gt=0.0, le=1.0)
    safe_noise_reduction_min_db: float = 1.0
    false_safe_max: float = Field(default=0.05, gt=0.0, lt=0.5)
    false_safe_upper_ci_max: float = Field(default=0.10, gt=0.0, lt=0.5)
    minimum_coverage: float = Field(default=0.20, gt=0.0, le=1.0)
    minimum_risk_spearman: float = Field(default=0.60, ge=-1.0, le=1.0)
    minimum_tail_loss_reduction: float = Field(default=0.50, ge=0.0, le=1.0)
    percentile_step: int = Field(default=5, ge=1, le=25)


class ReferenceTrainingConfig(BaseModel):
    """Exact official-compatible AE training and seed contract."""

    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=256, ge=1)
    learning_rate: float = Field(default=1.0e-3, gt=0.0)
    device: Literal["cuda", "cpu"] = "cuda"
    screening_seeds: tuple[int, ...] = (13711, 42, 2026)
    replication_seeds: tuple[int, ...] = (
        13711,
        42,
        2026,
        3407,
        777,
        11,
        23,
        101,
        314,
        2718,
    )

    @model_validator(mode="after")
    def validate_seeds(self) -> ReferenceTrainingConfig:
        if not self.screening_seeds or not self.replication_seeds:
            raise ValueError("screening_seeds and replication_seeds must not be empty")
        if any(seed < 0 for seed in (*self.screening_seeds, *self.replication_seeds)):
            raise ValueError("all seeds must be non-negative")
        if len(set(self.screening_seeds)) != len(self.screening_seeds):
            raise ValueError("screening_seeds must be unique")
        if len(set(self.replication_seeds)) != len(self.replication_seeds):
            raise ValueError("replication_seeds must be unique")
        return self


class ReferenceDevelopmentGateConfig(BaseModel):
    """Predeclared development go/no-go thresholds."""

    model_config = ConfigDict(extra="forbid")

    macro_noninferiority_margin: float = Field(default=0.005, ge=0.0)
    machine_harm_margin: float = Field(default=0.005, ge=0.0)
    maximum_machine_drop: float = Field(default=0.01, ge=0.0)
    minimum_harm_reduction: float = Field(default=0.50, ge=0.0, le=1.0)


class ReferenceSafetyExperimentConfig(BaseModel):
    """Top-level versioned SAFE-REF configuration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    experiment_id: str = "phase10_reference_safety"
    stft: ReferenceSTFTConfig = Field(default_factory=ReferenceSTFTConfig)
    refsub: ReferenceSubtractionConfig = Field(default_factory=ReferenceSubtractionConfig)
    profile: ReferenceProfileConfig = Field(default_factory=ReferenceProfileConfig)
    simulation: ReferenceSimulationConfig = Field(default_factory=ReferenceSimulationConfig)
    training: ReferenceTrainingConfig = Field(default_factory=ReferenceTrainingConfig)
    development_gate: ReferenceDevelopmentGateConfig = Field(
        default_factory=ReferenceDevelopmentGateConfig
    )


class ReferenceSafetyPolicy(BaseModel):
    """Immutable output of synthetic-only safety calibration."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    risk_max: float = Field(ge=0.0, le=1.0)
    benefit_min_db: float
    calibration_cases: int = Field(ge=1)
    holdout_cases: int = Field(ge=1)
    calibration_false_safe_rate: float = Field(ge=0.0, le=1.0)
    calibration_coverage: float = Field(ge=0.0, le=1.0)
    source: Literal["semi_synthetic_only"] = "semi_synthetic_only"


def load_reference_safety_config(path: str | Path) -> ReferenceSafetyExperimentConfig:
    """Load and validate one SAFE-REF YAML configuration."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Reference-safety config not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Reference-safety config root must be a mapping")
    return ReferenceSafetyExperimentConfig.model_validate(payload)


def load_reference_safety_policy(path: str | Path) -> ReferenceSafetyPolicy:
    """Load and validate a calibrated policy YAML."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Reference-safety policy not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Reference-safety policy root must be a mapping")
    return ReferenceSafetyPolicy.model_validate(payload)


def reference_safety_config_hash(config: ReferenceSafetyExperimentConfig) -> str:
    """Return a deterministic SHA-256 of the validated SAFE-REF config."""
    canonical = json.dumps(config.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
