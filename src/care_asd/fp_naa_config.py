"""Validated configuration for the FP-NAA successor track."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class ProvenanceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    beats_repository: HttpUrl
    beats_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    checkpoint_url: HttpUrl
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class FPFrontendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(gt=0)
    duration_seconds: float = Field(gt=0.0)
    channels: list[str]
    frequency_patches: int = Field(gt=0)
    embedding_dim: int = Field(gt=0)
    cache_dtype: str
    inference_batch_size: int = Field(gt=0)


class FPBackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temporal_pooling: str
    rdp_gamma: float = Field(ge=0.0)
    scorer: str
    cosine_distance_scale: float = Field(gt=0.0)
    local_density_neighbors: int = Field(gt=0)
    score_rescaling: str
    eps: float = Field(gt=0.0)


class FPAdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_dim: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    reference_dropout_probability: float = Field(ge=0.0, le=1.0)
    reference_corruption_probability: float = Field(ge=0.0, le=1.0)


class FPObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normal_mse_weight: float = Field(ge=0.0)
    fault_direction_weight: float = Field(ge=0.0)
    fault_magnitude_weight: float = Field(ge=0.0)
    reference_consistency_weight: float = Field(ge=0.0)
    magnitude_huber_delta: float = Field(gt=0.0)


class FPTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    workers: int = Field(ge=0, le=16)
    mixed_precision: bool
    screening_seeds: list[int]
    confirmatory_seeds: list[int]


class FPGatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    baseline_minimum_official_score: float = Field(ge=0.0, le=1.0)
    screening_minimum_official_score: float = Field(ge=0.0, le=1.0)
    screening_minimum_gain_over_c0: float
    screening_minimum_gain_over_c1: float
    confirmatory_minimum_ensemble_official_score: float = Field(ge=0.0, le=1.0)
    confirmatory_minimum_gain_over_c1: float
    confirmatory_bootstrap_ci_low_minimum: float
    fault_delta_retention_median_minimum: float = Field(ge=0.0)
    fault_delta_retention_q05_minimum: float = Field(ge=0.0)
    screening_maximum_machine_drop: float = Field(ge=0.0)
    confirmatory_maximum_machine_drop: float = Field(ge=0.0)
    screening_positive_lomo_folds_minimum: int = Field(gt=0)
    confirmatory_positive_lomo_folds_minimum: int = Field(gt=0)
    bootstrap_iterations: int = Field(gt=0)


class FPNAAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    experiment_id: str
    provenance: ProvenanceConfig
    frontend: FPFrontendConfig
    backend: FPBackendConfig
    adapter: FPAdapterConfig
    objective: FPObjectiveConfig
    training: FPTrainingConfig
    gates: FPGatesConfig


def load_fp_naa_config(path: str | Path) -> FPNAAConfig:
    """Load and strictly validate an FP-NAA YAML file."""
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"FP-NAA config must be a mapping: {source}")
    config = FPNAAConfig.model_validate(payload)
    if config.schema_version != 1:
        raise ValueError(f"Unsupported FP-NAA schema_version: {config.schema_version}")
    if config.frontend.channels != ["near", "far"]:
        raise ValueError("FP-NAA v1 requires channels [near, far] in that order")
    if config.frontend.cache_dtype != "float16":
        raise ValueError("FP-NAA v1 cache_dtype must be float16")
    return config

