"""Configuration loading and hashing for CARE-ASD experiments.

Uses OmegaConf for hierarchical YAML configs and Pydantic for validated
runtime views of the configuration schema.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExperimentConfig(BaseModel):
    """Top-level experiment identity and seed."""

    model_config = ConfigDict(extra="forbid")

    id: str = "care_dev_default"
    seed: int = 42
    dry_run: bool = False
    notes: str = ""


class ChannelMapConfig(BaseModel):
    """Stereo channel role mapping (DCASE 2026: 0=near, 1=far)."""

    model_config = ConfigDict(extra="forbid")

    near: int = 0
    far: int = 1


class DataConfig(BaseModel):
    """Data paths and channel conventions."""

    model_config = ConfigDict(extra="forbid")

    root: str = "data"
    manifest: str = "data/manifests/dcase2026_dev.parquet"
    sample_rate: int | None = None
    channel_map: ChannelMapConfig = Field(default_factory=ChannelMapConfig)


class SignalConfig(BaseModel):
    """STFT / spectral analysis parameters."""

    model_config = ConfigDict(extra="forbid")

    n_fft: int = Field(default=1024, gt=0)
    win_length: int = Field(default=1024, gt=0)
    hop_length: int = Field(default=512, gt=0)
    window: Literal["hann"] = "hann"
    center: bool = False
    eps: float = Field(default=1.0e-8, gt=0.0)


class TransferConfig(BaseModel):
    """Acoustic transfer-function estimation."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["causal_ema", "static_per_clip"] = "causal_ema"
    alpha: float = Field(default=0.95, ge=0.0, lt=1.0)
    reg_floor: float = Field(default=1.0e-5, gt=0.0)
    frequency_smoothing_bins: int = Field(default=1, ge=1)


class GateConfig(BaseModel):
    """Reliability gate over residual cancellation."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["semi_parametric"] = "semi_parametric"
    min_value: float = Field(default=0.0, ge=0.0, le=1.0)
    max_value: float = Field(default=0.9, ge=0.0, le=1.0)
    coherence_weight: float = 4.0
    snr_weight: float = -0.5
    bias: float = 0.0
    bypass: bool = False

    @model_validator(mode="after")
    def validate_bounds(self) -> GateConfig:
        if self.min_value > self.max_value:
            raise ValueError("min_value must not exceed max_value")
        return self


class ResidualConfig(BaseModel):
    """Residual generation constraints."""

    model_config = ConfigDict(extra="forbid")

    max_removed_energy_ratio: float = Field(default=0.8, gt=0.0, le=1.0)


class FrontendConfig(BaseModel):
    """Acoustic-path front-end configuration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "care"
    transfer: TransferConfig = Field(default_factory=TransferConfig)
    gate: GateConfig = Field(default_factory=GateConfig)
    residual: ResidualConfig = Field(default_factory=ResidualConfig)


class FeaturesConfig(BaseModel):
    """Multi-view feature flags."""

    model_config = ConfigDict(extra="forbid")

    near_logmel: bool = True
    far_logmel: bool = True
    residual_logmel: bool = True
    coherence: bool = True
    log_ratio: bool = True
    phase_sin_cos: bool = True
    n_mels: int = 128
    fmin: float = 0.0
    fmax: float | None = None


class ModelConfig(BaseModel):
    """Encoder / model hyperparameters."""

    model_config = ConfigDict(extra="forbid")

    name: str = "lightweight_encoder"
    embedding_dim: int = 128
    dropout: float = 0.1


class BaselineConfig(BaseModel):
    """Pinned external DCASE baseline execution contract."""

    model_config = ConfigDict(extra="forbid")

    reference_dir: str = "external/dcase2026_task2_baseline_ae"
    evaluator_dir: str = "external/dcase2026_task2_evaluator"
    official_python: str | None = None
    seed: int = 13711
    input_channel: int = 0


class TargetAdaptationConfig(BaseModel):
    """Source-target prototype shrinkage."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    lambda_: float = Field(default=0.2, alias="lambda")


class ScoringConfig(BaseModel):
    """Anomaly scoring configuration."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str = "shrinkage_mahalanobis"
    covariance_shrinkage: str | float = "auto"
    target_adaptation: TargetAdaptationConfig = Field(default_factory=TargetAdaptationConfig)


class CalibrationConfig(BaseModel):
    """Normal-only reliability calibration."""

    model_config = ConfigDict(extra="forbid")

    name: str = "hierarchical_conformal"
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    abstain_margin: float = Field(default=0.02, ge=0.0)


class DeploymentConfig(BaseModel):
    """Reproducible profile for a target deployment device."""

    model_config = ConfigDict(extra="forbid")

    device_id: str = "E2"
    platform: Literal["jetson_xavier_nx", "jetson_agx_xavier", "pi5_hailo"] = "jetson_xavier_nx"
    backend: Literal["tensorrt", "onnxruntime", "hailo"] = "tensorrt"
    runner: Literal["cpp_tensorrt", "python_onnxruntime"] = "cpp_tensorrt"
    precision: Literal["fp32", "fp16", "int8"] = "fp16"
    power_mode_w: int = Field(default=15, gt=0)
    warmup_windows: int = Field(default=100, ge=0)
    timed_windows: int = Field(default=1000, gt=0)
    repetitions: int = Field(default=3, gt=0)
    require_external_power_meter: bool = True

    @model_validator(mode="after")
    def validate_platform_profile(self) -> DeploymentConfig:
        allowed_power_modes = {
            "jetson_xavier_nx": {10, 15},
            "jetson_agx_xavier": {10, 15, 30},
        }
        if (
            self.platform in allowed_power_modes
            and self.power_mode_w not in allowed_power_modes[self.platform]
        ):
            raise ValueError(
                f"power_mode_w={self.power_mode_w} is unsupported for platform={self.platform}"
            )
        if self.platform.startswith("jetson_") and (
            self.backend != "tensorrt" or self.runner != "cpp_tensorrt"
        ):
            raise ValueError("Jetson Xavier profiles require the cpp_tensorrt runner")
        return self


class StreamingConfig(BaseModel):
    """Causal streaming inference."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    window_seconds: float = 1.0
    hop_seconds: float = 0.25
    aggregation: str = "topk_mean"
    topk: int = 3


class OutputConfig(BaseModel):
    """Experiment output paths and save flags."""

    model_config = ConfigDict(extra="forbid")

    root: str = "outputs"
    save_embeddings: bool = False
    save_scores: bool = True
    save_diagnostics: bool = True


class CareASDConfig(BaseModel):
    """Validated full CARE-ASD configuration."""

    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    frontend: FrontendConfig = Field(default_factory=FrontendConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    baseline: BaselineConfig = Field(default_factory=BaselineConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def default_config() -> DictConfig:
    """Return the default OmegaConf configuration."""
    validated = CareASDConfig()
    return OmegaConf.create(validated.model_dump(by_alias=True))


def load_config(
    config_path: str | Path | None = None,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load configuration from YAML with optional CLI-style overrides.

    Parameters
    ----------
    config_path:
        Path to a YAML config file. If None, uses built-in defaults.
    overrides:
        Dot-list overrides such as ``experiment.seed=123``.

    Returns
    -------
    DictConfig
        Merged configuration (defaults < file < overrides).
    """
    cfg = default_config()

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        file_cfg = OmegaConf.load(path)
        if not isinstance(file_cfg, DictConfig):
            raise TypeError(f"Config root must be a mapping, got {type(file_cfg)}")
        cfg = OmegaConf.merge(cfg, file_cfg)  # type: ignore[assignment]

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)  # type: ignore[assignment]

    # Validate merged config against the schema
    validate_config(cfg)
    return cfg


def validate_config(cfg: DictConfig | dict[str, Any]) -> CareASDConfig:
    """Validate a config dict/DictConfig against the Pydantic schema."""
    plain: Any = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    if not isinstance(plain, dict):
        raise TypeError("Config must resolve to a dictionary")
    return CareASDConfig.model_validate(plain)


def config_to_dict(cfg: DictConfig | CareASDConfig) -> dict[str, Any]:
    """Convert config to a plain JSON-serializable dictionary."""
    if isinstance(cfg, CareASDConfig):
        return cfg.model_dump(by_alias=True)
    container = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Config must resolve to a dictionary")
    return container  # type: ignore[return-value]


def config_hash(cfg: DictConfig | CareASDConfig | dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 hash of the configuration.

    The hash is stable across process restarts for the same resolved values.
    """
    if isinstance(cfg, CareASDConfig):
        payload = cfg.model_dump(by_alias=True)
    elif isinstance(cfg, DictConfig):
        payload = config_to_dict(cfg)
    else:
        payload = cfg

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_config(cfg: DictConfig | CareASDConfig, path: str | Path) -> Path:
    """Write configuration to YAML. Creates parent directories if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(cfg, CareASDConfig):
        omega = OmegaConf.create(cfg.model_dump(by_alias=True))
    else:
        omega = cfg

    OmegaConf.save(omega, out)
    return out
