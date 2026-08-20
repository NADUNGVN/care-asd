"""Validated configuration for the FP-NAA successor track."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

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
    inference_mixed_precision: bool
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


class FPAugmentationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = Field(ge=0)
    noise_snr_db_min: float
    noise_snr_db_max: float
    fault_delta_level_db_min: float
    fault_delta_level_db_max: float
    train_fault_families: list[str]
    heldout_fault_family: str
    heldout_fraction: float = Field(gt=0.0, lt=1.0)
    peak_limit: float = Field(gt=0.0, le=1.0)


class FPAdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_dim: int = Field(gt=0)
    attention_heads: int = Field(gt=0)
    dropout: float = Field(ge=0.0, lt=1.0)
    reference_dropout_probability: float = Field(ge=0.0, le=1.0)
    reference_corruption_probability: float = Field(ge=0.0, le=1.0)
    c2_conditioning_mode: Literal["target_conditioned", "reference_only_equivariant"] = (
        "target_conditioned"
    )
    reference_safety_mode: Literal["none", "rdp_salient_contraction"] = "none"
    reference_safety_fraction: float = Field(default=0.20, gt=0.0, le=1.0)
    maximum_reference_contraction: float = Field(default=1.0, ge=0.0, le=1.0)
    share_c1_weights_for_c2: bool = False


class FPObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normal_mse_weight: float = Field(ge=0.0)
    fault_direction_weight: float = Field(ge=0.0)
    fault_magnitude_weight: float = Field(ge=0.0)
    fault_separation_weight: float = Field(default=0.0, ge=0.0)
    reference_consistency_weight: float = Field(ge=0.0)
    magnitude_huber_delta: float = Field(gt=0.0)
    fault_loss_mode: Literal["exact", "tail_constrained", "anchored_tangent_transport"] = "exact"
    direction_cosine_floor: float = Field(default=0.0, ge=-1.0, le=1.0)
    gain_lower_bound: float = Field(default=1.0, gt=0.0)
    gain_upper_bound: float = Field(default=1.0, gt=0.0)
    tail_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    score_gain_lower_bound: float = Field(default=1.0, gt=0.0)
    score_patch_fraction: float = Field(default=0.10, gt=0.0, le=1.0)
    auxiliary_start_epoch: int = Field(default=0, ge=0)
    auxiliary_ramp_epochs: int = Field(default=0, ge=0)
    primary_safe_gradient_projection: bool = False
    tangent_transport_mean_weight: float = Field(default=0.0, ge=0.0)
    tangent_transport_tail_weight: float = Field(default=0.0, ge=0.0)
    tangent_relative_error_limit: float = Field(default=0.25, gt=0.0)
    function_anchor_weight: float = Field(default=0.0, ge=0.0)
    function_anchor_relative_limit: float = Field(default=0.10, gt=0.0)


class FPTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    warmup_epochs: int = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0.0)
    workers: int = Field(ge=0, le=16)
    mixed_precision: bool
    screening_seeds: list[int]
    confirmatory_seeds: list[int]
    c2_finetune_epochs: int | None = Field(default=None, gt=0)
    c2_finetune_learning_rate: float | None = Field(default=None, gt=0.0)
    c2_finetune_warmup_epochs: int | None = Field(default=None, ge=0)
    c2_finetune_disable_dropout: bool = False


class FPObservabilityConfig(BaseModel):
    """Normal-only encoder taps used to localize counterfactual information loss."""

    model_config = ConfigDict(extra="forbid")

    encoder_taps: list[int]
    selection_rule: Literal["deepest_eligible"] = "deepest_eligible"


class FPLayerwiseConfig(BaseModel):
    """Frozen architecture and bounded mechanism-preflight contract for FP-NAA v8."""

    model_config = ConfigDict(extra="forbid")

    insertion_layers: list[int]
    preflight_seed: int = Field(ge=0)
    preflight_train_clips: int = Field(gt=0)
    preflight_validation_clips: int = Field(gt=0)
    preflight_heldout_clips: int = Field(gt=0)
    common_epochs: int = Field(gt=0)
    branch_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    gradient_clip_norm: float = Field(gt=0.0)
    tangent_mean_weight: float = Field(ge=0.0)
    tangent_tail_weight: float = Field(gt=0.0)
    tangent_relative_error_limit: float = Field(gt=0.0)
    tangent_tail_fraction: float = Field(gt=0.0, le=1.0)
    function_anchor_weight: float = Field(gt=0.0)
    function_anchor_relative_limit: float = Field(gt=0.0)
    retention_median_minimum: float = Field(ge=0.0)
    retention_q05_minimum: float = Field(ge=0.0)
    retention_median_gain_minimum: float = Field(ge=0.0)
    retention_q05_gain_minimum: float = Field(ge=0.0)
    heldout_retention_median_minimum: float = Field(ge=0.0)
    heldout_retention_q05_minimum: float = Field(ge=0.0)


class FPTapRepairConfig(BaseModel):
    """Frozen pre-encoder repair and mechanism-preflight contract for FP-NAA v9."""

    model_config = ConfigDict(extra="forbid")

    tap: Literal[0]
    preflight_seed: int = Field(ge=0)
    preflight_train_clips: int = Field(gt=0)
    preflight_validation_clips: int = Field(gt=0)
    preflight_heldout_clips: int = Field(gt=0)
    common_epochs: int = Field(gt=0)
    branch_epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    weight_decay: float = Field(ge=0.0)
    gradient_clip_norm: float = Field(gt=0.0)
    tangent_mean_weight: float = Field(ge=0.0)
    tangent_tail_weight: float = Field(gt=0.0)
    tangent_relative_error_limit: float = Field(gt=0.0)
    tangent_tail_fraction: float = Field(gt=0.0, le=1.0)
    function_anchor_weight: float = Field(gt=0.0)
    function_anchor_relative_limit: float = Field(gt=0.0)
    retention_median_minimum: float = Field(ge=0.0)
    retention_q05_minimum: float = Field(ge=0.0)
    retention_median_gain_minimum: float = Field(ge=0.0)
    retention_q05_gain_minimum: float = Field(ge=0.0)
    heldout_retention_median_minimum: float = Field(ge=0.0)
    heldout_retention_q05_minimum: float = Field(ge=0.0)


class FPEvidenceUnionConfig(BaseModel):
    """Frozen score-space mechanism contract for FP-NAA v10.

    The base score is an immutable C1 ensemble.  Supplementary experts may only
    add evidence after cross-fitted normal-tail calibration; they can never
    reduce the base evidence for an individual clip.
    """

    model_config = ConfigDict(extra="forbid")

    tap_source_run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    c1_source_run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
    c1_seeds: list[int]
    supplementary_experts: list[Literal["tap0_rdp8_beam", "final_rdp4_beam", "final_global_ap"]]
    crossfit_folds: int = Field(ge=2)
    calibration_tail_probability: float = Field(gt=0.0, lt=0.5)
    calibration_epsilon: float = Field(gt=0.0, lt=0.1)
    minimum_in_support_evidence_gain_median: float = Field(ge=0.0)
    minimum_in_support_evidence_gain_q05: float
    minimum_heldout_evidence_gain_median: float = Field(ge=0.0)
    minimum_heldout_evidence_gain_q05: float
    maximum_clean_activation_fraction: float = Field(gt=0.0, lt=0.5)
    minimum_machine_pass_fraction: float = Field(gt=0.0, le=1.0)
    minimum_active_experts: int = Field(gt=0)
    screening_minimum_gain_over_raw_c1: float = Field(gt=0.0)
    screening_minimum_gain_over_calibrated_c1: float = Field(gt=0.0)
    screening_maximum_machine_drop: float = Field(ge=0.0)
    confirmatory_minimum_gain_over_calibrated_c1: float = Field(gt=0.0)
    confirmatory_bootstrap_ci_low_minimum: float


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
    heldout_fault_delta_retention_median_minimum: float = Field(ge=0.0)
    heldout_fault_delta_retention_q05_minimum: float = Field(ge=0.0)
    screening_maximum_machine_drop: float = Field(ge=0.0)
    confirmatory_maximum_machine_drop: float = Field(ge=0.0)
    screening_positive_lomo_folds_minimum: int = Field(gt=0)
    confirmatory_positive_lomo_folds_minimum: int = Field(gt=0)
    bootstrap_iterations: int = Field(gt=0)


class FPNAAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    experiment_id: str
    screening_c1_reuse_run_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    screening_c2_initialization_run_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    provenance: ProvenanceConfig
    frontend: FPFrontendConfig
    backend: FPBackendConfig
    augmentation: FPAugmentationConfig
    adapter: FPAdapterConfig
    objective: FPObjectiveConfig
    training: FPTrainingConfig
    observability: FPObservabilityConfig | None = None
    layerwise: FPLayerwiseConfig | None = None
    tap_repair: FPTapRepairConfig | None = None
    evidence_union: FPEvidenceUnionConfig | None = None
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
    if config.augmentation.noise_snr_db_min > config.augmentation.noise_snr_db_max:
        raise ValueError("noise_snr_db_min must not exceed noise_snr_db_max")
    if config.augmentation.fault_delta_level_db_min > config.augmentation.fault_delta_level_db_max:
        raise ValueError("fault_delta_level_db_min must not exceed fault_delta_level_db_max")
    allowed_faults = {
        "periodic_resonance",
        "amplitude_modulation",
        "frequency_modulation",
        "friction_burst",
    }
    train_faults = set(config.augmentation.train_fault_families)
    if not train_faults or not train_faults.issubset(allowed_faults):
        raise ValueError("train_fault_families contains an unsupported family")
    if len(train_faults) != len(config.augmentation.train_fault_families):
        raise ValueError("train_fault_families must not contain duplicates")
    if config.augmentation.heldout_fault_family not in allowed_faults:
        raise ValueError("heldout_fault_family is unsupported")
    if config.augmentation.heldout_fault_family in train_faults:
        raise ValueError("heldout_fault_family must not appear in train_fault_families")
    if config.adapter.hidden_dim % config.adapter.attention_heads != 0:
        raise ValueError("adapter hidden_dim must be divisible by attention_heads")
    if config.training.warmup_epochs >= config.training.epochs:
        raise ValueError("warmup_epochs must be smaller than epochs")
    if config.observability is not None:
        taps = config.observability.encoder_taps
        if not taps or taps != sorted(set(taps)):
            raise ValueError("observability encoder_taps must be sorted and unique")
        if taps[0] < 0 or taps[-1] > 12:
            raise ValueError("observability encoder_taps must be in [0, 12]")
    if config.layerwise is not None:
        layers = config.layerwise.insertion_layers
        if not layers or layers != sorted(set(layers)):
            raise ValueError("layerwise insertion_layers must be sorted and unique")
        if layers[0] < 1 or layers[-1] > 12:
            raise ValueError("layerwise insertion_layers must be in [1, 12]")
        if config.layerwise.preflight_heldout_clips > config.layerwise.preflight_validation_clips:
            raise ValueError("preflight heldout clips cannot exceed validation clips")
    if config.tap_repair is not None:
        if config.tap_repair.preflight_heldout_clips > config.tap_repair.preflight_validation_clips:
            raise ValueError("tap-repair heldout clips cannot exceed validation clips")
        if config.layerwise is not None:
            raise ValueError("tap-repair and layerwise preflights are mutually exclusive")
    if config.evidence_union is not None:
        union = config.evidence_union
        if not union.c1_seeds or union.c1_seeds != sorted(set(union.c1_seeds)):
            raise ValueError("evidence-union C1 seeds must be sorted and unique")
        if not union.supplementary_experts or len(union.supplementary_experts) != len(
            set(union.supplementary_experts)
        ):
            raise ValueError("evidence-union experts must be non-empty and unique")
        if union.minimum_active_experts > len(union.supplementary_experts):
            raise ValueError("minimum_active_experts exceeds registered experts")
        if union.crossfit_folds > 10:
            raise ValueError("evidence-union crossfit_folds must not exceed 10")
        if config.layerwise is not None or config.tap_repair is not None:
            raise ValueError("evidence-union and representation preflights are mutually exclusive")
    objective = config.objective
    if objective.gain_lower_bound > objective.gain_upper_bound:
        raise ValueError("gain_lower_bound must not exceed gain_upper_bound")
    if objective.auxiliary_start_epoch >= config.training.epochs:
        raise ValueError("auxiliary_start_epoch must be smaller than training epochs")
    if objective.auxiliary_start_epoch + objective.auxiliary_ramp_epochs > config.training.epochs:
        raise ValueError("auxiliary objective ramp must finish within training epochs")
    if (
        objective.primary_safe_gradient_projection
        and objective.fault_loss_mode != "tail_constrained"
    ):
        raise ValueError("primary-safe projection requires tail_constrained fault loss")
    adapter = config.adapter
    if adapter.share_c1_weights_for_c2:
        if adapter.reference_safety_mode == "none":
            raise ValueError("shared C1 weights require an active reference-safety projection")
        auxiliary_weights = (
            objective.fault_direction_weight,
            objective.fault_magnitude_weight,
            objective.fault_separation_weight,
            objective.reference_consistency_weight,
        )
        if any(weight != 0.0 for weight in auxiliary_weights):
            raise ValueError("shared C1 weights require zero auxiliary-objective weights")
        if objective.primary_safe_gradient_projection:
            raise ValueError("shared C1 weights cannot enable auxiliary-gradient projection")
        if adapter.c2_conditioning_mode != "target_conditioned":
            raise ValueError("shared C1 weights require target-conditioned C2 architecture")
    if adapter.c2_conditioning_mode == "reference_only_equivariant":
        if adapter.reference_safety_mode != "none":
            raise ValueError("reference-only C2 cannot combine with an output safety projection")
        if adapter.share_c1_weights_for_c2:
            raise ValueError("reference-only C2 must train its own capacity-matched weights")
        auxiliary_weights = (
            objective.fault_direction_weight,
            objective.fault_magnitude_weight,
            objective.fault_separation_weight,
            objective.reference_consistency_weight,
        )
        if any(weight != 0.0 for weight in auxiliary_weights):
            raise ValueError("reference-only C2 requires zero auxiliary-objective weights")
        if objective.primary_safe_gradient_projection:
            raise ValueError("reference-only C2 cannot enable auxiliary-gradient projection")
    if objective.fault_loss_mode == "anchored_tangent_transport":
        if config.screening_c2_initialization_run_id is None:
            raise ValueError("anchored tangent transport requires a registered C1 initialization")
        if config.screening_c2_initialization_run_id != config.screening_c1_reuse_run_id:
            raise ValueError("C1 comparator reuse and C2 initialization must use the same run")
        if adapter.c2_conditioning_mode != "target_conditioned":
            raise ValueError("anchored tangent transport requires target-conditioned C2")
        if adapter.reference_safety_mode != "none" or adapter.share_c1_weights_for_c2:
            raise ValueError("anchored tangent transport cannot use output safety projections")
        auxiliary_weights = (
            objective.fault_direction_weight,
            objective.fault_magnitude_weight,
            objective.fault_separation_weight,
            objective.reference_consistency_weight,
        )
        if any(weight != 0.0 for weight in auxiliary_weights):
            raise ValueError("anchored tangent transport owns the complete auxiliary objective")
        if objective.primary_safe_gradient_projection:
            raise ValueError("anchored tangent transport cannot use auxiliary-gradient projection")
        if objective.tangent_transport_tail_weight <= 0.0:
            raise ValueError("anchored tangent transport requires a positive tail weight")
        if objective.function_anchor_weight <= 0.0:
            raise ValueError("anchored tangent transport requires a positive function anchor")
        c2_epochs = config.training.c2_finetune_epochs
        c2_learning_rate = config.training.c2_finetune_learning_rate
        c2_warmup = config.training.c2_finetune_warmup_epochs
        c2_schedule = (c2_epochs, c2_learning_rate, c2_warmup)
        if any(value is None for value in c2_schedule):
            raise ValueError("anchored tangent transport requires a complete C2 fine-tune schedule")
        assert c2_epochs is not None and c2_learning_rate is not None and c2_warmup is not None
        if c2_warmup >= c2_epochs:
            raise ValueError("C2 fine-tune warmup must be shorter than its schedule")
    elif config.screening_c2_initialization_run_id is not None:
        raise ValueError(
            "registered C2 initialization is only valid for anchored tangent transport"
        )
    return config
