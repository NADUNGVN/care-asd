"""Strict configuration for the frozen FP-NAA reference-safety stress protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

ReferenceSafetyCondition = Literal[
    "matched",
    "unmatched",
    "dropout",
    "channel_swap",
    "leakage_low",
    "leakage_medium",
    "leakage_high",
]

FROZEN_CONDITIONS: tuple[ReferenceSafetyCondition, ...] = (
    "matched",
    "unmatched",
    "dropout",
    "channel_swap",
    "leakage_low",
    "leakage_medium",
    "leakage_high",
)


class ReferenceSafetyGateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retention_median_minimum: float = Field(ge=0.0, le=1.0)
    retention_worst_seed_q05_minimum: float = Field(ge=0.0, le=1.0)


class FPNAAReferenceSafetyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int
    experiment_id: str
    population: Literal["heldout_friction_burst_only"]
    unmatched_reference_seed: int = Field(ge=0)
    leakage_machine_to_noise_db: dict[Literal["low", "medium", "high"], float]
    conditions: list[ReferenceSafetyCondition]
    gate: ReferenceSafetyGateConfig


def load_fp_naa_reference_safety_config(
    path: str | Path,
) -> FPNAAReferenceSafetyConfig:
    """Load and validate the independently frozen safety protocol."""
    source = Path(path)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"FP-NAA reference-safety config must be a mapping: {source}")
    config = FPNAAReferenceSafetyConfig.model_validate(payload)
    if config.schema_version != 1:
        raise ValueError(f"Unsupported reference-safety schema_version: {config.schema_version}")
    if tuple(config.conditions) != FROZEN_CONDITIONS:
        raise ValueError("Reference-safety conditions must equal the frozen ordered protocol")
    leakage = config.leakage_machine_to_noise_db
    if set(leakage) != {"low", "medium", "high"}:
        raise ValueError("Reference leakage must define exactly low, medium, and high")
    if not leakage["low"] < leakage["medium"] < leakage["high"]:
        raise ValueError("Reference leakage levels must increase from low to medium to high")
    return config
