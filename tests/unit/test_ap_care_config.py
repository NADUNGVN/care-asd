"""Contract tests for the isolated AP-CARE v2 configuration."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from care_asd.ap_care_config import (
    APCAREExperimentConfig,
    ap_care_config_hash,
    load_ap_care_config,
)


def test_project_ap_care_config_loads_and_hashes_stably() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_ap_care_config(root / "configs" / "experiment" / "ap_care_v2.yaml")

    assert config.schema_version == 1
    assert config.experiment_id == "ap_care_v2_g1"
    assert len(ap_care_config_hash(config)) == 64
    assert ap_care_config_hash(config) == ap_care_config_hash(config.model_copy(deep=True))


def test_ap_care_config_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        APCAREExperimentConfig.model_validate({"controller": {"unknown": 1}})


def test_ap_care_config_rejects_uncovered_nyquist() -> None:
    with pytest.raises(ValidationError, match="Nyquist"):
        APCAREExperimentConfig.model_validate(
            {
                "simulation": {"sample_rate": 16000},
                "controller": {"band_edges_hz": [0.0, 1000.0, 4000.0]},
            }
        )


def test_ap_care_config_rejects_reversed_bands() -> None:
    with pytest.raises(ValidationError, match="strictly increasing"):
        APCAREExperimentConfig.model_validate(
            {"controller": {"band_edges_hz": [0.0, 1000.0, 500.0, 8000.0]}}
        )
