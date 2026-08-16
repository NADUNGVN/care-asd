from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from care_asd.fp_naa_reference_safety_config import (
    FROZEN_CONDITIONS,
    load_fp_naa_reference_safety_config,
)


def test_reference_safety_protocol_is_frozen_and_ordered() -> None:
    config = load_fp_naa_reference_safety_config(
        Path("configs/experiment/fp_naa_reference_safety_v1.yaml")
    )
    assert tuple(config.conditions) == FROZEN_CONDITIONS
    assert config.leakage_machine_to_noise_db == {
        "low": -20.0,
        "medium": -10.0,
        "high": 0.0,
    }
    assert config.gate.retention_median_minimum == 0.85
    assert config.gate.retention_worst_seed_q05_minimum == 0.65


def test_reference_safety_rejects_reordered_conditions(tmp_path: Path) -> None:
    source = Path("configs/experiment/fp_naa_reference_safety_v1.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["conditions"] = list(reversed(payload["conditions"]))
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen ordered protocol"):
        load_fp_naa_reference_safety_config(path)
