from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from care_asd.fp_naa_config import load_fp_naa_config


def test_v1_exact_and_v2_tail_safe_configs_are_versioned() -> None:
    v1 = load_fp_naa_config(Path("configs/experiment/fp_naa_v1.yaml"))
    v2 = load_fp_naa_config(Path("configs/experiment/fp_naa_v2.yaml"))

    assert v1.objective.fault_loss_mode == "exact"
    assert v1.objective.primary_safe_gradient_projection is False
    assert v2.experiment_id == "fp_naa_v2_tail_safe"
    assert v2.objective.fault_loss_mode == "tail_constrained"
    assert v2.objective.primary_safe_gradient_projection is True
    assert v2.frontend == v1.frontend
    assert v2.augmentation == v1.augmentation
    assert v2.backend == v1.backend
    assert v2.gates == v1.gates


def test_tail_safe_gain_bounds_must_be_ordered(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/experiment/fp_naa_v2.yaml").read_text())
    payload["objective"]["gain_lower_bound"] = 1.3
    payload["objective"]["gain_upper_bound"] = 1.1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="gain_lower_bound"):
        load_fp_naa_config(path)


def test_primary_safe_projection_requires_tail_constrained_loss(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/experiment/fp_naa_v2.yaml").read_text())
    payload["objective"]["fault_loss_mode"] = "exact"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="primary-safe projection"):
        load_fp_naa_config(path)
