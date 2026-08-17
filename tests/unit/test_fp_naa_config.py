from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from care_asd.fp_naa_config import load_fp_naa_config


def test_v1_through_v5_configs_are_versioned_without_changing_frozen_gates() -> None:
    v1 = load_fp_naa_config(Path("configs/experiment/fp_naa_v1.yaml"))
    v2 = load_fp_naa_config(Path("configs/experiment/fp_naa_v2.yaml"))
    v3 = load_fp_naa_config(Path("configs/experiment/fp_naa_v3.yaml"))
    v4 = load_fp_naa_config(Path("configs/experiment/fp_naa_v4.yaml"))
    v5 = load_fp_naa_config(Path("configs/experiment/fp_naa_v5.yaml"))

    assert v1.objective.fault_loss_mode == "exact"
    assert v1.objective.primary_safe_gradient_projection is False
    assert v2.experiment_id == "fp_naa_v2_tail_safe"
    assert v2.objective.fault_loss_mode == "tail_constrained"
    assert v2.objective.primary_safe_gradient_projection is True
    assert v2.frontend == v1.frontend
    assert v2.augmentation == v1.augmentation
    assert v2.backend == v1.backend
    assert v2.gates == v1.gates
    assert v3.experiment_id == "fp_naa_v3_reference_projection"
    assert v3.adapter.reference_safety_mode == "rdp_salient_contraction"
    assert v3.adapter.share_c1_weights_for_c2 is True
    assert v3.objective.fault_direction_weight == 0.0
    assert v3.objective.fault_magnitude_weight == 0.0
    assert v3.objective.fault_separation_weight == 0.0
    assert v3.frontend == v1.frontend
    assert v3.augmentation == v1.augmentation
    assert v3.backend == v1.backend
    assert v3.gates == v1.gates
    assert v4.experiment_id == "fp_naa_v4_reference_only_equivariant"
    assert (
        v4.screening_c1_reuse_run_id
        == "server02_fp_naa_screening_20260817T145911Z"
    )
    assert v4.adapter.c2_conditioning_mode == "reference_only_equivariant"
    assert v4.adapter.reference_safety_mode == "none"
    assert v4.adapter.share_c1_weights_for_c2 is False
    assert v4.objective.fault_direction_weight == 0.0
    assert v4.objective.fault_magnitude_weight == 0.0
    assert v4.objective.fault_separation_weight == 0.0
    assert v4.frontend == v1.frontend
    assert v4.augmentation == v1.augmentation
    assert v4.backend == v1.backend
    assert v4.gates == v1.gates
    assert v5.experiment_id == "fp_naa_v5_anchored_counterfactual_tangent_transport"
    assert v5.screening_c2_initialization_run_id == v5.screening_c1_reuse_run_id
    assert v5.adapter.c2_conditioning_mode == "target_conditioned"
    assert v5.objective.fault_loss_mode == "anchored_tangent_transport"
    assert v5.objective.tangent_relative_error_limit == 0.25
    assert v5.objective.function_anchor_relative_limit == 0.10
    assert v5.training.c2_finetune_epochs == 30
    assert v5.training.c2_finetune_disable_dropout is True
    assert v5.frontend == v1.frontend
    assert v5.augmentation == v1.augmentation
    assert v5.backend == v1.backend
    assert v5.gates == v1.gates


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


def test_shared_c1_weights_reject_nonzero_auxiliary_loss(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/experiment/fp_naa_v3.yaml").read_text())
    payload["objective"]["fault_magnitude_weight"] = 0.1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="zero auxiliary-objective weights"):
        load_fp_naa_config(path)


def test_reference_only_c2_rejects_auxiliary_loss(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/experiment/fp_naa_v4.yaml").read_text())
    payload["objective"]["fault_magnitude_weight"] = 0.1
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="zero auxiliary-objective weights"):
        load_fp_naa_config(path)


def test_c1_reuse_run_id_rejects_filesystem_paths(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/experiment/fp_naa_v4.yaml").read_text())
    payload["screening_c1_reuse_run_id"] = "../unregistered-run"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="screening_c1_reuse_run_id"):
        load_fp_naa_config(path)


def test_anchored_transport_requires_matching_registered_c1(tmp_path: Path) -> None:
    payload = yaml.safe_load(Path("configs/experiment/fp_naa_v5.yaml").read_text())
    payload["screening_c2_initialization_run_id"] = "another-run"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="same run"):
        load_fp_naa_config(path)
