from __future__ import annotations

# ruff: noqa: E402 -- optional Torch must be checked before importing Torch-backed modules.
import pytest

torch = pytest.importorskip("torch")

from care_asd.fp_naa_config import FPObjectiveConfig
from care_asd.models.fp_naa_adapter import (
    BandwiseReferenceAdapter,
    rdp_salient_contraction_projection,
    trainable_parameter_count,
)
from care_asd.models.fp_naa_objective import _upper_tail_mean, fault_delta_retention, fp_naa_loss


def _objective_config() -> FPObjectiveConfig:
    return FPObjectiveConfig(
        normal_mse_weight=1.0,
        fault_direction_weight=1.0,
        fault_magnitude_weight=0.5,
        reference_consistency_weight=0.25,
        magnitude_huber_delta=0.1,
    )


def test_adapter_is_initially_identity_and_preserves_shape() -> None:
    model = BandwiseReferenceAdapter(
        embedding_dim=32,
        hidden_dim=16,
        attention_heads=4,
        dropout=0.0,
    )
    target = torch.randn(2, 5, 8, 32)
    reference = torch.randn_like(target)
    output = model(target, reference)
    torch.testing.assert_close(output, target)
    assert output.shape == target.shape
    assert trainable_parameter_count(model) > 0


def test_c2_loss_rewards_direction_and_magnitude_preservation() -> None:
    teacher_clean = torch.randn(3, 2, 2, 4)
    teacher_delta = 0.1 * torch.randn_like(teacher_clean)
    teacher_fault = teacher_clean + teacher_delta
    perfect = fp_naa_loss(
        objective="c2_fault_preserving",
        student_clean=teacher_clean,
        teacher_clean=teacher_clean,
        student_fault=teacher_fault,
        teacher_fault=teacher_fault,
        config=_objective_config(),
    )
    erased = fp_naa_loss(
        objective="c2_fault_preserving",
        student_clean=teacher_clean,
        teacher_clean=teacher_clean,
        student_fault=teacher_clean,
        teacher_fault=teacher_fault,
        config=_objective_config(),
    )
    assert perfect.total.item() == pytest.approx(0.0, abs=1.0e-6)
    assert torch.all(perfect.retention > 0.999)
    assert erased.total > perfect.total
    assert torch.all(erased.retention < 0.01)


def test_c1_excludes_fault_terms_and_retention_is_bounded() -> None:
    clean = torch.zeros(2, 2, 2, 4)
    student = torch.ones_like(clean)
    result = fp_naa_loss(
        objective="c1_mse",
        student_clean=student,
        teacher_clean=clean,
        config=_objective_config(),
    )
    assert result.total.item() == pytest.approx(1.0)
    assert result.fault_direction.item() == 0.0
    retention = fault_delta_retention(clean, clean + 2.0, clean, clean + 1.0)
    assert torch.all((retention > 0.0) & (retention <= 1.0))
    assert retention.tolist() == pytest.approx([0.5, 0.5], abs=1.0e-6)


def test_tail_constrained_loss_prefers_safe_fault_gain() -> None:
    config = _objective_config().model_copy(
        update={
            "fault_loss_mode": "tail_constrained",
            "direction_cosine_floor": 0.5,
            "gain_lower_bound": 1.05,
            "gain_upper_bound": 1.20,
            "tail_fraction": 0.5,
            "fault_separation_weight": 2.0,
            "score_gain_lower_bound": 1.05,
            "score_patch_fraction": 0.5,
        }
    )
    torch.manual_seed(4)
    teacher_clean = torch.randn(4, 2, 2, 8)
    teacher_delta = 0.01 * torch.randn_like(teacher_clean)
    teacher_fault = teacher_clean + teacher_delta
    erased = fp_naa_loss(
        objective="c2_fault_preserving",
        student_clean=teacher_clean,
        teacher_clean=teacher_clean,
        student_fault=teacher_clean,
        teacher_fault=teacher_fault,
        config=config,
    )
    safe_gain = fp_naa_loss(
        objective="c2_fault_preserving",
        student_clean=teacher_clean,
        teacher_clean=teacher_clean,
        student_fault=teacher_clean + 1.10 * teacher_delta,
        teacher_fault=teacher_fault,
        config=config,
    )
    assert safe_gain.total < erased.total
    assert safe_gain.fault_direction.item() == pytest.approx(0.0, abs=1.0e-6)
    assert torch.all(safe_gain.retention > 0.90)


def test_upper_tail_mean_targets_worst_violations() -> None:
    values = torch.tensor([0.0, 1.0, 2.0, 3.0])
    assert _upper_tail_mean(values, 0.5).item() == pytest.approx(2.5)


def test_rdp_salient_projection_bounds_only_high_disagreement_rows() -> None:
    reference = torch.zeros(1, 4, 2, 3)
    target = torch.stack(
        [torch.full((1, 2, 3), float(index + 1)) for index in range(4)], dim=1
    )
    correction = -target
    projected = rdp_salient_contraction_projection(
        correction=correction,
        target=target,
        reference=reference,
        protected_fraction=0.5,
        maximum_contraction=0.1,
    )
    discrepancy = target - reference
    dot = (projected * discrepancy).sum(dim=(2, 3))
    norm_sq = discrepancy.square().sum(dim=(2, 3))
    torch.testing.assert_close(projected[:, :2], correction[:, :2])
    torch.testing.assert_close(dot[:, -2:], -0.1 * norm_sq[:, -2:])


def test_reference_safety_projection_adds_no_trainable_parameters() -> None:
    common = {
        "embedding_dim": 32,
        "hidden_dim": 16,
        "attention_heads": 4,
        "dropout": 0.0,
    }
    baseline = BandwiseReferenceAdapter(**common)
    safe = BandwiseReferenceAdapter(
        **common,
        reference_safety_mode="rdp_salient_contraction",
        reference_safety_fraction=0.2,
        maximum_reference_contraction=0.1,
    )
    assert trainable_parameter_count(safe) == trainable_parameter_count(baseline)


def test_reference_only_adapter_is_exactly_target_perturbation_equivariant() -> None:
    torch.manual_seed(11)
    model = BandwiseReferenceAdapter(
        embedding_dim=32,
        hidden_dim=16,
        attention_heads=4,
        dropout=0.0,
        conditioning_mode="reference_only_equivariant",
    ).eval()
    with torch.no_grad():
        output = model.fusion[-1]
        assert isinstance(output, torch.nn.Linear)
        torch.nn.init.normal_(output.weight, std=0.02)
        torch.nn.init.normal_(output.bias, std=0.02)
        target = torch.randn(2, 5, 8, 32)
        reference = torch.randn_like(target)
        perturbation = 0.05 * torch.randn_like(target)
        baseline = model(target, reference)
        perturbed = model(target + perturbation, reference)
        shifted_reference = model(target, reference + 0.2)
    torch.testing.assert_close(perturbed - baseline, perturbation, rtol=1.0e-5, atol=1.0e-6)
    assert not torch.allclose(shifted_reference, baseline)


def test_reference_only_adapter_preserves_capacity() -> None:
    common = {
        "embedding_dim": 32,
        "hidden_dim": 16,
        "attention_heads": 4,
        "dropout": 0.0,
    }
    target_conditioned = BandwiseReferenceAdapter(**common)
    reference_only = BandwiseReferenceAdapter(
        **common,
        conditioning_mode="reference_only_equivariant",
    )
    assert trainable_parameter_count(reference_only) == trainable_parameter_count(
        target_conditioned
    )
