from __future__ import annotations

# ruff: noqa: E402 -- optional Torch must be checked before importing Torch-backed modules.
import pytest

torch = pytest.importorskip("torch")

from care_asd.fp_naa_config import FPObjectiveConfig
from care_asd.models.fp_naa_adapter import BandwiseReferenceAdapter, trainable_parameter_count
from care_asd.models.fp_naa_objective import fault_delta_retention, fp_naa_loss


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
