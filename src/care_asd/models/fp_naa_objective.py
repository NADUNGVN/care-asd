"""Counterfactual representation objectives and safety metrics for FP-NAA."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as functional
from torch import Tensor

from care_asd.fp_naa_config import FPObjectiveConfig

FPNaaObjective = Literal["c1_mse", "c2_fault_preserving", "c3_reference_safe"]


@dataclass(frozen=True)
class FPNAALoss:
    total: Tensor
    normal_mse: Tensor
    fault_direction: Tensor
    fault_magnitude: Tensor
    fault_separation: Tensor
    reference_consistency: Tensor
    retention: Tensor


def fp_naa_loss(
    *,
    objective: FPNaaObjective,
    student_clean: Tensor,
    teacher_clean: Tensor,
    config: FPObjectiveConfig,
    student_fault: Tensor | None = None,
    teacher_fault: Tensor | None = None,
    student_clean_corrupted_reference: Tensor | None = None,
    eps: float = 1.0e-8,
) -> FPNAALoss:
    """Compute the capacity-controlled C1/C2/C3 objective.

    Fault preservation is measured on paired representation differences. The clean and faulty
    student examples must therefore have received the identical waveform noise realization.
    """
    _equal_shape(student_clean, teacher_clean, "student_clean", "teacher_clean")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    normal = functional.mse_loss(student_clean, teacher_clean)
    zero = normal.new_zeros(())
    direction = zero
    magnitude = zero
    separation = zero
    retention = normal.new_ones(student_clean.shape[0])
    if objective in {"c2_fault_preserving", "c3_reference_safe"}:
        if student_fault is None or teacher_fault is None:
            raise ValueError(f"{objective} requires student_fault and teacher_fault")
        _equal_shape(student_clean, student_fault, "student_clean", "student_fault")
        _equal_shape(teacher_clean, teacher_fault, "teacher_clean", "teacher_fault")
        student_delta = _flatten_per_item(student_fault - student_clean).float()
        teacher_delta = _flatten_per_item(teacher_fault - teacher_clean).float()
        cosine = functional.cosine_similarity(student_delta, teacher_delta, dim=1)
        student_norm = torch.linalg.vector_norm(student_delta, dim=1)
        teacher_norm = torch.linalg.vector_norm(teacher_delta, dim=1)
        log_ratio = torch.log((student_norm + eps) / (teacher_norm + eps))
        retention = torch.exp(-log_ratio.abs())
        if config.fault_loss_mode == "exact":
            direction = (1.0 - cosine).mean()
            magnitude = functional.smooth_l1_loss(
                log_ratio,
                torch.zeros_like(log_ratio),
                beta=config.magnitude_huber_delta,
            )
        else:
            direction_violation = functional.relu(config.direction_cosine_floor - cosine)
            lower_violation = functional.relu(math.log(config.gain_lower_bound) - log_ratio)
            upper_violation = functional.relu(log_ratio - math.log(config.gain_upper_bound))
            direction = _upper_tail_mean(direction_violation, config.tail_fraction)
            magnitude = _upper_tail_mean(
                functional.smooth_l1_loss(
                    lower_violation + upper_violation,
                    torch.zeros_like(log_ratio),
                    beta=config.magnitude_huber_delta,
                    reduction="none",
                ),
                config.tail_fraction,
            )
            teacher_distance = 1.0 - functional.cosine_similarity(
                teacher_fault.float(), teacher_clean.float(), dim=-1
            )
            student_distance = 1.0 - functional.cosine_similarity(
                student_fault.float(), student_clean.float(), dim=-1
            )
            separation = _salient_patch_deficit(
                teacher_distance=teacher_distance,
                student_distance=student_distance,
                gain=config.score_gain_lower_bound,
                patch_fraction=config.score_patch_fraction,
            )
    consistency = zero
    if objective == "c3_reference_safe":
        if student_clean_corrupted_reference is None:
            raise ValueError("c3_reference_safe requires a corrupted-reference prediction")
        _equal_shape(
            student_clean,
            student_clean_corrupted_reference,
            "student_clean",
            "student_clean_corrupted_reference",
        )
        consistency = functional.mse_loss(student_clean_corrupted_reference, student_clean.detach())
    total = (
        config.normal_mse_weight * normal
        + config.fault_direction_weight * direction
        + config.fault_magnitude_weight * magnitude
        + config.fault_separation_weight * separation
        + config.reference_consistency_weight * consistency
    )
    return FPNAALoss(total, normal, direction, magnitude, separation, consistency, retention)


def fault_delta_retention(
    student_clean: Tensor,
    student_fault: Tensor,
    teacher_clean: Tensor,
    teacher_fault: Tensor,
    *,
    eps: float = 1.0e-8,
) -> Tensor:
    """Return symmetric per-item delta-magnitude retention in ``(0, 1]``."""
    _equal_shape(student_clean, student_fault, "student_clean", "student_fault")
    _equal_shape(student_clean, teacher_clean, "student_clean", "teacher_clean")
    _equal_shape(student_clean, teacher_fault, "student_clean", "teacher_fault")
    student_norm = torch.linalg.vector_norm(
        _flatten_per_item(student_fault - student_clean).float(), dim=1
    )
    teacher_norm = torch.linalg.vector_norm(
        _flatten_per_item(teacher_fault - teacher_clean).float(), dim=1
    )
    return torch.exp(-torch.log((student_norm + eps) / (teacher_norm + eps)).abs())


def _flatten_per_item(value: Tensor) -> Tensor:
    if value.ndim < 2 or value.shape[0] < 1:
        raise ValueError("representation tensors must have a non-empty batch dimension")
    return value.reshape(value.shape[0], -1)


def _upper_tail_mean(value: Tensor, fraction: float) -> Tensor:
    """Return the empirical CVaR of the largest per-item violations."""
    if value.ndim != 1 or value.numel() < 1:
        raise ValueError("tail values must be a non-empty vector")
    count = max(1, math.ceil(fraction * value.numel()))
    return torch.topk(value, k=count, largest=True, sorted=False).values.mean()


def _salient_patch_deficit(
    *,
    teacher_distance: Tensor,
    student_distance: Tensor,
    gain: float,
    patch_fraction: float,
) -> Tensor:
    """Penalize attenuation on the teacher's most fault-responsive time-frequency patches."""
    if teacher_distance.shape != student_distance.shape or teacher_distance.ndim < 2:
        raise ValueError("teacher and student patch distances must have matching batch shapes")
    teacher = teacher_distance.reshape(teacher_distance.shape[0], -1)
    student = student_distance.reshape(student_distance.shape[0], -1)
    count = max(1, math.ceil(patch_fraction * teacher.shape[1]))
    teacher_salient, indices = torch.topk(teacher, k=count, dim=1, largest=True, sorted=False)
    student_salient = torch.gather(student, 1, indices)
    return functional.relu(gain * teacher_salient - student_salient).mean()


def _equal_shape(first: Tensor, second: Tensor, first_name: str, second_name: str) -> None:
    if first.shape != second.shape:
        raise ValueError(f"{first_name} and {second_name} must have equal shapes")
