"""Counterfactual-certified monotone score fusion for FP-NAA v10.

This module is intentionally model-agnostic.  Every expert supplies cross-fitted
normal scores and paired clean/fault scores.  Empirical tail calibration makes
heterogeneous experts comparable, a high normal quantile pays for each expert's
tail activation, and the final maximum can only add to the immutable base
evidence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpertCertificate:
    """Normal-tail and counterfactual evidence for one supplementary expert."""

    name: str
    penalty: float
    clean_activation_fraction: float
    in_support_gain_median: float
    in_support_gain_q05: float
    heldout_gain_median: float
    heldout_gain_q05: float
    eligible: bool


@dataclass(frozen=True)
class EvidenceUnionResult:
    """Fused evidence and the immutable certificates used to construct it."""

    evidence: np.ndarray
    certificates: tuple[ExpertCertificate, ...]


def empirical_tail_evidence(
    scores: np.ndarray,
    normal_reference_scores: np.ndarray,
    *,
    epsilon: float = 1.0e-6,
) -> np.ndarray:
    """Map high anomaly scores to finite conformal tail evidence ``-log(p)``.

    The add-one tail probability is valid for finite reference sets and avoids
    an infinite value above the largest observed normal score.  ``epsilon`` is
    only a numerical guard; the add-one floor normally dominates it.
    """
    query = _finite_vector(scores, "scores")
    reference = np.sort(_finite_vector(normal_reference_scores, "normal_reference_scores"))
    if not 0.0 < epsilon < 0.1:
        raise ValueError("epsilon must be in (0, 0.1)")
    insertion = np.searchsorted(reference, query, side="left")
    greater_or_equal = len(reference) - insertion
    probability = (1.0 + greater_or_equal) / (len(reference) + 1.0)
    return np.asarray(-np.log(np.maximum(probability, epsilon)), dtype=np.float64)


def fit_normal_tail_penalty(
    base_evidence: np.ndarray,
    expert_evidence: np.ndarray,
    *,
    tail_probability: float,
) -> float:
    """Return a conservative normal-only expert-minus-base tail penalty."""
    base, expert = _paired(base_evidence, expert_evidence, "normal evidence")
    if not 0.0 < tail_probability < 0.5:
        raise ValueError("tail_probability must be in (0, 0.5)")
    delta = expert - base
    penalty = float(np.quantile(delta, 1.0 - tail_probability, method="higher"))
    return max(0.0, penalty)


def monotone_evidence_union(
    base_evidence: np.ndarray,
    expert_evidence: Mapping[str, np.ndarray],
    penalties: Mapping[str, float],
    *,
    active_experts: Sequence[str] | None = None,
) -> np.ndarray:
    """Fuse expert evidence while preserving the base score for every item."""
    base = _finite_vector(base_evidence, "base_evidence")
    active = tuple(active_experts) if active_experts is not None else tuple(expert_evidence)
    unknown = sorted(set(active).difference(expert_evidence))
    if unknown:
        raise ValueError(f"Unknown active experts: {', '.join(unknown)}")
    missing_penalties = sorted(set(active).difference(penalties))
    if missing_penalties:
        raise ValueError(f"Missing expert penalties: {', '.join(missing_penalties)}")
    fused = base.copy()
    for name in active:
        expert = _finite_vector(expert_evidence[name], f"expert_evidence[{name}]")
        if expert.shape != base.shape:
            raise ValueError(f"Expert {name} does not match base evidence shape")
        penalty = float(penalties[name])
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError(f"Expert {name} penalty must be finite and non-negative")
        fused = np.maximum(fused, expert - penalty)
    if np.any(fused < base):
        raise RuntimeError("Monotone evidence union suppressed the immutable base")
    return fused


def certify_evidence_expert(
    *,
    name: str,
    normal_base: np.ndarray,
    normal_expert: np.ndarray,
    in_support_clean_base: np.ndarray,
    in_support_fault_base: np.ndarray,
    in_support_clean_expert: np.ndarray,
    in_support_fault_expert: np.ndarray,
    heldout_clean_base: np.ndarray,
    heldout_fault_base: np.ndarray,
    heldout_clean_expert: np.ndarray,
    heldout_fault_expert: np.ndarray,
    tail_probability: float,
    minimum_in_support_gain_median: float,
    minimum_in_support_gain_q05: float,
    minimum_heldout_gain_median: float,
    minimum_heldout_gain_q05: float,
    maximum_clean_activation_fraction: float,
) -> ExpertCertificate:
    """Certify one expert without anomaly labels or development-test scores."""
    penalty = fit_normal_tail_penalty(normal_base, normal_expert, tail_probability=tail_probability)
    normal_union = monotone_evidence_union(normal_base, {name: normal_expert}, {name: penalty})
    clean_activation = float(np.mean(normal_union > np.asarray(normal_base) + 1.0e-12))
    in_gain = _counterfactual_union_gain(
        name=name,
        penalty=penalty,
        clean_base=in_support_clean_base,
        fault_base=in_support_fault_base,
        clean_expert=in_support_clean_expert,
        fault_expert=in_support_fault_expert,
    )
    heldout_gain = _counterfactual_union_gain(
        name=name,
        penalty=penalty,
        clean_base=heldout_clean_base,
        fault_base=heldout_fault_base,
        clean_expert=heldout_clean_expert,
        fault_expert=heldout_fault_expert,
    )
    in_median = float(np.median(in_gain))
    in_q05 = float(np.quantile(in_gain, 0.05))
    heldout_median = float(np.median(heldout_gain))
    heldout_q05 = float(np.quantile(heldout_gain, 0.05))
    # The held-out family is a gate only.  It must never participate in expert
    # selection, otherwise the claimed family generalization would be circular.
    eligible = bool(
        clean_activation <= maximum_clean_activation_fraction
        and in_median >= minimum_in_support_gain_median
        and in_q05 >= minimum_in_support_gain_q05
    )
    return ExpertCertificate(
        name=name,
        penalty=penalty,
        clean_activation_fraction=clean_activation,
        in_support_gain_median=in_median,
        in_support_gain_q05=in_q05,
        heldout_gain_median=heldout_median,
        heldout_gain_q05=heldout_q05,
        eligible=eligible,
    )


def build_certified_union(
    *,
    base_evidence: np.ndarray,
    expert_evidence: Mapping[str, np.ndarray],
    certificates: Sequence[ExpertCertificate],
) -> EvidenceUnionResult:
    """Apply exactly the experts admitted by immutable certificates."""
    by_name = {item.name: item for item in certificates}
    if len(by_name) != len(certificates):
        raise ValueError("Expert certificates contain duplicate names")
    if set(by_name) != set(expert_evidence):
        raise ValueError("Certificates and supplied expert evidence differ")
    active = tuple(item.name for item in certificates if item.eligible)
    penalties = {item.name: item.penalty for item in certificates}
    evidence = monotone_evidence_union(
        base_evidence,
        expert_evidence,
        penalties,
        active_experts=active,
    )
    return EvidenceUnionResult(evidence, tuple(certificates))


def _counterfactual_union_gain(
    *,
    name: str,
    penalty: float,
    clean_base: np.ndarray,
    fault_base: np.ndarray,
    clean_expert: np.ndarray,
    fault_expert: np.ndarray,
) -> np.ndarray:
    clean_base_array, fault_base_array = _paired(clean_base, fault_base, "base pairs")
    clean_expert_array, fault_expert_array = _paired(clean_expert, fault_expert, "expert pairs")
    if clean_base_array.shape != clean_expert_array.shape:
        raise ValueError("Base and expert counterfactual pairs differ in shape")
    clean_union = monotone_evidence_union(
        clean_base_array, {name: clean_expert_array}, {name: penalty}
    )
    fault_union = monotone_evidence_union(
        fault_base_array, {name: fault_expert_array}, {name: penalty}
    )
    return np.asarray(
        (fault_union - clean_union) - (fault_base_array - clean_base_array),
        dtype=np.float64,
    )


def _finite_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a non-empty finite vector")
    return array


def _paired(left: np.ndarray, right: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    left_array = _finite_vector(left, f"{name} left")
    right_array = _finite_vector(right, f"{name} right")
    if left_array.shape != right_array.shape:
        raise ValueError(f"{name} vectors must have equal shapes")
    return left_array, right_array
