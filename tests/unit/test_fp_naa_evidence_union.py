from __future__ import annotations

import numpy as np
import pytest

from care_asd.evaluation.fp_naa_evidence_union import (
    build_certified_union,
    certify_evidence_expert,
    empirical_tail_evidence,
    fit_normal_tail_penalty,
    monotone_evidence_union,
)


def test_empirical_tail_evidence_is_finite_and_monotone() -> None:
    reference = np.asarray([0.1, 0.2, 0.3, 0.4])
    evidence = empirical_tail_evidence(np.asarray([0.0, 0.2, 0.5]), reference)

    assert np.isfinite(evidence).all()
    assert np.all(np.diff(evidence) >= 0.0)
    assert evidence[-1] == pytest.approx(-np.log(1.0 / 5.0))


def test_tail_penalty_uses_registered_upper_normal_quantile() -> None:
    base = np.zeros(100)
    expert = np.arange(100, dtype=np.float64)

    penalty = fit_normal_tail_penalty(base, expert, tail_probability=0.05)

    assert penalty == 95.0


def test_union_is_exactly_base_monotone() -> None:
    base = np.asarray([0.2, 1.0, 2.0])
    expert = np.asarray([2.0, 0.5, 5.0])

    fused = monotone_evidence_union(base, {"tap": expert}, {"tap": 1.5})

    np.testing.assert_allclose(fused, np.asarray([0.5, 1.0, 3.5]))
    assert np.all(fused >= base)


def test_certificate_rejects_clean_tail_inflation() -> None:
    normal_base = np.zeros(100)
    normal_expert = np.linspace(0.0, 1.0, 100)
    clean = np.zeros(20)
    fault = np.ones(20)

    certificate = certify_evidence_expert(
        name="tap",
        normal_base=normal_base,
        normal_expert=normal_expert,
        in_support_clean_base=clean,
        in_support_fault_base=clean,
        in_support_clean_expert=clean,
        in_support_fault_expert=fault + 2.0,
        heldout_clean_base=clean,
        heldout_fault_base=clean,
        heldout_clean_expert=clean,
        heldout_fault_expert=fault + 2.0,
        tail_probability=0.05,
        minimum_in_support_gain_median=0.5,
        minimum_in_support_gain_q05=0.0,
        minimum_heldout_gain_median=0.5,
        minimum_heldout_gain_q05=0.0,
        maximum_clean_activation_fraction=0.01,
    )

    assert certificate.clean_activation_fraction > 0.01
    assert certificate.eligible is False


def test_certified_union_ignores_ineligible_expert() -> None:
    base = np.asarray([0.0, 1.0])
    clean = np.zeros(100)
    fault = np.full(100, 4.0)
    accepted = certify_evidence_expert(
        name="accepted",
        normal_base=clean,
        normal_expert=clean,
        in_support_clean_base=clean,
        in_support_fault_base=clean,
        in_support_clean_expert=clean,
        in_support_fault_expert=fault,
        heldout_clean_base=clean,
        heldout_fault_base=clean,
        heldout_clean_expert=clean,
        heldout_fault_expert=fault,
        tail_probability=0.05,
        minimum_in_support_gain_median=1.0,
        minimum_in_support_gain_q05=1.0,
        minimum_heldout_gain_median=1.0,
        minimum_heldout_gain_q05=1.0,
        maximum_clean_activation_fraction=0.05,
    )
    rejected = certify_evidence_expert(
        name="rejected",
        normal_base=clean,
        normal_expert=clean,
        in_support_clean_base=clean,
        in_support_fault_base=clean,
        in_support_clean_expert=clean,
        in_support_fault_expert=clean,
        heldout_clean_base=clean,
        heldout_fault_base=clean,
        heldout_clean_expert=clean,
        heldout_fault_expert=clean,
        tail_probability=0.05,
        minimum_in_support_gain_median=1.0,
        minimum_in_support_gain_q05=1.0,
        minimum_heldout_gain_median=1.0,
        minimum_heldout_gain_q05=1.0,
        maximum_clean_activation_fraction=0.05,
    )

    result = build_certified_union(
        base_evidence=base,
        expert_evidence={
            "accepted": np.asarray([2.0, 4.0]),
            "rejected": np.asarray([100.0, 100.0]),
        },
        certificates=(accepted, rejected),
    )

    assert accepted.eligible is True
    assert rejected.eligible is False
    np.testing.assert_allclose(result.evidence, np.asarray([2.0, 4.0]))
