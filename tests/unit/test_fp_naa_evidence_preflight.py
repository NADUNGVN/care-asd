from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from care_asd.evaluation.fp_naa_evidence_preflight import (
    _atomic_json,
    _calibrate_score_set,
    _ScoreSet,
    _selection_frame,
    _sha256,
    _summarize_certificates,
    _validate_completed_outputs,
)
from care_asd.fp_naa_config import FPEvidenceUnionConfig


def _union_config() -> FPEvidenceUnionConfig:
    return FPEvidenceUnionConfig(
        tap_source_run_id="frozen-v9",
        c1_source_run_id="frozen-c1",
        c1_seeds=[42, 2026, 13711],
        supplementary_experts=["tap0_rdp8_beam", "final_rdp4_beam"],
        crossfit_folds=5,
        calibration_tail_probability=0.01,
        calibration_epsilon=1.0e-6,
        minimum_in_support_evidence_gain_median=0.05,
        minimum_in_support_evidence_gain_q05=0.0,
        minimum_heldout_evidence_gain_median=0.025,
        minimum_heldout_evidence_gain_q05=0.0,
        maximum_clean_activation_fraction=0.035,
        minimum_machine_pass_fraction=0.70,
        minimum_active_experts=1,
        screening_minimum_gain_over_raw_c1=0.0075,
        screening_minimum_gain_over_calibrated_c1=0.0025,
        screening_maximum_machine_drop=0.01,
        confirmatory_minimum_gain_over_calibrated_c1=0.0025,
        confirmatory_bootstrap_ci_low_minimum=0.0,
    )


def test_selection_frame_rejects_development_test_rows() -> None:
    contract = {
        "selection": [
            {"file_id": "train", "role": "train", "heldout": False},
            {"file_id": "test", "role": "validation", "heldout": False},
        ]
    }
    index = pd.DataFrame(
        [
            {
                "file_id": "train",
                "machine_type": "fan",
                "section": "00",
                "dataset_split": "dev_train",
                "condition": "normal",
            },
            {
                "file_id": "test",
                "machine_type": "fan",
                "section": "00",
                "dataset_split": "dev_test",
                "condition": "anomaly",
            },
        ]
    )
    try:
        _selection_frame(contract, index)
    except ValueError as exc:
        assert "training normals only" in str(exc)
    else:  # pragma: no cover - explicit guard against accidental label leakage
        raise AssertionError("development-test row was accepted")


def test_c1_calibration_averages_seed_evidence_not_raw_scores() -> None:
    first = _ScoreSet(
        normal=np.array([0.0, 1.0, 2.0]),
        in_clean=np.array([1.5]),
        in_fault=np.array([3.0]),
        heldout_clean=np.array([0.5]),
        heldout_fault=np.array([2.5]),
    )
    second = _ScoreSet(
        normal=np.array([10.0, 20.0, 30.0]),
        in_clean=np.array([15.0]),
        in_fault=np.array([40.0]),
        heldout_clean=np.array([5.0]),
        heldout_fault=np.array([35.0]),
    )
    result = _calibrate_score_set(
        {"a": first, "b": second},
        {"a": first.normal, "b": second.normal},
        ensemble=True,
    )
    assert result.in_clean.shape == (1,)
    assert np.isfinite(result.in_fault).all()
    assert result.in_fault[0] > result.in_clean[0]


def test_global_authorization_requires_registered_machine_fraction() -> None:
    config = _union_config()
    certificates = pd.DataFrame(
        [
            {"name": "tap0_rdp8_beam", "eligible": True},
            {"name": "tap0_rdp8_beam", "eligible": True},
            {"name": "tap0_rdp8_beam", "eligible": False},
            {"name": "final_rdp4_beam", "eligible": True},
            {"name": "final_rdp4_beam", "eligible": False},
            {"name": "final_rdp4_beam", "eligible": False},
        ]
    )
    for column in (
        "clean_activation_fraction",
        "in_support_gain_median",
        "in_support_gain_q05",
        "heldout_gain_median",
        "heldout_gain_q05",
    ):
        certificates[column] = 0.0
    machine_results = {
        "a": {
            "active_experts": ("tap0_rdp8_beam",),
            "normal_activation_fraction": 0.0,
            "in_support_gain": np.array([0.0]),
            "heldout_gain": np.array([0.0]),
        }
    }
    summary = _summarize_certificates(certificates, machine_results, config)
    expert_rows = summary.set_index("expert")
    assert bool(expert_rows.loc["tap0_rdp8_beam", "globally_authorized"]) is False
    assert bool(expert_rows.loc["final_rdp4_beam", "globally_authorized"]) is False


def test_completed_outputs_are_bound_to_contract_and_policy(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    summary = tmp_path / "summary.csv"
    certificates = tmp_path / "expert_certificates.csv"
    calibration = tmp_path / "normal_calibration.csv"
    policy = tmp_path / "policy.json"
    gate = tmp_path / "gate.json"
    contract.write_text("{}\n", encoding="utf-8")
    summary.write_text("expert,eligible\ncandidate_union,True\n", encoding="utf-8")
    certificates.write_text("name,eligible\ntap0,True\n", encoding="utf-8")
    calibration.write_text("branch,normal_score\ntap0,0.1\n", encoding="utf-8")
    _atomic_json(
        policy,
        {
            "contract_sha256": _sha256(contract),
            "artifacts": {
                "summary_sha256": _sha256(summary),
                "expert_certificates_sha256": _sha256(certificates),
                "normal_calibration_sha256": _sha256(calibration),
            },
        },
    )
    _atomic_json(gate, {"passed": True, "policy_sha256": _sha256(policy)})

    _validate_completed_outputs(
        contract_path=contract,
        gate_path=gate,
        summary_path=summary,
        certificates_path=certificates,
        policy_path=policy,
        calibration_path=calibration,
    )
    calibration.write_text("branch,normal_score\ntap0,9.9\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"normal_calibration\.csv"):
        _validate_completed_outputs(
            contract_path=contract,
            gate_path=gate,
            summary_path=summary,
            certificates_path=certificates,
            policy_path=policy,
            calibration_path=calibration,
        )
