from __future__ import annotations

import json
from pathlib import Path

import yaml

from care_asd.evaluation.reference_safety import create_reference_safety_freeze
from care_asd.reference_safety_config import (
    ReferenceSafetyExperimentConfig,
    ReferenceSafetyPolicy,
)


def test_freeze_records_hashes_after_passed_gate(tmp_path: Path) -> None:
    config = ReferenceSafetyExperimentConfig()
    config_path = tmp_path / "config.yaml"
    policy_path = tmp_path / "policy.yaml"
    gate_path = tmp_path / "gate.json"
    manifest_path = tmp_path / "manifest.parquet"
    output = tmp_path / "freeze.yaml"
    config_path.write_text(yaml.safe_dump(config.model_dump()), encoding="utf-8")
    policy_path.write_text(
        yaml.safe_dump(
            ReferenceSafetyPolicy(
                risk_max=0.2,
                benefit_min_db=1.0,
                calibration_cases=32,
                holdout_cases=32,
                calibration_false_safe_rate=0.0,
                calibration_coverage=0.5,
            ).model_dump()
        ),
        encoding="utf-8",
    )
    gate_path.write_text(json.dumps({"passed": True}), encoding="utf-8")
    manifest_path.write_bytes(b"immutable manifest fixture")

    create_reference_safety_freeze(
        config_path=config_path,
        policy_path=policy_path,
        development_gate_path=gate_path,
        development_manifest_path=manifest_path,
        output_path=output,
        config=config,
    )

    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert payload["ground_truth_access_during_scoring"] is False
    assert len(payload["evaluation_seeds"]) == 10
    assert payload["systems"] == ["near", "unconditional_refsub", "safe_ref"]
