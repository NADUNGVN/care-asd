"""Contract checks for the frozen identifiability/audit synthesis."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from care_asd.evaluation.audit_synthesis import (
    audit_synthesis_plan,
    load_audit_synthesis_config,
    portable_sha256,
    run_audit_synthesis,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment" / "audit_paper_v1.yaml"


def test_audit_plan_validates_and_hashes_frozen_sources() -> None:
    config = load_audit_synthesis_config(CONFIG_PATH)

    plan = audit_synthesis_plan(config, repository_root=REPOSITORY_ROOT)

    assert plan["study_id"] == "care_asd_identifiability_audit_v1"
    assert plan["decisions"]["method_route"] == "stopped"
    assert plan["decisions"]["evaluation_access"] == "prohibited"
    assert len(plan["source_hashes"]) == 9
    assert all(len(item["sha256"]) == 64 for item in plan["source_hashes"].values())


def test_audit_synthesis_writes_immutable_tables_figures_and_decision(tmp_path: Path) -> None:
    config = load_audit_synthesis_config(CONFIG_PATH)
    output = tmp_path / "audit"

    result = run_audit_synthesis(
        output_directory=output,
        config=config,
        repository_root=REPOSITORY_ROOT,
    )

    decision = json.loads(result.decision_path.read_text(encoding="utf-8"))
    assert decision["stop_rule_satisfied"] is True
    assert decision["decision"]["publication_route"] == "identifiability_audit"
    assert decision["headline_evidence"]["ap_care_holdout_cases"] == 256
    assert decision["headline_evidence"]["ap_care_holdout_cases_ge_1db"] == 0
    evidence = pd.read_csv(result.evidence_path)
    assert list(evidence["system"]) == ["B00", "B01", "B02"]
    assert evidence.loc[evidence["system"] == "B01", "pauc_ci95_high"].iloc[0] < 0.0
    identifiability = pd.read_csv(result.identifiability_path)
    assert int(identifiability["passed"].sum()) == 1
    diagnostics = pd.read_csv(result.diagnostics_path)
    assert len(diagnostics) == 8
    assert diagnostics["attenuation_ge_1db_fraction"].max() == 0.0
    assert result.summary_path.is_file()
    assert all(path.read_text(encoding="utf-8").startswith("<svg") for path in result.figure_paths)
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["artifacts"][result.decision_path.name] == portable_sha256(result.decision_path)
    with pytest.raises(FileExistsError):
        run_audit_synthesis(
            output_directory=output,
            config=config,
            repository_root=REPOSITORY_ROOT,
        )


def test_portable_hash_normalizes_text_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b'{\n  "ok": true\n}\n')
    crlf.write_bytes(b'{\r\n  "ok": true\r\n}\r\n')

    assert portable_sha256(lf) == portable_sha256(crlf)
