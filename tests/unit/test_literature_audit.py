"""Contract checks for the frozen Audit-A1 literature package."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from care_asd.evaluation.literature_audit import (
    literature_audit_plan,
    load_literature_audit_config,
    run_literature_audit,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "research" / "audit_literature_v1.yaml"
POST_V10_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs" / "research" / "fp_naa_post_v10_literature_delta.yaml"
)


def test_literature_contract_freezes_sources_status_and_claim_links() -> None:
    config = load_literature_audit_config(CONFIG_PATH)

    plan = literature_audit_plan(
        config,
        repository_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
    )

    assert plan["study_id"] == "care_asd_literature_audit_v1"
    assert plan["cutoff_date"] == "2026-08-16"
    assert plan["source_count"] == 13
    assert len(plan["config"]["sha256"]) == 64
    direct_asd = [source for source in config.sources if source.cluster == "direct_asd"]
    assert len(direct_asd) == 7
    assert not any(source.evaluates_known_component_safety for source in direct_asd)
    reports = [
        source
        for source in config.sources
        if source.publication_type == "challenge_technical_report"
    ]
    assert reports
    assert all(source.review_status == "not_peer_reviewed" for source in reports)


def test_post_v10_delta_contract_is_a_backward_compatible_audit_extension() -> None:
    legacy = load_literature_audit_config(CONFIG_PATH)
    delta = load_literature_audit_config(POST_V10_CONFIG_PATH)

    assert legacy.required_clusters == ("direct_asd", "adjacent_signal_processing")
    assert delta.study_id == "fp_naa_post_v10_literature_delta"
    assert delta.cutoff_date.isoformat() == "2026-08-21"
    assert delta.required_clusters == (
        "task_context",
        "direct_asd",
        "adjacent_model_selection",
    )
    assert len(delta.sources) == 9
    assert {source.method_family for source in delta.sources} >= {
        "backend_diversity_fusion",
        "domain_aware_score_calibration",
        "cross_channel_predictive_residual",
        "multi_encoder_multiview_fusion",
        "train_normal_profile_ensemble",
        "unsupervised_model_selection",
    }
    assert any(
        claim.claim_id == "cross_channel_residual_is_prior_art" for claim in delta.prohibited_claims
    )


def test_literature_audit_writes_immutable_matrix_and_boundary(tmp_path: Path) -> None:
    config = load_literature_audit_config(CONFIG_PATH)
    output = tmp_path / "literature"

    result = run_literature_audit(
        output_directory=output,
        config=config,
        repository_root=REPOSITORY_ROOT,
        config_path=CONFIG_PATH,
    )

    matrix = pd.read_csv(result.matrix_path)
    assert len(matrix) == 13
    assert matrix["source_id"].is_unique
    boundary = json.loads(result.boundary_path.read_text(encoding="utf-8"))
    assert boundary["working_title"].startswith("When the Noise Reference")
    prohibited = {claim["claim_id"] for claim in boundary["prohibited_claims"]}
    assert "no_dual_mic_benefit" in prohibited
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert set(run["artifacts"]) == {
        "claim_boundary.json",
        "literature_audit.md",
        "literature_matrix.csv",
    }
    assert result.summary_path.read_text(encoding="utf-8").count("not_peer_reviewed") >= 1
    with pytest.raises(FileExistsError):
        run_literature_audit(
            output_directory=output,
            config=config,
            repository_root=REPOSITORY_ROOT,
            config_path=CONFIG_PATH,
        )
