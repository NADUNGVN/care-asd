"""Contract and result checks for the frozen Audit-A2 appendix."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from care_asd.evaluation.robustness_appendix import (
    load_robustness_appendix_config,
    robustness_appendix_plan,
    run_robustness_appendix,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "experiment" / "audit_robustness_v1.yaml"


def test_robustness_plan_requires_identical_frozen_pairing() -> None:
    config = load_robustness_appendix_config(CONFIG_PATH)

    plan = robustness_appendix_plan(config, repository_root=REPOSITORY_ROOT)

    assert plan["study_id"] == "care_asd_robustness_appendix_v1"
    assert plan["coverage"]["systems"] == {"B00": 1400, "B01": 1400, "B02": 1400}
    assert plan["coverage"]["paired_clips"] == 1400
    assert plan["coverage"]["machine_sections"] == 7
    assert plan["coverage"]["clips_per_machine_domain_condition"] == [50]
    assert plan["decisions"] == {
        "audio_access": "prohibited",
        "evaluation_access": "prohibited",
        "model_training": "prohibited",
        "model_tuning": "prohibited",
    }
    assert len(plan["source_hashes"]) == 5


def test_robustness_appendix_writes_expected_tables_and_stable_result(tmp_path: Path) -> None:
    config = load_robustness_appendix_config(CONFIG_PATH)
    fast_analysis = config.analysis.model_copy(update={"bootstrap_iterations": 100})
    fast_config = config.model_copy(update={"analysis": fast_analysis})
    output = tmp_path / "robustness"

    result = run_robustness_appendix(
        output_directory=output,
        config=fast_config,
        repository_root=REPOSITORY_ROOT,
    )

    metrics = pd.read_csv(result.group_metrics_path)
    assert len(metrics) == 63
    bootstrap = pd.read_csv(result.bootstrap_path)
    assert len(bootstrap) == 84
    assert set(bootstrap["iterations"]) == {100}
    leave_one_out = pd.read_csv(result.leave_one_out_path)
    assert len(leave_one_out) == 84
    b01_pauc = leave_one_out.loc[
        (leave_one_out["candidate"] == "B01")
        & (leave_one_out["domain"] == "all")
        & (leave_one_out["metric"] == "pauc_max_fpr_0_1")
    ]
    assert (b01_pauc["leave_one_out_mean_delta"] < 0.0).all()
    b02_pauc = leave_one_out.loc[
        (leave_one_out["candidate"] == "B02")
        & (leave_one_out["domain"] == "all")
        & (leave_one_out["metric"] == "pauc_max_fpr_0_1")
    ]
    assert b02_pauc["leave_one_out_mean_delta"].min() < 0.0
    assert b02_pauc["leave_one_out_mean_delta"].max() > 0.0
    heterogeneity = json.loads(result.heterogeneity_path.read_text(encoding="utf-8"))
    assert len(heterogeneity["records"]) == 12
    assert all(path.read_text(encoding="utf-8").startswith("<svg") for path in result.figure_paths)
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert len(run["artifacts"]) == 9
    with pytest.raises(FileExistsError):
        run_robustness_appendix(
            output_directory=output,
            config=fast_config,
            repository_root=REPOSITORY_ROOT,
        )
