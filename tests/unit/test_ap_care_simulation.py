"""Small deterministic checks for the AP-CARE G1 synthetic runner."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from care_asd.ap_care_config import APCAREExperimentConfig
from care_asd.reproducibility import file_sha256
from care_asd.signal.ap_care_simulation import (
    ap_care_simulation_plan,
    run_ap_care_simulation,
    simulate_ap_care_case,
)


def _small_config() -> APCAREExperimentConfig:
    return APCAREExperimentConfig.model_validate(
        {
            "stft": {"n_fft": 128, "win_length": 128, "hop_length": 64},
            "transfer": {"frequency_smoothing_bins": 3},
            "controller": {
                "band_edges_hz": [0.0, 500.0, 2000.0],
                "warmup_frames": 1,
            },
            "simulation": {
                "cases": 32,
                "duration_seconds": 0.25,
                "sample_rate": 4000,
                "profile_clips": 2,
                "d00_alpha_grid": [0.5, 1.0],
            },
            "gate": {"bootstrap_iterations": 100},
        }
    )


def test_ap_care_plan_has_no_side_effect(tmp_path: Path) -> None:
    config = _small_config()
    output = tmp_path / "not-created"

    plan = ap_care_simulation_plan(config)

    assert plan["cases"] == 32
    assert plan["workers"] == 1
    assert len(plan["case_seeds"]) == 32
    assert plan["case_seeds"][1] - plan["case_seeds"][0] == 7919
    assert len(str(plan["config_hash"])) == 64
    assert not output.exists()

    with pytest.raises(ValueError, match="workers"):
        ap_care_simulation_plan(config, workers=0)


def test_synthetic_case_is_deterministic_with_exact_components() -> None:
    kwargs = {
        "sample_rate": 4000,
        "duration_seconds": 0.25,
        "seed": 9,
        "profile_clips": 2,
        "machine_leakage_gain": 0.2,
        "near_snr_db": 0.0,
        "far_environment_gain": 1.5,
        "path_delay_samples": 4,
        "path_gain_mismatch": 1.0,
        "path_delay_mismatch_samples": 2,
        "fault_amplitude": 0.2,
        "fault_support": "in_support",
        "fault_far_ratio": 1.0,
        "base_frequency_hz": 200.0,
    }

    first = simulate_ap_care_case(**kwargs)  # type: ignore[arg-type]
    second = simulate_ap_care_case(**kwargs)  # type: ignore[arg-type]

    assert first.profile_waveforms.tobytes() == second.profile_waveforms.tobytes()
    assert first.faulty_waveform.tobytes() == second.faulty_waveform.tobytes()
    assert (
        first.normal_waveform.tobytes()
        == (first.machine_component + first.environmental_component).tobytes()
    )


def test_ap_care_small_run_writes_auditable_artifacts(tmp_path: Path) -> None:
    config = _small_config()
    output = tmp_path / "g1"
    progress = tmp_path / "progress.env"

    result = run_ap_care_simulation(
        output_directory=output,
        config=config,
        progress_path=progress,
    )

    assert result.cases_path.is_file()
    assert result.summary_path.is_file()
    assert result.gate_path.is_file()
    assert result.config_path.is_file()
    assert result.run_path.is_file()
    assert result.environment_path.is_file()
    assert "completed_cases=32" in progress.read_text(encoding="utf-8")
    cases = pd.read_parquet(result.cases_path)
    assert len(cases) == 32
    assert set(cases["split"]) == {"calibration", "holdout"}
    assert {"ap_fault_retention", "d00_fault_retention", "matched_d00"}.issubset(cases.columns)
    gate_text = result.gate_path.read_text(encoding="utf-8")
    assert "NaN" not in gate_text
    gate = json.loads(gate_text)
    assert gate["passed"] == result.passed
    assert set(gate["checks"]) == {
        "eligible_noise_attenuation",
        "in_support_retention",
        "leakage_tracking",
        "matched_retention_improvement",
        "nontrivial_cancellation",
        "uncertainty_tracking",
    }
    run = json.loads(result.run_path.read_text(encoding="utf-8"))
    assert run["workers"] == 1
    assert run["manifest_sha256"] is None
    assert run["artifacts"]["config"]["sha256"] == file_sha256(result.config_path)
    assert run["artifacts"]["cases"]["sha256"] == file_sha256(result.cases_path)
    with pytest.raises(FileExistsError):
        run_ap_care_simulation(output_directory=output, config=config)
