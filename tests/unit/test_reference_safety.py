from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from care_asd.reference_safety_config import (
    ReferenceSafetyExperimentConfig,
    ReferenceSafetyPolicy,
)
from care_asd.signal.reference_safety import (
    aggregate_reference_profile,
    apply_reference_subtraction,
    diagnose_reference_pair,
    estimate_noise_transfer,
    noise_floor_spectrum,
    select_reference_view,
)
from care_asd.signal.reference_safety_simulation import (
    calibrate_reference_safety_policy,
    evaluate_reference_safety_policy,
    run_reference_safety_simulation,
)


def _pair() -> tuple[np.ndarray, np.ndarray, int]:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    machine = np.sin(2.0 * np.pi * 440.0 * time)
    noise = 0.2 * np.sin(2.0 * np.pi * 1300.0 * time + 0.4)
    near = machine + noise
    far = 0.15 * machine + noise
    return near, far, sample_rate


def test_reference_subtraction_is_finite_and_bounded() -> None:
    config = ReferenceSafetyExperimentConfig()
    near, far, sample_rate = _pair()
    near_floor = noise_floor_spectrum(near, sample_rate, config.stft, config.refsub)
    far_floor = noise_floor_spectrum(far, sample_rate, config.stft, config.refsub)
    transfer = estimate_noise_transfer(
        near_floor[None, :], far_floor[None, :], config.stft, config.refsub
    )

    enhanced = apply_reference_subtraction(
        near, far, sample_rate, transfer, config.stft, config.refsub
    )

    assert enhanced.shape == near.shape
    assert np.all(np.isfinite(enhanced))
    assert np.mean(enhanced**2) <= np.mean(near**2) * 1.05
    assert np.mean(enhanced**2) >= config.refsub.beta * np.mean(near**2) * 0.5


def test_profile_selects_near_when_risk_exceeds_policy() -> None:
    config = ReferenceSafetyExperimentConfig()
    near, far, sample_rate = _pair()
    near_floor = noise_floor_spectrum(near, sample_rate, config.stft, config.refsub)
    far_floor = noise_floor_spectrum(far, sample_rate, config.stft, config.refsub)
    transfer = estimate_noise_transfer(
        near_floor[None, :], far_floor[None, :], config.stft, config.refsub
    )
    enhanced = apply_reference_subtraction(
        near, far, sample_rate, transfer, config.stft, config.refsub
    )
    diagnostic = diagnose_reference_pair(
        near, far, enhanced, sample_rate, transfer, config.stft, config.refsub
    )
    profile = aggregate_reference_profile(transfer, [diagnostic], config.profile)
    policy = ReferenceSafetyPolicy(
        risk_max=0.0,
        benefit_min_db=-100.0,
        calibration_cases=16,
        holdout_cases=16,
        calibration_false_safe_rate=0.0,
        calibration_coverage=0.5,
    )

    assert select_reference_view(profile, policy) == "near"


def test_policy_calibration_is_deterministic() -> None:
    config = ReferenceSafetyExperimentConfig()
    frame = pd.DataFrame(
        {
            "risk_score": np.linspace(0.0, 1.0, 100),
            "estimated_noise_reduction_db": np.linspace(3.0, 0.0, 100),
            "is_safe": [index < 50 for index in range(100)],
            "anomaly_loss": np.linspace(0.0, 1.0, 100),
        }
    )

    first = calibrate_reference_safety_policy(frame, config)
    second = calibrate_reference_safety_policy(frame, config)
    metrics = evaluate_reference_safety_policy(frame, first, config)

    assert first == second
    assert metrics["coverage"] > 0.0
    assert metrics["false_safe_rate"] <= config.simulation.false_safe_max
    assert metrics["safe_cases"] == 50
    assert metrics["safe_prevalence"] == 0.5


def test_simulation_writes_policy_and_gate(tmp_path: Path) -> None:
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    random = np.random.default_rng(2026)
    source_a = (
        np.sin(2.0 * np.pi * 220.0 * time) + 0.1 * random.normal(size=len(time)),
        np.sin(2.0 * np.pi * 900.0 * time) + 0.1 * random.normal(size=len(time)),
        sample_rate,
    )
    source_b = (
        np.sin(2.0 * np.pi * 330.0 * time) + 0.1 * random.normal(size=len(time)),
        np.sin(2.0 * np.pi * 1200.0 * time) + 0.1 * random.normal(size=len(time)),
        sample_rate,
    )
    config = ReferenceSafetyExperimentConfig(
        simulation={"cases": 32, "duration_seconds": 0.5, "sample_rate": sample_rate}
    )

    result = run_reference_safety_simulation(
        sources=[source_a, source_b], output_directory=tmp_path / "simulation", config=config
    )

    assert result.policy_path.is_file()
    assert result.gate_path.is_file()
    cases = pd.read_parquet(result.cases_path)
    assert len(cases) == 32
    assert cases["anomaly_far_ratio"].eq(1.0).all()
    assert cases["transfer_instability"].max() > 0.0
