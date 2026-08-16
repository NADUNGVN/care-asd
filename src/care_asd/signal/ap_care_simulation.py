"""Controlled synthetic mechanism benchmark for AP-CARE v2."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import signal as scipy_signal
from scipy.stats import qmc, spearmanr

from care_asd.ap_care_config import APCAREExperimentConfig, ap_care_config_hash
from care_asd.reproducibility import collect_environment_report, get_git_commit
from care_asd.signal.ap_care import APCAREController, causal_stft

FloatArray = NDArray[np.float64]
FaultSupport = Literal["in_support", "out_of_support"]


@dataclass(frozen=True)
class APCARESyntheticCase:
    """Stereo mixture and exact additive components for one controlled case."""

    sample_rate: int
    profile_waveforms: FloatArray
    normal_waveform: FloatArray
    faulty_waveform: FloatArray
    machine_component: FloatArray
    fault_component: FloatArray
    environmental_component: FloatArray
    true_reference_leakage: float
    path_mismatch: float
    near_environment_to_machine_db: float
    fault_support: FaultSupport
    machine_leakage_gain: float
    near_snr_db: float
    far_environment_gain: float
    path_delay_samples: int
    path_gain_mismatch: float
    path_delay_mismatch_samples: int
    fault_amplitude: float
    fault_far_ratio: float
    base_frequency_hz: float


@dataclass(frozen=True)
class APCARESimulationResult:
    """Immutable artifact paths from one G1 synthetic run."""

    output_directory: Path
    cases_path: Path
    summary_path: Path
    gate_path: Path
    run_path: Path
    environment_path: Path
    passed: bool


def simulate_ap_care_case(
    *,
    sample_rate: int,
    duration_seconds: float,
    seed: int,
    profile_clips: int,
    machine_leakage_gain: float,
    near_snr_db: float,
    far_environment_gain: float,
    path_delay_samples: int,
    path_gain_mismatch: float,
    path_delay_mismatch_samples: int,
    fault_amplitude: float,
    fault_support: FaultSupport,
    fault_far_ratio: float,
    base_frequency_hz: float,
) -> APCARESyntheticCase:
    """Generate one independent-profile case with known stereo components."""
    if sample_rate <= 1000:
        raise ValueError("sample_rate must exceed 1000 Hz")
    if duration_seconds <= 0.1:
        raise ValueError("duration_seconds must exceed 0.1 seconds")
    if profile_clips < 2:
        raise ValueError("profile_clips must be at least two")
    if machine_leakage_gain < 0.0 or far_environment_gain <= 0.0:
        raise ValueError("path gains must be non-negative and far_environment_gain positive")
    if path_delay_samples < 0 or path_delay_mismatch_samples < 0:
        raise ValueError("path delays must be non-negative")
    if path_gain_mismatch <= 0.0 or fault_amplitude <= 0.0 or fault_far_ratio < 0.0:
        raise ValueError("mismatch and fault parameters are outside their valid range")
    if fault_support not in {"in_support", "out_of_support"}:
        raise ValueError(f"Unsupported fault_support: {fault_support}")

    length = round(sample_rate * duration_seconds)
    if length < 64:
        raise ValueError("duration_seconds must produce at least 64 samples")
    rng = np.random.default_rng(seed)
    machine = _machine_waveform(length, sample_rate, base_frequency_hz, phase=0.0)
    machine = _unit_rms(machine)
    near_noise_gain = 10.0 ** (-near_snr_db / 20.0)
    evaluation_delay = path_delay_samples + path_delay_mismatch_samples

    shared_noise = _colored_noise(rng, length)
    independent_near = _colored_noise(rng, length)
    independent_far = _colored_noise(rng, length)
    near_environment = near_noise_gain * (0.85 * shared_noise + 0.15 * independent_near)
    far_environment = (
        far_environment_gain
        * path_gain_mismatch
        * (0.85 * _delay(shared_noise, evaluation_delay) + 0.15 * independent_far)
    )
    machine_far = machine_leakage_gain * path_gain_mismatch * _delay(machine, evaluation_delay)
    machine_component = np.stack((machine, machine_far))
    environmental_component = np.stack((near_environment, far_environment))

    fault = _fault_waveform(
        length,
        sample_rate,
        base_frequency_hz=base_frequency_hz,
        amplitude=fault_amplitude,
        support=fault_support,
    )
    fault_far = (
        machine_leakage_gain
        * fault_far_ratio
        * path_gain_mismatch
        * _delay(fault, evaluation_delay)
    )
    fault_component = np.stack((fault, fault_far))
    normal = machine_component + environmental_component
    faulty = normal + fault_component

    profiles: list[FloatArray] = []
    for profile_index in range(profile_clips):
        profile_machine = _unit_rms(
            _machine_waveform(
                length,
                sample_rate,
                base_frequency_hz,
                phase=2.0 * np.pi * profile_index / profile_clips,
            )
        )
        profile_shared = _colored_noise(rng, length)
        profile_near_independent = _colored_noise(rng, length)
        profile_far_independent = _colored_noise(rng, length)
        profile_near_noise = near_noise_gain * (
            0.85 * profile_shared + 0.15 * profile_near_independent
        )
        profile_far_noise = far_environment_gain * (
            0.85 * _delay(profile_shared, path_delay_samples) + 0.15 * profile_far_independent
        )
        profile_far_machine = machine_leakage_gain * _delay(profile_machine, path_delay_samples)
        profiles.append(
            np.stack(
                (
                    profile_machine + profile_near_noise,
                    profile_far_machine + profile_far_noise,
                )
            )
        )

    machine_far_energy = _time_energy(machine_far)
    far_environment_energy = _time_energy(far_environment)
    true_leakage = machine_far_energy / max(
        machine_far_energy + far_environment_energy,
        1.0e-12,
    )
    near_environment_to_machine_db = 10.0 * math.log10(
        max(_time_energy(near_environment), 1.0e-12) / max(_time_energy(machine), 1.0e-12)
    )
    path_mismatch = abs(math.log(path_gain_mismatch)) + (
        path_delay_mismatch_samples / max(sample_rate * 0.02, 1.0)
    )
    return APCARESyntheticCase(
        sample_rate=sample_rate,
        profile_waveforms=np.stack(profiles),
        normal_waveform=np.asarray(normal, dtype=np.float64),
        faulty_waveform=np.asarray(faulty, dtype=np.float64),
        machine_component=np.asarray(machine_component, dtype=np.float64),
        fault_component=np.asarray(fault_component, dtype=np.float64),
        environmental_component=np.asarray(environmental_component, dtype=np.float64),
        true_reference_leakage=float(true_leakage),
        path_mismatch=float(path_mismatch),
        near_environment_to_machine_db=float(near_environment_to_machine_db),
        fault_support=fault_support,
        machine_leakage_gain=machine_leakage_gain,
        near_snr_db=near_snr_db,
        far_environment_gain=far_environment_gain,
        path_delay_samples=path_delay_samples,
        path_gain_mismatch=path_gain_mismatch,
        path_delay_mismatch_samples=path_delay_mismatch_samples,
        fault_amplitude=fault_amplitude,
        fault_far_ratio=fault_far_ratio,
        base_frequency_hz=base_frequency_hz,
    )


def ap_care_simulation_plan(
    config: APCAREExperimentConfig,
    cases: int | None = None,
) -> dict[str, Any]:
    """Return the validated, side-effect-free G1 execution plan."""
    case_count = config.simulation.cases if cases is None else cases
    if case_count < 32:
        raise ValueError("AP-CARE simulation requires at least 32 cases")
    return {
        "schema_version": 1,
        "experiment_id": config.experiment_id,
        "config_hash": ap_care_config_hash(config),
        "cases": case_count,
        "calibration_cases": round(case_count * config.simulation.calibration_fraction),
        "sample_rate": config.simulation.sample_rate,
        "duration_seconds": config.simulation.duration_seconds,
        "profile_clips_per_case": config.simulation.profile_clips,
        "seed": config.simulation.seed,
        "d00_alpha_grid": list(config.simulation.d00_alpha_grid),
    }


def run_ap_care_simulation(
    *,
    output_directory: str | Path,
    config: APCAREExperimentConfig,
    cases: int | None = None,
) -> APCARESimulationResult:
    """Run one immutable AP-CARE G1 calibration/holdout sweep."""
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite AP-CARE simulation: {output}")
    plan = ap_care_simulation_plan(config, cases)
    frame = _simulate_cases(config, int(plan["cases"]))
    calibration_count = int(plan["calibration_cases"])
    calibration_count = min(max(calibration_count, 1), len(frame) - 1)
    frame["split"] = "holdout"
    frame.loc[: calibration_count - 1, "split"] = "calibration"
    calibration = frame.iloc[:calibration_count].copy()
    holdout = frame.iloc[calibration_count:].copy()
    calibration_summary = summarize_ap_care_cases(calibration, config, seed_offset=0)
    holdout_summary = summarize_ap_care_cases(holdout, config, seed_offset=1)
    checks = _gate_checks(holdout_summary, config)
    passed = all(checks.values())

    output.mkdir(parents=True)
    cases_path = output / "synthetic_cases.parquet"
    frame.to_parquet(cases_path, index=False)
    summary_path = output / "summary.csv"
    pd.DataFrame(
        [
            {"split": "calibration", **calibration_summary, "passed": passed},
            {"split": "holdout", **holdout_summary, "passed": passed},
        ]
    ).to_csv(summary_path, index=False)
    gate_payload = {
        "schema_version": 1,
        "passed": passed,
        "checks": checks,
        "criteria": config.gate.model_dump(),
        "calibration": calibration_summary,
        "holdout": holdout_summary,
    }
    gate_path = output / "gate.json"
    gate_path.write_text(
        json.dumps(_json_ready(gate_payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    run_payload = {
        **plan,
        "git_commit": get_git_commit(),
        "artifacts": {
            "cases": cases_path.name,
            "summary": summary_path.name,
            "gate": gate_path.name,
            "environment": "environment.json",
        },
    }
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    environment_path = output / "environment.json"
    environment_path.write_text(
        collect_environment_report().to_json() + "\n",
        encoding="utf-8",
    )
    return APCARESimulationResult(
        output_directory=output,
        cases_path=cases_path,
        summary_path=summary_path,
        gate_path=gate_path,
        run_path=run_path,
        environment_path=environment_path,
        passed=passed,
    )


def summarize_ap_care_cases(
    frame: pd.DataFrame,
    config: APCAREExperimentConfig,
    *,
    seed_offset: int,
) -> dict[str, float | int]:
    """Summarize one fixed split without selecting controller thresholds."""
    required = {
        "true_reference_leakage",
        "path_mismatch",
        "leakage_risk",
        "transfer_uncertainty",
        "ap_fault_retention",
        "ap_noise_attenuation_db",
        "eligible",
        "active_cancellation",
        "fault_support",
        "matched_d00",
        "retention_improvement_vs_d00",
    }
    missing = sorted(required.difference(frame.columns))
    if frame.empty or missing:
        raise ValueError(f"AP-CARE cases are empty or missing columns: {', '.join(missing)}")
    leakage_rho, leakage_low, leakage_high = _spearman_interval(
        frame["leakage_risk"].to_numpy(dtype=float),
        frame["true_reference_leakage"].to_numpy(dtype=float),
        iterations=config.gate.bootstrap_iterations,
        seed=config.simulation.seed + seed_offset,
    )
    uncertainty_rho, uncertainty_low, uncertainty_high = _spearman_interval(
        frame["transfer_uncertainty"].to_numpy(dtype=float),
        frame["path_mismatch"].to_numpy(dtype=float),
        iterations=config.gate.bootstrap_iterations,
        seed=config.simulation.seed + 10 + seed_offset,
    )
    in_support = frame.loc[frame["fault_support"] == "in_support"]
    eligible = frame.loc[frame["eligible"].astype(bool)]
    active = eligible.loc[eligible["active_cancellation"].astype(bool)]
    matched_high = frame.loc[
        frame["matched_d00"].astype(bool)
        & (frame["true_reference_leakage"] >= config.gate.medium_high_leakage_min)
    ]
    improvements = matched_high["retention_improvement_vs_d00"].dropna().to_numpy(dtype=float)
    improvement_low, improvement_high = _mean_interval(
        improvements,
        iterations=config.gate.bootstrap_iterations,
        seed=config.simulation.seed + 20 + seed_offset,
    )
    return {
        "cases": len(frame),
        "leakage_spearman": leakage_rho,
        "leakage_spearman_ci95_low": leakage_low,
        "leakage_spearman_ci95_high": leakage_high,
        "uncertainty_spearman": uncertainty_rho,
        "uncertainty_spearman_ci95_low": uncertainty_low,
        "uncertainty_spearman_ci95_high": uncertainty_high,
        "in_support_cases": len(in_support),
        "in_support_fault_retention_median": _quantile_or_nan(
            in_support["ap_fault_retention"], 0.50
        ),
        "in_support_fault_retention_q05": _quantile_or_nan(in_support["ap_fault_retention"], 0.05),
        "eligible_cases": len(eligible),
        "eligible_noise_attenuation_median_db": _quantile_or_nan(
            eligible["ap_noise_attenuation_db"], 0.50
        ),
        "active_cases": len(active),
        "active_case_fraction": len(active) / max(len(eligible), 1),
        "active_noise_attenuation_median_db": _quantile_or_nan(
            active["ap_noise_attenuation_db"], 0.50
        ),
        "matched_medium_high_cases": len(improvements),
        "retention_improvement_median": (
            float(np.median(improvements)) if len(improvements) else float("nan")
        ),
        "retention_improvement_mean_ci95_low": improvement_low,
        "retention_improvement_mean_ci95_high": improvement_high,
        "out_of_support_cases": int((frame["fault_support"] == "out_of_support").sum()),
        "out_of_support_fault_retention_median": _quantile_or_nan(
            frame.loc[frame["fault_support"] == "out_of_support", "ap_fault_retention"],
            0.50,
        ),
    }


def _simulate_cases(config: APCAREExperimentConfig, cases: int) -> pd.DataFrame:
    exponent = math.ceil(math.log2(cases))
    design = qmc.Sobol(d=10, scramble=True, seed=config.simulation.seed).random_base2(exponent)[
        :cases
    ]
    records: list[dict[str, float | int | str | bool]] = []
    for case_id, point in enumerate(design):
        sample_rate = config.simulation.sample_rate
        leakage_gain = math.exp(math.log(0.02) + point[0] * math.log(0.80 / 0.02))
        near_snr_db = -10.0 + 20.0 * point[1]
        far_environment_gain = 0.5 + 2.5 * point[2]
        path_delay = round(point[3] * 0.020 * sample_rate)
        path_gain_mismatch = math.exp(math.log(0.5) + point[4] * math.log(4.0))
        delay_mismatch = round(point[5] * 0.020 * sample_rate)
        fault_amplitude = 0.05 + 0.45 * point[6]
        fault_support: FaultSupport = "in_support" if point[7] < 0.5 else "out_of_support"
        fault_far_ratio = 0.5 + point[8]
        base_frequency = 160.0 + 360.0 * point[9]
        case = simulate_ap_care_case(
            sample_rate=sample_rate,
            duration_seconds=config.simulation.duration_seconds,
            seed=config.simulation.seed + case_id * 7919,
            profile_clips=config.simulation.profile_clips,
            machine_leakage_gain=leakage_gain,
            near_snr_db=near_snr_db,
            far_environment_gain=far_environment_gain,
            path_delay_samples=path_delay,
            path_gain_mismatch=path_gain_mismatch,
            path_delay_mismatch_samples=delay_mismatch,
            fault_amplitude=fault_amplitude,
            fault_support=fault_support,
            fault_far_ratio=fault_far_ratio,
            base_frequency_hz=base_frequency,
        )
        records.append({"case_id": case_id, **_measure_case(case, config)})
    return pd.DataFrame.from_records(records)


def _measure_case(
    case: APCARESyntheticCase,
    config: APCAREExperimentConfig,
) -> dict[str, float | int | str | bool]:
    controller = APCAREController(config)
    profile = controller.fit(case.profile_waveforms, case.sample_rate)
    output = controller.transform(case.faulty_waveform, case.sample_rate, profile)
    fault_output = controller.apply_frozen(output, case.fault_component, case.sample_rate)
    machine_output = controller.apply_frozen(output, case.machine_component, case.sample_rate)
    noise_output = controller.apply_frozen(output, case.environmental_component, case.sample_rate)
    fault_input = causal_stft(case.fault_component[0], config.stft)
    machine_input = causal_stft(case.machine_component[0], config.stft)
    noise_input = causal_stft(case.environmental_component[0], config.stft)
    fault_retention = _spectral_energy(fault_output) / max(
        _spectral_energy(fault_input), config.stft.eps
    )
    machine_retention = _spectral_energy(machine_output) / max(
        _spectral_energy(machine_input), config.stft.eps
    )
    noise_attenuation = 10.0 * math.log10(
        max(_spectral_energy(noise_input), config.stft.eps)
        / max(_spectral_energy(noise_output), config.stft.eps)
    )
    far_power = np.abs(output.far_stft) ** 2
    leakage_risk = float(
        np.sum(output.leakage_risk * far_power) / max(float(np.sum(far_power)), config.stft.eps)
    )
    converged_start = max(0, output.transfer_uncertainty.shape[0] * 3 // 4)
    transfer_uncertainty = float(np.median(output.transfer_uncertainty[converged_start:]))
    estimated_gain_mismatch = float(np.median(output.transfer_gain_mismatch[converged_start:]))
    estimated_delay_mismatch = float(
        np.median(output.transfer_delay_mismatch_seconds[converged_start:])
    )
    active_fraction = float(np.mean(output.controller_gain > 1.0e-8))
    baseline = _match_d00(
        case,
        config,
        profile_power_transfer=_power_transfer(case, config),
        target_attenuation_db=noise_attenuation,
    )
    matched = bool(
        abs(noise_attenuation - baseline["noise_attenuation_db"])
        <= config.gate.matched_attenuation_tolerance_db
    )
    return {
        "fault_support": case.fault_support,
        "machine_leakage_gain": case.machine_leakage_gain,
        "near_snr_db": case.near_snr_db,
        "far_environment_gain": case.far_environment_gain,
        "path_delay_samples": case.path_delay_samples,
        "path_gain_mismatch": case.path_gain_mismatch,
        "path_delay_mismatch_samples": case.path_delay_mismatch_samples,
        "fault_amplitude": case.fault_amplitude,
        "fault_far_ratio": case.fault_far_ratio,
        "base_frequency_hz": case.base_frequency_hz,
        "true_reference_leakage": case.true_reference_leakage,
        "path_mismatch": case.path_mismatch,
        "near_environment_to_machine_db": case.near_environment_to_machine_db,
        "eligible": case.near_environment_to_machine_db >= -10.0,
        "leakage_risk": leakage_risk,
        "transfer_uncertainty": transfer_uncertainty,
        "estimated_gain_mismatch": estimated_gain_mismatch,
        "estimated_delay_mismatch_seconds": estimated_delay_mismatch,
        "mean_noise_utility": float(np.mean(output.noise_utility)),
        "mean_coherence": float(np.mean(output.coherence)),
        "active_gain_fraction": active_fraction,
        "active_cancellation": active_fraction > 0.0,
        "bound_active_fraction": float(np.mean(output.bound_active)),
        "ap_fault_retention": float(fault_retention),
        "ap_machine_retention": float(machine_retention),
        "ap_noise_attenuation_db": float(noise_attenuation),
        "d00_alpha": baseline["alpha"],
        "d00_fault_retention": baseline["fault_retention"],
        "d00_noise_attenuation_db": baseline["noise_attenuation_db"],
        "matched_d00": matched,
        "retention_improvement_vs_d00": (
            float(fault_retention - baseline["fault_retention"]) if matched else float("nan")
        ),
    }


def _power_transfer(case: APCARESyntheticCase, config: APCAREExperimentConfig) -> FloatArray:
    near_floors: list[FloatArray] = []
    far_floors: list[FloatArray] = []
    for profile in case.profile_waveforms:
        near = causal_stft(profile[0], config.stft)
        far = causal_stft(profile[1], config.stft)
        near_floors.append(np.quantile(np.abs(near) ** 2, 0.10, axis=0))
        far_floors.append(np.quantile(np.abs(far) ** 2, 0.10, axis=0))
    ratio = np.stack(near_floors) / np.maximum(np.stack(far_floors), config.stft.eps)
    return np.asarray(np.clip(np.median(ratio, axis=0), 0.10, 10.0), dtype=np.float64)


def _match_d00(
    case: APCARESyntheticCase,
    config: APCAREExperimentConfig,
    *,
    profile_power_transfer: FloatArray,
    target_attenuation_db: float,
) -> dict[str, float]:
    near = causal_stft(case.faulty_waveform[0], config.stft)
    far = causal_stft(case.faulty_waveform[1], config.stft)
    fault = causal_stft(case.fault_component[0], config.stft)
    noise = causal_stft(case.environmental_component[0], config.stft)
    near_power = np.abs(near) ** 2
    far_power = np.abs(far) ** 2
    fault_energy = max(_spectral_energy(fault), config.stft.eps)
    noise_energy = max(_spectral_energy(noise), config.stft.eps)
    candidates: list[dict[str, float]] = []
    for alpha in config.simulation.d00_alpha_grid:
        enhanced_power = np.maximum(
            near_power - alpha * profile_power_transfer[np.newaxis, :] * far_power,
            config.simulation.d00_floor_beta * near_power,
        )
        amplitude_mask = np.sqrt(enhanced_power / np.maximum(near_power, config.stft.eps))
        fault_retention = _spectral_energy(amplitude_mask * fault) / fault_energy
        noise_attenuation = 10.0 * math.log10(
            noise_energy / max(_spectral_energy(amplitude_mask * noise), config.stft.eps)
        )
        candidates.append(
            {
                "alpha": float(alpha),
                "fault_retention": float(fault_retention),
                "noise_attenuation_db": float(noise_attenuation),
            }
        )
    return min(
        candidates,
        key=lambda item: (
            abs(item["noise_attenuation_db"] - target_attenuation_db),
            item["alpha"],
        ),
    )


def _gate_checks(
    summary: dict[str, float | int],
    config: APCAREExperimentConfig,
) -> dict[str, bool]:
    gate = config.gate
    return {
        "leakage_tracking": bool(
            summary["leakage_spearman"] >= gate.leakage_spearman_min
            and summary["leakage_spearman_ci95_low"] > gate.leakage_spearman_ci_low_min
        ),
        "uncertainty_tracking": bool(
            summary["uncertainty_spearman"] >= gate.uncertainty_spearman_min
            and summary["uncertainty_spearman_ci95_low"] > gate.uncertainty_spearman_ci_low_min
        ),
        "matched_retention_improvement": bool(
            summary["matched_medium_high_cases"] > 0
            and summary["retention_improvement_median"] >= gate.retention_improvement_min
            and summary["retention_improvement_mean_ci95_low"] > 0.0
        ),
        "in_support_retention": bool(
            summary["in_support_fault_retention_median"] >= gate.fault_retention_median_min
            and summary["in_support_fault_retention_q05"] >= gate.fault_retention_q05_min
        ),
        "eligible_noise_attenuation": bool(
            summary["eligible_noise_attenuation_median_db"] >= gate.noise_attenuation_median_min_db
        ),
        "nontrivial_cancellation": bool(
            summary["active_case_fraction"] >= gate.active_case_fraction_min
            and summary["active_noise_attenuation_median_db"]
            >= gate.active_attenuation_median_min_db
        ),
    }


def _spearman_interval(
    x: FloatArray,
    y: FloatArray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float, float]:
    point = _safe_spearman(x, y)
    if len(x) < 2:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(x), size=len(x))
        if np.ptp(x[indices]) > 0.0 and np.ptp(y[indices]) > 0.0:
            values.append(_safe_spearman(x[indices], y[indices]))
    if not values:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _safe_spearman(x: FloatArray, y: FloatArray) -> float:
    if len(x) < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return 0.0
    statistic = spearmanr(x, y, nan_policy="omit").statistic
    return float(statistic) if np.isfinite(statistic) else 0.0


def _mean_interval(
    values: FloatArray,
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    if not len(values):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = values[rng.integers(0, len(values), size=len(values))]
        means[index] = np.mean(sample)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _machine_waveform(
    length: int,
    sample_rate: int,
    base_frequency_hz: float,
    *,
    phase: float,
) -> FloatArray:
    time = np.arange(length, dtype=np.float64) / sample_rate
    return np.asarray(
        np.sin(2.0 * np.pi * base_frequency_hz * time + phase)
        + 0.35 * np.sin(2.0 * np.pi * 2.0 * base_frequency_hz * time + 0.5 * phase)
        + 0.15 * np.sin(2.0 * np.pi * 3.0 * base_frequency_hz * time + 0.25 * phase),
        dtype=np.float64,
    )


def _fault_waveform(
    length: int,
    sample_rate: int,
    *,
    base_frequency_hz: float,
    amplitude: float,
    support: FaultSupport,
) -> FloatArray:
    time = np.arange(length, dtype=np.float64) / sample_rate
    start = length // 3
    active_time = time[: length - start]
    if support == "in_support":
        carrier = 2.0 * base_frequency_hz
    else:
        carrier = min(max(6.0 * base_frequency_hz, 2600.0), 0.42 * sample_rate)
    active = scipy_signal.chirp(
        active_time,
        f0=0.90 * carrier,
        f1=1.10 * carrier,
        t1=max(active_time[-1], 1.0 / sample_rate),
        method="linear",
    )
    envelope = np.minimum(np.arange(len(active), dtype=np.float64) / (0.02 * sample_rate), 1.0)
    fault = np.zeros(length, dtype=np.float64)
    fault[start:] = amplitude * envelope * active
    return fault


def _colored_noise(rng: np.random.Generator, length: int) -> FloatArray:
    white = rng.standard_normal(length + 8)
    filtered = scipy_signal.lfilter([1.0], [1.0, -0.82], white)[8:]
    return _unit_rms(np.asarray(filtered, dtype=np.float64))


def _unit_rms(values: NDArray[np.floating]) -> FloatArray:
    waveform = np.asarray(values, dtype=np.float64)
    waveform = waveform - np.mean(waveform)
    rms = math.sqrt(float(np.mean(waveform**2)))
    return waveform / max(rms, 1.0e-12)


def _delay(values: FloatArray, samples: int) -> FloatArray:
    if samples <= 0:
        return values.copy()
    delayed = np.zeros_like(values)
    if samples < len(values):
        delayed[samples:] = values[:-samples]
    return delayed


def _time_energy(values: FloatArray) -> float:
    return float(np.sum(np.asarray(values, dtype=np.float64) ** 2))


def _spectral_energy(values: NDArray[np.complexfloating]) -> float:
    return float(np.sum(np.abs(np.asarray(values, dtype=np.complex128)) ** 2))


def _quantile_or_nan(values: pd.Series, quantile: float) -> float:
    clean = values.dropna()
    return float(clean.quantile(quantile)) if len(clean) else float("nan")


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, float | np.floating) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
