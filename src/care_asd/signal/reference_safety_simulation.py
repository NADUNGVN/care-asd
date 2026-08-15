"""Deterministic semi-synthetic calibration for SAFE-REF."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml
from numpy.typing import NDArray
from scipy import signal as scipy_signal
from scipy.stats import qmc, spearmanr

from care_asd.reference_safety_config import (
    ReferenceSafetyExperimentConfig,
    ReferenceSafetyPolicy,
)
from care_asd.signal.reference_safety import (
    apply_reference_subtraction,
    diagnose_reference_pair,
    estimate_noise_transfer,
    noise_floor_spectrum,
)

FloatArray = NDArray[np.float64]
StereoSource = tuple[NDArray[np.floating], NDArray[np.floating], int]


@dataclass(frozen=True)
class ReferenceSimulationResult:
    """Immutable outputs from one synthetic calibration run."""

    output_directory: Path
    cases_path: Path
    policy_path: Path
    gate_path: Path
    summary_path: Path
    passed: bool


def load_normal_stereo_sources(
    *, manifest_path: str | Path, audio_root: str | Path, limit: int = 64
) -> list[StereoSource]:
    """Load a deterministic machine-balanced subset of normal training clips."""
    if limit < 2:
        raise ValueError("At least two simulation source clips are required")
    manifest = Path(manifest_path)
    root = Path(audio_root)
    if not manifest.is_file() or not root.is_dir():
        raise FileNotFoundError("Simulation manifest or audio root does not exist")
    frame = pd.read_parquet(manifest)
    required = {"relative_path", "machine_type", "condition", "dataset_split"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Simulation manifest is missing: {', '.join(missing)}")
    normal = frame.loc[
        frame["dataset_split"].isin({"dev_train", "add_train"}) & (frame["condition"] == "normal")
    ].sort_values(["machine_type", "relative_path"], kind="stable")
    if len(normal) < 2:
        raise ValueError("Simulation manifest has fewer than two normal training clips")
    per_group = max(math.ceil(limit / normal["machine_type"].nunique()), 1)
    selected = normal.groupby("machine_type", sort=True).head(per_group).head(limit)
    resolved_root = root.resolve()
    sources: list[StereoSource] = []
    for row in selected.itertuples(index=False):
        path = (resolved_root / str(row.relative_path)).resolve()
        if resolved_root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Unsafe or missing simulation audio: {path}")
        values, sample_rate = sf.read(path, dtype="float64", always_2d=True)
        if values.ndim != 2 or values.shape[1] < 2 or not len(values):
            raise ValueError(f"Expected stereo simulation audio: {path}")
        sources.append((values[:, 0], values[:, 1], int(sample_rate)))
    return sources


def run_reference_safety_simulation(
    *,
    sources: Sequence[StereoSource],
    output_directory: str | Path,
    config: ReferenceSafetyExperimentConfig,
    cases: int | None = None,
) -> ReferenceSimulationResult:
    """Calibrate SAFE-REF on synthetic anomalies mixed with real normal carriers."""
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite reference-safety simulation: {output}")
    if len(sources) < 2:
        raise ValueError("Simulation requires at least two normal stereo source clips")
    case_count = config.simulation.cases if cases is None else cases
    if case_count < 32:
        raise ValueError("Simulation requires at least 32 cases")
    prepared = [_prepare_source(item, config) for item in sources]
    frame = _simulate_cases(prepared, config, case_count)
    calibration_count = round(case_count * config.simulation.calibration_fraction)
    calibration_count = min(max(calibration_count, 1), case_count - 1)
    calibration = frame.iloc[:calibration_count].copy()
    holdout = frame.iloc[calibration_count:].copy()
    policy = calibrate_reference_safety_policy(calibration, config)
    holdout_metrics = evaluate_reference_safety_policy(holdout, policy, config)
    calibration_metrics = evaluate_reference_safety_policy(calibration, policy, config)
    passed = bool(
        holdout_metrics["false_safe_rate"] <= config.simulation.false_safe_max
        and holdout_metrics["false_safe_upper_ci"] <= config.simulation.false_safe_upper_ci_max
        and holdout_metrics["coverage"] >= config.simulation.minimum_coverage
        and holdout_metrics["risk_spearman"] >= config.simulation.minimum_risk_spearman
        and holdout_metrics["tail_loss_reduction"] >= config.simulation.minimum_tail_loss_reduction
    )

    output.mkdir(parents=True)
    cases_path = output / "synthetic_cases.parquet"
    frame.to_parquet(cases_path, index=False)
    policy_path = output / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy.model_dump(), sort_keys=True), encoding="utf-8")
    gate = {
        "schema_version": 1,
        "passed": passed,
        "criteria": {
            "false_safe_max": config.simulation.false_safe_max,
            "false_safe_upper_ci_max": config.simulation.false_safe_upper_ci_max,
            "minimum_coverage": config.simulation.minimum_coverage,
            "minimum_risk_spearman": config.simulation.minimum_risk_spearman,
            "minimum_tail_loss_reduction": config.simulation.minimum_tail_loss_reduction,
        },
        "calibration": calibration_metrics,
        "holdout": holdout_metrics,
    }
    gate_path = output / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path = output / "summary.csv"
    pd.DataFrame(
        [
            {"split": "calibration", **calibration_metrics, "passed": passed},
            {"split": "holdout", **holdout_metrics, "passed": passed},
        ]
    ).to_csv(summary_path, index=False)
    return ReferenceSimulationResult(
        output,
        cases_path,
        policy_path,
        gate_path,
        summary_path,
        passed,
    )


def calibrate_reference_safety_policy(
    calibration: pd.DataFrame, config: ReferenceSafetyExperimentConfig
) -> ReferenceSafetyPolicy:
    """Choose the highest-coverage policy satisfying the synthetic false-safe budget."""
    _validate_case_frame(calibration)
    step = config.simulation.percentile_step
    percentiles = np.arange(step, 100, step, dtype=float)
    risk_candidates = np.unique(np.percentile(calibration["risk_score"], percentiles))
    benefit_candidates = np.unique(
        np.percentile(calibration["estimated_noise_reduction_db"], percentiles)
    )
    candidates: list[tuple[bool, float, float, float, float, float]] = []
    for risk_max in risk_candidates:
        for benefit_min in benefit_candidates:
            accepted = (calibration["risk_score"] <= risk_max) & (
                calibration["estimated_noise_reduction_db"] >= benefit_min
            )
            accepted_count = int(accepted.sum())
            false_safe = int((accepted & ~calibration["is_safe"]).sum())
            rate = false_safe / max(accepted_count, 1)
            upper = _wilson_upper(false_safe, accepted_count)
            coverage = accepted_count / len(calibration)
            is_feasible = bool(
                accepted_count > 0
                and rate <= config.simulation.false_safe_max
                and upper <= config.simulation.false_safe_upper_ci_max
            )
            candidates.append(
                (
                    is_feasible,
                    coverage,
                    -upper,
                    -rate,
                    float(risk_max),
                    float(benefit_min),
                )
            )
    if not candidates:
        raise ValueError("Synthetic calibration produced no threshold candidates")
    feasible_candidates = [item for item in candidates if item[0]]
    if feasible_candidates:
        selected = max(
            feasible_candidates,
            key=lambda item: (item[1], item[2], item[3], -item[4], item[5]),
        )
    else:
        selected = max(
            candidates,
            key=lambda item: (item[2], item[1], item[3], -item[4], item[5]),
        )
    _, coverage, _, negative_rate, risk_max, benefit_min = selected
    calibration_count = round(len(calibration) / config.simulation.calibration_fraction)
    return ReferenceSafetyPolicy(
        risk_max=risk_max,
        benefit_min_db=benefit_min,
        calibration_cases=len(calibration),
        holdout_cases=max(calibration_count - len(calibration), 1),
        calibration_false_safe_rate=-negative_rate,
        calibration_coverage=coverage,
    )


def evaluate_reference_safety_policy(
    frame: pd.DataFrame,
    policy: ReferenceSafetyPolicy,
    config: ReferenceSafetyExperimentConfig,
) -> dict[str, float | int]:
    """Evaluate a fixed policy without changing its thresholds."""
    _validate_case_frame(frame)
    accepted = (frame["risk_score"] <= policy.risk_max) & (
        frame["estimated_noise_reduction_db"] >= policy.benefit_min_db
    )
    accepted_count = int(accepted.sum())
    false_safe = int((accepted & ~frame["is_safe"]).sum())
    false_safe_rate = false_safe / max(accepted_count, 1)
    false_safe_upper = _wilson_upper(false_safe, accepted_count)
    correlation = spearmanr(frame["risk_score"], frame["anomaly_loss"], nan_policy="omit")
    risk_spearman = float(correlation.statistic) if np.isfinite(correlation.statistic) else 0.0
    unconditional_tail = float(np.quantile(frame["anomaly_loss"], 0.95))
    policy_loss = np.where(accepted, frame["anomaly_loss"], 0.0)
    policy_tail = float(np.quantile(policy_loss, 0.95))
    tail_reduction = 1.0 - policy_tail / max(unconditional_tail, 1.0e-12)
    safe_cases = int(frame["is_safe"].sum())
    return {
        "cases": len(frame),
        "safe_cases": safe_cases,
        "safe_prevalence": safe_cases / len(frame),
        "accepted": accepted_count,
        "coverage": accepted_count / len(frame),
        "false_safe_count": false_safe,
        "false_safe_rate": false_safe_rate,
        "false_safe_upper_ci": false_safe_upper,
        "risk_spearman": risk_spearman,
        "tail_loss_reduction": float(np.clip(tail_reduction, -1.0, 1.0)),
        "unconditional_loss_q95": unconditional_tail,
        "policy_loss_q95": policy_tail,
    }


def _simulate_cases(
    sources: Sequence[tuple[FloatArray, FloatArray]],
    config: ReferenceSafetyExperimentConfig,
    cases: int,
) -> pd.DataFrame:
    dimensions = 11
    exponent = math.ceil(math.log2(cases))
    samples = qmc.Sobol(d=dimensions, scramble=True, seed=config.simulation.seed).random_base2(
        exponent
    )[:cases]
    records: list[dict[str, float | int | str | bool]] = []
    for index, point in enumerate(samples):
        carrier_index = min(int(point[0] * len(sources)), len(sources) - 1)
        carrier = sources[carrier_index][0]
        noise_index = min(int(point[1] * len(sources)), len(sources) - 1)
        if noise_index == carrier_index:
            noise_index = (noise_index + 1) % len(sources)
        noise = sources[noise_index][1]
        snr_db = -10.0 + 20.0 * point[2]
        leakage_db = -30.0 + 30.0 * point[3]
        coherence = point[4]
        delay_ms = 20.0 * point[5]
        gain_db = -12.0 + 24.0 * point[6]
        profile_carrier_index = min(int(point[7] * len(sources)), len(sources) - 1)
        if profile_carrier_index == carrier_index:
            profile_carrier_index = (profile_carrier_index + 1) % len(sources)
        profile_noise_index = min(int(point[8] * len(sources)), len(sources) - 1)
        if profile_noise_index == noise_index:
            profile_noise_index = (profile_noise_index + 1) % len(sources)
        family_index = min(int(point[9] * 4), 3)
        severity = 0.05 + 0.45 * point[10]
        anomaly_family = ("impulsive", "sideband", "rubbing", "speed_drift")[family_index]
        record = _simulate_one(
            carrier=carrier,
            noise=noise,
            profile_carrier=sources[profile_carrier_index][0],
            profile_noise=sources[profile_noise_index][1],
            sample_rate=config.simulation.sample_rate,
            snr_db=snr_db,
            leakage_db=leakage_db,
            coherence=coherence,
            delay_ms=delay_ms,
            gain_db=gain_db,
            anomaly_far_ratio=1.0,
            anomaly_family=anomaly_family,
            severity=severity,
            phase=2.0 * np.pi * ((index * 0.61803398875) % 1.0),
            config=config,
        )
        records.append({"case_id": index, **record})
    return pd.DataFrame.from_records(records)


def _simulate_one(
    *,
    carrier: FloatArray,
    noise: FloatArray,
    profile_carrier: FloatArray,
    profile_noise: FloatArray,
    sample_rate: int,
    snr_db: float,
    leakage_db: float,
    coherence: float,
    delay_ms: float,
    gain_db: float,
    anomaly_far_ratio: float,
    anomaly_family: str,
    severity: float,
    phase: float,
    config: ReferenceSafetyExperimentConfig,
) -> dict[str, float | str | bool]:
    carrier = _unit_rms(carrier)
    noise = _unit_rms(noise)
    profile_carrier = _unit_rms(profile_carrier)
    profile_noise = _unit_rms(profile_noise)
    normal_near, normal_far = _normal_pair(
        carrier,
        noise,
        snr_db=snr_db,
        leakage_db=leakage_db,
        coherence=coherence,
        delay_ms=delay_ms,
        gain_db=gain_db,
        sample_rate=sample_rate,
    )
    profile_near, profile_far = _normal_pair(
        profile_carrier,
        profile_noise,
        snr_db=snr_db,
        leakage_db=leakage_db,
        coherence=coherence,
        delay_ms=delay_ms,
        gain_db=gain_db,
        sample_rate=sample_rate,
    )
    leakage_gain = 10.0 ** (leakage_db / 20.0)
    delay = round(delay_ms * sample_rate / 1000.0)
    anomaly = _anomaly_waveform(
        anomaly_family, len(carrier), sample_rate, severity=severity, phase=phase
    )
    delayed_anomaly = _delay(anomaly, delay)
    anomaly_near = normal_near + anomaly
    anomaly_far = normal_far + leakage_gain * anomaly_far_ratio * delayed_anomaly

    near_floor = noise_floor_spectrum(profile_near, sample_rate, config.stft, config.refsub)[None, :]
    far_floor = noise_floor_spectrum(profile_far, sample_rate, config.stft, config.refsub)[None, :]
    transfer = estimate_noise_transfer(near_floor, far_floor, config.stft, config.refsub)
    enhanced_normal = apply_reference_subtraction(
        normal_near, normal_far, sample_rate, transfer, config.stft, config.refsub
    )
    enhanced_anomaly = apply_reference_subtraction(
        anomaly_near, anomaly_far, sample_rate, transfer, config.stft, config.refsub
    )
    diagnostics = diagnose_reference_pair(
        normal_near,
        normal_far,
        enhanced_normal,
        sample_rate,
        transfer,
        config.stft,
        config.refsub,
    )
    enhanced_anomaly_component = enhanced_anomaly - enhanced_normal
    retention = float(
        np.sum(enhanced_anomaly_component**2) / max(float(np.sum(anomaly**2)), config.stft.eps)
    )
    before_error = float(np.mean((normal_near - carrier) ** 2))
    after_error = float(np.mean((enhanced_normal - carrier) ** 2))
    true_noise_reduction = 10.0 * math.log10(
        max(before_error, config.stft.eps) / max(after_error, config.stft.eps)
    )
    anomaly_loss = float(np.clip(1.0 - retention, 0.0, 1.0))
    is_safe = bool(
        retention >= config.simulation.safe_retention_min
        and true_noise_reduction >= config.simulation.safe_noise_reduction_min_db
    )
    return {
        "snr_db": snr_db,
        "leakage_db": leakage_db,
        "coherence": coherence,
        "delay_ms": delay_ms,
        "gain_db": gain_db,
        "anomaly_far_ratio": anomaly_far_ratio,
        "anomaly_family": anomaly_family,
        "severity": severity,
        "leakage_index": diagnostics.leakage_index,
        "transfer_instability": diagnostics.transfer_instability,
        "spectral_drift": diagnostics.spectral_drift,
        "estimated_noise_reduction_db": diagnostics.noise_reduction_db,
        "risk_score": diagnostics.risk_score,
        "anomaly_retention": retention,
        "anomaly_loss": anomaly_loss,
        "true_noise_reduction_db": true_noise_reduction,
        "is_safe": is_safe,
    }


def _normal_pair(
    carrier: FloatArray,
    noise: FloatArray,
    *,
    snr_db: float,
    leakage_db: float,
    coherence: float,
    delay_ms: float,
    gain_db: float,
    sample_rate: int,
) -> tuple[FloatArray, FloatArray]:
    """Mix one normal pair for either profile fitting or held-out diagnosis."""
    shared_noise = coherence * noise + math.sqrt(max(1.0 - coherence**2, 0.0)) * np.roll(
        noise, len(noise) // 3
    )
    noise_scale = 10.0 ** (-snr_db / 20.0)
    leakage_gain = 10.0 ** (leakage_db / 20.0)
    far_gain = 10.0 ** (gain_db / 20.0)
    delay = round(delay_ms * sample_rate / 1000.0)
    normal_near = carrier + noise_scale * noise
    normal_far = far_gain * shared_noise + leakage_gain * _delay(carrier, delay)
    return normal_near, normal_far


def _prepare_source(
    source: StereoSource, config: ReferenceSafetyExperimentConfig
) -> tuple[FloatArray, FloatArray]:
    near, far, sample_rate = source
    near_values = np.asarray(near, dtype=np.float64)
    far_values = np.asarray(far, dtype=np.float64)
    if near_values.ndim != 1 or far_values.ndim != 1 or near_values.shape != far_values.shape:
        raise ValueError("Synthetic source clips must be equal-length mono near/far arrays")
    target_rate = config.simulation.sample_rate
    if sample_rate != target_rate:
        divisor = math.gcd(sample_rate, target_rate)
        near_values = scipy_signal.resample_poly(
            near_values, target_rate // divisor, sample_rate // divisor
        )
        far_values = scipy_signal.resample_poly(
            far_values, target_rate // divisor, sample_rate // divisor
        )
    length = round(config.simulation.duration_seconds * target_rate)
    return _fit_length(near_values, length), _fit_length(far_values, length)


def _anomaly_waveform(
    family: str, length: int, sample_rate: int, *, severity: float, phase: float
) -> FloatArray:
    time = np.arange(length, dtype=np.float64) / sample_rate
    if family == "impulsive":
        period = max(int(sample_rate / (20.0 + 80.0 * severity)), 1)
        values = np.zeros(length, dtype=np.float64)
        values[np.arange(0, length, period)] = 1.0
        kernel = np.exp(-np.arange(max(sample_rate // 200, 2)) / max(sample_rate / 2000.0, 1.0))
        values = np.convolve(values, kernel, mode="same")
    elif family == "sideband":
        carrier = 400.0 + 1200.0 * severity
        modulation = 8.0 + 32.0 * severity
        values = np.sin(2.0 * np.pi * carrier * time + phase) * (
            1.0 + 0.7 * np.sin(2.0 * np.pi * modulation * time)
        )
    elif family == "rubbing":
        values = np.tanh(3.0 * np.sin(2.0 * np.pi * (80.0 + 200.0 * severity) * time + phase))
        values *= 0.6 + 0.4 * np.sin(2.0 * np.pi * 3.0 * time) ** 2
    elif family == "speed_drift":
        start = 150.0 + 300.0 * severity
        values = scipy_signal.chirp(time, f0=start, f1=1.6 * start, t1=time[-1], method="linear")
    else:
        raise ValueError(f"Unsupported anomaly family: {family}")
    return np.asarray(severity * _unit_rms(values), dtype=np.float64)


def _fit_length(values: FloatArray, length: int) -> FloatArray:
    if not len(values):
        raise ValueError("Synthetic source waveform is empty")
    if len(values) >= length:
        return np.asarray(values[:length], dtype=np.float64)
    repeats = math.ceil(length / len(values))
    return np.asarray(np.tile(values, repeats)[:length], dtype=np.float64)


def _unit_rms(values: NDArray[np.floating]) -> FloatArray:
    waveform = np.asarray(values, dtype=np.float64)
    waveform = waveform - np.mean(waveform)
    rms = math.sqrt(float(np.mean(waveform**2)))
    return waveform / max(rms, 1.0e-8)


def _delay(values: FloatArray, samples: int) -> FloatArray:
    if samples <= 0:
        return values.copy()
    delayed = np.zeros_like(values)
    if samples < len(values):
        delayed[samples:] = values[:-samples]
    return delayed


def _wilson_upper(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return 1.0
    probability = successes / trials
    denominator = 1.0 + z**2 / trials
    center = probability + z**2 / (2.0 * trials)
    radius = z * math.sqrt(probability * (1.0 - probability) / trials + z**2 / (4.0 * trials**2))
    return float(np.clip((center + radius) / denominator, 0.0, 1.0))


def _validate_case_frame(frame: pd.DataFrame) -> None:
    required = {
        "risk_score",
        "estimated_noise_reduction_db",
        "is_safe",
        "anomaly_loss",
    }
    missing = sorted(required.difference(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"Synthetic case frame is empty or missing columns: {', '.join(missing)}")
