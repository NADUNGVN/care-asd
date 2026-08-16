"""Staged five-seed confirmatory evaluation for FP-NAA C1/C2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from care_asd.evaluation.dcase2026_metrics import (
    calculate_dcase2026_official_metrics,
    read_official_score,
)
from care_asd.evaluation.fp_naa_candidate import (
    CANDIDATES,
    _atomic_csv,
    _atomic_json,
    _cuda_device,
    _ensure_contract,
    _exclude_machine,
    _load_or_train_model,
    _machine_scores,
    _preload_base_store,
    _preload_training_arrays,
    _score_adapter,
    _sha256,
    _validate_caches,
    _write_progress,
    _write_retention_diagnostics,
)
from care_asd.evaluation.fp_naa_statistics import write_exact_official_paired_bootstrap
from care_asd.evaluation.paired_bootstrap import write_seed_ensemble_scores
from care_asd.fp_naa_config import FPNAAConfig, load_fp_naa_config
from care_asd.models.fp_naa_adapter import trainable_parameter_count


@dataclass(frozen=True)
class FPNaaConfirmatoryResult:
    output_directory: Path
    summary_path: Path
    gate_path: Path
    core_gate_passed: bool


@dataclass(frozen=True)
class FPNaaConfirmatoryLomoResult:
    output_directory: Path
    summary_path: Path
    gate_path: Path
    gate_passed: bool


def run_fp_naa_confirmatory(
    *,
    base_cache_directory: str | Path,
    augmentation_cache_directory: str | Path,
    c0_score_path: str | Path,
    screening_directory: str | Path,
    lomo_directory: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config_path: str | Path,
    experiment_id: str,
    device: str = "cuda",
    preload_workers: int | None = None,
) -> FPNaaConfirmatoryResult:
    """Reuse screening seeds, train two new seeds, ensemble, and bootstrap the exact score."""
    base_cache = Path(base_cache_directory).resolve()
    augmentation_cache = Path(augmentation_cache_directory).resolve()
    c0_scores = Path(c0_score_path).resolve()
    screening = Path(screening_directory).resolve()
    lomo = Path(lomo_directory).resolve()
    output = Path(output_directory).resolve()
    checkpoints = Path(checkpoint_directory).resolve()
    config = load_fp_naa_config(Path(config_path).resolve())
    workers = config.training.workers if preload_workers is None else preload_workers
    if not 1 <= workers <= 16:
        raise ValueError("preload_workers must be in [1, 16]")
    screening_gate = screening / "gate.json"
    lomo_gate = lomo / "gate.json"
    if not screening_gate.is_file() or not lomo_gate.is_file():
        raise FileNotFoundError("Confirmatory run requires completed screening and LOMO gates")
    screening_payload = json.loads(screening_gate.read_text(encoding="utf-8"))
    lomo_payload = json.loads(lomo_gate.read_text(encoding="utf-8"))
    if not bool(screening_payload.get("checks", {}).get("core_screening")) or not bool(
        lomo_payload.get("passed")
    ):
        raise ValueError("Confirmatory run is blocked until both G2 core and LOMO pass")
    if not c0_scores.is_file():
        raise FileNotFoundError(f"C0 score file not found: {c0_scores}")
    _validate_caches(base_cache, augmentation_cache, config)
    torch_device = _cuda_device(device)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    _ensure_contract(
        output,
        {
            "schema_version": 1,
            "kind": "fp_naa_confirmatory",
            "experiment_id": experiment_id,
            "config": config.model_dump(mode="json"),
            "base_cache_metadata_sha256": _sha256(base_cache / "cache.json"),
            "augmentation_cache_metadata_sha256": _sha256(augmentation_cache / "cache.json"),
            "c0_scores_sha256": _sha256(c0_scores),
            "screening_gate_sha256": _sha256(screening_gate),
            "lomo_gate_sha256": _sha256(lomo_gate),
        },
    )
    summary_path = output / "confirmatory_summary.csv"
    gate_path = output / "gate.json"
    if summary_path.is_file() and gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        return FPNaaConfirmatoryResult(
            output,
            summary_path,
            gate_path,
            bool(gate["checks"]["core_confirmatory"]),
        )
    screening_seeds = set(config.training.screening_seeds)
    confirmatory_seeds = config.training.confirmatory_seeds
    if not screening_seeds.issubset(confirmatory_seeds) or len(confirmatory_seeds) != 5:
        raise ValueError("Confirmatory seeds must contain all three screening seeds and total five")
    new_seeds = [seed for seed in confirmatory_seeds if seed not in screening_seeds]
    if len(new_seeds) != 2:
        raise ValueError("Confirmatory protocol requires exactly two new seeds")
    _validate_screening_artifacts(screening, config.training.screening_seeds)
    _write_progress(output, stage="preload_base", completed=0, total=len(new_seeds) * 2)
    base_store = _preload_base_store(base_cache, workers=workers)
    _write_progress(output, stage="preload_augmentation", completed=0, total=len(new_seeds) * 2)
    training = _preload_training_arrays(augmentation_cache, base_store, workers=workers)
    rows = _screening_rows(screening, config.training.screening_seeds)
    total_new_runs = len(new_seeds) * len(CANDIDATES)
    run_number = 0
    for seed in new_seeds:
        for candidate in CANDIDATES:
            run_number += 1
            variant = output / f"seed{seed}" / candidate
            variant.mkdir(parents=True, exist_ok=True)
            checkpoint = checkpoints / f"seed{seed}" / f"{candidate}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            _write_progress(
                output,
                stage=f"train:{seed}:{candidate}",
                completed=run_number - 1,
                total=total_new_runs,
            )
            model = _load_or_train_model(
                checkpoint=checkpoint,
                history_path=variant / "training_history.csv",
                arrays=training,
                candidate=candidate,
                seed=seed,
                config=config,
                device=torch_device,
                progress_output=output,
                run_number=run_number,
                total_runs=total_new_runs,
            )
            retention_path = variant / "retention.csv"
            if not retention_path.is_file():
                _write_retention_diagnostics(
                    model=model,
                    arrays=training,
                    augmentation_cache=augmentation_cache,
                    base_store=base_store,
                    output_path=retention_path,
                    config=config,
                    device=torch_device,
                    workers=workers,
                )
            score_path = variant / "scores.csv"
            metrics_path = variant / "metrics.json"
            if metrics_path.is_file() and not score_path.is_file():
                raise ValueError(f"Metrics exist without scores: {variant}")
            if score_path.is_file() and not metrics_path.is_file():
                calculate_dcase2026_official_metrics(score_path, metrics_path)
            elif not score_path.is_file():
                _score_adapter(
                    model=model,
                    base_store=base_store,
                    score_path=score_path,
                    metrics_path=metrics_path,
                    model_id=f"fp_naa_confirmatory_{candidate}_seed{seed}",
                    experiment_id=experiment_id,
                    config=config,
                    device=torch_device,
                )
            rows.append(
                _summary_row(
                    seed=seed,
                    candidate=candidate,
                    metrics_path=metrics_path,
                    retention_path=retention_path,
                    source="confirmatory",
                    trainable_parameters=trainable_parameter_count(model),
                )
            )
            _write_progress(
                output,
                stage=f"complete:{seed}:{candidate}",
                completed=run_number,
                total=total_new_runs,
            )
    summary = pd.DataFrame(rows).sort_values(["seed", "candidate"], kind="stable")
    _atomic_csv(summary_path, summary)
    ensemble_paths: dict[str, Path] = {}
    metrics_paths: dict[str, Path] = {}
    for candidate in CANDIDATES:
        seed_scores = [
            _seed_score_path(screening, output, seed, candidate, screening_seeds)
            for seed in confirmatory_seeds
        ]
        ensemble_dir = output / f"{candidate}_ensemble"
        ensemble_dir.mkdir(exist_ok=True)
        ensemble_path = ensemble_dir / "scores.csv"
        metrics_path = ensemble_dir / "metrics.json"
        if not ensemble_path.is_file():
            write_seed_ensemble_scores(
                score_paths=seed_scores,
                output_path=ensemble_path,
                model_id=f"fp_naa_{candidate}_five_seed_ensemble",
                experiment_id=experiment_id,
            )
        if not metrics_path.is_file():
            calculate_dcase2026_official_metrics(ensemble_path, metrics_path)
        ensemble_paths[candidate] = ensemble_path
        metrics_paths[candidate] = metrics_path
    bootstrap_path = output / "exact_paired_bootstrap_c2_vs_c1.json"
    if not bootstrap_path.is_file():
        write_exact_official_paired_bootstrap(
            reference_scores=ensemble_paths["c1_mse"],
            candidate_scores=ensemble_paths["c2_fault_preserving"],
            output_path=bootstrap_path,
            iterations=config.gates.bootstrap_iterations,
            seed=2608,
        )
    gate = _confirmatory_gate(
        summary=summary,
        c0_scores=c0_scores,
        c1_metrics=metrics_paths["c1_mse"],
        c2_metrics=metrics_paths["c2_fault_preserving"],
        bootstrap_path=bootstrap_path,
        config=config,
    )
    _atomic_json(gate_path, gate)
    _atomic_json(
        output / "run.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_cache": str(base_cache),
            "augmentation_cache": str(augmentation_cache),
            "c0_scores": str(c0_scores),
            "screening_directory": str(screening),
            "lomo_directory": str(lomo),
            "config": config.model_dump(mode="json"),
            "device": str(torch_device),
            "preload_workers": workers,
            "reused_seeds": sorted(screening_seeds),
            "new_seeds": new_seeds,
        },
    )
    _write_progress(output, stage="complete", completed=total_new_runs, total=total_new_runs)
    return FPNaaConfirmatoryResult(
        output,
        summary_path,
        gate_path,
        bool(gate["checks"]["core_confirmatory"]),
    )


def run_fp_naa_confirmatory_lomo(
    *,
    base_cache_directory: str | Path,
    augmentation_cache_directory: str | Path,
    screening_lomo_directory: str | Path,
    confirmatory_directory: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config_path: str | Path,
    experiment_id: str,
    device: str = "cuda",
    preload_workers: int | None = None,
) -> FPNaaConfirmatoryLomoResult:
    """Extend the passed three-seed LOMO experiment with the two new frozen seeds."""
    base_cache = Path(base_cache_directory).resolve()
    augmentation_cache = Path(augmentation_cache_directory).resolve()
    screening_lomo = Path(screening_lomo_directory).resolve()
    confirmatory = Path(confirmatory_directory).resolve()
    output = Path(output_directory).resolve()
    checkpoints = Path(checkpoint_directory).resolve()
    config = load_fp_naa_config(Path(config_path).resolve())
    workers = config.training.workers if preload_workers is None else preload_workers
    if not 1 <= workers <= 16:
        raise ValueError("preload_workers must be in [1, 16]")
    screening_lomo_gate = screening_lomo / "gate.json"
    confirmatory_gate = confirmatory / "gate.json"
    if not screening_lomo_gate.is_file() or not confirmatory_gate.is_file():
        raise FileNotFoundError("Confirmatory LOMO requires both prior gate artifacts")
    screening_gate_payload = json.loads(screening_lomo_gate.read_text(encoding="utf-8"))
    confirmatory_gate_payload = json.loads(confirmatory_gate.read_text(encoding="utf-8"))
    if not bool(screening_gate_payload.get("passed")):
        raise ValueError("Confirmatory LOMO is blocked because screening LOMO failed")
    if not bool(confirmatory_gate_payload.get("checks", {}).get("core_confirmatory")):
        raise ValueError("Confirmatory LOMO is blocked because the five-seed core gate failed")
    _validate_caches(base_cache, augmentation_cache, config)
    torch_device = _cuda_device(device)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    _ensure_contract(
        output,
        {
            "schema_version": 1,
            "kind": "fp_naa_confirmatory_lomo",
            "experiment_id": experiment_id,
            "config": config.model_dump(mode="json"),
            "base_cache_metadata_sha256": _sha256(base_cache / "cache.json"),
            "augmentation_cache_metadata_sha256": _sha256(augmentation_cache / "cache.json"),
            "screening_lomo_gate_sha256": _sha256(screening_lomo_gate),
            "confirmatory_gate_sha256": _sha256(confirmatory_gate),
        },
    )
    summary_path = output / "confirmatory_lomo_summary.csv"
    gate_path = output / "gate.json"
    if summary_path.is_file() and gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        return FPNaaConfirmatoryLomoResult(
            output,
            summary_path,
            gate_path,
            bool(gate["passed"]),
        )
    old_summary_path = screening_lomo / "lomo_summary.csv"
    if not old_summary_path.is_file():
        raise FileNotFoundError(f"Screening LOMO summary not found: {old_summary_path}")
    old_summary = pd.read_csv(old_summary_path)
    _validate_lomo_summary(
        old_summary,
        expected_seeds=set(config.training.screening_seeds),
        expected_machines=None,
    )
    screening_seeds = set(config.training.screening_seeds)
    confirmatory_seeds = config.training.confirmatory_seeds
    if not screening_seeds.issubset(confirmatory_seeds) or len(confirmatory_seeds) != 5:
        raise ValueError("Confirmatory seeds must contain all three screening seeds and total five")
    new_seeds = [seed for seed in confirmatory_seeds if seed not in screening_seeds]
    if len(new_seeds) != 2:
        raise ValueError("Confirmatory LOMO protocol requires exactly two new seeds")
    _write_progress(output, stage="preload_base", completed=0, total=len(new_seeds) * 14)
    base_store = _preload_base_store(base_cache, workers=workers)
    _write_progress(output, stage="preload_augmentation", completed=0, total=len(new_seeds) * 14)
    training = _preload_training_arrays(augmentation_cache, base_store, workers=workers)
    machines = sorted(training.frame["machine_type"].astype(str).unique())
    if len(machines) != 7:
        raise ValueError(
            f"Preregistered DCASE development LOMO requires 7 machines, found {len(machines)}"
        )
    _validate_lomo_summary(
        old_summary,
        expected_seeds=screening_seeds,
        expected_machines=set(machines),
    )
    reused = old_summary.copy()
    reused["source"] = "screening_reuse"
    rows = reused.to_dict(orient="records")
    total_new_runs = len(machines) * len(new_seeds) * len(CANDIDATES)
    run_number = 0
    for heldout_machine in machines:
        fold_training = _exclude_machine(training, heldout_machine)
        for seed in new_seeds:
            for candidate in CANDIDATES:
                run_number += 1
                variant = output / heldout_machine / f"seed{seed}" / candidate
                variant.mkdir(parents=True, exist_ok=True)
                checkpoint = checkpoints / heldout_machine / f"seed{seed}" / f"{candidate}.pt"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                _write_progress(
                    output,
                    stage=f"lomo:{heldout_machine}:{seed}:{candidate}",
                    completed=run_number - 1,
                    total=total_new_runs,
                )
                model = _load_or_train_model(
                    checkpoint=checkpoint,
                    history_path=variant / "training_history.csv",
                    arrays=fold_training,
                    candidate=candidate,
                    seed=seed,
                    config=config,
                    device=torch_device,
                    progress_output=output,
                    run_number=run_number,
                    total_runs=total_new_runs,
                )
                score_path = variant / "scores.csv"
                metrics_path = variant / "metrics.json"
                if metrics_path.is_file() and not score_path.is_file():
                    raise ValueError(f"Metrics exist without scores: {variant}")
                if score_path.is_file() and not metrics_path.is_file():
                    calculate_dcase2026_official_metrics(score_path, metrics_path)
                elif not score_path.is_file():
                    _score_adapter(
                        model=model,
                        base_store=base_store,
                        score_path=score_path,
                        metrics_path=metrics_path,
                        model_id=(
                            f"fp_naa_confirmatory_lomo_{candidate}_{heldout_machine}_seed{seed}"
                        ),
                        experiment_id=experiment_id,
                        config=config,
                        device=torch_device,
                        machine_filter=heldout_machine,
                    )
                rows.append(
                    {
                        "heldout_machine": heldout_machine,
                        "seed": seed,
                        "candidate": candidate,
                        "official_score": read_official_score(metrics_path),
                        "official_score_percent": 100.0 * read_official_score(metrics_path),
                        "train_machines": len(machines) - 1,
                        "train_clips": len(fold_training.frame),
                        "source": "confirmatory",
                    }
                )
                _write_progress(
                    output,
                    stage=f"complete:{heldout_machine}:{seed}:{candidate}",
                    completed=run_number,
                    total=total_new_runs,
                )
    summary = pd.DataFrame(rows).sort_values(
        ["heldout_machine", "seed", "candidate"], kind="stable"
    )
    _validate_lomo_summary(
        summary,
        expected_seeds=set(confirmatory_seeds),
        expected_machines=set(machines),
    )
    _atomic_csv(summary_path, summary)
    pivot = summary.pivot_table(
        index="heldout_machine",
        columns="candidate",
        values="official_score",
        aggfunc="mean",
    )
    pivot["delta_c2_minus_c1"] = pivot["c2_fault_preserving"] - pivot["c1_mse"]
    positive_folds = int((pivot["delta_c2_minus_c1"] > 0.0).sum())
    passed = positive_folds >= config.gates.confirmatory_positive_lomo_folds_minimum
    gate: dict[str, object] = {
        "schema_version": 1,
        "gate": "G3_confirmatory_lomo",
        "confirmatory_core_passed": True,
        "fold_delta_c2_minus_c1": {
            str(machine): float(value) for machine, value in pivot["delta_c2_minus_c1"].items()
        },
        "positive_folds": positive_folds,
        "minimum_positive_folds": config.gates.confirmatory_positive_lomo_folds_minimum,
        "passed": passed,
    }
    _atomic_json(gate_path, gate)
    _atomic_csv(output / "confirmatory_lomo_fold_means.csv", pivot.reset_index())
    _atomic_json(
        output / "run.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "screening_lomo_directory": str(screening_lomo),
            "confirmatory_directory": str(confirmatory),
            "reused_seeds": sorted(screening_seeds),
            "new_seeds": new_seeds,
            "device": str(torch_device),
            "preload_workers": workers,
        },
    )
    _write_progress(output, stage="complete", completed=total_new_runs, total=total_new_runs)
    return FPNaaConfirmatoryLomoResult(output, summary_path, gate_path, passed)


def _validate_lomo_summary(
    summary: pd.DataFrame,
    *,
    expected_seeds: set[int],
    expected_machines: set[str] | None,
) -> None:
    required = {"heldout_machine", "seed", "candidate", "official_score"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"LOMO summary is missing columns: {', '.join(missing)}")
    if summary.duplicated(["heldout_machine", "seed", "candidate"]).any():
        raise ValueError("LOMO summary contains duplicate fold/seed/candidate rows")
    if set(summary["seed"].astype(int)) != expected_seeds:
        raise ValueError("LOMO summary seed coverage does not match the frozen protocol")
    if set(summary["candidate"].astype(str)) != set(CANDIDATES):
        raise ValueError("LOMO summary candidate coverage does not match the frozen protocol")
    machines = set(summary["heldout_machine"].astype(str))
    if expected_machines is not None and machines != expected_machines:
        raise ValueError("LOMO summary machine coverage does not match the development cache")
    expected_rows = len(machines) * len(expected_seeds) * len(CANDIDATES)
    if len(summary) != expected_rows:
        raise ValueError("LOMO summary does not contain the complete factorial design")


def _validate_screening_artifacts(screening: Path, seeds: list[int]) -> None:
    for seed in seeds:
        for candidate in CANDIDATES:
            variant = screening / f"seed{seed}" / candidate
            for name in ("scores.csv", "metrics.json", "retention.csv"):
                if not (variant / name).is_file():
                    raise FileNotFoundError(f"Screening artifact missing: {variant / name}")


def _screening_rows(screening: Path, seeds: list[int]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for candidate in CANDIDATES:
            variant = screening / f"seed{seed}" / candidate
            rows.append(
                _summary_row(
                    seed=seed,
                    candidate=candidate,
                    metrics_path=variant / "metrics.json",
                    retention_path=variant / "retention.csv",
                    source="screening_reuse",
                    trainable_parameters=None,
                )
            )
    return rows


def _summary_row(
    *,
    seed: int,
    candidate: str,
    metrics_path: Path,
    retention_path: Path,
    source: str,
    trainable_parameters: int | None,
) -> dict[str, object]:
    retention = pd.read_csv(retention_path)
    in_support = retention.loc[retention["fault_set"] == "in_support", "retention"]
    heldout = retention.loc[retention["fault_set"] == "heldout", "retention"]
    if in_support.empty or heldout.empty:
        raise ValueError(f"Retention diagnostics are incomplete: {retention_path}")
    score = read_official_score(metrics_path)
    return {
        "seed": seed,
        "candidate": candidate,
        "source": source,
        "official_score": score,
        "official_score_percent": 100.0 * score,
        "in_support_retention_median": float(in_support.median()),
        "in_support_retention_q05": float(in_support.quantile(0.05)),
        "heldout_retention_median": float(heldout.median()),
        "heldout_retention_q05": float(heldout.quantile(0.05)),
        "trainable_parameters": trainable_parameters,
    }


def _seed_score_path(
    screening: Path,
    output: Path,
    seed: int,
    candidate: str,
    screening_seeds: set[int],
) -> Path:
    root = screening if seed in screening_seeds else output
    path = root / f"seed{seed}" / candidate / "scores.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Confirmatory seed score missing: {path}")
    return path


def _confirmatory_gate(
    *,
    summary: pd.DataFrame,
    c0_scores: Path,
    c1_metrics: Path,
    c2_metrics: Path,
    bootstrap_path: Path,
    config: FPNAAConfig,
) -> dict[str, object]:
    gates = config.gates
    c1_ensemble = read_official_score(c1_metrics)
    c2_ensemble = read_official_score(c2_metrics)
    pivot = summary.pivot(index="seed", columns="candidate", values="official_score")
    if set(pivot.index.astype(int)) != set(config.training.confirmatory_seeds) or set(
        pivot.columns.astype(str)
    ) != set(CANDIDATES):
        raise ValueError("Confirmatory summary does not cover the frozen seed/candidate design")
    seed_deltas = pivot["c2_fault_preserving"] - pivot["c1_mse"]
    mean_delta = float(seed_deltas.mean())
    c2_rows = summary.loc[summary["candidate"] == "c2_fault_preserving"]
    in_median = float(c2_rows["in_support_retention_median"].median())
    in_q05 = float(c2_rows["in_support_retention_q05"].min())
    heldout_median = float(c2_rows["heldout_retention_median"].median())
    heldout_q05 = float(c2_rows["heldout_retention_q05"].min())
    bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    observed = bootstrap["observed"]
    if (
        abs(float(observed["reference_official_score"]) - c1_ensemble) > 1.0e-12
        or abs(float(observed["candidate_official_score"]) - c2_ensemble) > 1.0e-12
    ):
        raise ValueError("Exact bootstrap and ensemble metric artifacts do not match")
    interval = bootstrap["bootstrap"]["delta_candidate_minus_reference"]
    c0_metrics_path = bootstrap_path.parent / "c0_metrics.json"
    if not c0_metrics_path.is_file():
        calculate_dcase2026_official_metrics(c0_scores, c0_metrics_path)
    c0_machines = _machine_scores(c0_metrics_path)
    c2_machines = _machine_scores(c2_metrics)
    if set(c0_machines) != set(c2_machines):
        raise ValueError("C0 and confirmatory C2 metrics cover different machines")
    machine_drops = {
        machine: c2_machines[machine] - value for machine, value in c0_machines.items()
    }
    worst_machine_drop = float(min(machine_drops.values()))
    checks = {
        "ensemble_score": c2_ensemble >= gates.confirmatory_minimum_ensemble_official_score,
        "mean_gain_over_c1": mean_delta >= gates.confirmatory_minimum_gain_over_c1,
        "bootstrap_ci_low": (
            float(interval["ci95_low"]) > gates.confirmatory_bootstrap_ci_low_minimum
        ),
        "worst_machine_drop": worst_machine_drop >= -gates.confirmatory_maximum_machine_drop,
        "in_support_retention_median": (in_median >= gates.fault_delta_retention_median_minimum),
        "in_support_retention_q05": in_q05 >= gates.fault_delta_retention_q05_minimum,
        "heldout_retention_median": (
            heldout_median >= gates.heldout_fault_delta_retention_median_minimum
        ),
        "heldout_retention_q05": (heldout_q05 >= gates.heldout_fault_delta_retention_q05_minimum),
    }
    core = all(checks.values())
    return {
        "schema_version": 1,
        "gate": "G3_confirmatory_core",
        "scores": {
            "c1_ensemble": c1_ensemble,
            "c2_ensemble": c2_ensemble,
            "c2_minus_c1_ensemble": c2_ensemble - c1_ensemble,
            "mean_seed_delta_c2_minus_c1": mean_delta,
            "seed_deltas": {str(seed): float(value) for seed, value in seed_deltas.items()},
        },
        "bootstrap_delta_c2_minus_c1": interval,
        "machine_delta_c2_ensemble_minus_c0": machine_drops,
        "retention": {
            "in_support_median_across_seeds": in_median,
            "in_support_worst_seed_q05": in_q05,
            "heldout_median_across_seeds": heldout_median,
            "heldout_worst_seed_q05": heldout_q05,
        },
        "checks": {
            **checks,
            "core_confirmatory": core,
            "confirmatory_lomo": None,
            "reference_safety": None,
        },
        "passed": False,
        "note": "passed remains false until confirmatory LOMO and reference-safety gates pass",
    }
