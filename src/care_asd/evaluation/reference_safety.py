"""SAFE-REF development, freeze, evaluation, and official-scoring contracts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import hmean
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from care_asd.data.reference_safety_cache import load_reference_vectors
from care_asd.evaluation.official_baseline import (
    SCORE_COLUMNS,
    calculate_development_auc_metrics,
)
from care_asd.evaluation.paired_bootstrap import write_paired_bootstrap_comparison
from care_asd.models.official_baseline import OFFICIAL_EVALUATOR_COMMIT
from care_asd.models.official_compatible import OfficialCompatibleAutoencoder
from care_asd.reference_safety_config import (
    ReferenceSafetyExperimentConfig,
    ReferenceSafetyPolicy,
    load_reference_safety_policy,
    reference_safety_config_hash,
)

SystemName = Literal["near", "unconditional_refsub", "safe_ref"]
StageName = Literal["screening", "replication"]
SYSTEMS: tuple[SystemName, ...] = ("near", "unconditional_refsub", "safe_ref")


@dataclass(frozen=True)
class ReferenceSafetyDevelopmentResult:
    """Immutable development evidence paths."""

    output_directory: Path
    summary_path: Path
    decisions_path: Path
    gate_path: Path
    passed: bool


@dataclass(frozen=True)
class ReferenceSafetyEvaluationResult:
    """Evaluation scores written without accessing ground truth."""

    output_directory: Path
    complete_path: Path
    official_scores_directory: Path


def run_reference_safety_development(
    *,
    cache_directory: str | Path,
    policy_path: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config: ReferenceSafetyExperimentConfig,
    stage: StageName = "screening",
) -> ReferenceSafetyDevelopmentResult:
    """Train capacity-matched near/RefSub AEs and evaluate a fixed SAFE-REF policy."""
    _validate_official_training(config)
    cache, index, profiles = _load_cache(cache_directory)
    train = index.loc[
        index["dataset_split"].isin({"dev_train"}) & (index["condition"] == "normal")
    ].copy()
    test = index.loc[index["dataset_split"] == "dev_test"].copy()
    if train.empty or test.empty:
        raise ValueError("Development SAFE-REF requires dev_train and dev_test cache rows")
    policy = load_reference_safety_policy(policy_path)
    decisions = _build_decisions(profiles, policy)
    output = Path(output_directory)
    checkpoints = Path(checkpoint_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite SAFE-REF development output: {output}")
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    seeds = (
        config.training.screening_seeds
        if stage == "screening"
        else config.training.replication_seeds
    )
    seed_scores: dict[SystemName, list[Path]] = {system: [] for system in SYSTEMS}
    threshold_records: list[dict[str, object]] = []
    for seed in seeds:
        seed_directory = output / f"seed_{seed}"
        results, thresholds = _run_seed(
            cache=cache,
            train=train,
            test=test,
            decisions=decisions,
            seed=seed,
            output_directory=seed_directory,
            checkpoint_directory=checkpoints / f"seed_{seed}",
            config=config,
            calculate_metrics=True,
        )
        for system, path in results.items():
            seed_scores[system].append(path)
        threshold_records.extend(thresholds)
    pd.DataFrame.from_records(threshold_records).to_csv(output / "thresholds.csv", index=False)
    decisions_path = output / "decisions.csv"
    decisions.to_csv(decisions_path, index=False)
    ensemble_paths: dict[SystemName, Path] = {}
    metrics_paths: dict[SystemName, Path] = {}
    for system in SYSTEMS:
        system_directory = output / system
        system_directory.mkdir()
        ensemble_path = system_directory / "scores.csv"
        _write_score_ensemble(seed_scores[system], ensemble_path, system, config.experiment_id)
        metrics_path = system_directory / "metrics.json"
        calculate_development_auc_metrics(ensemble_path, metrics_path)
        ensemble_paths[system] = ensemble_path
        metrics_paths[system] = metrics_path
    summary = _development_summary(metrics_paths)
    summary_path = output / "summary.csv"
    summary.to_csv(summary_path, index=False)
    gate = _development_gate(summary, decisions, config)
    gate_path = output / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_paired_bootstrap_comparison(
        reference_scores=ensemble_paths["near"],
        candidate_scores=ensemble_paths["unconditional_refsub"],
        output_path=output / "paired_bootstrap_unconditional_vs_near.json",
        iterations=5000,
        seed=2026,
    )
    write_paired_bootstrap_comparison(
        reference_scores=ensemble_paths["near"],
        candidate_scores=ensemble_paths["safe_ref"],
        output_path=output / "paired_bootstrap_safe_ref_vs_near.json",
        iterations=5000,
        seed=2026,
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": stage,
                "seeds": list(seeds),
                "cache": str(cache),
                "policy": str(policy_path),
                "config": config.model_dump(),
                "capacity_contract": "two independent official-compatible AEs per group and seed",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return ReferenceSafetyDevelopmentResult(
        output, summary_path, decisions_path, gate_path, bool(gate["passed"])
    )


def create_reference_safety_freeze(
    *,
    config_path: str | Path,
    policy_path: str | Path,
    development_gate_path: str | Path,
    development_manifest_path: str | Path,
    output_path: str | Path,
    config: ReferenceSafetyExperimentConfig,
) -> Path:
    """Write a freeze manifest only after the replication development gate passes."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite freeze manifest: {output}")
    gate_path = Path(development_gate_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not bool(gate.get("passed")):
        raise ValueError("Cannot freeze a SAFE-REF configuration whose development gate failed")
    payload = {
        "schema_version": 1,
        "freeze_id": f"safe_ref_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "config_sha256": _sha256_file(config_path),
        "validated_config_sha256": reference_safety_config_hash(config),
        "policy_sha256": _sha256_file(policy_path),
        "development_gate_sha256": _sha256_file(gate_path),
        "development_manifest_sha256": _sha256_file(development_manifest_path),
        "evaluation_seeds": list(config.training.replication_seeds),
        "systems": list(SYSTEMS),
        "ground_truth_access_during_scoring": False,
        "official_evaluator_commit": OFFICIAL_EVALUATOR_COMMIT,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    return output


def run_reference_safety_evaluation(
    *,
    cache_directory: str | Path,
    policy_path: str | Path,
    freeze_path: str | Path,
    config_path: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config: ReferenceSafetyExperimentConfig,
) -> ReferenceSafetyEvaluationResult:
    """Generate frozen evaluation scores without accepting any ground-truth path."""
    _validate_official_training(config)
    freeze = _load_and_validate_freeze(freeze_path, config_path, policy_path, config)
    cache, index, profiles = _load_cache(cache_directory)
    train = index.loc[
        index["dataset_split"].isin({"add_train"}) & (index["condition"] == "normal")
    ].copy()
    test = index.loc[index["dataset_split"] == "eval_test"].copy()
    if train.empty or test.empty:
        raise ValueError("Evaluation SAFE-REF requires add_train and eval_test cache rows")
    if set(test["condition"]) != {"unknown"}:
        raise ValueError("Evaluation scoring refuses known normal/anomaly test labels")
    policy = load_reference_safety_policy(policy_path)
    decisions = _build_decisions(profiles, policy)
    output = Path(output_directory)
    checkpoints = Path(checkpoint_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite SAFE-REF evaluation output: {output}")
    output.mkdir(parents=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    frozen_seeds = freeze.get("evaluation_seeds")
    if not isinstance(frozen_seeds, list) or not frozen_seeds:
        raise ValueError("SAFE-REF freeze has no evaluation seeds")
    seeds = tuple(int(seed) for seed in frozen_seeds)
    seed_scores: dict[SystemName, list[Path]] = {system: [] for system in SYSTEMS}
    threshold_records: list[dict[str, object]] = []
    for seed in seeds:
        results, thresholds = _run_seed(
            cache=cache,
            train=train,
            test=test,
            decisions=decisions,
            seed=seed,
            output_directory=output / f"seed_{seed}",
            checkpoint_directory=checkpoints / f"seed_{seed}",
            config=config,
            calculate_metrics=False,
        )
        for system, path in results.items():
            seed_scores[system].append(path)
        threshold_records.extend(thresholds)
    thresholds = pd.DataFrame.from_records(threshold_records)
    thresholds.to_csv(output / "thresholds.csv", index=False)
    decisions.to_csv(output / "decisions.csv", index=False)
    official_root = output / "official_scores"
    official_root.mkdir()
    written_files: list[Path] = []
    for system in SYSTEMS:
        system_directory = output / system
        system_directory.mkdir()
        ensemble_path = system_directory / "scores.csv"
        frame = _write_score_ensemble(
            seed_scores[system], ensemble_path, system, config.experiment_id
        )
        system_official = official_root / system
        system_official.mkdir()
        written_files.extend(
            _write_official_score_files(
                scores=frame,
                index=test,
                thresholds=thresholds.loc[thresholds["system"] == system],
                output_directory=system_official,
            )
        )
    hashes = {
        str(path.relative_to(output)).replace("\\", "/"): _sha256_file(path)
        for path in written_files
    }
    complete = {
        "schema_version": 1,
        "completed_utc": datetime.now(UTC).isoformat(),
        "freeze_sha256": _sha256_file(freeze_path),
        "policy_sha256": _sha256_file(policy_path),
        "ground_truth_accessed": False,
        "score_files": hashes,
    }
    complete_path = output / "score_complete.json"
    complete_path.write_text(
        json.dumps(complete, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return ReferenceSafetyEvaluationResult(output, complete_path, official_root)


def run_official_reference_safety_scoring(
    *,
    evaluation_output_directory: str | Path,
    evaluator_directory: str | Path,
    output_directory: str | Path,
) -> Path:
    """Run the separately pinned evaluator only after score-completion validation."""
    evaluation_output = Path(evaluation_output_directory)
    evaluator = Path(evaluator_directory)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite official evaluator output: {output}")
    complete_path = evaluation_output / "score_complete.json"
    if not complete_path.is_file():
        raise FileNotFoundError("Evaluation output lacks score_complete.json")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    score_root = evaluation_output / "official_scores"
    for relative, expected_hash in complete.get("score_files", {}).items():
        path = evaluation_output / str(relative)
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"Evaluation score hash mismatch: {relative}")
    required = [
        evaluator / "dcase2026_task2_evaluator.py",
        evaluator / "ground_truth_data",
        evaluator / "ground_truth_domain",
        evaluator / "ground_truth_attributes",
    ]
    if not all(path.exists() for path in required):
        raise FileNotFoundError("Pinned evaluator or its ground-truth directories are incomplete")
    commit = subprocess.run(
        ["git", "-C", str(evaluator), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != OFFICIAL_EVALUATOR_COMMIT:
        raise ValueError(
            f"Evaluator commit mismatch: expected {OFFICIAL_EVALUATOR_COMMIT}, got {commit}"
        )
    output.mkdir(parents=True)
    result_directory = output / "teams_result"
    command = [
        sys.executable,
        "dcase2026_task2_evaluator.py",
        "--teams_root_dir",
        str(score_root.resolve()),
        "--result_dir",
        str(result_directory.resolve()),
        "--dir_depth",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=evaluator,
        env={**os.environ, "LD_LIBRARY_PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    (output / "evaluator.log").write_text(
        completed.stdout + "\nSTDERR:\n" + completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Official evaluator failed with status {completed.returncode}")
    results = sorted(result_directory.glob("*_result.csv"))
    if len(results) != len(SYSTEMS):
        raise RuntimeError(
            f"Official evaluator wrote {len(results)} results; expected {len(SYSTEMS)}"
        )
    return result_directory


def _run_seed(
    *,
    cache: Path,
    train: pd.DataFrame,
    test: pd.DataFrame,
    decisions: pd.DataFrame,
    seed: int,
    output_directory: Path,
    checkpoint_directory: Path,
    config: ReferenceSafetyExperimentConfig,
    calculate_metrics: bool,
) -> tuple[dict[SystemName, Path], list[dict[str, object]]]:
    output_directory.mkdir(parents=True)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.training.device)
    records: dict[SystemName, list[dict[str, str | float]]] = {system: [] for system in SYSTEMS}
    thresholds: list[dict[str, object]] = []
    decision_lookup = decisions.set_index(["machine_type", "section"])["decision"]
    for group, group_train in train.groupby(["machine_type", "section"], sort=True):
        machine, section = str(group[0]), str(group[1])
        group_test = test.loc[
            (test["machine_type"] == machine) & (test["section"] == section)
        ].sort_values("file_id", kind="stable")
        if group_test.empty:
            raise ValueError(f"No test rows for training group {machine}/{section}")
        models: dict[str, nn.Module] = {}
        for view in ("near", "refsub"):
            _set_seed(seed)
            vectors = _concatenate_vectors(cache, group_train, view)
            model, final_loss = _fit_model(vectors, config, device)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "machine_type": machine,
                    "section": section,
                    "seed": seed,
                    "view": view,
                    "final_loss": final_loss,
                },
                checkpoint_directory / f"{machine}_{section}_{view}.pt",
            )
            models[view] = model
        group_scores: dict[str, list[float]] = {"near": [], "refsub": []}
        for row in group_test.itertuples(index=False):
            for view in ("near", "refsub"):
                values = load_reference_vectors(cache / str(row.cache_file), view)
                group_scores[view].append(
                    _score_clip(models[view], values, device, config.training.batch_size)
                )
        chosen = str(decision_lookup.loc[(machine, section)])
        for row_index, row in enumerate(group_test.itertuples(index=False)):
            base = {
                "file_id": str(row.file_id),
                "machine_type": machine,
                "section": section,
                "domain": str(row.domain),
                "condition": str(row.condition),
                "experiment_id": config.experiment_id,
            }
            values_by_system: dict[SystemName, float] = {
                "near": group_scores["near"][row_index],
                "unconditional_refsub": group_scores["refsub"][row_index],
                "safe_ref": group_scores[chosen][row_index],
            }
            for system, score in values_by_system.items():
                records[system].append(
                    {
                        **base,
                        "anomaly_score": score,
                        "model_id": f"official_compatible_ae_{system}",
                    }
                )
        train_scores = {
            view: [
                _score_clip(
                    models[view],
                    load_reference_vectors(cache / str(row.cache_file), view),
                    device,
                    config.training.batch_size,
                )
                for row in group_train.itertuples(index=False)
            ]
            for view in ("near", "refsub")
        }
        threshold_by_system = {
            "near": float(np.quantile(train_scores["near"], 0.99)),
            "unconditional_refsub": float(np.quantile(train_scores["refsub"], 0.99)),
            "safe_ref": float(np.quantile(train_scores[chosen], 0.99)),
        }
        thresholds.extend(
            {
                "seed": seed,
                "machine_type": machine,
                "section": section,
                "system": system,
                "threshold": threshold,
                "selected_view": chosen if system == "safe_ref" else system,
            }
            for system, threshold in threshold_by_system.items()
        )
    paths: dict[SystemName, Path] = {}
    for system in SYSTEMS:
        system_directory = output_directory / system
        system_directory.mkdir()
        path = system_directory / "scores.csv"
        frame = pd.DataFrame.from_records(records[system], columns=SCORE_COLUMNS).sort_values(
            "file_id", kind="stable"
        )
        if len(frame) != len(test) or frame["file_id"].duplicated().any():
            raise ValueError(f"{system} score coverage does not match test cache")
        frame.to_csv(path, index=False)
        if calculate_metrics:
            calculate_development_auc_metrics(path, system_directory / "metrics.json")
        paths[system] = path
    return paths, thresholds


def _load_cache(path: str | Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    cache = Path(path)
    index_path = cache / "index.parquet"
    profiles_path = cache / "profiles.parquet"
    metadata_path = cache / "cache.json"
    if not index_path.is_file() or not profiles_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("SAFE-REF cache requires index, profiles, and cache metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("views") != ["near", "refsub"] or metadata.get("feature_dim") != 640:
        raise ValueError("SAFE-REF cache metadata violates the official vector contract")
    index = pd.read_parquet(index_path)
    profiles = pd.read_parquet(profiles_path)
    required_index = {
        "file_id",
        "relative_path",
        "machine_type",
        "section",
        "domain",
        "condition",
        "dataset_split",
        "cache_file",
    }
    required_profiles = {
        "machine_type",
        "section",
        "risk_score",
        "noise_reduction_l05_db",
    }
    if required_index.difference(index.columns) or required_profiles.difference(profiles.columns):
        raise ValueError("SAFE-REF cache index/profile schema is incomplete")
    return cache, index, profiles


def _build_decisions(profiles: pd.DataFrame, policy: ReferenceSafetyPolicy) -> pd.DataFrame:
    decisions = profiles.copy()
    decisions["decision"] = np.where(
        (decisions["risk_score"] <= policy.risk_max)
        & (decisions["noise_reduction_l05_db"] >= policy.benefit_min_db),
        "refsub",
        "near",
    )
    decisions["policy_risk_max"] = policy.risk_max
    decisions["policy_benefit_min_db"] = policy.benefit_min_db
    return decisions.sort_values(["machine_type", "section"], kind="stable")


def _concatenate_vectors(cache: Path, rows: pd.DataFrame, view: str) -> np.ndarray:
    values = [
        load_reference_vectors(cache / str(row.cache_file), view)
        for row in rows.itertuples(index=False)
    ]
    if not values or any(not len(item) for item in values):
        raise ValueError(f"Training cache contains an empty {view} vector matrix")
    return np.concatenate(values, axis=0)


def _fit_model(
    vectors: np.ndarray, config: ReferenceSafetyExperimentConfig, device: torch.device
) -> tuple[OfficialCompatibleAutoencoder, float]:
    held_out = int(np.ceil(0.1 * len(vectors)))
    permutation = np.random.permutation(len(vectors))
    train_values = torch.from_numpy(vectors[permutation[held_out:]]).float()
    loader = DataLoader(
        TensorDataset(train_values),
        batch_size=config.training.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    model = OfficialCompatibleAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    final_loss = float("nan")
    model.train()
    for _ in range(config.training.epochs):
        total, count = 0.0, 0
        for (batch,) in loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, _ = model(batch)
            loss = torch.mean((reconstruction - batch) ** 2)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            total += float(loss.detach()) * len(batch)
            count += len(batch)
        final_loss = total / max(count, 1)
    return model.eval(), final_loss


def _score_clip(
    model: nn.Module, values: np.ndarray, device: torch.device, batch_size: int
) -> float:
    if not len(values):
        raise ValueError("Cannot score an empty official feature matrix")
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            reconstruction, _ = model(batch)
            total += float(torch.sum((reconstruction - batch) ** 2).cpu())
            count += reconstruction.numel()
    return total / count


def _write_score_ensemble(
    paths: list[Path], output: Path, system: SystemName, experiment_id: str
) -> pd.DataFrame:
    if not paths:
        raise ValueError("Score ensemble requires at least one seed")
    frames = [pd.read_csv(path).sort_values("file_id", kind="stable") for path in paths]
    reference_ids = frames[0]["file_id"].tolist()
    if any(frame["file_id"].tolist() != reference_ids for frame in frames[1:]):
        raise ValueError("Seed score files do not have identical file coverage/order")
    result = frames[0].copy()
    result["anomaly_score"] = np.mean(
        np.stack([frame["anomaly_score"].to_numpy(dtype=float) for frame in frames]), axis=0
    )
    result["model_id"] = f"official_compatible_ae_{system}_{len(paths)}seed_ensemble"
    result["experiment_id"] = experiment_id
    result.to_csv(output, index=False)
    return result


def _development_summary(metrics_paths: dict[SystemName, Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        metrics = json.loads(metrics_paths[system].read_text(encoding="utf-8"))
        group_values = metrics["groups"]
        official_values: list[float] = []
        machine_scores: dict[str, float] = {}
        for group_name, values in group_values.items():
            components = [
                float(values["auc_source"]),
                float(values["auc_target"]),
                float(values["pauc_all_max_fpr_0_1"]),
            ]
            official_values.extend(components)
            machine_scores[group_name] = float(
                hmean(np.maximum(components, sys.float_info.epsilon))
            )
        rows.append(
            {
                "system": system,
                "official_score": float(hmean(np.maximum(official_values, sys.float_info.epsilon))),
                "mean_auc_all": float(
                    np.mean([float(values["auc_all"]) for values in group_values.values()])
                ),
                "mean_pauc_all_max_fpr_0_1": float(
                    np.mean(
                        [float(values["pauc_all_max_fpr_0_1"]) for values in group_values.values()]
                    )
                ),
                "machine_scores_json": json.dumps(machine_scores, sort_keys=True),
            }
        )
    return pd.DataFrame.from_records(rows)


def _development_gate(
    summary: pd.DataFrame,
    decisions: pd.DataFrame,
    config: ReferenceSafetyExperimentConfig,
) -> dict[str, object]:
    scores = summary.set_index("system")["official_score"].astype(float)
    machine = {
        system: json.loads(
            str(summary.loc[summary["system"] == system, "machine_scores_json"].iloc[0])
        )
        for system in SYSTEMS
    }
    margin = config.development_gate.machine_harm_margin
    unconditional_harms = sum(
        float(machine["unconditional_refsub"][key]) - float(value) < -margin
        for key, value in machine["near"].items()
    )
    safe_harms = sum(
        float(machine["safe_ref"][key]) - float(value) < -margin
        for key, value in machine["near"].items()
    )
    harm_reduction = (
        (unconditional_harms - safe_harms) / unconditional_harms if unconditional_harms else 0.0
    )
    minimum_delta = min(
        float(machine["safe_ref"][key]) - float(value) for key, value in machine["near"].items()
    )
    accepted = int((decisions["decision"] == "refsub").sum())
    rejected = int((decisions["decision"] == "near").sum())
    passed = bool(
        scores["safe_ref"] >= scores["near"] - config.development_gate.macro_noninferiority_margin
        and harm_reduction >= config.development_gate.minimum_harm_reduction
        and minimum_delta >= -config.development_gate.maximum_machine_drop
        and accepted >= 1
        and rejected >= 1
    )
    return {
        "schema_version": 1,
        "passed": passed,
        "near_official_score": float(scores["near"]),
        "unconditional_official_score": float(scores["unconditional_refsub"]),
        "safe_ref_official_score": float(scores["safe_ref"]),
        "safe_ref_delta": float(scores["safe_ref"] - scores["near"]),
        "unconditional_harm_events": unconditional_harms,
        "safe_ref_harm_events": safe_harms,
        "harm_reduction": harm_reduction,
        "minimum_machine_delta": minimum_delta,
        "accepted_groups": accepted,
        "rejected_groups": rejected,
    }


def _write_official_score_files(
    *,
    scores: pd.DataFrame,
    index: pd.DataFrame,
    thresholds: pd.DataFrame,
    output_directory: Path,
) -> list[Path]:
    lookup = index.set_index("file_id", verify_integrity=True)
    threshold_lookup = thresholds.groupby(["machine_type", "section"], sort=True)[
        "threshold"
    ].mean()
    written: list[Path] = []
    for (machine, section), group in scores.groupby(["machine_type", "section"], sort=True):
        machine_name, section_name = str(machine), str(section)
        threshold = float(threshold_lookup.loc[(machine_name, section_name)])
        rows: list[tuple[str, float]] = []
        decisions: list[tuple[str, int]] = []
        for row in group.itertuples(index=False):
            basename = Path(str(lookup.loc[str(row.file_id), "relative_path"])).name
            score = float(row.anomaly_score)
            rows.append((basename, score))
            decisions.append((basename, int(score > threshold)))
        score_path = output_directory / f"anomaly_score_{machine_name}_{section_name}_test.csv"
        decision_path = output_directory / f"decision_result_{machine_name}_{section_name}_test.csv"
        for path, values in ((score_path, rows), (decision_path, decisions)):
            with path.open("w", newline="", encoding="utf-8") as handle:
                csv.writer(handle, lineterminator="\n").writerows(values)
            written.append(path)
    return written


def _load_and_validate_freeze(
    freeze_path: str | Path,
    config_path: str | Path,
    policy_path: str | Path,
    config: ReferenceSafetyExperimentConfig,
) -> dict[str, object]:
    source = Path(freeze_path)
    if not source.is_file():
        raise FileNotFoundError(f"SAFE-REF freeze not found: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("SAFE-REF freeze schema is invalid")
    expected = {
        "config_sha256": _sha256_file(config_path),
        "validated_config_sha256": reference_safety_config_hash(config),
        "policy_sha256": _sha256_file(policy_path),
        "official_evaluator_commit": OFFICIAL_EVALUATOR_COMMIT,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"SAFE-REF freeze mismatch for {field}")
    if payload.get("ground_truth_access_during_scoring") is not False:
        raise ValueError("SAFE-REF freeze must prohibit ground-truth access during scoring")
    if payload.get("systems") != list(SYSTEMS):
        raise ValueError("SAFE-REF freeze systems do not match the runtime contract")
    return payload


def _validate_official_training(config: ReferenceSafetyExperimentConfig) -> None:
    if config.training.epochs != 100 or config.training.batch_size != 256:
        raise ValueError("SAFE-REF official alignment requires 100 epochs and batch size 256")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SAFE-REF configuration requests CUDA, but no CUDA GPU is available")
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()
