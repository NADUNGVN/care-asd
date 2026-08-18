"""Resumable, capacity-matched C1/C2 FP-NAA development screening."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
import pandas as pd

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from care_asd.evaluation.dcase2026_metrics import (
    calculate_dcase2026_official_metrics,
    read_official_score,
)
from care_asd.evaluation.fp_naa_backend import accelerated_beam_scores, accelerated_rdp_pool
from care_asd.evaluation.official_baseline import SCORE_COLUMNS
from care_asd.fp_naa_config import FPNAAConfig, load_fp_naa_config
from care_asd.models.fp_naa_adapter import BandwiseReferenceAdapter, trainable_parameter_count
from care_asd.models.fp_naa_objective import FPNaaObjective, fp_naa_loss

Candidate = Literal["c1_mse", "c2_fault_preserving"]
CANDIDATES: tuple[Candidate, ...] = ("c1_mse", "c2_fault_preserving")


@dataclass(frozen=True)
class FPNaaScreeningResult:
    output_directory: Path
    summary_path: Path
    gate_path: Path
    core_gate_passed: bool


@dataclass(frozen=True)
class FPNaaLomoResult:
    output_directory: Path
    summary_path: Path
    gate_path: Path
    gate_passed: bool


@dataclass(frozen=True)
class _BaseTokenStore:
    frame: pd.DataFrame
    near: np.ndarray
    far: np.ndarray
    index_by_file_id: dict[str, int]

    def select(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        indices = np.asarray(
            [self.index_by_file_id[str(value)] for value in rows["file_id"]],
            dtype=np.int64,
        )
        return self.near[indices], self.far[indices]


@dataclass(frozen=True)
class _TrainingArrays:
    frame: pd.DataFrame
    noisy_clean: np.ndarray
    reference: np.ndarray
    teacher_clean: np.ndarray
    fault_noisy: np.ndarray
    teacher_fault: np.ndarray


class _FaultDiagnosticRow(Protocol):
    retention: float
    delta_gain: float
    direction_cosine: float
    teacher_delta_norm: float
    student_delta_norm: float
    salient_distance_gain: float
    frontend_observability: float
    frontend_retention: float
    adapter_delta_gain: float
    adapter_retention: float
    transport_relative_error: float


@dataclass(frozen=True)
class _C1Initialization:
    source_path: Path
    source_sha256: str
    model_state: dict[str, Tensor]


class _CounterfactualDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]):
    def __init__(self, arrays: _TrainingArrays) -> None:
        self._arrays = arrays

    def __len__(self) -> int:
        return len(self._arrays.frame)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        return (
            torch.from_numpy(self._arrays.noisy_clean[index]),
            torch.from_numpy(self._arrays.reference[index]),
            torch.from_numpy(self._arrays.teacher_clean[index]),
            torch.from_numpy(self._arrays.fault_noisy[index]),
            torch.from_numpy(self._arrays.teacher_fault[index]),
        )


def run_fp_naa_screening(
    *,
    base_cache_directory: str | Path,
    augmentation_cache_directory: str | Path,
    c0_score_path: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config_path: str | Path,
    experiment_id: str,
    device: str = "cuda",
    preload_workers: int | None = None,
) -> FPNaaScreeningResult:
    """Train and score C1/C2 over the three frozen screening seeds.

    Completed seed/candidate checkpoints, scores, and retention files are reused, allowing a
    server job to resume after SSH or process interruption without changing the experiment.
    """
    base_cache = Path(base_cache_directory).resolve()
    augmentation_cache = Path(augmentation_cache_directory).resolve()
    c0_scores = Path(c0_score_path).resolve()
    output = Path(output_directory).resolve()
    checkpoints = Path(checkpoint_directory).resolve()
    config_source = Path(config_path).resolve()
    config = load_fp_naa_config(config_source)
    workers = config.training.workers if preload_workers is None else preload_workers
    if not 1 <= workers <= 16:
        raise ValueError("preload_workers must be in [1, 16]")
    torch_device = _cuda_device(device)
    _validate_caches(base_cache, augmentation_cache, config)
    if not c0_scores.is_file():
        raise FileNotFoundError(f"C0 score file not found: {c0_scores}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    c1_reuse = _screening_c1_reuse_manifest(
        output=output,
        config=config,
        base_cache_sha256=_sha256(base_cache / "cache.json"),
        augmentation_cache_sha256=_sha256(augmentation_cache / "cache.json"),
        c0_scores_sha256=_sha256(c0_scores),
    )
    _ensure_contract(
        output,
        {
            "schema_version": 1,
            "kind": "fp_naa_screening",
            "experiment_id": experiment_id,
            "config": config.model_dump(mode="json"),
            "base_cache_metadata_sha256": _sha256(base_cache / "cache.json"),
            "augmentation_cache_metadata_sha256": _sha256(augmentation_cache / "cache.json"),
            "c0_scores_sha256": _sha256(c0_scores),
            "screening_c1_reuse": c1_reuse,
        },
    )
    completed_gate = output / "gate.json"
    completed_summary = output / "screening_summary.csv"
    if completed_gate.is_file() and completed_summary.is_file():
        gate = json.loads(completed_gate.read_text(encoding="utf-8"))
        return FPNaaScreeningResult(
            output,
            completed_summary,
            completed_gate,
            bool(gate["checks"]["core_screening"]),
        )

    _write_progress(output, stage="preload_base", completed=0, total=1)
    base_store = _preload_base_store(base_cache, workers=workers)
    _write_progress(output, stage="preload_augmentation", completed=0, total=1)
    training = _preload_training_arrays(
        augmentation_cache,
        base_store,
        workers=workers,
    )
    _write_progress(output, stage="preload_complete", completed=1, total=1)

    summary_rows: list[dict[str, object]] = []
    total_runs = len(config.training.screening_seeds) * len(CANDIDATES)
    run_number = 0
    for seed in config.training.screening_seeds:
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
                total=total_runs,
            )
            reused_parameters: int | None = None
            if candidate == "c1_mse" and c1_reuse is not None:
                reused_parameters = _materialize_reused_c1_variant(
                    source_directory=Path(str(c1_reuse["source_directory"])),
                    destination=variant,
                    seed=seed,
                    manifest=c1_reuse,
                )
                model = None
            else:
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
                    total_runs=total_runs,
                )
            retention_path = variant / "retention.csv"
            if not retention_path.is_file():
                if model is None:
                    raise ValueError("Reused C1 retention artifact is missing")
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
                raise ValueError(f"Metrics exist without an immutable score file in {variant}")
            if score_path.is_file() and not metrics_path.is_file():
                calculate_dcase2026_official_metrics(score_path, metrics_path)
            elif not score_path.is_file():
                if model is None:
                    raise ValueError("Reused C1 score artifact is missing")
                _score_adapter(
                    model=model,
                    base_store=base_store,
                    score_path=score_path,
                    metrics_path=metrics_path,
                    model_id=f"fp_naa_{candidate}_seed{seed}",
                    experiment_id=experiment_id,
                    config=config,
                    device=torch_device,
                )
            retention = pd.read_csv(retention_path)
            in_support = retention.loc[retention["fault_set"] == "in_support", "retention"]
            heldout = retention.loc[retention["fault_set"] == "heldout", "retention"]
            if in_support.empty or heldout.empty:
                raise ValueError(f"Retention diagnostics are incomplete: {retention_path}")
            summary_rows.append(
                {
                    "seed": seed,
                    "candidate": candidate,
                    "official_score": read_official_score(metrics_path),
                    "official_score_percent": 100.0 * read_official_score(metrics_path),
                    "in_support_retention_median": float(in_support.median()),
                    "in_support_retention_q05": float(in_support.quantile(0.05)),
                    "heldout_retention_median": float(heldout.median()),
                    "heldout_retention_q05": float(heldout.quantile(0.05)),
                    "in_support_frontend_retention_median": _diagnostic_quantile(
                        retention, fault_set="in_support", column="frontend_retention", q=0.50
                    ),
                    "in_support_frontend_retention_q05": _diagnostic_quantile(
                        retention, fault_set="in_support", column="frontend_retention", q=0.05
                    ),
                    "in_support_adapter_retention_median": _diagnostic_quantile(
                        retention, fault_set="in_support", column="adapter_retention", q=0.50
                    ),
                    "in_support_adapter_retention_q05": _diagnostic_quantile(
                        retention, fault_set="in_support", column="adapter_retention", q=0.05
                    ),
                    "in_support_transport_error_median": _diagnostic_quantile(
                        retention,
                        fault_set="in_support",
                        column="transport_relative_error",
                        q=0.50,
                    ),
                    "in_support_transport_error_q90": _diagnostic_quantile(
                        retention,
                        fault_set="in_support",
                        column="transport_relative_error",
                        q=0.90,
                    ),
                    "trainable_parameters": (
                        reused_parameters
                        if reused_parameters is not None
                        else trainable_parameter_count(model)
                    ),
                    "score_path": str(score_path.relative_to(output)),
                }
            )
            _write_progress(
                output,
                stage=f"complete:{seed}:{candidate}",
                completed=run_number,
                total=total_runs,
            )

    summary = pd.DataFrame(summary_rows)
    _atomic_csv(completed_summary, summary)
    gate = _screening_gate(
        summary=summary,
        output=output,
        c0_scores=c0_scores,
        config=config,
    )
    _atomic_json(completed_gate, gate)
    _atomic_json(
        output / "run.json",
        {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "base_cache": str(base_cache),
            "augmentation_cache": str(augmentation_cache),
            "c0_scores": str(c0_scores),
            "deterministic_runtime": _deterministic_runtime_metadata(),
            "config": config.model_dump(mode="json"),
            "device": str(torch_device),
            "preload_workers": workers,
        },
    )
    _write_progress(output, stage="complete", completed=total_runs, total=total_runs)
    return FPNaaScreeningResult(
        output,
        completed_summary,
        completed_gate,
        bool(gate["checks"]["core_screening"]),
    )


def _screening_c1_reuse_manifest(
    *,
    output: Path,
    config: FPNAAConfig,
    base_cache_sha256: str,
    augmentation_cache_sha256: str,
    c0_scores_sha256: str,
) -> dict[str, object] | None:
    """Validate a frozen C1 screening result for byte-identical artifact reuse."""
    run_id = config.screening_c1_reuse_run_id
    if run_id is None:
        return None
    source = output.parent.parent / run_id / "screening"
    contract_path = source / "contract.json"
    summary_path = source / "screening_summary.csv"
    if not contract_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Registered C1 reuse run is incomplete: {source}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_hashes = {
        "base_cache_metadata_sha256": base_cache_sha256,
        "augmentation_cache_metadata_sha256": augmentation_cache_sha256,
        "c0_scores_sha256": c0_scores_sha256,
    }
    for key, expected in expected_hashes.items():
        if contract.get(key) != expected:
            raise ValueError(f"Registered C1 reuse {key} mismatch: {source}")
    source_config = contract.get("config")
    if not isinstance(source_config, dict):
        raise ValueError(f"Registered C1 reuse config is invalid: {source}")
    if _c1_reuse_signature(source_config) != _c1_reuse_signature(config.model_dump(mode="json")):
        raise ValueError(f"Registered C1 training/scoring contract mismatch: {source}")

    summary = pd.read_csv(summary_path)
    c1 = summary.loc[summary["candidate"] == "c1_mse"].copy()
    if set(c1["seed"].astype(int)) != set(config.training.screening_seeds):
        raise ValueError(f"Registered C1 reuse seeds are incomplete: {source}")
    artifacts: dict[str, str] = {}
    required = ("training_history.csv", "retention.csv", "scores.csv", "metrics.json")
    for seed in config.training.screening_seeds:
        for name in required:
            relative = Path(f"seed{seed}") / "c1_mse" / name
            artifact = source / relative
            if not artifact.is_file():
                raise FileNotFoundError(f"Registered C1 artifact is missing: {artifact}")
            artifacts[relative.as_posix()] = _sha256(artifact)
    return {
        "source_run_id": run_id,
        "source_directory": str(source),
        "source_contract_sha256": _sha256(contract_path),
        "source_summary_sha256": _sha256(summary_path),
        "artifacts": artifacts,
    }


def _diagnostic_quantile(frame: pd.DataFrame, *, fault_set: str, column: str, q: float) -> float:
    """Summarize a new diagnostic while allowing immutable older C1 artifacts."""
    if column not in frame.columns:
        return float("nan")
    values = frame.loc[frame["fault_set"] == fault_set, column]
    return float(values.quantile(q)) if not values.empty else float("nan")


def _c1_reuse_signature(config: dict[str, object]) -> dict[str, object]:
    """Return only fields that can affect C1 training, diagnostics, or scoring."""
    adapter = cast(dict[str, object], config["adapter"])
    objective = cast(dict[str, object], config["objective"])
    training = cast(dict[str, object], config["training"])
    return {
        "provenance": config["provenance"],
        "frontend": config["frontend"],
        "backend": config["backend"],
        "augmentation": config["augmentation"],
        "adapter": {
            key: adapter[key]
            for key in (
                "hidden_dim",
                "attention_heads",
                "dropout",
                "reference_dropout_probability",
                "reference_corruption_probability",
            )
        },
        "objective": {"normal_mse_weight": objective["normal_mse_weight"]},
        "training": {
            key: training[key]
            for key in (
                "epochs",
                "batch_size",
                "learning_rate",
                "weight_decay",
                "warmup_epochs",
                "gradient_clip_norm",
                "workers",
                "mixed_precision",
                "screening_seeds",
            )
        },
    }


def _materialize_reused_c1_variant(
    *,
    source_directory: Path,
    destination: Path,
    seed: int,
    manifest: dict[str, object],
) -> int:
    """Copy a validated C1 variant and preserve per-file provenance."""
    artifacts = cast(dict[str, str], manifest["artifacts"])
    names = ("training_history.csv", "retention.csv", "scores.csv", "metrics.json")
    copied: dict[str, str] = {}
    for name in names:
        relative = Path(f"seed{seed}") / "c1_mse" / name
        expected = artifacts[relative.as_posix()]
        source = source_directory / relative
        target = destination / name
        if target.is_file():
            if _sha256(target) != expected:
                raise ValueError(f"Reused C1 artifact hash mismatch: {target}")
        else:
            shutil.copy2(source, target)
        if _sha256(target) != expected:
            raise ValueError(f"Reused C1 artifact copy failed: {target}")
        copied[name] = expected
    summary = pd.read_csv(source_directory / "screening_summary.csv")
    row = summary.loc[(summary["candidate"] == "c1_mse") & (summary["seed"].astype(int) == seed)]
    if len(row) != 1:
        raise ValueError(f"Reused C1 summary row is invalid for seed {seed}")
    parameters = int(row.iloc[0]["trainable_parameters"])
    _atomic_json(
        destination / "reuse.json",
        {
            "schema_version": 1,
            "source_run_id": manifest["source_run_id"],
            "source_contract_sha256": manifest["source_contract_sha256"],
            "seed": seed,
            "candidate": "c1_mse",
            "trainable_parameters": parameters,
            "artifacts": copied,
        },
    )
    return parameters


def run_fp_naa_lomo(
    *,
    base_cache_directory: str | Path,
    augmentation_cache_directory: str | Path,
    screening_directory: str | Path,
    output_directory: str | Path,
    checkpoint_directory: str | Path,
    config_path: str | Path,
    experiment_id: str,
    device: str = "cuda",
    preload_workers: int | None = None,
) -> FPNaaLomoResult:
    """Run leave-one-machine-out adapter training after the core screening gate passes."""
    base_cache = Path(base_cache_directory).resolve()
    augmentation_cache = Path(augmentation_cache_directory).resolve()
    screening = Path(screening_directory).resolve()
    output = Path(output_directory).resolve()
    checkpoints = Path(checkpoint_directory).resolve()
    config = load_fp_naa_config(Path(config_path).resolve())
    workers = config.training.workers if preload_workers is None else preload_workers
    if not 1 <= workers <= 16:
        raise ValueError("preload_workers must be in [1, 16]")
    core_gate_path = screening / "gate.json"
    if not core_gate_path.is_file():
        raise FileNotFoundError(f"Core screening gate not found: {core_gate_path}")
    core_gate = json.loads(core_gate_path.read_text(encoding="utf-8"))
    if not bool(core_gate["checks"]["core_screening"]):
        raise ValueError("LOMO is blocked because the preregistered core screening gate failed")
    torch_device = _cuda_device(device)
    _validate_caches(base_cache, augmentation_cache, config)
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    _ensure_contract(
        output,
        {
            "schema_version": 1,
            "kind": "fp_naa_lomo",
            "experiment_id": experiment_id,
            "config": config.model_dump(mode="json"),
            "base_cache_metadata_sha256": _sha256(base_cache / "cache.json"),
            "augmentation_cache_metadata_sha256": _sha256(augmentation_cache / "cache.json"),
            "screening_gate_sha256": _sha256(core_gate_path),
        },
    )
    summary_path = output / "lomo_summary.csv"
    gate_path = output / "gate.json"
    if summary_path.is_file() and gate_path.is_file():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        return FPNaaLomoResult(output, summary_path, gate_path, bool(gate["passed"]))
    _write_progress(output, stage="preload_base", completed=0, total=1)
    base_store = _preload_base_store(base_cache, workers=workers)
    _write_progress(output, stage="preload_augmentation", completed=0, total=1)
    training = _preload_training_arrays(augmentation_cache, base_store, workers=workers)
    machines = sorted(training.frame["machine_type"].astype(str).unique())
    if len(machines) != 7:
        raise ValueError(
            f"Preregistered DCASE development LOMO requires 7 machines, found {len(machines)}"
        )
    total_runs = len(machines) * len(config.training.screening_seeds) * len(CANDIDATES)
    run_number = 0
    rows: list[dict[str, object]] = []
    for heldout_machine in machines:
        fold_training = _exclude_machine(training, heldout_machine)
        for seed in config.training.screening_seeds:
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
                    total=total_runs,
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
                    total_runs=total_runs,
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
                        model_id=f"fp_naa_lomo_{candidate}_{heldout_machine}_seed{seed}",
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
                    }
                )
                _write_progress(
                    output,
                    stage=f"complete:{heldout_machine}:{seed}:{candidate}",
                    completed=run_number,
                    total=total_runs,
                )
    summary = pd.DataFrame(rows)
    _atomic_csv(summary_path, summary)
    pivot = summary.pivot_table(
        index="heldout_machine",
        columns="candidate",
        values="official_score",
        aggfunc="mean",
    )
    pivot["delta_c2_minus_c1"] = pivot["c2_fault_preserving"] - pivot["c1_mse"]
    fold_deltas = {
        str(machine): float(value) for machine, value in pivot["delta_c2_minus_c1"].items()
    }
    positive_folds = int((pivot["delta_c2_minus_c1"] > 0.0).sum())
    lomo_passed = positive_folds >= config.gates.screening_positive_lomo_folds_minimum
    gate = {
        "schema_version": 1,
        "gate": "G2_screening_lomo",
        "screening_core_passed": True,
        "fold_delta_c2_minus_c1": fold_deltas,
        "positive_folds": positive_folds,
        "minimum_positive_folds": config.gates.screening_positive_lomo_folds_minimum,
        "passed": lomo_passed,
    }
    _atomic_json(gate_path, gate)
    _atomic_csv(output / "lomo_fold_means.csv", pivot.reset_index())
    _write_progress(output, stage="complete", completed=total_runs, total=total_runs)
    return FPNaaLomoResult(output, summary_path, gate_path, lomo_passed)


def _load_or_train_model(
    *,
    checkpoint: Path,
    history_path: Path,
    arrays: _TrainingArrays,
    candidate: Candidate,
    seed: int,
    config: FPNAAConfig,
    device: torch.device,
    progress_output: Path,
    run_number: int,
    total_runs: int,
) -> BandwiseReferenceAdapter:
    model = _new_model(config, candidate=candidate).to(device)
    initialization = (
        _load_c1_initialization(checkpoint=checkpoint, seed=seed, config=config, device=device)
        if candidate == "c2_fault_preserving"
        and config.objective.fault_loss_mode == "anchored_tangent_transport"
        else None
    )
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload.get("candidate") != candidate or int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Checkpoint contract mismatch: {checkpoint}")
        if payload.get("config") != config.model_dump(mode="json"):
            raise ValueError(f"Checkpoint config mismatch: {checkpoint}")
        if candidate == "c2_fault_preserving" and config.adapter.share_c1_weights_for_c2:
            source_checkpoint = checkpoint.with_name("c1_mse.pt")
            if not source_checkpoint.is_file():
                raise FileNotFoundError(f"Shared C1 checkpoint not found: {source_checkpoint}")
            if payload.get("derived_from_c1_sha256") != _sha256(source_checkpoint):
                raise ValueError(f"Shared C1 checkpoint hash mismatch: {checkpoint}")
        if initialization is not None:
            if payload.get("derived_from_c1_sha256") != initialization.source_sha256:
                raise ValueError(f"Anchored C1 checkpoint hash mismatch: {checkpoint}")
            _write_c1_initialization_contract(
                history_path.parent / "initialization.json", initialization, seed=seed
            )
        model.load_state_dict(payload["model_state"])
        return model.eval()
    if candidate == "c2_fault_preserving" and config.adapter.share_c1_weights_for_c2:
        return _materialize_reference_safe_c2(
            checkpoint=checkpoint,
            history_path=history_path,
            seed=seed,
            config=config,
            device=device,
        )
    _set_seed(seed)
    model = _new_model(config, candidate=candidate).to(device)
    anchor_model: BandwiseReferenceAdapter | None = None
    if initialization is not None:
        model.load_state_dict(initialization.model_state)
        anchor_model = _new_model(config, candidate="c1_mse").to(device)
        anchor_model.load_state_dict(initialization.model_state)
        anchor_model.requires_grad_(False).eval()
        _write_c1_initialization_contract(
            history_path.parent / "initialization.json", initialization, seed=seed
        )
    dataset = _CounterfactualDataset(arrays)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    anchored_transport = initialization is not None
    epochs = (
        cast(int, config.training.c2_finetune_epochs)
        if anchored_transport
        else config.training.epochs
    )
    learning_rate = (
        cast(float, config.training.c2_finetune_learning_rate)
        if anchored_transport
        else config.training.learning_rate
    )
    warmup_epochs = (
        cast(int, config.training.c2_finetune_warmup_epochs)
        if anchored_transport
        else config.training.warmup_epochs
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _cosine_learning_rate_factor(
            epoch, epochs=epochs, warmup_epochs=warmup_epochs
        ),
    )
    use_amp = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    objective = cast(FPNaaObjective, candidate)
    if anchored_transport and config.training.c2_finetune_disable_dropout:
        model.eval()
    else:
        model.train()
    for epoch in range(epochs):
        totals = torch.zeros(12, dtype=torch.float64, device=device)
        examples = 0
        auxiliary_scale = _auxiliary_scale(epoch, candidate=candidate, config=config)
        for noisy_clean, reference, teacher_clean, fault_noisy, teacher_fault in loader:
            noisy_clean = noisy_clean.to(device, non_blocking=True)
            reference = reference.to(device, non_blocking=True)
            teacher_clean = teacher_clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                student_clean = model(noisy_clean, reference)
                if candidate == "c1_mse" or auxiliary_scale == 0.0:
                    loss = fp_naa_loss(
                        objective="c1_mse",
                        student_clean=student_clean,
                        teacher_clean=teacher_clean,
                        config=config.objective,
                    )
                else:
                    fault_noisy = fault_noisy.to(device, non_blocking=True)
                    teacher_fault = teacher_fault.to(device, non_blocking=True)
                    student_fault = model(fault_noisy, reference)
                    anchor_clean = None
                    if anchor_model is not None:
                        with torch.no_grad():
                            anchor_clean = anchor_model(noisy_clean, reference)
                    loss = fp_naa_loss(
                        objective=objective,
                        student_clean=student_clean,
                        teacher_clean=teacher_clean,
                        student_fault=student_fault,
                        teacher_fault=teacher_fault,
                        anchor_clean=anchor_clean,
                        config=config.objective,
                    )
                primary = config.objective.normal_mse_weight * loss.normal_mse
                auxiliary = loss.total - primary
                effective_total = primary + auxiliary_scale * auxiliary
            gradient_cosine = loss.total.new_zeros(())
            gradient_conflict = loss.total.new_zeros(())
            if (
                candidate == "c2_fault_preserving"
                and auxiliary_scale > 0.0
                and config.objective.primary_safe_gradient_projection
            ):
                gradient_cosine, gradient_conflict = _primary_safe_backward(
                    primary=primary,
                    auxiliary=auxiliary,
                    parameters=list(model.parameters()),
                    auxiliary_scale=auxiliary_scale,
                )
            else:
                scaler.scale(effective_total).backward()
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.training.gradient_clip_norm,
                error_if_nonfinite=True,
            )
            if (
                candidate == "c2_fault_preserving"
                and auxiliary_scale > 0.0
                and config.objective.primary_safe_gradient_projection
            ):
                optimizer.step()
            else:
                scaler.step(optimizer)
                scaler.update()
            batch = len(noisy_clean)
            totals += batch * torch.stack(
                (
                    effective_total.detach(),
                    loss.normal_mse.detach(),
                    loss.fault_direction.detach(),
                    loss.fault_magnitude.detach(),
                    loss.fault_separation.detach(),
                    loss.retention.detach().mean(),
                    gradient_cosine.detach(),
                    gradient_conflict.detach(),
                    loss.tangent_transport.detach(),
                    loss.function_anchor.detach(),
                    loss.tangent_relative_error.detach().mean(),
                    loss.function_anchor_ratio.detach().mean(),
                )
            ).to(dtype=torch.float64)
            examples += batch
        epoch_totals = (totals / examples).cpu().tolist()
        history.append(
            {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "total_loss": epoch_totals[0],
                "normal_mse": epoch_totals[1],
                "fault_direction": epoch_totals[2],
                "fault_magnitude": epoch_totals[3],
                "fault_separation": epoch_totals[4],
                "retention": epoch_totals[5],
                "gradient_cosine": epoch_totals[6],
                "gradient_conflict_fraction": epoch_totals[7],
                "auxiliary_scale": auxiliary_scale,
                "tangent_transport": epoch_totals[8],
                "function_anchor": epoch_totals[9],
                "tangent_relative_error": epoch_totals[10],
                "function_anchor_ratio": epoch_totals[11],
            }
        )
        scheduler.step()
        _write_progress(
            progress_output,
            stage=f"train:{seed}:{candidate}:epoch{epoch + 1}",
            completed=run_number - 1,
            total=total_runs,
        )
    _atomic_csv(history_path, pd.DataFrame(history))
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "candidate": candidate,
            "seed": seed,
            "model_state": model.state_dict(),
            "config": config.model_dump(mode="json"),
            "deterministic_runtime": _deterministic_runtime_metadata(),
            "derived_from_c1_sha256": (
                initialization.source_sha256 if initialization is not None else None
            ),
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    return model.eval()


def _load_c1_initialization(
    *,
    checkpoint: Path,
    seed: int,
    config: FPNAAConfig,
    device: torch.device,
) -> _C1Initialization:
    """Load the exact matching C1 state used to initialize anchored C2."""
    sibling = checkpoint.with_name("c1_mse.pt")
    if sibling.is_file():
        source = sibling
    else:
        run_id = config.screening_c2_initialization_run_id
        if run_id is None or len(checkpoint.parents) < 3:
            raise FileNotFoundError("Anchored C2 cannot resolve its registered C1 checkpoint")
        source = checkpoint.parents[2] / run_id / f"seed{seed}" / "c1_mse.pt"
    if not source.is_file():
        raise FileNotFoundError(f"Registered C1 initialization checkpoint is missing: {source}")
    payload = torch.load(source, map_location=device, weights_only=True)
    if payload.get("candidate") != "c1_mse" or int(payload.get("seed", -1)) != seed:
        raise ValueError(f"C1 initialization checkpoint contract mismatch: {source}")
    source_config = payload.get("config")
    if not isinstance(source_config, dict):
        raise ValueError(f"C1 initialization config is invalid: {source}")
    if _c1_reuse_signature(source_config) != _c1_reuse_signature(config.model_dump(mode="json")):
        raise ValueError(f"C1 initialization training/scoring contract mismatch: {source}")
    model_state = payload.get("model_state")
    if not isinstance(model_state, dict):
        raise ValueError(f"C1 initialization model state is invalid: {source}")
    return _C1Initialization(
        source_path=source.resolve(),
        source_sha256=_sha256(source),
        model_state=cast(dict[str, Tensor], model_state),
    )


def _write_c1_initialization_contract(
    path: Path, initialization: _C1Initialization, *, seed: int
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": 1,
            "candidate": "c2_fault_preserving",
            "seed": seed,
            "source_candidate": "c1_mse",
            "source_checkpoint": str(initialization.source_path),
            "source_checkpoint_sha256": initialization.source_sha256,
        },
    )


def _materialize_reference_safe_c2(
    *,
    checkpoint: Path,
    history_path: Path,
    seed: int,
    config: FPNAAConfig,
    device: torch.device,
) -> BandwiseReferenceAdapter:
    """Reuse C1 weights so C2 differs only by the parameter-free safety projection."""
    source_checkpoint = checkpoint.with_name("c1_mse.pt")
    source_history = history_path.parent.parent / "c1_mse" / history_path.name
    if not source_checkpoint.is_file() or not source_history.is_file():
        raise FileNotFoundError(
            "Reference-safe C2 requires the matching C1 checkpoint and training history"
        )
    source = torch.load(source_checkpoint, map_location=device, weights_only=True)
    if source.get("candidate") != "c1_mse" or int(source.get("seed", -1)) != seed:
        raise ValueError(f"C1 source checkpoint contract mismatch: {source_checkpoint}")
    if source.get("config") != config.model_dump(mode="json"):
        raise ValueError(f"C1 source checkpoint config mismatch: {source_checkpoint}")

    model = _new_model(config, candidate="c2_fault_preserving").to(device)
    model.load_state_dict(source["model_state"])
    history = pd.read_csv(source_history)
    history["derived_from_candidate"] = "c1_mse"
    history["reference_safety_mode"] = config.adapter.reference_safety_mode
    history["maximum_reference_contraction"] = config.adapter.maximum_reference_contraction
    _atomic_csv(history_path, history)
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "candidate": "c2_fault_preserving",
            "seed": seed,
            "model_state": source["model_state"],
            "config": config.model_dump(mode="json"),
            "deterministic_runtime": _deterministic_runtime_metadata(),
            "derived_from_c1_sha256": _sha256(source_checkpoint),
            "reference_safety_mode": config.adapter.reference_safety_mode,
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    return model.eval()


def _auxiliary_scale(epoch: int, *, candidate: Candidate, config: FPNAAConfig) -> float:
    if candidate == "c1_mse":
        return 0.0
    if config.objective.fault_loss_mode == "anchored_tangent_transport":
        return 1.0
    auxiliary_weights = (
        config.objective.fault_direction_weight,
        config.objective.fault_magnitude_weight,
        config.objective.fault_separation_weight,
        config.objective.reference_consistency_weight,
    )
    if all(weight == 0.0 for weight in auxiliary_weights):
        return 0.0
    start = config.objective.auxiliary_start_epoch
    if epoch < start:
        return 0.0
    ramp = config.objective.auxiliary_ramp_epochs
    if ramp == 0:
        return 1.0
    return min(1.0, (epoch - start + 1) / ramp)


def _primary_safe_backward(
    *,
    primary: torch.Tensor,
    auxiliary: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    auxiliary_scale: float,
    eps: float = 1.0e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply PCGrad to the auxiliary gradient while preserving primary-loss descent."""
    primary_gradients = torch.autograd.grad(
        primary, parameters, retain_graph=True, allow_unused=True
    )
    auxiliary_gradients = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
    primary_norm_sq = primary.new_zeros(())
    auxiliary_norm_sq = primary.new_zeros(())
    dot = primary.new_zeros(())
    for primary_gradient, auxiliary_gradient in zip(
        primary_gradients, auxiliary_gradients, strict=True
    ):
        if primary_gradient is not None:
            primary_norm_sq += primary_gradient.float().square().sum()
        if auxiliary_gradient is not None:
            auxiliary_norm_sq += auxiliary_gradient.float().square().sum()
        if primary_gradient is not None and auxiliary_gradient is not None:
            dot += (primary_gradient.float() * auxiliary_gradient.float()).sum()
    cosine = dot / (primary_norm_sq.sqrt() * auxiliary_norm_sq.sqrt() + eps)
    conflict = (dot < 0.0).to(dtype=primary.dtype)
    coefficient = torch.where(
        dot < 0.0,
        dot / (primary_norm_sq + eps),
        dot.new_zeros(()),
    )
    for parameter, primary_gradient, auxiliary_gradient in zip(
        parameters, primary_gradients, auxiliary_gradients, strict=True
    ):
        if primary_gradient is None and auxiliary_gradient is None:
            parameter.grad = None
            continue
        primary_value = (
            torch.zeros_like(parameter) if primary_gradient is None else primary_gradient
        )
        auxiliary_value = (
            torch.zeros_like(parameter) if auxiliary_gradient is None else auxiliary_gradient
        )
        if primary_gradient is not None:
            auxiliary_value = auxiliary_value - coefficient * primary_gradient
        parameter.grad = primary_value + auxiliary_scale * auxiliary_value
    return cosine, conflict


def _write_retention_diagnostics(
    *,
    model: BandwiseReferenceAdapter,
    arrays: _TrainingArrays,
    augmentation_cache: Path,
    base_store: _BaseTokenStore,
    output_path: Path,
    config: FPNAAConfig,
    device: torch.device,
    workers: int,
) -> None:
    rows: list[dict[str, object]] = []
    in_support = _fault_diagnostics(
        model=model,
        noisy_clean=arrays.noisy_clean,
        reference=arrays.reference,
        teacher_clean=arrays.teacher_clean,
        fault_noisy=arrays.fault_noisy,
        teacher_fault=arrays.teacher_fault,
        config=config,
        device=device,
    )
    for row, diagnostic in zip(
        arrays.frame.itertuples(index=False), in_support.itertuples(index=False), strict=True
    ):
        rows.append(
            _fault_diagnostic_row(
                file_id=str(row.file_id),
                fault_set="in_support",
                fault_family=str(row.fault_family),
                diagnostic=diagnostic,
            )
        )
    heldout_frame = arrays.frame.loc[arrays.frame["heldout"].astype(bool)].reset_index(drop=True)
    heldout = _load_named_augmentation_arrays(
        augmentation_cache,
        heldout_frame,
        names=(
            "heldout_noisy_clean",
            "heldout_reference",
            "heldout_fault_noisy",
            "heldout_fault_teacher",
        ),
        workers=workers,
    )
    teacher_clean, _ = base_store.select(heldout_frame)
    heldout_values = _fault_diagnostics(
        model=model,
        noisy_clean=heldout[0],
        reference=heldout[1],
        teacher_clean=teacher_clean,
        fault_noisy=heldout[2],
        teacher_fault=heldout[3],
        config=config,
        device=device,
    )
    for row, diagnostic in zip(
        heldout_frame.itertuples(index=False),
        heldout_values.itertuples(index=False),
        strict=True,
    ):
        rows.append(
            _fault_diagnostic_row(
                file_id=str(row.file_id),
                fault_set="heldout",
                fault_family=str(row.heldout_fault_family),
                diagnostic=diagnostic,
            )
        )
    _atomic_csv(output_path, pd.DataFrame(rows))


def _fault_diagnostic_row(
    *, file_id: str, fault_set: str, fault_family: str, diagnostic: _FaultDiagnosticRow
) -> dict[str, object]:
    return {
        "file_id": file_id,
        "fault_set": fault_set,
        "fault_family": fault_family,
        "retention": float(diagnostic.retention),
        "delta_gain": float(diagnostic.delta_gain),
        "direction_cosine": float(diagnostic.direction_cosine),
        "teacher_delta_norm": float(diagnostic.teacher_delta_norm),
        "student_delta_norm": float(diagnostic.student_delta_norm),
        "salient_distance_gain": float(diagnostic.salient_distance_gain),
        "frontend_observability": float(diagnostic.frontend_observability),
        "frontend_retention": float(diagnostic.frontend_retention),
        "adapter_delta_gain": float(diagnostic.adapter_delta_gain),
        "adapter_retention": float(diagnostic.adapter_retention),
        "transport_relative_error": float(diagnostic.transport_relative_error),
    }


def _fault_diagnostics(
    *,
    model: BandwiseReferenceAdapter,
    noisy_clean: np.ndarray,
    reference: np.ndarray,
    teacher_clean: np.ndarray,
    fault_noisy: np.ndarray,
    teacher_fault: np.ndarray,
    config: FPNAAConfig,
    device: torch.device,
) -> pd.DataFrame:
    values: dict[str, list[np.ndarray]] = {
        "retention": [],
        "delta_gain": [],
        "direction_cosine": [],
        "teacher_delta_norm": [],
        "student_delta_norm": [],
        "salient_distance_gain": [],
        "frontend_observability": [],
        "frontend_retention": [],
        "adapter_delta_gain": [],
        "adapter_retention": [],
        "transport_relative_error": [],
    }
    batch_size = min(config.training.batch_size, 64)
    use_amp = config.training.mixed_precision and device.type == "cuda"
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(noisy_clean), batch_size):
            stop = start + batch_size
            clean_input = torch.as_tensor(noisy_clean[start:stop], device=device)
            fault_input = torch.as_tensor(fault_noisy[start:stop], device=device)
            ref = torch.as_tensor(reference[start:stop], device=device)
            clean_teacher = torch.as_tensor(teacher_clean[start:stop], device=device)
            fault_teacher = torch.as_tensor(teacher_fault[start:stop], device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                clean_student = model(clean_input, ref)
                fault_student = model(fault_input, ref)
            clean_student = clean_student.float()
            fault_student = fault_student.float()
            clean_teacher = clean_teacher.float()
            fault_teacher = fault_teacher.float()
            student_delta = (fault_student - clean_student).reshape(len(clean_student), -1)
            teacher_delta = (fault_teacher - clean_teacher).reshape(len(clean_teacher), -1)
            frontend_delta = (fault_input.float() - clean_input.float()).reshape(
                len(clean_input), -1
            )
            student_norm = torch.linalg.vector_norm(student_delta, dim=1)
            teacher_norm = torch.linalg.vector_norm(teacher_delta, dim=1)
            frontend_norm = torch.linalg.vector_norm(frontend_delta, dim=1)
            gain = (student_norm + 1.0e-8) / (teacher_norm + 1.0e-8)
            retention = torch.exp(-torch.log(gain).abs())
            frontend_gain = (frontend_norm + 1.0e-8) / (teacher_norm + 1.0e-8)
            frontend_retention = torch.exp(-torch.log(frontend_gain).abs())
            adapter_gain = (student_norm + 1.0e-8) / (frontend_norm + 1.0e-8)
            adapter_retention = torch.exp(-torch.log(adapter_gain).abs())
            transport_error = torch.linalg.vector_norm(student_delta - teacher_delta, dim=1) / (
                teacher_norm + 1.0e-8
            )
            direction = torch.nn.functional.cosine_similarity(student_delta, teacher_delta, dim=1)
            teacher_distance = (
                1.0 - torch.nn.functional.cosine_similarity(fault_teacher, clean_teacher, dim=-1)
            ).clamp_min(0.0)
            student_distance = (
                1.0 - torch.nn.functional.cosine_similarity(fault_student, clean_student, dim=-1)
            ).clamp_min(0.0)
            teacher_patches = teacher_distance.reshape(len(clean_teacher), -1)
            student_patches = student_distance.reshape(len(clean_student), -1)
            patch_count = max(
                1, math.ceil(config.objective.score_patch_fraction * teacher_patches.shape[1])
            )
            teacher_salient, indices = torch.topk(
                teacher_patches, k=patch_count, dim=1, largest=True, sorted=False
            )
            student_salient = torch.gather(student_patches, 1, indices)
            salient_gain = (student_salient.sum(dim=1) + 1.0e-8) / (
                teacher_salient.sum(dim=1) + 1.0e-8
            )
            batch_values = {
                "retention": retention,
                "delta_gain": gain,
                "direction_cosine": direction,
                "teacher_delta_norm": teacher_norm,
                "student_delta_norm": student_norm,
                "salient_distance_gain": salient_gain,
                "frontend_observability": frontend_gain,
                "frontend_retention": frontend_retention,
                "adapter_delta_gain": adapter_gain,
                "adapter_retention": adapter_retention,
                "transport_relative_error": transport_error,
            }
            for name, value in batch_values.items():
                values[name].append(value.cpu().numpy())
    return pd.DataFrame(
        {name: np.concatenate(chunks).astype(np.float64) for name, chunks in values.items()}
    )


def _score_adapter(
    *,
    model: BandwiseReferenceAdapter,
    base_store: _BaseTokenStore,
    score_path: Path,
    metrics_path: Path,
    model_id: str,
    experiment_id: str,
    config: FPNAAConfig,
    device: torch.device,
    machine_filter: str | None = None,
) -> None:
    frame = base_store.frame
    train = frame.loc[frame["dataset_split"] == "dev_train"]
    test = frame.loc[frame["dataset_split"] == "dev_test"].reset_index(drop=True)
    if machine_filter is not None:
        train = train.loc[train["machine_type"].astype(str) == machine_filter]
        test = test.loc[test["machine_type"].astype(str) == machine_filter].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError(f"No train/test rows for machine filter: {machine_filter}")
    scores = np.full(len(test), np.nan, dtype=np.float64)
    groups = sorted(
        set(zip(test["machine_type"].astype(str), test["section"].astype(str), strict=True))
    )
    for machine, section in groups:
        train_rows = train.loc[
            (train["machine_type"].astype(str) == machine)
            & (train["section"].astype(str) == section)
        ]
        test_mask = (test["machine_type"].astype(str) == machine) & (
            test["section"].astype(str) == section
        )
        test_rows = test.loc[test_mask]
        train_tokens = _adapt_rows(model, base_store, train_rows, config, device)
        test_tokens = _adapt_rows(model, base_store, test_rows, config, device)
        train_pooled = accelerated_rdp_pool(
            train_tokens,
            gamma=config.backend.rdp_gamma,
            device=str(device),
            batch_size=config.frontend.inference_batch_size * 4,
            eps=config.backend.eps,
        )
        test_pooled = accelerated_rdp_pool(
            test_tokens,
            gamma=config.backend.rdp_gamma,
            device=str(device),
            batch_size=config.frontend.inference_batch_size * 4,
            eps=config.backend.eps,
        )
        group_scores, _ = accelerated_beam_scores(
            test_pooled,
            train_pooled,
            neighbors=config.backend.local_density_neighbors,
            device=str(device),
            eps=config.backend.eps,
        )
        scores[test.index[test_mask].to_numpy()] = group_scores
    if not np.isfinite(scores).all():
        raise RuntimeError("Candidate scoring did not cover every development-test clip")
    result = test[["file_id", "machine_type", "section", "domain", "condition"]].copy()
    result["anomaly_score"] = scores
    result["model_id"] = model_id
    result["experiment_id"] = experiment_id
    _atomic_csv(score_path, result[SCORE_COLUMNS])
    calculate_dcase2026_official_metrics(score_path, metrics_path)


def _adapt_rows(
    model: BandwiseReferenceAdapter,
    store: _BaseTokenStore,
    rows: pd.DataFrame,
    config: FPNAAConfig,
    device: torch.device,
) -> np.ndarray:
    near, far = store.select(rows)
    outputs: list[np.ndarray] = []
    batch_size = min(config.training.batch_size, 64)
    use_amp = config.training.mixed_precision and device.type == "cuda"
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(near), batch_size):
            target = torch.as_tensor(near[start : start + batch_size], device=device)
            reference = torch.as_tensor(far[start : start + batch_size], device=device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(target, reference)
            outputs.append(output.float().cpu().numpy())
    return np.concatenate(outputs)


def _screening_gate(
    *,
    summary: pd.DataFrame,
    output: Path,
    c0_scores: Path,
    config: FPNAAConfig,
) -> dict[str, object]:
    c0_metrics = output / "c0_metrics.json"
    if not c0_metrics.is_file():
        calculate_dcase2026_official_metrics(c0_scores, c0_metrics)
    c0_score = read_official_score(c0_metrics)
    means = summary.groupby("candidate", sort=True)["official_score"].mean()
    c1_score = float(means["c1_mse"])
    c2_score = float(means["c2_fault_preserving"])
    c2_rows = summary.loc[summary["candidate"] == "c2_fault_preserving"]
    in_median = float(c2_rows["in_support_retention_median"].median())
    in_q05 = float(c2_rows["in_support_retention_q05"].min())
    heldout_median = float(c2_rows["heldout_retention_median"].median())
    heldout_q05 = float(c2_rows["heldout_retention_q05"].min())
    decomposition = {
        "frontend_retention_median": float(
            c2_rows["in_support_frontend_retention_median"].median()
        ),
        "frontend_retention_worst_seed_q05": float(
            c2_rows["in_support_frontend_retention_q05"].min()
        ),
        "adapter_retention_median": float(c2_rows["in_support_adapter_retention_median"].median()),
        "adapter_retention_worst_seed_q05": float(
            c2_rows["in_support_adapter_retention_q05"].min()
        ),
        "transport_error_median": float(c2_rows["in_support_transport_error_median"].median()),
        "transport_error_worst_seed_q90": float(c2_rows["in_support_transport_error_q90"].max()),
        "note": "diagnostic only; the frozen combined retention checks remain authoritative",
    }
    machine_drops = _mean_machine_drops(output, c0_metrics, config.training.screening_seeds)
    worst_machine_drop = float(min(machine_drops.values()))
    checks = {
        "absolute_score": c2_score >= config.gates.screening_minimum_official_score,
        "gain_over_c0": c2_score - c0_score >= config.gates.screening_minimum_gain_over_c0,
        "gain_over_c1": c2_score - c1_score >= config.gates.screening_minimum_gain_over_c1,
        "in_support_retention_median": (
            in_median >= config.gates.fault_delta_retention_median_minimum
        ),
        "in_support_retention_q05": in_q05 >= config.gates.fault_delta_retention_q05_minimum,
        "heldout_retention_median": (
            heldout_median >= config.gates.heldout_fault_delta_retention_median_minimum
        ),
        "heldout_retention_q05": (
            heldout_q05 >= config.gates.heldout_fault_delta_retention_q05_minimum
        ),
        "worst_machine_drop": worst_machine_drop >= -config.gates.screening_maximum_machine_drop,
    }
    core = all(checks.values())
    return {
        "schema_version": 1,
        "gate": "G2_screening_core",
        "scores": {
            "c0": c0_score,
            "c1_mean": c1_score,
            "c2_mean": c2_score,
            "c2_minus_c0": c2_score - c0_score,
            "c2_minus_c1": c2_score - c1_score,
        },
        "retention": {
            "in_support_median_across_seeds": in_median,
            "in_support_worst_seed_q05": in_q05,
            "heldout_median_across_seeds": heldout_median,
            "heldout_worst_seed_q05": heldout_q05,
        },
        "observability_decomposition": decomposition,
        "machine_delta_c2_minus_c0": machine_drops,
        "checks": {**checks, "core_screening": core, "lomo": None},
        "passed": False,
        "note": "passed remains false until the preregistered LOMO gate is run",
    }


def _mean_machine_drops(
    output: Path,
    c0_metrics: Path,
    seeds: list[int],
) -> dict[str, float]:
    c0 = _machine_scores(c0_metrics)
    by_machine: dict[str, list[float]] = {machine: [] for machine in c0}
    for seed in seeds:
        metrics = output / f"seed{seed}" / "c2_fault_preserving" / "metrics.json"
        values = _machine_scores(metrics)
        if set(values) != set(c0):
            raise ValueError("Candidate and C0 metrics cover different machines")
        for machine, value in values.items():
            by_machine[machine].append(value)
    return {machine: float(np.mean(values) - c0[machine]) for machine, values in by_machine.items()}


def _machine_scores(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells: dict[str, list[float]] = {}
    for group, metrics in payload["groups"].items():
        machine = str(group).split("/", maxsplit=1)[0]
        cells.setdefault(machine, []).extend(float(value) for value in metrics.values())
    return {machine: _harmonic_mean(values) for machine, values in cells.items()}


def _preload_base_store(cache: Path, *, workers: int) -> _BaseTokenStore:
    frame = pd.read_parquet(cache / "index.parquet").sort_values("file_id", kind="stable")
    frame = frame.reset_index(drop=True)
    if frame.empty or frame["file_id"].duplicated().any():
        raise ValueError("Base cache index must contain unique rows")

    def load(relative: str) -> tuple[np.ndarray, np.ndarray]:
        path = cache / relative
        with np.load(path, allow_pickle=False) as payload:
            near = payload["near"].copy()
            far = payload["far"].copy()
        _validate_grid(near, path)
        _validate_grid(far, path)
        return near, far

    relatives = frame["feature_file"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = list(executor.map(load, relatives))
    near = np.stack([item[0] for item in loaded])
    far = np.stack([item[1] for item in loaded])
    mapping = {str(file_id): index for index, file_id in enumerate(frame["file_id"])}
    return _BaseTokenStore(frame, near, far, mapping)


def _preload_training_arrays(
    augmentation_cache: Path,
    base_store: _BaseTokenStore,
    *,
    workers: int,
) -> _TrainingArrays:
    frame = pd.read_parquet(augmentation_cache / "index.parquet")
    frame = frame.sort_values("file_id", kind="stable").reset_index(drop=True)
    names = ("noisy_clean", "reference", "fault_noisy", "fault_teacher")
    values = _load_named_augmentation_arrays(
        augmentation_cache,
        frame,
        names=names,
        workers=workers,
    )
    teacher_clean, _ = base_store.select(frame)
    return _TrainingArrays(frame, values[0], values[1], teacher_clean, values[2], values[3])


def _exclude_machine(arrays: _TrainingArrays, machine: str) -> _TrainingArrays:
    mask = arrays.frame["machine_type"].astype(str).to_numpy() != machine
    frame = arrays.frame.loc[mask].reset_index(drop=True)
    if frame.empty or machine in set(frame["machine_type"].astype(str)):
        raise ValueError(f"Failed to construct LOMO training fold for {machine}")
    return _TrainingArrays(
        frame=frame,
        noisy_clean=arrays.noisy_clean[mask],
        reference=arrays.reference[mask],
        teacher_clean=arrays.teacher_clean[mask],
        fault_noisy=arrays.fault_noisy[mask],
        teacher_fault=arrays.teacher_fault[mask],
    )


def _load_named_augmentation_arrays(
    cache: Path,
    frame: pd.DataFrame,
    *,
    names: tuple[str, ...],
    workers: int,
) -> tuple[np.ndarray, ...]:
    def load(relative: str) -> tuple[np.ndarray, ...]:
        path = cache / relative
        with np.load(path, allow_pickle=False) as payload:
            result = tuple(payload[name].copy() for name in names)
        for value in result:
            _validate_grid(value, path)
        return result

    with ThreadPoolExecutor(max_workers=workers) as executor:
        loaded = list(executor.map(load, frame["augmentation_file"].astype(str)))
    return tuple(np.stack([row[index] for row in loaded]) for index in range(len(names)))


def _validate_caches(base: Path, augmentation: Path, config: FPNAAConfig) -> None:
    for cache in (base, augmentation):
        if not (cache / "cache.json").is_file() or not (cache / "index.parquet").is_file():
            raise FileNotFoundError(f"Completed FP-NAA cache not found: {cache}")
    base_metadata = json.loads((base / "cache.json").read_text(encoding="utf-8"))
    augmentation_metadata = json.loads((augmentation / "cache.json").read_text(encoding="utf-8"))
    if base_metadata.get("checkpoint_sha256") != config.provenance.checkpoint_sha256:
        raise ValueError("Base cache checkpoint does not match config")
    if augmentation_metadata.get("checkpoint_sha256") != config.provenance.checkpoint_sha256:
        raise ValueError("Augmentation cache checkpoint does not match config")


def _new_model(config: FPNAAConfig, *, candidate: Candidate) -> BandwiseReferenceAdapter:
    safety_mode = (
        config.adapter.reference_safety_mode if candidate == "c2_fault_preserving" else "none"
    )
    return BandwiseReferenceAdapter(
        embedding_dim=config.frontend.embedding_dim,
        hidden_dim=config.adapter.hidden_dim,
        attention_heads=config.adapter.attention_heads,
        dropout=config.adapter.dropout,
        conditioning_mode=(
            config.adapter.c2_conditioning_mode
            if candidate == "c2_fault_preserving"
            else "target_conditioned"
        ),
        reference_safety_mode=safety_mode,
        reference_safety_fraction=config.adapter.reference_safety_fraction,
        maximum_reference_contraction=config.adapter.maximum_reference_contraction,
    )


def _learning_rate_factor(epoch: int, config: FPNAAConfig) -> float:
    return _cosine_learning_rate_factor(
        epoch,
        epochs=config.training.epochs,
        warmup_epochs=config.training.warmup_epochs,
    )


def _cosine_learning_rate_factor(epoch: int, *, epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return (epoch + 1) / max(warmup_epochs, 1)
    progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _configure_deterministic_runtime()


def _configure_deterministic_runtime() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _deterministic_runtime_metadata() -> dict[str, object]:
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "memory_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
    }


def _cuda_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FP-NAA screening requires a CUDA device")
    _configure_deterministic_runtime()
    return resolved


def _validate_grid(value: np.ndarray, path: Path) -> None:
    if value.ndim != 3 or value.dtype != np.float16 or not np.isfinite(value).all():
        raise ValueError(f"Invalid cached token grid: {path}")


def _harmonic_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or (array <= 0.0).any() or not np.isfinite(array).all():
        raise ValueError("Machine score requires positive finite cells")
    return float(len(array) / np.reciprocal(array).sum())


def _write_progress(output: Path, *, stage: str, completed: int, total: int) -> None:
    temporary = output / "progress.env.tmp"
    temporary.write_text(
        f"stage={stage}\ncompleted_runs={completed}\ntotal_runs={total}\n"
        f"updated_utc={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
        encoding="utf-8",
    )
    os.replace(temporary, output / "progress.env")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _ensure_contract(output: Path, contract: dict[str, object]) -> None:
    path = output / "contract.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(f"Immutable run contract mismatch: {path}")
        return
    _atomic_json(path, contract)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
