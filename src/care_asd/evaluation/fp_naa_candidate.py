"""Resumable, capacity-matched C1/C2 FP-NAA development screening."""

from __future__ import annotations

import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
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
from care_asd.models.fp_naa_objective import FPNaaObjective, fault_delta_retention, fp_naa_loss

Candidate = Literal["c1_mse", "c2_fault_preserving"]
CANDIDATES: tuple[Candidate, ...] = ("c1_mse", "c2_fault_preserving")


@dataclass(frozen=True)
class FPNaaScreeningResult:
    output_directory: Path
    summary_path: Path
    gate_path: Path
    core_gate_passed: bool


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
                    "trainable_parameters": trainable_parameter_count(model),
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
    model = _new_model(config).to(device)
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        if payload.get("candidate") != candidate or int(payload.get("seed", -1)) != seed:
            raise ValueError(f"Checkpoint contract mismatch: {checkpoint}")
        model.load_state_dict(payload["model_state"])
        return model.eval()
    _set_seed(seed)
    model = _new_model(config).to(device)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: _learning_rate_factor(epoch, config),
    )
    use_amp = config.training.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    objective = cast(FPNaaObjective, candidate)
    model.train()
    for epoch in range(config.training.epochs):
        totals = torch.zeros(5, dtype=torch.float64, device=device)
        examples = 0
        for noisy_clean, reference, teacher_clean, fault_noisy, teacher_fault in loader:
            noisy_clean = noisy_clean.to(device, non_blocking=True)
            reference = reference.to(device, non_blocking=True)
            teacher_clean = teacher_clean.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                student_clean = model(noisy_clean, reference)
                if candidate == "c1_mse":
                    loss = fp_naa_loss(
                        objective=objective,
                        student_clean=student_clean,
                        teacher_clean=teacher_clean,
                        config=config.objective,
                    )
                else:
                    fault_noisy = fault_noisy.to(device, non_blocking=True)
                    teacher_fault = teacher_fault.to(device, non_blocking=True)
                    student_fault = model(fault_noisy, reference)
                    loss = fp_naa_loss(
                        objective=objective,
                        student_clean=student_clean,
                        teacher_clean=teacher_clean,
                        student_fault=student_fault,
                        teacher_fault=teacher_fault,
                        config=config.objective,
                    )
            scaler.scale(loss.total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            batch = len(noisy_clean)
            totals += batch * torch.stack(
                (
                    loss.total.detach(),
                    loss.normal_mse.detach(),
                    loss.fault_direction.detach(),
                    loss.fault_magnitude.detach(),
                    loss.retention.detach().mean(),
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
                "retention": epoch_totals[4],
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
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    return model.eval()


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
    in_support = _retention_values(
        model=model,
        noisy_clean=arrays.noisy_clean,
        reference=arrays.reference,
        teacher_clean=arrays.teacher_clean,
        fault_noisy=arrays.fault_noisy,
        teacher_fault=arrays.teacher_fault,
        config=config,
        device=device,
    )
    for row, value in zip(arrays.frame.itertuples(index=False), in_support, strict=True):
        rows.append(
            {
                "file_id": str(row.file_id),
                "fault_set": "in_support",
                "fault_family": str(row.fault_family),
                "retention": float(value),
            }
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
    heldout_values = _retention_values(
        model=model,
        noisy_clean=heldout[0],
        reference=heldout[1],
        teacher_clean=teacher_clean,
        fault_noisy=heldout[2],
        teacher_fault=heldout[3],
        config=config,
        device=device,
    )
    for row, value in zip(heldout_frame.itertuples(index=False), heldout_values, strict=True):
        rows.append(
            {
                "file_id": str(row.file_id),
                "fault_set": "heldout",
                "fault_family": str(row.heldout_fault_family),
                "retention": float(value),
            }
        )
    _atomic_csv(output_path, pd.DataFrame(rows))


def _retention_values(
    *,
    model: BandwiseReferenceAdapter,
    noisy_clean: np.ndarray,
    reference: np.ndarray,
    teacher_clean: np.ndarray,
    fault_noisy: np.ndarray,
    teacher_fault: np.ndarray,
    config: FPNAAConfig,
    device: torch.device,
) -> np.ndarray:
    values: list[np.ndarray] = []
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
            retention = fault_delta_retention(
                clean_student.float(),
                fault_student.float(),
                clean_teacher.float(),
                fault_teacher.float(),
            )
            values.append(retention.cpu().numpy())
    return np.concatenate(values).astype(np.float64)


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
) -> None:
    frame = base_store.frame
    train = frame.loc[frame["dataset_split"] == "dev_train"]
    test = frame.loc[frame["dataset_split"] == "dev_test"].reset_index(drop=True)
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


def _new_model(config: FPNAAConfig) -> BandwiseReferenceAdapter:
    return BandwiseReferenceAdapter(
        embedding_dim=config.frontend.embedding_dim,
        hidden_dim=config.adapter.hidden_dim,
        attention_heads=config.adapter.attention_heads,
        dropout=config.adapter.dropout,
    )


def _learning_rate_factor(epoch: int, config: FPNAAConfig) -> float:
    warmup = config.training.warmup_epochs
    if epoch < warmup:
        return (epoch + 1) / max(warmup, 1)
    progress = (epoch - warmup) / max(config.training.epochs - warmup - 1, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _cuda_device(device: str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("FP-NAA screening requires a CUDA device")
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
