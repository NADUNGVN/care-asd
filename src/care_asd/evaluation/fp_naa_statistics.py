"""Exact-score paired inference for confirmatory FP-NAA comparisons."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from care_asd.evaluation.dcase2026_metrics import _partial_auc, _roc_auc
from care_asd.evaluation.official_baseline import SCORE_COLUMNS


@dataclass(frozen=True)
class _PairedGroup:
    machine: str
    section: str
    domain: np.ndarray
    condition: np.ndarray
    reference: np.ndarray
    candidate: np.ndarray
    strata: tuple[np.ndarray, ...]


def write_exact_official_paired_bootstrap(
    *,
    reference_scores: str | Path,
    candidate_scores: str | Path,
    output_path: str | Path,
    iterations: int = 10_000,
    seed: int = 2608,
) -> Path:
    """Bootstrap the paired delta of the exact DCASE 2026 harmonic score.

    Clips are resampled with replacement inside machine/section/domain/condition strata. The same
    sampled clip indices are used for both systems, and every replicate recomputes all source AUC,
    target AUC, and pooled pAUC cells before taking the official harmonic mean.
    """
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bootstrap output: {output}")
    reference = _load_scores(Path(reference_scores), "reference")
    candidate = _load_scores(Path(candidate_scores), "candidate")
    keys = ["file_id", "machine_type", "section", "domain", "condition"]
    merged = reference.merge(
        candidate,
        on=keys,
        how="inner",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("Reference and candidate scores must cover identical development clips")
    groups = _paired_groups(merged)
    observed_reference = _score_groups(groups, score_name="reference")
    observed_candidate = _score_groups(groups, score_name="candidate")
    rng = np.random.default_rng(seed)
    deltas = np.empty(iterations, dtype=np.float64)
    reference_distribution = np.empty(iterations, dtype=np.float64)
    candidate_distribution = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        reference_cells: list[float] = []
        candidate_cells: list[float] = []
        for group in groups:
            sampled = np.concatenate(
                [
                    indices[rng.integers(0, len(indices), size=len(indices))]
                    for indices in group.strata
                ]
            )
            reference_cells.extend(_group_cells(group, group.reference, sampled))
            candidate_cells.extend(_group_cells(group, group.candidate, sampled))
        reference_value = _harmonic_mean(reference_cells)
        candidate_value = _harmonic_mean(candidate_cells)
        reference_distribution[iteration] = reference_value
        candidate_distribution[iteration] = candidate_value
        deltas[iteration] = candidate_value - reference_value
    payload: dict[str, object] = {
        "schema_version": 1,
        "metric_contract": (
            "exact DCASE2026 official harmonic mean over source AUC, target AUC, and pAUC@0.1"
        ),
        "stratification": "machine_type/section/domain/condition",
        "paired_resampling": True,
        "reference": str(reference_scores),
        "candidate": str(candidate_scores),
        "iterations": iterations,
        "seed": seed,
        "observed": {
            "reference_official_score": observed_reference,
            "candidate_official_score": observed_candidate,
            "delta_candidate_minus_reference": observed_candidate - observed_reference,
        },
        "bootstrap": {
            "reference_official_score": _interval(reference_distribution),
            "candidate_official_score": _interval(candidate_distribution),
            "delta_candidate_minus_reference": {
                **_interval(deltas),
                "probability_greater_than_zero": float(np.mean(deltas > 0.0)),
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def _paired_groups(frame: pd.DataFrame) -> tuple[_PairedGroup, ...]:
    groups: list[_PairedGroup] = []
    for (machine, section), group in frame.groupby(["machine_type", "section"], sort=True):
        group = group.reset_index(drop=True)
        domain = group["domain"].astype(str).to_numpy()
        condition = group["condition"].astype(str).to_numpy()
        strata = tuple(
            np.flatnonzero((domain == domain_name) & (condition == condition_name))
            for domain_name in ("source", "target")
            for condition_name in ("normal", "anomaly")
            if np.any((domain == domain_name) & (condition == condition_name))
        )
        if not np.any(condition == "anomaly"):
            raise ValueError(f"No anomaly rows for {machine}/{section}")
        for domain_name in ("source", "target"):
            if not np.any((domain == domain_name) & (condition == "normal")):
                raise ValueError(f"No {domain_name} normal rows for {machine}/{section}")
        groups.append(
            _PairedGroup(
                machine=str(machine),
                section=str(section),
                domain=domain,
                condition=condition,
                reference=group["anomaly_score_reference"].to_numpy(dtype=np.float64),
                candidate=group["anomaly_score_candidate"].to_numpy(dtype=np.float64),
                strata=strata,
            )
        )
    if not groups:
        raise ValueError("Paired bootstrap requires at least one machine/section group")
    return tuple(groups)


def _score_groups(groups: tuple[_PairedGroup, ...], *, score_name: str) -> float:
    cells: list[float] = []
    for group in groups:
        scores = group.reference if score_name == "reference" else group.candidate
        cells.extend(_group_cells(group, scores, np.arange(len(scores))))
    return _harmonic_mean(cells)


def _group_cells(group: _PairedGroup, scores: np.ndarray, indices: np.ndarray) -> list[float]:
    domain = group.domain[indices]
    condition = group.condition[indices]
    values = scores[indices]
    anomaly = condition == "anomaly"
    cells: list[float] = []
    for domain_name in ("source", "target"):
        normal = (condition == "normal") & (domain == domain_name)
        labels = np.concatenate(
            [np.zeros(int(normal.sum()), dtype=np.int8), np.ones(int(anomaly.sum()), dtype=np.int8)]
        )
        selected_scores = np.concatenate([values[normal], values[anomaly]])
        cells.append(_roc_auc(labels, selected_scores))
    cells.append(_partial_auc(anomaly.astype(np.int8), values, max_fpr=0.1))
    return cells


def _harmonic_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Official score requires finite metric cells")
    if (array <= 0.0).any():
        return 0.0
    return float(array.size / np.reciprocal(array).sum())


def _interval(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
    }


def _load_scores(path: Path, name: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{name} score file not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(SCORE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} scores are missing columns: {', '.join(missing)}")
    if frame["file_id"].duplicated().any() or not np.isfinite(frame["anomaly_score"]).all():
        raise ValueError(f"{name} scores require unique IDs and finite anomaly scores")
    return frame
