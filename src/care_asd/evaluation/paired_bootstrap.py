"""Paired, stratified development bootstrap for final MVP ablation comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from care_asd.evaluation.official_baseline import SCORE_COLUMNS, _partial_auc, _roc_auc


def write_paired_bootstrap_comparison(
    *,
    reference_scores: str | Path,
    candidate_scores: str | Path,
    output_path: str | Path,
    iterations: int = 2000,
    seed: int = 2026,
) -> Path:
    """Write a paired AUC/pAUC delta CI, resampling normal/anomaly within group."""
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite bootstrap report: {output}")
    if iterations < 100:
        raise ValueError("iterations must be at least 100")
    reference = _load_scores(Path(reference_scores))
    candidate = _load_scores(Path(candidate_scores))
    keys = ["file_id", "machine_type", "section", "domain", "condition"]
    merged = reference.merge(candidate, on=keys, how="inner", suffixes=("_reference", "_candidate"))
    if len(merged) != len(reference) or len(merged) != len(candidate):
        raise ValueError("Score files must cover the identical file_id set")
    rng = np.random.default_rng(seed)
    auc_deltas = np.empty(iterations, dtype=np.float64)
    pauc_deltas = np.empty(iterations, dtype=np.float64)
    grouped = list(merged.groupby(["machine_type", "section"], sort=True))
    for iteration in range(iterations):
        auc_reference: list[float] = []
        auc_candidate: list[float] = []
        pauc_reference: list[float] = []
        pauc_candidate: list[float] = []
        for _, group in grouped:
            sampled = _stratified_resample(group, rng)
            labels = (sampled["condition"] == "anomaly").to_numpy(dtype=int)
            reference_values = sampled["anomaly_score_reference"].to_numpy(dtype=float)
            candidate_values = sampled["anomaly_score_candidate"].to_numpy(dtype=float)
            auc_reference.append(_roc_auc(labels, reference_values))
            auc_candidate.append(_roc_auc(labels, candidate_values))
            pauc_reference.append(_partial_auc(labels, reference_values, max_fpr=0.1))
            pauc_candidate.append(_partial_auc(labels, candidate_values, max_fpr=0.1))
        auc_deltas[iteration] = float(np.mean(auc_candidate) - np.mean(auc_reference))
        pauc_deltas[iteration] = float(np.mean(pauc_candidate) - np.mean(pauc_reference))
    payload = {
        "candidate": str(candidate_scores),
        "iterations": iterations,
        "metric_delta_candidate_minus_reference": {
            "mean_auc": _interval(auc_deltas),
            "mean_pauc_max_fpr_0_1": _interval(pauc_deltas),
        },
        "reference": str(reference_scores),
        "seed": seed,
        "stratification": "machine_type/section/condition",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _load_scores(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Score file not found: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(SCORE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Score file missing required columns: {', '.join(missing)}")
    if frame["file_id"].duplicated().any() or not set(frame["condition"]).issubset(
        {"normal", "anomaly"}
    ):
        raise ValueError("Score file must have unique IDs and known development labels")
    return frame


def _stratified_resample(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for condition in ("normal", "anomaly"):
        subset = group.loc[group["condition"] == condition]
        if subset.empty:
            raise ValueError("Every bootstrap group requires normal and anomaly rows")
        parts.append(subset.iloc[rng.integers(0, len(subset), size=len(subset))])
    return pd.concat(parts, ignore_index=True)


def _interval(values: np.ndarray) -> dict[str, float]:
    return {
        "ci95_high": float(np.quantile(values, 0.975)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "mean": float(np.mean(values)),
    }
