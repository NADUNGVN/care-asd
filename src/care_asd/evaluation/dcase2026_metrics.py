"""Exact DCASE 2026 Task 2 development metric contract.

This module intentionally coexists with ``official_baseline.py``. Historical CARE-ASD artifacts
used a same-domain positive subset for domain AUCs; changing that function would invalidate those
artifacts. DCASE 2026 instead compares normal clips from one domain with anomalous clips from both
domains.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from care_asd.evaluation.official_baseline import SCORE_COLUMNS


def calculate_dcase2026_official_metrics(
    score_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write per-section cells and the exact official harmonic-mean score."""
    source = Path(score_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {output}")
    frame = pd.read_csv(source)
    missing = sorted(set(SCORE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Normalized scores are missing required columns: {', '.join(missing)}")
    if frame["file_id"].duplicated().any():
        raise ValueError("Scores contain duplicate file_id values")
    if not set(frame["condition"]).issubset({"normal", "anomaly"}):
        raise ValueError("Development metrics require known normal/anomaly labels")
    if not set(frame["domain"]).issubset({"source", "target"}):
        raise ValueError("DCASE 2026 metrics require source/target domains")

    groups: dict[str, dict[str, float]] = {}
    official_cells: list[float] = []
    for (machine, section), group in frame.groupby(["machine_type", "section"], sort=True):
        anomaly = group.loc[group["condition"] == "anomaly"]
        if anomaly.empty:
            raise ValueError(f"No anomalous clips for {machine}/{section}")
        values: dict[str, float] = {}
        for domain in ("source", "target"):
            normal = group.loc[
                (group["condition"] == "normal") & (group["domain"] == domain)
            ]
            if normal.empty:
                raise ValueError(f"No {domain} normal clips for {machine}/{section}")
            labels = np.concatenate(
                [np.zeros(len(normal), dtype=np.int8), np.ones(len(anomaly), dtype=np.int8)]
            )
            scores = np.concatenate(
                [
                    normal["anomaly_score"].to_numpy(dtype=np.float64),
                    anomaly["anomaly_score"].to_numpy(dtype=np.float64),
                ]
            )
            value = _roc_auc(labels, scores)
            values[f"auc_{domain}"] = value
            official_cells.append(value)

        labels_all = (group["condition"] == "anomaly").to_numpy(dtype=np.int8)
        scores_all = group["anomaly_score"].to_numpy(dtype=np.float64)
        pauc = _partial_auc(labels_all, scores_all, max_fpr=0.1)
        values["pauc_all_max_fpr_0_1"] = pauc
        official_cells.append(pauc)
        groups[f"{machine}/{section}"] = values

    official_score = _harmonic_mean(official_cells)
    payload: dict[str, object] = {
        "metric_contract": (
            "DCASE2026: domain normal versus all anomalies; pooled pAUC@0.1; "
            "harmonic mean over every cell"
        ),
        "score_file": source.name,
        "official_score": official_score,
        "official_score_percent": 100.0 * official_score,
        "cell_count": len(official_cells),
        "groups": groups,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def read_official_score(metrics_path: str | Path) -> float:
    """Read a validated official score from a metrics artifact."""
    payload = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    value = float(payload["official_score"])
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid official_score in {metrics_path}: {value}")
    return value


def _harmonic_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Official score requires finite metric cells")
    if (array <= 0.0).any():
        return 0.0
    return float(array.size / np.reciprocal(array).sum())


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr = _roc_curve(labels, scores)
    return float(np.trapezoid(tpr, fpr))


def _partial_auc(labels: np.ndarray, scores: np.ndarray, *, max_fpr: float) -> float:
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    fpr, tpr = _roc_curve(labels, scores)
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    truncated_fpr = np.append(fpr[:stop], max_fpr)
    truncated_tpr = np.append(tpr[:stop], np.interp(max_fpr, fpr, tpr))
    partial = float(np.trapezoid(truncated_tpr, truncated_fpr))
    # McClish standardization, matching sklearn.metrics.roc_auc_score(max_fpr=...).
    minimum_area = 0.5 * max_fpr**2
    return 0.5 * (1.0 + (partial - minimum_area) / (max_fpr - minimum_area))


def _roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUC requires both normal and anomaly examples")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    change = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    true_positive = np.cumsum(sorted_labels)[change]
    false_positive = 1 + change - true_positive
    return np.r_[0.0, false_positive / negatives], np.r_[0.0, true_positive / positives]

