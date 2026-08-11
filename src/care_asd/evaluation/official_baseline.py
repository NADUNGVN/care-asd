"""Normalize and score development outputs from the pinned official baseline."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ScoreMode = Literal["mse", "mahala"]
SCORE_COLUMNS = [
    "file_id",
    "machine_type",
    "section",
    "domain",
    "condition",
    "anomaly_score",
    "model_id",
    "experiment_id",
]
_OFFICIAL_SCORE_NAME = re.compile(
    r"^anomaly_score_(?:DCASE2026T2)?(?P<machine>.+?)_(?P<section>section_\d+)_test.*\.csv$"
)


def normalize_official_development_scores(
    *,
    official_score_directory: str | Path,
    manifest_path: str | Path,
    score_mode: ScoreMode,
    experiment_id: str,
    output_path: str | Path,
) -> Path:
    """Map official two-column score files into CARE-ASD's immutable score schema."""
    source = Path(official_score_directory)
    if not source.is_dir():
        raise FileNotFoundError(f"Official score directory not found: {source}")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite normalized scores: {output}")
    manifest = pd.read_parquet(manifest_path)
    expected = manifest.loc[manifest["dataset_split"] == "dev_test"].copy()
    required = {"file_id", "relative_path", "machine_type", "section", "domain", "condition"}
    missing = sorted(required.difference(expected.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    expected["basename"] = expected["relative_path"].map(lambda value: Path(str(value)).name)
    if expected.duplicated(["machine_type", "section", "basename"]).any():
        raise ValueError("Manifest score-matching keys are not unique")

    rows: list[dict[str, str | float]] = []
    for score_file in sorted(source.glob("anomaly_score_*.csv")):
        match = _OFFICIAL_SCORE_NAME.match(score_file.name)
        if match is None:
            continue
        machine = match.group("machine")
        section = match.group("section")
        lookup = expected.loc[
            (expected["machine_type"] == machine) & (expected["section"] == section)
        ].set_index("basename", verify_integrity=True)
        if lookup.empty:
            raise ValueError(f"No development test manifest rows match {score_file.name}")
        for basename, score in _read_official_score_file(score_file):
            if basename not in lookup.index:
                raise ValueError(f"Official score basename is absent from manifest: {basename}")
            item = lookup.loc[basename]
            rows.append(
                {
                    "file_id": str(item["file_id"]),
                    "machine_type": str(item["machine_type"]),
                    "section": str(item["section"]),
                    "domain": str(item["domain"]),
                    "condition": str(item["condition"]),
                    "anomaly_score": score,
                    "model_id": f"official_dcase2026_ae_{score_mode}",
                    "experiment_id": experiment_id,
                }
            )
    if not rows:
        raise FileNotFoundError(f"No parseable official anomaly-score CSV files in {source}")
    normalized = pd.DataFrame(rows, columns=SCORE_COLUMNS).sort_values("file_id", kind="stable")
    if normalized["file_id"].duplicated().any():
        raise ValueError("Official scores contain duplicate file_id values")
    if set(normalized["file_id"]) != set(expected["file_id"]):
        missing_scores = len(set(expected["file_id"]).difference(normalized["file_id"]))
        unexpected_scores = len(set(normalized["file_id"]).difference(expected["file_id"]))
        raise ValueError(
            f"Official score coverage mismatch: missing={missing_scores}, unexpected={unexpected_scores}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output, index=False)
    return output


def calculate_development_auc_metrics(score_path: str | Path, output_path: str | Path) -> Path:
    """Compute deterministic development AUC/pAUC directly from normalized scores."""
    source = Path(score_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite metrics: {output}")
    frame = pd.read_csv(source)
    missing = sorted(set(SCORE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Normalized scores are missing required columns: {', '.join(missing)}")
    if not set(frame["condition"]).issubset({"normal", "anomaly"}):
        raise ValueError("Development metrics require known normal/anomaly labels")
    metrics: dict[str, object] = {"score_file": source.name, "groups": {}}
    groups: dict[str, object] = {}
    for (machine, section), group in frame.groupby(["machine_type", "section"], sort=True):
        labels = (group["condition"] == "anomaly").to_numpy(dtype=int)
        scores = group["anomaly_score"].to_numpy(dtype=float)
        values: dict[str, float] = {
            "auc_all": _roc_auc(labels, scores),
            "pauc_all_max_fpr_0_1": _partial_auc(labels, scores, max_fpr=0.1),
        }
        for domain in ("source", "target"):
            domain_group = group.loc[group["domain"] == domain]
            domain_labels = (domain_group["condition"] == "anomaly").to_numpy(dtype=int)
            domain_scores = domain_group["anomaly_score"].to_numpy(dtype=float)
            values[f"auc_{domain}"] = _roc_auc(domain_labels, domain_scores)
            values[f"pauc_{domain}_max_fpr_0_1"] = _partial_auc(
                domain_labels, domain_scores, max_fpr=0.1
            )
        groups[f"{machine}/{section}"] = values
    metrics["groups"] = groups
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _read_official_score_file(path: Path) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            if len(row) != 2:
                raise ValueError(f"Expected two columns in {path.name}")
            rows.append((row[0], float(row[1])))
    return rows


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr = _roc_curve(labels, scores)
    return float(np.trapezoid(tpr, fpr))


def _partial_auc(labels: np.ndarray, scores: np.ndarray, *, max_fpr: float) -> float:
    fpr, tpr = _roc_curve(labels, scores)
    stop = int(np.searchsorted(fpr, max_fpr, side="right"))
    truncated_fpr = np.append(fpr[:stop], max_fpr)
    truncated_tpr = np.append(tpr[:stop], np.interp(max_fpr, fpr, tpr))
    partial = float(np.trapezoid(truncated_tpr, truncated_fpr))
    min_area = 0.5 * max_fpr**2
    return 0.5 * (1.0 + (partial - min_area) / (max_fpr - min_area))


def _roc_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be one-dimensional arrays of equal length")
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
