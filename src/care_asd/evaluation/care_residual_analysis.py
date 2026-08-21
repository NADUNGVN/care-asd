"""Post-hoc evidence for explaining the Phase 7 CARE residual regression."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from care_asd.data.official_vector_cache import OFFICIAL_FEATURE_DIM, load_official_vectors
from care_asd.evaluation.official_baseline import SCORE_COLUMNS


@dataclass(frozen=True)
class CareResidualAnalysisResult:
    """Immutable paths emitted by Phase 8 post-hoc residual analysis."""

    output_directory: Path
    per_clip_path: Path
    strata_path: Path
    correlations_path: Path
    report_path: Path


def analyze_care_residual_development(
    *,
    near_cache_directory: str | Path,
    residual_cache_directory: str | Path,
    reference_scores: str | Path,
    candidate_scores: str | Path,
    output_directory: str | Path,
) -> CareResidualAnalysisResult:
    """Join locked B00/B01 scores with per-clip log-Mel feature displacement.

    This is a post-hoc explanatory analysis of already frozen development scores.
    It must not be used to select new hyperparameters on the same development set.
    """
    near_cache = Path(near_cache_directory)
    residual_cache = Path(residual_cache_directory)
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite CARE residual analysis: {output}")
    near = _load_cache_index(near_cache, label="near")
    residual = _load_cache_index(residual_cache, label="residual")
    reference = _load_scores(Path(reference_scores), label="reference")
    candidate = _load_scores(Path(candidate_scores), label="candidate")
    analysis_index = _validate_and_join(near, residual, reference, candidate)
    output.mkdir(parents=True)

    feature_records = [
        _feature_displacement(
            near_cache / str(row.near_cache_file),
            residual_cache / str(row.residual_cache_file),
            file_id=str(row.file_id),
        )
        for row in analysis_index.itertuples(index=False)
    ]
    per_clip = analysis_index.merge(pd.DataFrame(feature_records), on="file_id", validate="one_to_one")
    per_clip["score_delta_b01_minus_b00"] = (
        per_clip["candidate_anomaly_score"] - per_clip["reference_anomaly_score"]
    )
    per_clip["absolute_score_delta"] = per_clip["score_delta_b01_minus_b00"].abs()
    per_clip_path = output / "per_clip_analysis.csv"
    per_clip.sort_values("file_id", kind="stable").to_csv(per_clip_path, index=False)

    strata = _summarize_strata(per_clip)
    strata_path = output / "strata.csv"
    strata.to_csv(strata_path, index=False)
    correlations = _summarize_correlations(per_clip)
    correlations_path = output / "correlations.csv"
    correlations.to_csv(correlations_path, index=False)

    report_path = output / "analysis.md"
    report_path.write_text(_render_report(strata, correlations), encoding="utf-8")
    (output / "run.json").write_text(
        json.dumps(
            {
                "candidate_scores": str(Path(candidate_scores)),
                "near_cache": str(near_cache),
                "purpose": "post_hoc_explanation_only_not_model_selection",
                "reference_scores": str(Path(reference_scores)),
                "residual_cache": str(residual_cache),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return CareResidualAnalysisResult(
        output, per_clip_path, strata_path, correlations_path, report_path
    )


def _load_cache_index(directory: Path, *, label: str) -> pd.DataFrame:
    index_path = directory / "index.parquet"
    metadata_path = directory / "cache.json"
    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"{label} cache requires index.parquet and cache.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("feature_dim", -1)) != OFFICIAL_FEATURE_DIM:
        raise ValueError(f"{label} cache does not have the official 640-vector contract")
    frame = pd.read_parquet(index_path)
    required = {
        "file_id",
        "machine_type",
        "section",
        "domain",
        "condition",
        "dataset_split",
        "cache_file",
        "vector_count",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} cache index missing: {', '.join(missing)}")
    if frame["file_id"].duplicated().any():
        raise ValueError(f"{label} cache index has duplicate file_id values")
    suffix = "near" if label == "near" else "residual"
    return frame.rename(columns={"cache_file": f"{suffix}_cache_file", "vector_count": f"{suffix}_vector_count"})


def _load_scores(path: Path, *, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{label} score file does not exist: {path}")
    frame = pd.read_csv(path)
    missing = sorted(set(SCORE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{label} score file missing: {', '.join(missing)}")
    if frame["file_id"].duplicated().any():
        raise ValueError(f"{label} score file has duplicate file_id values")
    return frame[["file_id", "anomaly_score"]].rename(
        columns={"anomaly_score": f"{label}_anomaly_score"}
    )


def _validate_and_join(
    near: pd.DataFrame,
    residual: pd.DataFrame,
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    near_test = near.loc[near["dataset_split"] == "dev_test"].copy()
    residual_test = residual.loc[residual["dataset_split"] == "dev_test"].copy()
    if near_test.empty or residual_test.empty:
        raise ValueError("Both caches must contain development-test clips")
    identity = ["file_id", "machine_type", "section", "domain", "condition"]
    joined = near_test.merge(
        residual_test[[*identity, "residual_cache_file", "residual_vector_count"]],
        on=identity,
        how="inner",
        validate="one_to_one",
    ).merge(reference, on="file_id", how="inner", validate="one_to_one").merge(
        candidate, on="file_id", how="inner", validate="one_to_one"
    )
    expected = len(near_test)
    if len(joined) != expected or len(residual_test) != expected or len(reference) != expected or len(candidate) != expected:
        raise ValueError("B00/B01 caches and score files must cover the identical development test set")
    if (joined["near_vector_count"] != joined["residual_vector_count"]).any():
        raise ValueError("B00/B01 vector counts differ; comparison is not frame-aligned")
    return joined


def _feature_displacement(near_path: Path, residual_path: Path, *, file_id: str) -> dict[str, str | float]:
    near = load_official_vectors(near_path)
    residual = load_official_vectors(residual_path)
    if near.shape != residual.shape:
        raise ValueError(f"B00/B01 vector shape differs for {file_id}: {near.shape} vs {residual.shape}")
    difference = residual.astype(np.float64) - near.astype(np.float64)
    return {
        "file_id": file_id,
        "near_logmel_mean_db": float(np.mean(near)),
        "residual_logmel_mean_db": float(np.mean(residual)),
        "residual_minus_near_logmel_db": float(np.mean(difference)),
        "feature_shift_l1_db": float(np.mean(np.abs(difference))),
    }


def _summarize_strata(per_clip: pd.DataFrame) -> pd.DataFrame:
    groups = ["machine_type", "section", "domain", "condition"]
    return (
        per_clip.groupby(groups, sort=True, dropna=False)
        .agg(
            clips=("file_id", "count"),
            score_delta_mean=("score_delta_b01_minus_b00", "mean"),
            score_delta_median=("score_delta_b01_minus_b00", "median"),
            residual_minus_near_logmel_db_mean=("residual_minus_near_logmel_db", "mean"),
            feature_shift_l1_db_mean=("feature_shift_l1_db", "mean"),
        )
        .reset_index()
    )


def _summarize_correlations(per_clip: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str | float | int]] = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", per_clip)]
    groups.extend(
        (f"{machine_type}/{section}", frame)
        for (machine_type, section), frame in per_clip.groupby(["machine_type", "section"], sort=True)
    )
    for group, frame in groups:
        for outcome in ("score_delta_b01_minus_b00", "absolute_score_delta"):
            rho, p_value = _safe_spearman(frame["residual_minus_near_logmel_db"], frame[outcome])
            rows.append(
                {
                    "group": group,
                    "outcome": outcome,
                    "clips": len(frame),
                    "spearman_rho": rho,
                    "two_sided_p_value": p_value,
                }
            )
    return pd.DataFrame(rows)


def _safe_spearman(left: pd.Series, right: pd.Series) -> tuple[float, float]:
    if left.nunique() < 2 or right.nunique() < 2:
        return float("nan"), float("nan")
    result = spearmanr(left, right)
    return float(result.statistic), float(result.pvalue)


def _render_report(strata: pd.DataFrame, correlations: pd.DataFrame) -> str:
    all_correlation = correlations.loc[
        (correlations["group"] == "all")
        & (correlations["outcome"] == "score_delta_b01_minus_b00")
    ].iloc[0]
    worst = strata.sort_values("score_delta_mean", kind="stable").head(5)
    lines = [
        "# Phase 8 CARE residual failure analysis",
        "",
        "This is a post-hoc explanation of frozen B00/B01 development results; it is not tuning evidence.",
        "",
        "## Global association",
        "",
        f"- Spearman rho between residual-minus-near mean log-Mel displacement and B01-B00 score delta: {all_correlation['spearman_rho']:.4f} (two-sided p={all_correlation['two_sided_p_value']:.4g}).",
        "- A more negative residual-minus-near value means CARE removed more log-Mel energy. It is a feature-displacement diagnostic, not a calibrated physical energy ratio.",
        "",
        "## Strata with most negative mean score shift",
        "",
        "| machine / section / domain / condition | clips | mean score delta | mean log-Mel displacement (dB) |",
        "|---|---:|---:|---:|",
    ]
    for row in worst.itertuples(index=False):
        label = f"{row.machine_type} / {row.section} / {row.domain} / {row.condition}"
        lines.append(
            f"| {label} | {row.clips} | {row.score_delta_mean:.6f} | {row.residual_minus_near_logmel_db_mean:.4f} |"
        )
    lines.extend(
        [
            "",
            "See `per_clip_analysis.csv`, `strata.csv`, and `correlations.csv` for machine-readable evidence.",
            "",
        ]
    )
    return "\n".join(lines)
