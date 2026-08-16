"""Frozen machine/domain robustness appendix for Audit-A2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from care_asd.evaluation.audit_synthesis import portable_sha256
from care_asd.evaluation.official_baseline import SCORE_COLUMNS, _partial_auc, _roc_auc
from care_asd.reproducibility import get_git_commit

KEY_COLUMNS = ["file_id", "machine_type", "section", "domain", "condition"]
SYSTEM_ORDER = ("B00", "B01", "B02")
CANDIDATE_ORDER = ("B01", "B02")
DOMAIN_ORDER = ("all", "source", "target")
METRIC_ORDER = ("auc", "pauc_max_fpr_0_1")


class RobustnessSourceConfig(BaseModel):
    """Frozen score and global-bootstrap sources accepted by Audit-A2."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    b00_scores: Path
    b01_scores: Path
    b02_scores: Path
    b01_bootstrap: Path
    b02_bootstrap: Path

    @field_validator("*")
    @classmethod
    def source_paths_are_repository_relative(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("robustness source paths must be repository-relative without '..'")
        return value


class RobustnessAnalysisConfig(BaseModel):
    """Statistical contract; values are descriptive and cannot tune a model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reference: Literal["B00"] = "B00"
    candidates: tuple[Literal["B01", "B02"], Literal["B01", "B02"]] = ("B01", "B02")
    bootstrap_iterations: int = Field(default=2000, ge=100)
    seed: int = Field(default=2026, ge=0)
    max_fpr: Literal[0.1] = 0.1
    resampling: Literal["paired_within_machine_domain_condition"] = (
        "paired_within_machine_domain_condition"
    )
    aggregation: Literal["arithmetic_mean_over_machine_sections"] = (
        "arithmetic_mean_over_machine_sections"
    )

    @field_validator("candidates")
    @classmethod
    def candidate_order_is_frozen(
        cls, value: tuple[Literal["B01", "B02"], Literal["B01", "B02"]]
    ) -> tuple[Literal["B01", "B02"], Literal["B01", "B02"]]:
        if value != CANDIDATE_ORDER:
            raise ValueError("Audit-A2 candidates must be exactly [B01, B02] in that order")
        return value


class RobustnessDecisionConfig(BaseModel):
    """Actions prohibited while producing the frozen appendix."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    audio_access: Literal["prohibited"] = "prohibited"
    evaluation_access: Literal["prohibited"] = "prohibited"
    model_tuning: Literal["prohibited"] = "prohibited"
    model_training: Literal["prohibited"] = "prohibited"


class RobustnessAppendixConfig(BaseModel):
    """Versioned Audit-A2 input contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    study_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    sources: RobustnessSourceConfig
    analysis: RobustnessAnalysisConfig = RobustnessAnalysisConfig()
    decisions: RobustnessDecisionConfig = RobustnessDecisionConfig()


@dataclass(frozen=True)
class RobustnessAppendixResult:
    """Paths written by an immutable Audit-A2 synthesis."""

    output_directory: Path
    coverage_path: Path
    group_metrics_path: Path
    bootstrap_path: Path
    global_inference_path: Path
    leave_one_out_path: Path
    heterogeneity_path: Path
    summary_path: Path
    figure_paths: tuple[Path, ...]
    run_path: Path


def load_robustness_appendix_config(path: str | Path) -> RobustnessAppendixConfig:
    """Load and strictly validate the Audit-A2 contract."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Robustness config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Robustness config root must be a mapping")
    return RobustnessAppendixConfig.model_validate(payload)


def robustness_appendix_plan(
    config: RobustnessAppendixConfig,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate frozen sources and return a side-effect-free plan."""
    root = Path(repository_root).resolve()
    sources = _resolved_sources(config, root)
    scores = _load_aligned_scores(sources)
    global_inference = _load_global_inference(sources)
    coverage = _coverage_payload(scores)
    return {
        "schema_version": 1,
        "study_id": config.study_id,
        "git_commit": get_git_commit(root),
        "analysis": config.analysis.model_dump(mode="json"),
        "decisions": config.decisions.model_dump(mode="json"),
        "coverage": coverage,
        "frozen_global_inference": global_inference,
        "source_hashes": {
            name: {
                "path": config.sources.model_dump(mode="json")[name],
                "sha256": portable_sha256(path),
            }
            for name, path in sources.items()
        },
    }


def run_robustness_appendix(
    *,
    output_directory: str | Path,
    config: RobustnessAppendixConfig,
    repository_root: str | Path,
) -> RobustnessAppendixResult:
    """Create immutable machine/domain robustness tables and figures."""
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite robustness appendix: {output}")
    root = Path(repository_root).resolve()
    sources = _resolved_sources(config, root)
    scores = _load_aligned_scores(sources)
    plan = robustness_appendix_plan(config, repository_root=root)

    group_metrics = _group_metrics(scores, max_fpr=config.analysis.max_fpr)
    bootstrap = _group_bootstrap(
        scores,
        iterations=config.analysis.bootstrap_iterations,
        seed=config.analysis.seed,
        max_fpr=config.analysis.max_fpr,
    )
    global_inference = _global_inference_table(plan["frozen_global_inference"])
    leave_one_out = _leave_one_machine_out(group_metrics)
    heterogeneity = _heterogeneity_payload(group_metrics, leave_one_out)

    output.mkdir(parents=True)
    coverage_path = output / "score_coverage.json"
    group_metrics_path = output / "machine_domain_metrics.csv"
    bootstrap_path = output / "machine_domain_bootstrap.csv"
    global_inference_path = output / "frozen_global_inference.csv"
    leave_one_out_path = output / "leave_one_machine_out.csv"
    heterogeneity_path = output / "heterogeneity.json"
    summary_path = output / "robustness_appendix.md"
    forest_path = output / "machine_pauc_forest.svg"
    heatmap_path = output / "machine_domain_pauc_heatmap.svg"

    coverage_path.write_text(
        json.dumps(plan["coverage"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    group_metrics.to_csv(group_metrics_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    global_inference.to_csv(global_inference_path, index=False)
    leave_one_out.to_csv(leave_one_out_path, index=False)
    heterogeneity_path.write_text(
        json.dumps(heterogeneity, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        _summary_markdown(group_metrics, global_inference, leave_one_out), encoding="utf-8"
    )
    forest_path.write_text(_forest_svg(bootstrap), encoding="utf-8")
    heatmap_path.write_text(_heatmap_svg(group_metrics), encoding="utf-8")
    figure_paths = (forest_path, heatmap_path)

    artifact_paths = (
        coverage_path,
        group_metrics_path,
        bootstrap_path,
        global_inference_path,
        leave_one_out_path,
        heterogeneity_path,
        summary_path,
        *figure_paths,
    )
    run_payload = {
        **plan,
        "artifacts": {path.name: portable_sha256(path) for path in artifact_paths},
    }
    run_path = output / "run.json"
    run_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RobustnessAppendixResult(
        output,
        coverage_path,
        group_metrics_path,
        bootstrap_path,
        global_inference_path,
        leave_one_out_path,
        heterogeneity_path,
        summary_path,
        figure_paths,
        run_path,
    )


def _resolved_sources(config: RobustnessAppendixConfig, root: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, relative in config.sources.model_dump().items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Robustness source escapes repository root: {relative}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"Robustness source does not exist: {relative}")
        resolved[name] = path
    return resolved


def _load_aligned_scores(sources: dict[str, Path]) -> dict[str, pd.DataFrame]:
    paths = {
        "B00": sources["b00_scores"],
        "B01": sources["b01_scores"],
        "B02": sources["b02_scores"],
    }
    frames: dict[str, pd.DataFrame] = {}
    for system, path in paths.items():
        frame = pd.read_csv(path)
        missing = sorted(set(SCORE_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"{system} score file missing columns: {', '.join(missing)}")
        if frame[KEY_COLUMNS].duplicated().any():
            raise ValueError(f"{system} score file contains duplicate paired keys")
        if not set(frame["domain"]).issubset({"source", "target"}):
            raise ValueError(f"{system} score file contains an unknown domain")
        if not set(frame["condition"]).issubset({"normal", "anomaly"}):
            raise ValueError(f"{system} score file contains an unknown condition")
        if not np.isfinite(frame["anomaly_score"].to_numpy(dtype=float)).all():
            raise ValueError(f"{system} score file contains non-finite scores")
        frames[system] = frame.sort_values(KEY_COLUMNS, kind="mergesort").reset_index(drop=True)
    reference = frames["B00"][KEY_COLUMNS]
    for system in CANDIDATE_ORDER:
        if not reference.equals(frames[system][KEY_COLUMNS]):
            raise ValueError(f"{system} does not cover the identical paired clip metadata as B00")
    counts = (
        frames["B00"].groupby(["machine_type", "section", "domain", "condition"], sort=True).size()
    )
    if counts.nunique() != 1 or int(counts.iloc[0]) < 2:
        raise ValueError("Every machine/section/domain/condition stratum must be balanced")
    return frames


def _load_global_inference(sources: dict[str, Path]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for system, source_name in (("B01", "b01_bootstrap"), ("B02", "b02_bootstrap")):
        payload = json.loads(sources[source_name].read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{system} bootstrap root must be a mapping")
        metrics = payload.get("metric_delta_candidate_minus_reference")
        if not isinstance(metrics, dict) or set(metrics) != {
            "mean_auc",
            "mean_pauc_max_fpr_0_1",
        }:
            raise ValueError(f"{system} bootstrap has an unexpected metric contract")
        for metric in metrics.values():
            if not isinstance(metric, dict) or set(metric) != {"ci95_high", "ci95_low", "mean"}:
                raise ValueError(f"{system} bootstrap interval contract is invalid")
            if not all(np.isfinite(float(value)) for value in metric.values()):
                raise ValueError(f"{system} bootstrap contains non-finite values")
        payloads[system] = payload
    return payloads


def _coverage_payload(scores: dict[str, pd.DataFrame]) -> dict[str, Any]:
    reference = scores["B00"]
    strata = reference.groupby(["machine_type", "section", "domain", "condition"], sort=True).size()
    return {
        "systems": {system: len(frame) for system, frame in scores.items()},
        "paired_clips": int(reference["file_id"].nunique()),
        "machine_sections": int(reference.groupby(["machine_type", "section"]).ngroups),
        "domains": sorted(reference["domain"].unique().tolist()),
        "conditions": sorted(reference["condition"].unique().tolist()),
        "clips_per_machine_domain_condition": sorted({int(value) for value in strata}),
        "identical_pairing": True,
    }


def _group_metrics(scores: dict[str, pd.DataFrame], *, max_fpr: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for system in SYSTEM_ORDER:
        for (machine, section), machine_group in scores[system].groupby(
            ["machine_type", "section"], sort=True
        ):
            for domain in DOMAIN_ORDER:
                group = (
                    machine_group
                    if domain == "all"
                    else machine_group.loc[machine_group["domain"] == domain]
                )
                values = _metric_values(group, "anomaly_score", max_fpr=max_fpr)
                rows.append(
                    {
                        "system": system,
                        "machine_type": machine,
                        "section": section,
                        "domain": domain,
                        "normal_clips": int((group["condition"] == "normal").sum()),
                        "anomaly_clips": int((group["condition"] == "anomaly").sum()),
                        **values,
                    }
                )
    return pd.DataFrame(rows)


def _group_bootstrap(
    scores: dict[str, pd.DataFrame],
    *,
    iterations: int,
    seed: int,
    max_fpr: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    reference = scores["B00"]
    for candidate in CANDIDATE_ORDER:
        merged = reference[[*KEY_COLUMNS, "anomaly_score"]].merge(
            scores[candidate][[*KEY_COLUMNS, "anomaly_score"]],
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
            suffixes=("_reference", "_candidate"),
        )
        for (machine, section), machine_group in merged.groupby(
            ["machine_type", "section"], sort=True
        ):
            for domain in DOMAIN_ORDER:
                group = (
                    machine_group
                    if domain == "all"
                    else machine_group.loc[machine_group["domain"] == domain]
                ).reset_index(drop=True)
                reference_metrics = _metric_values(
                    group, "anomaly_score_reference", max_fpr=max_fpr
                )
                candidate_metrics = _metric_values(
                    group, "anomaly_score_candidate", max_fpr=max_fpr
                )
                strata = [
                    indices.to_numpy(dtype=int)
                    for _, indices in group.groupby(
                        ["domain", "condition"], sort=True
                    ).groups.items()
                ]
                deltas = {metric: np.empty(iterations, dtype=np.float64) for metric in METRIC_ORDER}
                for iteration in range(iterations):
                    sampled_indices = np.concatenate(
                        [
                            indices[rng.integers(0, len(indices), size=len(indices))]
                            for indices in strata
                        ]
                    )
                    sampled = group.iloc[sampled_indices]
                    sampled_reference = _metric_values(
                        sampled, "anomaly_score_reference", max_fpr=max_fpr
                    )
                    sampled_candidate = _metric_values(
                        sampled, "anomaly_score_candidate", max_fpr=max_fpr
                    )
                    for metric in METRIC_ORDER:
                        deltas[metric][iteration] = (
                            sampled_candidate[metric] - sampled_reference[metric]
                        )
                for metric in METRIC_ORDER:
                    values = deltas[metric]
                    rows.append(
                        {
                            "reference": "B00",
                            "candidate": candidate,
                            "machine_type": machine,
                            "section": section,
                            "domain": domain,
                            "metric": metric,
                            "observed_reference": reference_metrics[metric],
                            "observed_candidate": candidate_metrics[metric],
                            "observed_delta": candidate_metrics[metric] - reference_metrics[metric],
                            "bootstrap_mean_delta": float(np.mean(values)),
                            "ci95_low": float(np.quantile(values, 0.025)),
                            "ci95_high": float(np.quantile(values, 0.975)),
                            "probability_delta_gt_zero": float(np.mean(values > 0.0)),
                            "iterations": iterations,
                            "seed": seed,
                        }
                    )
    return pd.DataFrame(rows)


def _metric_values(group: pd.DataFrame, score_column: str, *, max_fpr: float) -> dict[str, float]:
    labels = (group["condition"] == "anomaly").to_numpy(dtype=int)
    values = group[score_column].to_numpy(dtype=float)
    return {
        "auc": _roc_auc(labels, values),
        "pauc_max_fpr_0_1": _partial_auc(labels, values, max_fpr=max_fpr),
    }


def _global_inference_table(payloads: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = {"mean_auc": "auc", "mean_pauc_max_fpr_0_1": "pauc_max_fpr_0_1"}
    for candidate in CANDIDATE_ORDER:
        payload = payloads[candidate]
        for source_metric, metric in names.items():
            interval = payload["metric_delta_candidate_minus_reference"][source_metric]
            rows.append(
                {
                    "reference": "B00",
                    "candidate": candidate,
                    "metric": metric,
                    "bootstrap_mean_delta": float(interval["mean"]),
                    "ci95_low": float(interval["ci95_low"]),
                    "ci95_high": float(interval["ci95_high"]),
                    "iterations": int(payload["iterations"]),
                    "seed": int(payload["seed"]),
                    "stratification": str(payload["stratification"]),
                }
            )
    return pd.DataFrame(rows)


def _leave_one_machine_out(group_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_ORDER:
        for domain in DOMAIN_ORDER:
            reference = group_metrics.loc[
                (group_metrics["system"] == "B00") & (group_metrics["domain"] == domain)
            ].set_index(["machine_type", "section"])
            comparison = group_metrics.loc[
                (group_metrics["system"] == candidate) & (group_metrics["domain"] == domain)
            ].set_index(["machine_type", "section"])
            if not reference.index.equals(comparison.index):
                raise ValueError(f"{candidate}/{domain} group metrics are not aligned")
            for metric in METRIC_ORDER:
                delta = comparison[metric] - reference[metric]
                full_delta = float(delta.mean())
                for excluded_machine, excluded_section in delta.index:
                    retained = delta.drop((excluded_machine, excluded_section))
                    leave_out_delta = float(retained.mean())
                    rows.append(
                        {
                            "reference": "B00",
                            "candidate": candidate,
                            "domain": domain,
                            "metric": metric,
                            "excluded_machine_type": excluded_machine,
                            "excluded_section": excluded_section,
                            "retained_machine_sections": len(retained),
                            "full_mean_delta": full_delta,
                            "leave_one_out_mean_delta": leave_out_delta,
                            "deviation_from_full": leave_out_delta - full_delta,
                            "sign_matches_full": bool(
                                np.sign(leave_out_delta) == np.sign(full_delta)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def _heterogeneity_payload(
    group_metrics: pd.DataFrame, leave_one_out: pd.DataFrame
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for candidate in CANDIDATE_ORDER:
        for domain in DOMAIN_ORDER:
            reference = group_metrics.loc[
                (group_metrics["system"] == "B00") & (group_metrics["domain"] == domain)
            ].set_index(["machine_type", "section"])
            comparison = group_metrics.loc[
                (group_metrics["system"] == candidate) & (group_metrics["domain"] == domain)
            ].set_index(["machine_type", "section"])
            for metric in METRIC_ORDER:
                delta = (comparison[metric] - reference[metric]).to_numpy(dtype=float)
                lomo = leave_one_out.loc[
                    (leave_one_out["candidate"] == candidate)
                    & (leave_one_out["domain"] == domain)
                    & (leave_one_out["metric"] == metric),
                    "leave_one_out_mean_delta",
                ].to_numpy(dtype=float)
                records.append(
                    {
                        "reference": "B00",
                        "candidate": candidate,
                        "domain": domain,
                        "metric": metric,
                        "machine_sections": len(delta),
                        "mean_delta": float(np.mean(delta)),
                        "median_delta": float(np.median(delta)),
                        "min_delta": float(np.min(delta)),
                        "max_delta": float(np.max(delta)),
                        "improved_machine_sections": int(np.sum(delta > 0.0)),
                        "harmed_machine_sections": int(np.sum(delta < 0.0)),
                        "tied_machine_sections": int(np.sum(delta == 0.0)),
                        "leave_one_out_min_delta": float(np.min(lomo)),
                        "leave_one_out_max_delta": float(np.max(lomo)),
                        "leave_one_out_sign_stable": bool(
                            np.all(np.sign(lomo) == np.sign(np.mean(delta)))
                        ),
                    }
                )
    return {"schema_version": 1, "records": records}


def _summary_markdown(
    metrics: pd.DataFrame,
    global_inference: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> str:
    lines = [
        "# Audit-A2 machine/domain robustness appendix",
        "",
        "This appendix uses only the frozen B00, B01, and B02 development score tables. "
        "No audio, training, tuning, or evaluation-set access occurred.",
        "",
        "## Headline robustness results",
        "",
    ]
    for candidate in CANDIDATE_ORDER:
        reference = metrics.loc[
            (metrics["system"] == "B00") & (metrics["domain"] == "all")
        ].set_index(["machine_type", "section"])
        comparison = metrics.loc[
            (metrics["system"] == candidate) & (metrics["domain"] == "all")
        ].set_index(["machine_type", "section"])
        pauc_delta = comparison["pauc_max_fpr_0_1"] - reference["pauc_max_fpr_0_1"]
        inference = global_inference.loc[
            (global_inference["candidate"] == candidate)
            & (global_inference["metric"] == "pauc_max_fpr_0_1")
        ].iloc[0]
        lomo = leave_one_out.loc[
            (leave_one_out["candidate"] == candidate)
            & (leave_one_out["domain"] == "all")
            & (leave_one_out["metric"] == "pauc_max_fpr_0_1"),
            "leave_one_out_mean_delta",
        ]
        lines.append(
            f"- **{candidate} vs B00:** observed mean pAUC delta "
            f"{100.0 * pauc_delta.mean():+.2f} percentage points; frozen paired-bootstrap "
            f"95% CI [{100.0 * inference['ci95_low']:+.2f}, "
            f"{100.0 * inference['ci95_high']:+.2f}]; "
            f"{int((pauc_delta > 0.0).sum())}/{len(pauc_delta)} machine sections improved; "
            f"leave-one-machine-out range [{100.0 * lomo.min():+.2f}, "
            f"{100.0 * lomo.max():+.2f}]."
        )
    lines.extend(["", "## Domain split", ""])
    for candidate in CANDIDATE_ORDER:
        values: list[str] = []
        for domain in ("source", "target"):
            reference = metrics.loc[
                (metrics["system"] == "B00") & (metrics["domain"] == domain)
            ].set_index(["machine_type", "section"])
            comparison = metrics.loc[
                (metrics["system"] == candidate) & (metrics["domain"] == domain)
            ].set_index(["machine_type", "section"])
            delta = comparison["pauc_max_fpr_0_1"] - reference["pauc_max_fpr_0_1"]
            values.append(f"{domain} {100.0 * delta.mean():+.2f} pp")
        lines.append(f"- **{candidate}:** {'; '.join(values)}.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "B01's overall pAUC harm keeps the same sign after every one-machine deletion, "
            "but its AUC effects are heterogeneous. B02 remains statistically inconclusive and "
            "its pAUC sign is not stable to every one-machine deletion. These are development-set "
            "robustness statements, not claims about hidden evaluation machines or all dual-mic methods.",
        ]
    )
    return "\n".join(lines) + "\n"


def _forest_svg(bootstrap: pd.DataFrame) -> str:
    data = bootstrap.loc[
        (bootstrap["domain"] == "all") & (bootstrap["metric"] == "pauc_max_fpr_0_1")
    ].copy()
    machines = sorted(data["machine_type"].unique())
    width, height = 920, 110 + 42 * len(machines)
    left, right = 210.0, 875.0
    minimum = min(-0.12, float(data["ci95_low"].min()) - 0.01)
    maximum = max(0.12, float(data["ci95_high"].max()) + 0.01)

    def x(value: float) -> float:
        return left + (value - minimum) / (maximum - minimum) * (right - left)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="sans-serif" font-size="18" font-weight="700">Per-machine pAUC delta vs B00 (95% paired bootstrap CI)</text>',
        f'<line x1="{x(0):.2f}" y1="48" x2="{x(0):.2f}" y2="{height - 38}" stroke="#374151" stroke-width="1"/>',
    ]
    colors = {"B01": "#b91c1c", "B02": "#1d4ed8"}
    for index, machine in enumerate(machines):
        base_y = 72 + index * 42
        lines.append(
            f'<text x="24" y="{base_y + 8}" font-family="sans-serif" font-size="13">{escape(machine)}</text>'
        )
        for offset, candidate in ((-7, "B01"), (7, "B02")):
            row = data.loc[
                (data["machine_type"] == machine) & (data["candidate"] == candidate)
            ].iloc[0]
            y = base_y + offset
            color = colors[candidate]
            lines.extend(
                [
                    f'<line x1="{x(float(row["ci95_low"])):.2f}" y1="{y}" x2="{x(float(row["ci95_high"])):.2f}" y2="{y}" stroke="{color}" stroke-width="2"/>',
                    f'<circle cx="{x(float(row["observed_delta"])):.2f}" cy="{y}" r="4" fill="{color}"/>',
                ]
            )
    for value in np.linspace(minimum, maximum, 7):
        lines.append(
            f'<text x="{x(float(value)):.2f}" y="{height - 16}" text-anchor="middle" font-family="sans-serif" font-size="11">{100 * value:+.1f} pp</text>'
        )
    lines.extend(
        [
            f'<circle cx="{width - 145}" cy="28" r="4" fill="#b91c1c"/><text x="{width - 135}" y="32" font-family="sans-serif" font-size="12">B01</text>',
            f'<circle cx="{width - 78}" cy="28" r="4" fill="#1d4ed8"/><text x="{width - 68}" y="32" font-family="sans-serif" font-size="12">B02</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines) + "\n"


def _heatmap_svg(metrics: pd.DataFrame) -> str:
    machines = sorted(metrics["machine_type"].unique())
    columns = [(candidate, domain) for candidate in CANDIDATE_ORDER for domain in DOMAIN_ORDER]
    width, height = 940, 135 + 48 * len(machines)
    left, top, cell_w, cell_h = 190, 92, 116, 42
    reference = metrics.loc[metrics["system"] == "B00"].set_index(
        ["machine_type", "section", "domain"]
    )
    deltas: dict[tuple[str, str, str], float] = {}
    for candidate, domain in columns:
        comparison = metrics.loc[
            (metrics["system"] == candidate) & (metrics["domain"] == domain)
        ].set_index(["machine_type", "section", "domain"])
        for index, value in (
            comparison["pauc_max_fpr_0_1"] - reference["pauc_max_fpr_0_1"]
        ).items():
            deltas[(candidate, domain, index[0])] = float(value)
    scale = max(0.01, max(abs(value) for value in deltas.values()))

    def color(value: float) -> str:
        strength = min(1.0, abs(value) / scale)
        if value < 0:
            red, green, blue = 185, int(245 - 130 * strength), int(245 - 130 * strength)
        else:
            red, green, blue = int(235 - 130 * strength), int(245 - 80 * strength), 205
        return f"rgb({red},{green},{blue})"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="sans-serif" font-size="18" font-weight="700">Observed pAUC delta by machine and domain</text>',
        '<text x="24" y="50" font-family="sans-serif" font-size="12" fill="#4b5563">Red = harm; green = improvement; values are percentage points vs B00</text>',
    ]
    for column, (candidate, domain) in enumerate(columns):
        x = left + column * cell_w
        lines.append(
            f'<text x="{x + cell_w / 2:.1f}" y="78" text-anchor="middle" font-family="sans-serif" font-size="12">{candidate} {domain}</text>'
        )
    for row, machine in enumerate(machines):
        y = top + row * cell_h
        lines.append(
            f'<text x="24" y="{y + 26}" font-family="sans-serif" font-size="13">{escape(machine)}</text>'
        )
        for column, (candidate, domain) in enumerate(columns):
            x = left + column * cell_w
            value = deltas[(candidate, domain, machine)]
            lines.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="3" fill="{color(value)}"/>',
                    f'<text x="{x + (cell_w - 4) / 2:.1f}" y="{y + 25}" text-anchor="middle" font-family="sans-serif" font-size="12">{100 * value:+.2f}</text>',
                ]
            )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"
