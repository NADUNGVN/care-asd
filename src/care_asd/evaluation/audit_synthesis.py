"""Frozen evidence synthesis for the CARE-ASD identifiability/audit paper."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from care_asd.reproducibility import get_git_commit

TEXT_SUFFIXES = {".csv", ".json", ".md", ".svg", ".yaml", ".yml"}


class AuditSourceConfig(BaseModel):
    """Frozen relative paths to the accepted evidence artifacts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    b00_summary: Path
    b01_summary: Path
    b01_bootstrap: Path
    phase8_correlations: Path
    b02_summary: Path
    b02_bootstrap: Path
    safe_ref_gate: Path
    ap_care_gate: Path
    ap_care_cases: Path

    @field_validator("*")
    @classmethod
    def source_paths_are_repository_relative(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("audit source paths must be repository-relative without '..'")
        return value


class AuditDecisionConfig(BaseModel):
    """Pre-committed stop and publication decisions after AP-G1."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    method_route: Literal["stopped"] = "stopped"
    publication_route: Literal["identifiability_audit"] = "identifiability_audit"
    evaluation_access: Literal["prohibited"] = "prohibited"
    gpu_replication: Literal["prohibited"] = "prohibited"


class AuditSynthesisConfig(BaseModel):
    """Versioned input contract for one immutable audit synthesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    study_id: str = Field(min_length=3, pattern=r"^[a-z0-9][a-z0-9_-]+$")
    sources: AuditSourceConfig
    decisions: AuditDecisionConfig


@dataclass(frozen=True)
class AuditSynthesisResult:
    """Paths produced by one immutable audit synthesis."""

    output_directory: Path
    decision_path: Path
    evidence_path: Path
    identifiability_path: Path
    diagnostics_path: Path
    summary_path: Path
    run_path: Path
    figure_paths: tuple[Path, ...]


def load_audit_synthesis_config(path: str | Path) -> AuditSynthesisConfig:
    """Load and strictly validate the audit evidence contract."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Audit synthesis config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Audit synthesis config root must be a mapping")
    return AuditSynthesisConfig.model_validate(payload)


def audit_synthesis_plan(
    config: AuditSynthesisConfig,
    *,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate source boundaries and return a side-effect-free execution plan."""
    root = Path(repository_root).resolve()
    sources = _resolved_sources(config, root)
    _load_evidence(sources)
    return {
        "schema_version": 1,
        "study_id": config.study_id,
        "git_commit": get_git_commit(root),
        "decisions": config.decisions.model_dump(mode="json"),
        "source_hashes": {
            name: {"path": str(config.sources.model_dump()[name]), "sha256": portable_sha256(path)}
            for name, path in sources.items()
        },
    }


def run_audit_synthesis(
    *,
    output_directory: str | Path,
    config: AuditSynthesisConfig,
    repository_root: str | Path,
) -> AuditSynthesisResult:
    """Create tables, figures, and a frozen stop decision from committed evidence."""
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit synthesis: {output}")
    root = Path(repository_root).resolve()
    sources = _resolved_sources(config, root)
    loaded = _load_evidence(sources)
    plan = audit_synthesis_plan(config, repository_root=root)

    evidence = _performance_evidence(loaded)
    identifiability = _identifiability_evidence(loaded)
    diagnostics = _ap_care_diagnostics(loaded["ap_care_cases"])
    decision = _decision_payload(config, loaded, evidence, identifiability)

    output.mkdir(parents=True)
    evidence_path = output / "performance_evidence.csv"
    identifiability_path = output / "identifiability_evidence.csv"
    diagnostics_path = output / "ap_care_holdout_diagnostics.csv"
    decision_path = output / "decision.json"
    summary_path = output / "audit_summary.md"
    evidence.to_csv(evidence_path, index=False)
    identifiability.to_csv(identifiability_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    decision_path.write_text(
        json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_summary_markdown(decision, evidence), encoding="utf-8")

    performance_figure = output / "performance_deltas.svg"
    identifiability_figure = output / "identifiability_gates.svg"
    frontier_figure = output / "ap_care_mechanism_frontier.svg"
    performance_figure.write_text(_performance_svg(evidence), encoding="utf-8")
    identifiability_figure.write_text(_identifiability_svg(identifiability), encoding="utf-8")
    frontier_figure.write_text(
        _frontier_svg(loaded["ap_care_cases"]),
        encoding="utf-8",
    )
    figure_paths = (performance_figure, identifiability_figure, frontier_figure)

    artifact_paths = (
        evidence_path,
        identifiability_path,
        diagnostics_path,
        decision_path,
        summary_path,
        *figure_paths,
    )
    run_payload = {
        **plan,
        "hash_mode": "LF-normalized UTF-8 for text; raw bytes for binary",
        "artifacts": {path.name: portable_sha256(path) for path in artifact_paths},
    }
    run_path = output / "run.json"
    run_path.write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return AuditSynthesisResult(
        output_directory=output,
        decision_path=decision_path,
        evidence_path=evidence_path,
        identifiability_path=identifiability_path,
        diagnostics_path=diagnostics_path,
        summary_path=summary_path,
        run_path=run_path,
        figure_paths=figure_paths,
    )


def portable_sha256(path: str | Path) -> str:
    """Hash text with LF normalization and binary artifacts byte-for-byte."""
    source = Path(path)
    if source.suffix.lower() in TEXT_SUFFIXES:
        content = source.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        data = content.encode("utf-8")
    else:
        data = source.read_bytes()
    return hashlib.sha256(data).hexdigest()


def _resolved_sources(config: AuditSynthesisConfig, root: Path) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for name, relative in config.sources.model_dump().items():
        path = (root / Path(relative)).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Audit source escapes repository root: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"Missing audit source {name}: {path}")
        resolved[name] = path
    return resolved


def _load_evidence(sources: dict[str, Path]) -> dict[str, Any]:
    loaded: dict[str, Any] = {
        "b00_summary": pd.read_csv(sources["b00_summary"]),
        "b01_summary": pd.read_csv(sources["b01_summary"]),
        "b01_bootstrap": _read_json(sources["b01_bootstrap"]),
        "phase8_correlations": pd.read_csv(sources["phase8_correlations"]),
        "b02_summary": pd.read_csv(sources["b02_summary"]),
        "b02_bootstrap": _read_json(sources["b02_bootstrap"]),
        "safe_ref_gate": _read_json(sources["safe_ref_gate"]),
        "ap_care_gate": _read_json(sources["ap_care_gate"]),
        "ap_care_cases": pd.read_parquet(sources["ap_care_cases"]),
    }
    metric_columns = {"mean_auc_all", "mean_pauc_all_max_fpr_0_1"}
    for name in ("b00_summary", "b01_summary", "b02_summary"):
        frame = loaded[name]
        if len(frame) != 1 or not metric_columns.issubset(frame.columns):
            raise ValueError(f"{name} must contain one row and the aligned AUC/pAUC columns")
    correlations = loaded["phase8_correlations"]
    correlation_columns = {"group", "outcome", "clips", "spearman_rho", "two_sided_p_value"}
    if not correlation_columns.issubset(correlations.columns):
        raise ValueError("phase8_correlations has an incompatible schema")
    cases = loaded["ap_care_cases"]
    case_columns = {
        "case_id",
        "case_seed",
        "split",
        "fault_support",
        "path_mismatch",
        "ap_fault_retention",
        "ap_noise_attenuation_db",
    }
    if len(cases) != 512 or not case_columns.issubset(cases.columns):
        raise ValueError("ap_care_cases must contain the frozen 512-case G1 schema")
    if cases["case_seed"].nunique() != 512 or set(cases["split"]) != {"calibration", "holdout"}:
        raise ValueError("ap_care_cases seed or split contract is invalid")
    if loaded["safe_ref_gate"].get("passed") is not False:
        raise ValueError("SAFE-REF source is not the frozen failed gate")
    if loaded["ap_care_gate"].get("passed") is not False:
        raise ValueError("AP-CARE source is not the frozen failed gate")
    return loaded


def _performance_evidence(loaded: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    b00 = loaded["b00_summary"].iloc[0]
    rows.append(
        {
            "system": "B00",
            "description": "unchanged near-only official-compatible reference",
            "mean_auc": float(b00["mean_auc_all"]),
            "mean_pauc": float(b00["mean_pauc_all_max_fpr_0_1"]),
            "auc_delta_mean": 0.0,
            "auc_ci95_low": 0.0,
            "auc_ci95_high": 0.0,
            "pauc_delta_mean": 0.0,
            "pauc_ci95_low": 0.0,
            "pauc_ci95_high": 0.0,
            "decision": "reference",
        }
    )
    for system, summary_name, bootstrap_name, description, decision in (
        (
            "B01",
            "b01_summary",
            "b01_bootstrap",
            "CARE residual replacement",
            "rejected_harmful_pauc",
        ),
        (
            "B02",
            "b02_summary",
            "b02_bootstrap",
            "near-primary reliability-gated residual",
            "rejected_no_improvement",
        ),
    ):
        summary = loaded[summary_name].iloc[0]
        deltas = loaded[bootstrap_name]["metric_delta_candidate_minus_reference"]
        auc = deltas["mean_auc"]
        pauc = deltas["mean_pauc_max_fpr_0_1"]
        rows.append(
            {
                "system": system,
                "description": description,
                "mean_auc": float(summary["mean_auc_all"]),
                "mean_pauc": float(summary["mean_pauc_all_max_fpr_0_1"]),
                "auc_delta_mean": float(auc["mean"]),
                "auc_ci95_low": float(auc["ci95_low"]),
                "auc_ci95_high": float(auc["ci95_high"]),
                "pauc_delta_mean": float(pauc["mean"]),
                "pauc_ci95_low": float(pauc["ci95_low"]),
                "pauc_ci95_high": float(pauc["ci95_high"]),
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def _identifiability_evidence(loaded: dict[str, Any]) -> pd.DataFrame:
    safe = loaded["safe_ref_gate"]["holdout"]
    ap = loaded["ap_care_gate"]
    holdout = ap["holdout"]
    criteria = ap["criteria"]
    checks = ap["checks"]
    return pd.DataFrame(
        [
            {
                "test": "SAFE-REF risk tracking",
                "observed": float(safe["risk_spearman"]),
                "threshold": 0.60,
                "ci95_low": np.nan,
                "passed": False,
                "unit": "Spearman rho",
            },
            {
                "test": "AP-CARE leakage tracking",
                "observed": float(holdout["leakage_spearman"]),
                "threshold": float(criteria["leakage_spearman_min"]),
                "ci95_low": float(holdout["leakage_spearman_ci95_low"]),
                "passed": bool(checks["leakage_tracking"]),
                "unit": "Spearman rho",
            },
            {
                "test": "AP-CARE uncertainty tracking",
                "observed": float(holdout["uncertainty_spearman"]),
                "threshold": float(criteria["uncertainty_spearman_min"]),
                "ci95_low": float(holdout["uncertainty_spearman_ci95_low"]),
                "passed": bool(checks["uncertainty_tracking"]),
                "unit": "Spearman rho",
            },
            {
                "test": "AP-CARE eligible attenuation",
                "observed": float(holdout["eligible_noise_attenuation_median_db"]),
                "threshold": float(criteria["noise_attenuation_median_min_db"]),
                "ci95_low": np.nan,
                "passed": bool(checks["eligible_noise_attenuation"]),
                "unit": "dB",
            },
            {
                "test": "AP-CARE matched retention improvement",
                "observed": float(holdout["retention_improvement_median"]),
                "threshold": float(criteria["retention_improvement_min"]),
                "ci95_low": float(holdout["retention_improvement_mean_ci95_low"]),
                "passed": bool(checks["matched_retention_improvement"]),
                "unit": "ratio",
            },
            {
                "test": "AP-CARE in-support retention median",
                "observed": float(holdout["in_support_fault_retention_median"]),
                "threshold": float(criteria["fault_retention_median_min"]),
                "ci95_low": np.nan,
                "passed": bool(checks["in_support_retention"]),
                "unit": "ratio",
            },
        ]
    )


def _ap_care_diagnostics(cases: pd.DataFrame) -> pd.DataFrame:
    holdout = cases.loc[cases["split"] == "holdout"].copy()
    holdout["mismatch_quartile"] = pd.qcut(
        holdout["path_mismatch"],
        4,
        labels=("Q1_low", "Q2", "Q3", "Q4_high"),
    )
    rows: list[dict[str, Any]] = []
    for (support, quartile), group in holdout.groupby(
        ["fault_support", "mismatch_quartile"], observed=True
    ):
        rows.append(
            {
                "fault_support": str(support),
                "mismatch_quartile": str(quartile),
                "cases": len(group),
                "path_mismatch_median": float(group["path_mismatch"].median()),
                "fault_retention_median": float(group["ap_fault_retention"].median()),
                "noise_attenuation_median_db": float(group["ap_noise_attenuation_db"].median()),
                "positive_attenuation_fraction": float(
                    (group["ap_noise_attenuation_db"] > 0.0).mean()
                ),
                "attenuation_ge_1db_fraction": float(
                    (group["ap_noise_attenuation_db"] >= 1.0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _decision_payload(
    config: AuditSynthesisConfig,
    loaded: dict[str, Any],
    evidence: pd.DataFrame,
    identifiability: pd.DataFrame,
) -> dict[str, Any]:
    phase8 = loaded["phase8_correlations"]
    association = phase8.loc[
        (phase8["group"] == "all") & (phase8["outcome"] == "score_delta_b01_minus_b00")
    ]
    if len(association) != 1:
        raise ValueError("Phase 8 global score-delta association is missing or ambiguous")
    ap_holdout = loaded["ap_care_cases"].loc[loaded["ap_care_cases"]["split"] == "holdout"]
    return {
        "schema_version": 1,
        "study_id": config.study_id,
        "decision": config.decisions.model_dump(mode="json"),
        "stop_rule_satisfied": True,
        "failed_method_gates": ["SAFE-REF Phase 10", "AP-CARE G1"],
        "headline_evidence": {
            "b01_pauc_delta": float(
                evidence.loc[evidence["system"] == "B01", "pauc_delta_mean"].iloc[0]
            ),
            "b01_pauc_ci95": [
                float(evidence.loc[evidence["system"] == "B01", "pauc_ci95_low"].iloc[0]),
                float(evidence.loc[evidence["system"] == "B01", "pauc_ci95_high"].iloc[0]),
            ],
            "b02_pauc_delta": float(
                evidence.loc[evidence["system"] == "B02", "pauc_delta_mean"].iloc[0]
            ),
            "phase8_displacement_score_rho": float(association.iloc[0]["spearman_rho"]),
            "safe_ref_false_safe_rate": float(
                loaded["safe_ref_gate"]["holdout"]["false_safe_rate"]
            ),
            "safe_ref_risk_spearman": float(loaded["safe_ref_gate"]["holdout"]["risk_spearman"]),
            "ap_care_leakage_spearman": float(
                loaded["ap_care_gate"]["holdout"]["leakage_spearman"]
            ),
            "ap_care_uncertainty_spearman": float(
                loaded["ap_care_gate"]["holdout"]["uncertainty_spearman"]
            ),
            "ap_care_noise_attenuation_median_db": float(
                loaded["ap_care_gate"]["holdout"]["eligible_noise_attenuation_median_db"]
            ),
            "ap_care_holdout_cases": len(ap_holdout),
            "ap_care_holdout_cases_ge_1db": int(
                (ap_holdout["ap_noise_attenuation_db"] >= 1.0).sum()
            ),
        },
        "claim_boundary": (
            "Normal-only contaminated-reference heuristics did not identify a reliable safe "
            "cancellation regime under the tested assumptions; this is an empirical audit, "
            "not a distribution-free impossibility theorem."
        ),
        "prohibited_next_actions": [
            "retune AP-CARE thresholds on the frozen G1 holdout",
            "run AP-G2 or AP-G3 GPU experiments",
            "access unseen evaluation labels or scores",
            "run Jetson claims as a substitute for method evidence",
        ],
    }


def _summary_markdown(decision: dict[str, Any], evidence: pd.DataFrame) -> str:
    headline = decision["headline_evidence"]
    table_rows = "\n".join(
        f"| {row.system} | {row.mean_auc:.5f} | {row.mean_pauc:.5f} | "
        f"{row.pauc_delta_mean:+.5f} | {row.decision} |"
        for row in evidence.itertuples(index=False)
    )
    return f"""# CARE-ASD identifiability audit synthesis

## Frozen decision

The AP-CARE method route is stopped. AP-G2--G5, unseen evaluation access, and
board-kit claims are prohibited. The retained publication route is an
identifiability/audit study.

## Aligned development evidence

| System | Mean AUC | Mean pAUC | Bootstrap pAUC delta | Decision |
|---|---:|---:|---:|---|
{table_rows}

Phase 8 associated stronger residual-induced log-Mel displacement with a lower
B01-minus-B00 anomaly score (Spearman rho
{headline["phase8_displacement_score_rho"]:.4f}).

## Identifiability evidence

- SAFE-REF false-safe rate: {headline["safe_ref_false_safe_rate"]:.4f}; risk
  Spearman rho: {headline["safe_ref_risk_spearman"]:.4f}.
- AP-CARE leakage Spearman rho: {headline["ap_care_leakage_spearman"]:.4f};
  uncertainty Spearman rho: {headline["ap_care_uncertainty_spearman"]:.4f}.
- AP-CARE eligible median attenuation:
  {headline["ap_care_noise_attenuation_median_db"]:.4f} dB; holdout cases at or
  above 1 dB: {headline["ap_care_holdout_cases_ge_1db"]}/
  {headline["ap_care_holdout_cases"]}.

## Claim boundary

{decision["claim_boundary"]}

The CSV files and SVG figures in this directory are generated directly from the
frozen source artifacts listed and hashed in `run.json`.
"""


def _performance_svg(evidence: pd.DataFrame) -> str:
    rows: list[tuple[str, float, float, float, str]] = []
    for system in ("B01", "B02"):
        row = evidence.loc[evidence["system"] == system].iloc[0]
        rows.extend(
            [
                (
                    f"{system} mean AUC",
                    float(row["auc_delta_mean"]),
                    float(row["auc_ci95_low"]),
                    float(row["auc_ci95_high"]),
                    "#2563eb",
                ),
                (
                    f"{system} pAUC@0.1",
                    float(row["pauc_delta_mean"]),
                    float(row["pauc_ci95_low"]),
                    float(row["pauc_ci95_high"]),
                    "#dc2626",
                ),
            ]
        )
    width, height = 900, 430
    left, right = 210, 850
    x_min, x_max = -0.04, 0.02

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    elements = _svg_header(width, height, "Aligned B01/B02 performance deltas vs B00")
    zero = x(0.0)
    elements.append(
        f'<line x1="{zero:.1f}" y1="75" x2="{zero:.1f}" y2="360" stroke="#111827" stroke-width="2"/>'
    )
    for tick in np.linspace(x_min, x_max, 7):
        px = x(float(tick))
        elements.append(f'<line x1="{px:.1f}" y1="360" x2="{px:.1f}" y2="366" stroke="#374151"/>')
        elements.append(
            f'<text x="{px:.1f}" y="386" text-anchor="middle" class="tick">{tick:+.2f}</text>'
        )
    for index, (label, point, low, high, color) in enumerate(rows):
        y = 105 + index * 72
        elements.append(
            f'<text x="195" y="{y + 5}" text-anchor="end" class="label">{escape(label)}</text>'
        )
        elements.append(
            f'<line x1="{x(low):.1f}" y1="{y}" x2="{x(high):.1f}" y2="{y}" stroke="{color}" stroke-width="5" stroke-linecap="round"/>'
        )
        elements.append(f'<circle cx="{x(point):.1f}" cy="{y}" r="7" fill="{color}"/>')
    elements.append(
        '<text x="530" y="414" text-anchor="middle" class="axis">Bootstrap candidate-minus-B00 delta (95% CI)</text>'
    )
    return "\n".join((*elements, "</svg>", ""))


def _identifiability_svg(evidence: pd.DataFrame) -> str:
    rows = evidence.iloc[:3]
    width, height = 900, 355
    left, right = 260, 850
    x_min, x_max = 0.0, 0.7

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    elements = _svg_header(
        width, height, "Normal-only risk proxies miss preregistered identifiability gates"
    )
    threshold = x(0.60)
    elements.append(
        f'<line x1="{threshold:.1f}" y1="75" x2="{threshold:.1f}" y2="280" stroke="#111827" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    elements.append(
        f'<text x="{threshold - 6:.1f}" y="72" text-anchor="end" class="tick">gate 0.60</text>'
    )
    for index, row in enumerate(rows.itertuples(index=False)):
        y = 110 + index * 72
        elements.append(
            f'<text x="245" y="{y + 5}" text-anchor="end" class="label">{escape(row.test)}</text>'
        )
        if np.isfinite(row.ci95_low):
            elements.append(
                f'<line x1="{x(float(row.ci95_low)):.1f}" y1="{y}" x2="{x(float(row.observed)):.1f}" y2="{y}" stroke="#dc2626" stroke-width="5"/>'
            )
        elements.append(
            f'<circle cx="{x(float(row.observed)):.1f}" cy="{y}" r="8" fill="#dc2626"/>'
        )
        elements.append(
            f'<text x="{x(float(row.observed)) + 12:.1f}" y="{y + 5}" class="value">{float(row.observed):.3f}</text>'
        )
    elements.append(
        '<text x="555" y="330" text-anchor="middle" class="axis">Spearman rho (AP bars extend to bootstrap 95% lower bound)</text>'
    )
    return "\n".join((*elements, "</svg>", ""))


def _frontier_svg(cases: pd.DataFrame) -> str:
    holdout = cases.loc[cases["split"] == "holdout"]
    width, height = 900, 530
    left, right, top, bottom = 95, 855, 75, 450
    x_min = min(-2.0, float(holdout["ap_noise_attenuation_db"].min()) - 0.1)
    x_max = max(1.2, float(holdout["ap_noise_attenuation_db"].max()) + 0.1)
    y_min = min(0.75, float(holdout["ap_fault_retention"].min()) - 0.02)
    y_max = max(1.05, float(holdout["ap_fault_retention"].max()) + 0.02)

    def x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    elements = _svg_header(
        width, height, "AP-CARE G1 holdout: preservation without useful attenuation"
    )
    elements.append(
        f'<line x1="{x(1.0):.1f}" y1="{top}" x2="{x(1.0):.1f}" y2="{bottom}" stroke="#111827" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    elements.append(
        f'<line x1="{left}" y1="{y(0.90):.1f}" x2="{right}" y2="{y(0.90):.1f}" stroke="#111827" stroke-width="2" stroke-dasharray="7 6"/>'
    )
    for row in holdout.itertuples(index=False):
        color = "#2563eb" if row.fault_support == "in_support" else "#f59e0b"
        elements.append(
            f'<circle cx="{x(float(row.ap_noise_attenuation_db)):.1f}" cy="{y(float(row.ap_fault_retention)):.1f}" r="3.1" fill="{color}" fill-opacity="0.55"/>'
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#374151"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#374151"/>',
            '<text x="475" y="500" text-anchor="middle" class="axis">Environmental-noise attenuation (dB; gate ≥ 1 dB)</text>',
            f'<text x="24" y="{(top + bottom) / 2:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 24 {(top + bottom) / 2:.1f})">Fault retention ratio (gate ≥ 0.90)</text>',
            '<circle cx="650" cy="55" r="5" fill="#2563eb"/><text x="662" y="60" class="tick">in-support</text>',
            '<circle cx="755" cy="55" r="5" fill="#f59e0b"/><text x="767" y="60" class="tick">out-of-support</text>',
        ]
    )
    return "\n".join((*elements, "</svg>", ""))


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>.title{font:600 20px system-ui,sans-serif;fill:#111827}.label{font:14px system-ui,sans-serif;fill:#1f2937}.tick,.value{font:12px system-ui,sans-serif;fill:#4b5563}.axis{font:14px system-ui,sans-serif;fill:#111827}</style>",
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" class="title">{escape(title)}</text>',
    ]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload
