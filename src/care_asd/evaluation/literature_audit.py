"""Reproducible literature and claim-boundary audits for CARE-ASD."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from care_asd.evaluation.audit_synthesis import portable_sha256
from care_asd.reproducibility import get_git_commit


class LiteratureSource(BaseModel):
    """One primary source and its audited relationship to the paper."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_]+$")
    year: int = Field(ge=1970, le=2100)
    title: str = Field(min_length=8)
    authors: str = Field(min_length=3)
    venue: str = Field(min_length=2)
    publication_type: Literal[
        "benchmark_specification",
        "journal_article",
        "conference_paper",
        "workshop_paper",
        "arxiv_preprint",
        "challenge_technical_report",
    ]
    review_status: Literal["peer_reviewed", "not_peer_reviewed", "official_specification"]
    url: str
    cluster: Literal[
        "task_context",
        "direct_asd",
        "adjacent_signal_processing",
        "adjacent_model_selection",
    ]
    method_family: Literal[
        "benchmark_definition",
        "learned_dual_channel_representation",
        "deterministic_signal_subtraction",
        "embedding_residual",
        "spatial_masking",
        "adaptive_cancellation",
        "active_noise_control",
        "learned_reference_purification",
        "backend_diversity_fusion",
        "domain_aware_score_calibration",
        "cross_channel_predictive_residual",
        "multi_encoder_multiview_fusion",
        "train_normal_profile_ensemble",
        "heterogeneous_score_fusion",
        "unsupervised_model_selection",
    ]
    normal_only_asd: bool
    signal_level_transformation: bool
    evaluates_asd_metrics: bool
    evaluates_known_component_safety: bool
    reports_reference_contamination: bool
    result_direction: Literal["context", "benefit", "mixed", "risk_mitigation"]
    key_finding: str = Field(min_length=20)
    audit_relationship: str = Field(min_length=20)

    @field_validator("url")
    @classmethod
    def url_is_public_http(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("source URL must be an absolute HTTP(S) URL")
        return value

    @model_validator(mode="after")
    def publication_status_is_consistent(self) -> LiteratureSource:
        if (
            self.publication_type == "challenge_technical_report"
            and self.review_status != "not_peer_reviewed"
        ):
            raise ValueError("DCASE challenge technical reports must be marked not_peer_reviewed")
        if (
            self.publication_type == "benchmark_specification"
            and self.review_status != "official_specification"
        ):
            raise ValueError("benchmark specifications must be marked official_specification")
        return self


class ClaimBoundary(BaseModel):
    """One manuscript claim with explicit supporting or counterexample sources."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=r"^[a-z][a-z0-9_]+$")
    text: str = Field(min_length=20)
    source_ids: tuple[str, ...] = Field(min_length=1)
    evidence_scope: str = Field(min_length=20)


class LiteratureAuditConfig(BaseModel):
    """Strict, versioned contract for one CARE-ASD literature package."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    study_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]+$")
    cutoff_date: date
    source_policy: Literal["primary_sources_only"] = "primary_sources_only"
    required_clusters: tuple[
        Literal[
            "task_context",
            "direct_asd",
            "adjacent_signal_processing",
            "adjacent_model_selection",
        ],
        ...,
    ] = ("direct_asd", "adjacent_signal_processing")
    working_title: str = Field(min_length=20)
    central_question: str = Field(min_length=20)
    contribution_statement: str = Field(min_length=20)
    sources: tuple[LiteratureSource, ...] = Field(min_length=3)
    supported_claims: tuple[ClaimBoundary, ...] = Field(min_length=1)
    prohibited_claims: tuple[ClaimBoundary, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def identifiers_and_claim_links_are_valid(self) -> LiteratureAuditConfig:
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("literature source_id values must be unique")
        urls = [source.url for source in self.sources]
        if len(urls) != len(set(urls)):
            raise ValueError("literature source URLs must be unique")
        claim_ids = [claim.claim_id for claim in (*self.supported_claims, *self.prohibited_claims)]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique")
        known_sources = set(source_ids)
        for claim in (*self.supported_claims, *self.prohibited_claims):
            unknown = sorted(set(claim.source_ids) - known_sources)
            if unknown:
                raise ValueError(f"claim {claim.claim_id} cites unknown sources: {unknown}")
        if not self.required_clusters:
            raise ValueError("literature audit requires at least one required cluster")
        if len(self.required_clusters) != len(set(self.required_clusters)):
            raise ValueError("required_clusters values must be unique")
        observed_clusters = {source.cluster for source in self.sources}
        missing_clusters = sorted(set(self.required_clusters) - observed_clusters)
        if missing_clusters:
            raise ValueError(f"literature audit is missing required clusters: {missing_clusters}")
        return self


@dataclass(frozen=True)
class LiteratureAuditResult:
    """Paths written by one immutable literature synthesis."""

    output_directory: Path
    matrix_path: Path
    boundary_path: Path
    summary_path: Path
    run_path: Path


def load_literature_audit_config(path: str | Path) -> LiteratureAuditConfig:
    """Load and strictly validate the curated literature contract."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Literature audit config does not exist: {source}")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Literature audit config root must be a mapping")
    return LiteratureAuditConfig.model_validate(payload)


def literature_audit_plan(
    config: LiteratureAuditConfig,
    *,
    repository_root: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Return the validated, side-effect-free Audit-A1 execution plan."""
    root = Path(repository_root).resolve()
    resolved_config = Path(config_path).resolve()
    try:
        relative_config = resolved_config.relative_to(root)
    except ValueError as exc:
        raise ValueError("literature audit config must be inside repository_root") from exc
    return {
        "schema_version": 1,
        "study_id": config.study_id,
        "cutoff_date": config.cutoff_date.isoformat(),
        "source_policy": config.source_policy,
        "source_count": len(config.sources),
        "supported_claim_count": len(config.supported_claims),
        "prohibited_claim_count": len(config.prohibited_claims),
        "git_commit": get_git_commit(root),
        "config": {
            "path": relative_config.as_posix(),
            "sha256": portable_sha256(resolved_config),
        },
    }


def run_literature_audit(
    *,
    output_directory: str | Path,
    config: LiteratureAuditConfig,
    repository_root: str | Path,
    config_path: str | Path,
) -> LiteratureAuditResult:
    """Write an immutable literature matrix and manuscript claim boundary."""
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite literature audit: {output}")
    plan = literature_audit_plan(
        config,
        repository_root=repository_root,
        config_path=config_path,
    )
    output.mkdir(parents=True)

    matrix_path = output / "literature_matrix.csv"
    boundary_path = output / "claim_boundary.json"
    summary_path = output / "literature_audit.md"
    run_path = output / "run.json"

    matrix = pd.DataFrame([source.model_dump(mode="json") for source in config.sources])
    matrix.to_csv(matrix_path, index=False)
    boundary = {
        "study_id": config.study_id,
        "cutoff_date": config.cutoff_date.isoformat(),
        "working_title": config.working_title,
        "central_question": config.central_question,
        "contribution_statement": config.contribution_statement,
        "supported_claims": [claim.model_dump(mode="json") for claim in config.supported_claims],
        "prohibited_claims": [claim.model_dump(mode="json") for claim in config.prohibited_claims],
    }
    boundary_path.write_text(
        json.dumps(boundary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_summary_markdown(config), encoding="utf-8")

    artifacts = (matrix_path, boundary_path, summary_path)
    run_payload = {
        **plan,
        "artifacts": {path.name: portable_sha256(path) for path in artifacts},
    }
    run_path.write_text(
        json.dumps(run_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return LiteratureAuditResult(output, matrix_path, boundary_path, summary_path, run_path)


def _summary_markdown(config: LiteratureAuditConfig) -> str:
    reviewed = sum(source.review_status == "peer_reviewed" for source in config.sources)
    technical = sum(
        source.publication_type == "challenge_technical_report" for source in config.sources
    )
    safety = sum(source.evaluates_known_component_safety for source in config.sources)
    lines = [
        f"# {config.study_id}",
        "",
        f"Literature cutoff: **{config.cutoff_date.isoformat()}**. Curated sources: "
        f"**{len(config.sources)}** ({reviewed} peer-reviewed; {technical} non-peer-reviewed "
        "DCASE technical reports).",
        "",
        "## Positioning decision",
        "",
        config.contribution_statement,
        "",
        f"Known-component fault/noise safety evaluated by directly reviewed sources: **{safety}**.",
        "",
        "## Supported claims",
        "",
    ]
    lines.extend(
        f"- `{claim.claim_id}`: {claim.text} Sources: {', '.join(claim.source_ids)}."
        for claim in config.supported_claims
    )
    lines.extend(["", "## Prohibited claims", ""])
    lines.extend(
        f"- `{claim.claim_id}`: {claim.text} Counterexamples: {', '.join(claim.source_ids)}."
        for claim in config.prohibited_claims
    )
    lines.extend(
        [
            "",
            "## Source matrix",
            "",
            "| ID | Cluster | Method | Review | Safety audit |",
            "|---|---|---|---|---:|",
        ]
    )
    lines.extend(
        f"| {source.source_id} | {source.cluster} | {source.method_family} | "
        f"{source.review_status} | {str(source.evaluates_known_component_safety).lower()} |"
        for source in config.sources
    )
    return "\n".join(lines) + "\n"
