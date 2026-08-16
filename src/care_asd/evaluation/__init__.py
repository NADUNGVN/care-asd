"""Metrics, statistical tests, and report generation (Phases 2+)."""

from care_asd.evaluation.audit_synthesis import (
    AuditSynthesisConfig,
    AuditSynthesisResult,
    audit_synthesis_plan,
    load_audit_synthesis_config,
    run_audit_synthesis,
)
from care_asd.evaluation.dsp_benchmark import (
    DspBenchmarkResult,
    run_care_development_benchmark,
    run_dsp_development_benchmark,
)
from care_asd.evaluation.literature_audit import (
    LiteratureAuditConfig,
    LiteratureAuditResult,
    literature_audit_plan,
    load_literature_audit_config,
    run_literature_audit,
)
from care_asd.evaluation.official_baseline import (
    SCORE_COLUMNS,
    ScoreMode,
    calculate_development_auc_metrics,
    normalize_official_development_scores,
)
from care_asd.evaluation.paired_bootstrap import (
    write_paired_bootstrap_comparison,
    write_seed_ensemble_scores,
)

__all__ = [
    "SCORE_COLUMNS",
    "AuditSynthesisConfig",
    "AuditSynthesisResult",
    "DspBenchmarkResult",
    "LiteratureAuditConfig",
    "LiteratureAuditResult",
    "ScoreMode",
    "audit_synthesis_plan",
    "calculate_development_auc_metrics",
    "literature_audit_plan",
    "load_audit_synthesis_config",
    "load_literature_audit_config",
    "normalize_official_development_scores",
    "run_audit_synthesis",
    "run_care_development_benchmark",
    "run_dsp_development_benchmark",
    "run_literature_audit",
    "write_paired_bootstrap_comparison",
    "write_seed_ensemble_scores",
]
