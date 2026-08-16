"""Metrics, statistical tests, and report generation (Phases 2+)."""

from care_asd.evaluation.audit_synthesis import (
    AuditSynthesisConfig,
    AuditSynthesisResult,
    audit_synthesis_plan,
    load_audit_synthesis_config,
    run_audit_synthesis,
)
from care_asd.evaluation.dcase2026_metrics import (
    calculate_dcase2026_official_metrics,
    read_official_score,
)
from care_asd.evaluation.dsp_benchmark import (
    DspBenchmarkResult,
    run_care_development_benchmark,
    run_dsp_development_benchmark,
)
from care_asd.evaluation.fp_naa_baseline import FPNaaBaselineResult, run_fp_naa_baseline
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
from care_asd.evaluation.robustness_appendix import (
    RobustnessAppendixConfig,
    RobustnessAppendixResult,
    load_robustness_appendix_config,
    robustness_appendix_plan,
    run_robustness_appendix,
)

__all__ = [
    "SCORE_COLUMNS",
    "AuditSynthesisConfig",
    "AuditSynthesisResult",
    "DspBenchmarkResult",
    "FPNaaBaselineResult",
    "LiteratureAuditConfig",
    "LiteratureAuditResult",
    "RobustnessAppendixConfig",
    "RobustnessAppendixResult",
    "ScoreMode",
    "audit_synthesis_plan",
    "calculate_dcase2026_official_metrics",
    "calculate_development_auc_metrics",
    "literature_audit_plan",
    "load_audit_synthesis_config",
    "load_literature_audit_config",
    "load_robustness_appendix_config",
    "normalize_official_development_scores",
    "read_official_score",
    "robustness_appendix_plan",
    "run_audit_synthesis",
    "run_care_development_benchmark",
    "run_dsp_development_benchmark",
    "run_fp_naa_baseline",
    "run_literature_audit",
    "run_robustness_appendix",
    "write_paired_bootstrap_comparison",
    "write_seed_ensemble_scores",
]
