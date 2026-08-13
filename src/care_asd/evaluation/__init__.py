"""Metrics, statistical tests, and report generation (Phases 2+)."""

from care_asd.evaluation.dsp_benchmark import (
    DspBenchmarkResult,
    run_care_development_benchmark,
    run_dsp_development_benchmark,
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
    "DspBenchmarkResult",
    "ScoreMode",
    "calculate_development_auc_metrics",
    "normalize_official_development_scores",
    "run_care_development_benchmark",
    "run_dsp_development_benchmark",
    "write_paired_bootstrap_comparison",
    "write_seed_ensemble_scores",
]
