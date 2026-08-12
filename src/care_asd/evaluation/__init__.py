"""Metrics, statistical tests, and report generation (Phases 2+)."""

from care_asd.evaluation.dsp_benchmark import (
    DspBenchmarkResult,
    run_care_development_benchmark,
    run_dsp_development_benchmark,
)
from care_asd.evaluation.mvp_neural import (
    MvpAblation,
    MvpNeuralResult,
    available_mvp_ablations,
    run_mvp_neural_development,
)
from care_asd.evaluation.official_baseline import (
    SCORE_COLUMNS,
    ScoreMode,
    calculate_development_auc_metrics,
    normalize_official_development_scores,
)
from care_asd.evaluation.paired_bootstrap import write_paired_bootstrap_comparison

__all__ = [
    "SCORE_COLUMNS",
    "DspBenchmarkResult",
    "MvpAblation",
    "MvpNeuralResult",
    "ScoreMode",
    "available_mvp_ablations",
    "calculate_development_auc_metrics",
    "normalize_official_development_scores",
    "run_care_development_benchmark",
    "run_dsp_development_benchmark",
    "run_mvp_neural_development",
    "write_paired_bootstrap_comparison",
]
