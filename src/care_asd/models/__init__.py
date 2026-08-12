"""Models: official baseline adapter, lightweight encoder, CARE front-end (Phases 2-5)."""

from care_asd.models.mvp_autoencoder import LightweightNearAutoencoder, approximate_parameter_count
from care_asd.models.official_baseline import (
    OFFICIAL_BASELINE_COMMIT,
    OFFICIAL_BASELINE_REPOSITORY,
    OFFICIAL_EVALUATOR_COMMIT,
    OFFICIAL_EVALUATOR_REPOSITORY,
    BaselineMode,
    checkout_pinned_reference,
    run_official_development_baseline,
    stage_official_development_data,
    verify_pinned_reference,
)

__all__ = [
    "OFFICIAL_BASELINE_COMMIT",
    "OFFICIAL_BASELINE_REPOSITORY",
    "OFFICIAL_EVALUATOR_COMMIT",
    "OFFICIAL_EVALUATOR_REPOSITORY",
    "BaselineMode",
    "LightweightNearAutoencoder",
    "approximate_parameter_count",
    "checkout_pinned_reference",
    "run_official_development_baseline",
    "stage_official_development_data",
    "verify_pinned_reference",
]
