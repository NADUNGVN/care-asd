"""Dataset acquisition, manifests, and loaders (Phase 1+)."""

from care_asd.data.beats_cache import BEATsCache, build_beats_token_cache
from care_asd.data.care_residual_vector_cache import (
    CareResidualVectorCache,
    build_care_residual_vector_cache,
)
from care_asd.data.dcase2026 import (
    DatasetAudit,
    DatasetSplit,
    DownloadedArchive,
    ExtractionSummary,
    audit_dcase2026_manifest,
    build_dcase2026_manifest,
    dcase2026_manifest_path,
    download_dcase2026_split,
    extract_dcase2026_split,
    normalize_split,
)
from care_asd.data.fp_naa_augmentation_cache import (
    FPNaaAugmentationCache,
    build_fp_naa_augmentation_cache,
)
from care_asd.data.fp_naa_observability import (
    FPNaaObservabilityResult,
    run_fp_naa_observability_probe,
)
from care_asd.data.neural_cache import BASE_CHANNELS, NeuralFeatureCache, build_neural_feature_cache
from care_asd.data.official_vector_cache import (
    OFFICIAL_FEATURE_DIM,
    OfficialVectorCache,
    build_official_vector_cache,
)
from care_asd.data.reliability_index import ReliabilityIndex, build_reliability_index

__all__ = [
    "BASE_CHANNELS",
    "OFFICIAL_FEATURE_DIM",
    "BEATsCache",
    "CareResidualVectorCache",
    "DatasetAudit",
    "DatasetSplit",
    "DownloadedArchive",
    "ExtractionSummary",
    "FPNaaAugmentationCache",
    "FPNaaObservabilityResult",
    "NeuralFeatureCache",
    "OfficialVectorCache",
    "ReliabilityIndex",
    "audit_dcase2026_manifest",
    "build_beats_token_cache",
    "build_care_residual_vector_cache",
    "build_dcase2026_manifest",
    "build_fp_naa_augmentation_cache",
    "build_neural_feature_cache",
    "build_official_vector_cache",
    "build_reliability_index",
    "dcase2026_manifest_path",
    "download_dcase2026_split",
    "extract_dcase2026_split",
    "normalize_split",
    "run_fp_naa_observability_probe",
]
