"""Dataset acquisition, manifests, and loaders (Phase 1+)."""

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
    "CareResidualVectorCache",
    "DatasetAudit",
    "DatasetSplit",
    "DownloadedArchive",
    "ExtractionSummary",
    "NeuralFeatureCache",
    "OfficialVectorCache",
    "ReliabilityIndex",
    "audit_dcase2026_manifest",
    "build_care_residual_vector_cache",
    "build_dcase2026_manifest",
    "build_neural_feature_cache",
    "build_official_vector_cache",
    "build_reliability_index",
    "dcase2026_manifest_path",
    "download_dcase2026_split",
    "extract_dcase2026_split",
    "normalize_split",
]
