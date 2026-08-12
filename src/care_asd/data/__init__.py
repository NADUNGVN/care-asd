"""Dataset acquisition, manifests, and loaders (Phase 1+)."""

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

__all__ = [
    "BASE_CHANNELS",
    "DatasetAudit",
    "DatasetSplit",
    "DownloadedArchive",
    "ExtractionSummary",
    "NeuralFeatureCache",
    "audit_dcase2026_manifest",
    "build_dcase2026_manifest",
    "build_neural_feature_cache",
    "dcase2026_manifest_path",
    "download_dcase2026_split",
    "extract_dcase2026_split",
    "normalize_split",
]
