"""Validated, portable deployment bundles.

The bundle is the boundary between the Python research environment and a
Jetson TensorRT runner. Every external artifact is hashed and validated before
it is accepted, so deployment cannot silently use a stale model, scorer, or
threshold policy.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from care_asd.config import CareASDConfig, config_hash
from care_asd.reproducibility import file_sha256

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST_NAME = "bundle.json"


class BundleArtifact(BaseModel):
    """A bundle-local immutable artifact reference."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    sha256: str

    @field_validator("filename")
    @classmethod
    def filename_must_be_local(cls, value: str) -> str:
        path = Path(value)
        if path.name != value or value in {"", ".", ".."}:
            raise ValueError("artifact filename must be a plain bundle-local filename")
        return value

    @field_validator("sha256")
    @classmethod
    def sha256_must_be_digest(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        return value


class ModelInputSpec(BaseModel):
    """Tensor contract consumed by the deployment encoder."""

    model_config = ConfigDict(extra="forbid")

    name: str = "features"
    shape: list[int] = Field(min_length=1)
    dtype: Literal["float32"] = "float32"

    @field_validator("shape")
    @classmethod
    def dimensions_must_be_positive_or_dynamic(cls, value: list[int]) -> list[int]:
        if any(dimension == 0 or dimension < -1 for dimension in value):
            raise ValueError("input dimensions must be positive or -1 for a dynamic dimension")
        return value


class MahalanobisScorerArtifact(BaseModel):
    """Normal-reference scorer consumed after TensorRT inference."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["mahalanobis"] = "mahalanobis"
    mean: list[float] = Field(min_length=1)
    precision_matrix: list[list[float]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_square_precision(self) -> MahalanobisScorerArtifact:
        dimension = len(self.mean)
        if len(self.precision_matrix) != dimension or any(
            len(row) != dimension for row in self.precision_matrix
        ):
            raise ValueError("precision_matrix must be square with dimension len(mean)")
        return self


class SelectiveCalibrationArtifact(BaseModel):
    """Normal-only threshold policy for deployment decisions."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["normal_only_threshold"] = "normal_only_threshold"
    anomaly_threshold: float
    abstain_margin: float = Field(default=0.0, ge=0.0)


class DeploymentBundleManifest(BaseModel):
    """Versioned contract for a portable CARE-ASD inference bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    config_hash: str
    model_input: ModelInputSpec
    model: BundleArtifact
    scorer: BundleArtifact
    calibration: BundleArtifact
    output_name: str = "embedding"

    @field_validator("config_hash")
    @classmethod
    def config_hash_must_be_digest(cls, value: str) -> str:
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError("config_hash must be a lowercase SHA-256 digest")
        return value


def create_deployment_bundle(
    output_dir: str | Path,
    *,
    onnx_model: str | Path,
    scorer: MahalanobisScorerArtifact,
    calibration: SelectiveCalibrationArtifact,
    model_input: ModelInputSpec,
    config: CareASDConfig,
) -> Path:
    """Create a no-overwrite, hash-verified deployment bundle."""
    destination = Path(output_dir)
    source_model = Path(onnx_model)
    if not source_model.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source_model}")
    if destination.exists():
        raise FileExistsError(f"Bundle directory already exists: {destination}")

    destination.mkdir(parents=True)
    model_path = destination / "model.onnx"
    scorer_path = destination / "scorer.json"
    calibration_path = destination / "calibration.json"
    shutil.copy2(source_model, model_path)
    _write_json_no_overwrite(scorer_path, scorer.model_dump())
    _write_json_no_overwrite(calibration_path, calibration.model_dump())

    manifest = DeploymentBundleManifest(
        config_hash=config_hash(config),
        model_input=model_input,
        model=BundleArtifact(filename=model_path.name, sha256=file_sha256(model_path)),
        scorer=BundleArtifact(filename=scorer_path.name, sha256=file_sha256(scorer_path)),
        calibration=BundleArtifact(
            filename=calibration_path.name,
            sha256=file_sha256(calibration_path),
        ),
    )
    _write_json_no_overwrite(destination / BUNDLE_MANIFEST_NAME, manifest.model_dump())
    validate_deployment_bundle(destination)
    return destination


def validate_deployment_bundle(bundle_dir: str | Path) -> DeploymentBundleManifest:
    """Validate the bundle manifest, referenced hashes, and scorer contracts."""
    directory = Path(bundle_dir)
    manifest_path = directory / BUNDLE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Bundle manifest not found: {manifest_path}")
    manifest = DeploymentBundleManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))

    for artifact in (manifest.model, manifest.scorer, manifest.calibration):
        path = directory / artifact.filename
        if not path.is_file():
            raise FileNotFoundError(f"Bundle artifact not found: {path}")
        if file_sha256(path) != artifact.sha256:
            raise ValueError(f"Bundle artifact hash mismatch: {artifact.filename}")

    MahalanobisScorerArtifact.model_validate_json(
        (directory / manifest.scorer.filename).read_text(encoding="utf-8")
    )
    SelectiveCalibrationArtifact.model_validate_json(
        (directory / manifest.calibration.filename).read_text(encoding="utf-8")
    )
    return manifest


def load_bundle_artifact_json(bundle_dir: str | Path, artifact: BundleArtifact) -> dict[str, Any]:
    """Load a verified JSON artifact after validating its parent bundle."""
    directory = Path(bundle_dir)
    validate_deployment_bundle(directory)
    payload = json.loads((directory / artifact.filename).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Bundle JSON artifact must be an object: {artifact.filename}")
    return payload


def _write_json_no_overwrite(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
