"""Deployment bundle contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from care_asd.config import CareASDConfig
from care_asd.deployment import (
    MahalanobisScorerArtifact,
    ModelInputSpec,
    SelectiveCalibrationArtifact,
    create_deployment_bundle,
    validate_deployment_bundle,
)


def _write_model(path: Path) -> Path:
    path.write_bytes(b"not-a-real-onnx-model-but-a-stable-test-artifact")
    return path


def _create_bundle(tmp_path: Path) -> Path:
    return create_deployment_bundle(
        tmp_path / "bundle",
        onnx_model=_write_model(tmp_path / "model.onnx"),
        scorer=MahalanobisScorerArtifact(
            mean=[0.0, 1.0],
            precision_matrix=[[1.0, 0.0], [0.0, 1.0]],
        ),
        calibration=SelectiveCalibrationArtifact(anomaly_threshold=2.0, abstain_margin=0.2),
        model_input=ModelInputSpec(shape=[1, 4, 32, 64]),
        config=CareASDConfig(),
    )


def test_bundle_round_trip_is_hash_verified(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path)

    manifest = validate_deployment_bundle(bundle)

    assert manifest.schema_version == 1
    assert manifest.model.filename == "model.onnx"
    assert (bundle / "calibration.json").is_file()


def test_bundle_rejects_tampered_model(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path)
    (bundle / "model.onnx").write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_deployment_bundle(bundle)


def test_bundle_refuses_overwrite(tmp_path: Path) -> None:
    bundle = _create_bundle(tmp_path)

    with pytest.raises(FileExistsError):
        create_deployment_bundle(
            bundle,
            onnx_model=tmp_path / "model.onnx",
            scorer=MahalanobisScorerArtifact(mean=[0.0], precision_matrix=[[1.0]]),
            calibration=SelectiveCalibrationArtifact(anomaly_threshold=1.0),
            model_input=ModelInputSpec(shape=[1, 1]),
            config=CareASDConfig(),
        )
