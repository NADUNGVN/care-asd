"""Unit tests for the network-free DCASE 2026 dataset pipeline."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from care_asd.data.dcase2026 import (
    audit_dcase2026_manifest,
    build_dcase2026_manifest,
    download_dcase2026_split,
    extract_dcase2026_split,
    normalize_split,
)


def _write_audio(root: Path, channels: int = 2) -> Path:
    audio_path = (
        root
        / "raw"
        / "dcase2026"
        / "dev"
        / "extracted"
        / "ToyCar"
        / "test"
        / "normal_id_00_source_test_section_00.wav"
    )
    audio_path.parent.mkdir(parents=True)
    frames = (
        np.zeros((160, channels), dtype=np.float32)
        if channels == 2
        else np.zeros(160, dtype=np.float32)
    )
    sf.write(audio_path, frames, 16000)
    return audio_path


def test_build_and_audit_stereo_manifest(tmp_path: Path) -> None:
    _write_audio(tmp_path)

    manifest = build_dcase2026_manifest(tmp_path, "dev")
    audit = audit_dcase2026_manifest(manifest)

    assert audit.clips == 1
    assert audit.stereo_clips == 1
    assert audit.sample_rates == (16000,)
    assert audit.conditions == ("normal",)
    assert audit.domains == ("source",)


def test_audit_rejects_mono_audio(tmp_path: Path) -> None:
    _write_audio(tmp_path, channels=1)
    manifest = build_dcase2026_manifest(tmp_path, "dev")

    with pytest.raises(ValueError, match="stereo"):
        audit_dcase2026_manifest(manifest)


def test_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "raw" / "dcase2026" / "dev" / "archives" / "unsafe.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr("../outside.txt", "must not be written")

    with pytest.raises(ValueError, match="Unsafe archive member"):
        extract_dcase2026_split(tmp_path, "dev")


def test_evaluation_download_requires_explicit_acceptance(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="policy acceptance"):
        download_dcase2026_split(tmp_path, "evaluation")


def test_normalize_split_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="split must be"):
        normalize_split("invalid")
