from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from care_asd.data.reference_safety_cache import (
    build_reference_safety_vector_cache,
    load_reference_vectors,
)
from care_asd.reference_safety_config import ReferenceSafetyExperimentConfig


def test_reference_cache_builds_paired_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "care_asd.data.reference_safety_cache.official_waveform_to_vectors",
        lambda waveform, sample_rate: np.zeros((2, 640), dtype=np.float32),
    )
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    rows: list[dict[str, object]] = []
    sample_rate = 8_000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    for index, (split, condition, domain) in enumerate(
        [
            ("dev_train", "normal", "source"),
            ("dev_train", "normal", "target"),
            ("dev_test", "normal", "source"),
            ("dev_test", "anomaly", "target"),
        ]
    ):
        relative = Path("machine") / split / f"clip_{index}.wav"
        path = audio_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        machine = np.sin(2.0 * np.pi * (300.0 + index) * time)
        noise = 0.1 * np.sin(2.0 * np.pi * 1100.0 * time + 0.2)
        sf.write(path, np.column_stack([machine + noise, 0.2 * machine + noise]), sample_rate)
        rows.append(
            {
                "file_id": relative.as_posix(),
                "relative_path": relative.as_posix(),
                "machine_type": "machine",
                "section": "section_00",
                "domain": domain,
                "condition": condition,
                "dataset_split": split,
            }
        )
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame.from_records(rows).to_parquet(manifest, index=False)

    result = build_reference_safety_vector_cache(
        train_manifest_path=manifest,
        train_audio_root=audio_root,
        test_manifest_path=manifest,
        test_audio_root=audio_root,
        output_directory=tmp_path / "cache",
        config=ReferenceSafetyExperimentConfig(),
        workers=1,
    )

    index = pd.read_parquet(result.index_path)
    profiles = pd.read_parquet(result.profiles_path)
    first = result.directory / str(index.iloc[0]["cache_file"])
    assert result.clips == 4
    assert len(profiles) == 1
    assert load_reference_vectors(first, "near").shape[1] == 640
    assert load_reference_vectors(first, "refsub").shape[1] == 640


def test_evaluation_cache_rejects_known_test_labels(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    path = audio_root / "clip.wav"
    sf.write(path, np.zeros((2048, 2)), 8_000)
    common = {
        "file_id": "clip.wav",
        "relative_path": "clip.wav",
        "machine_type": "machine",
        "section": "section_00",
        "domain": "source",
    }
    train_manifest = tmp_path / "train.parquet"
    test_manifest = tmp_path / "test.parquet"
    pd.DataFrame([{**common, "condition": "normal", "dataset_split": "add_train"}]).to_parquet(
        train_manifest, index=False
    )
    pd.DataFrame(
        [{**common, "file_id": "test.wav", "condition": "anomaly", "dataset_split": "eval_test"}]
    ).to_parquet(test_manifest, index=False)

    with pytest.raises(ValueError, match="refuses manifests containing"):
        build_reference_safety_vector_cache(
            train_manifest_path=train_manifest,
            train_audio_root=audio_root,
            test_manifest_path=test_manifest,
            test_audio_root=audio_root,
            output_directory=tmp_path / "cache",
            config=ReferenceSafetyExperimentConfig(),
        )
