from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import yaml

from care_asd.data.fp_naa_reference_safety_cache import (
    _reference_leakage_pair,
    build_fp_naa_reference_safety_cache,
)


class _FakeFrontend:
    def extract(self, waveforms: np.ndarray) -> np.ndarray:
        output = np.zeros((len(waveforms), 2, 8, 3), dtype=np.float32)
        output += waveforms.mean(axis=1)[:, None, None, None]
        return output


def test_reference_leakage_pair_preserves_requested_ratio_and_shared_delta() -> None:
    time = np.arange(800, dtype=np.float64) / 8000
    clean = 0.1 * np.sin(2.0 * np.pi * 300.0 * time)
    faulty = clean + 0.01 * np.sin(2.0 * np.pi * 1200.0 * time)
    noise = 0.03 * np.sin(2.0 * np.pi * 700.0 * time)
    clean_ref, fault_ref, actual = _reference_leakage_pair(
        noise=noise,
        clean=clean,
        faulty=faulty,
        machine_to_noise_db=-10.0,
        peak_limit=0.999,
    )
    assert actual == pytest.approx(-10.0, abs=1.0e-6)
    delta = fault_ref - clean_ref
    assert np.linalg.norm(delta) > 0.0
    assert np.isfinite(clean_ref).all() and np.isfinite(fault_ref).all()


def test_reference_safety_cache_is_waveform_grounded_and_resumable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    main_payload = yaml.safe_load(Path("configs/experiment/fp_naa_v1.yaml").read_text())
    main_payload["provenance"]["checkpoint_sha256"] = checkpoint_sha
    main_payload["frontend"].update(
        {
            "sample_rate": 8000,
            "duration_seconds": 0.1,
            "embedding_dim": 3,
            "inference_batch_size": 2,
        }
    )
    main_config = tmp_path / "main.yaml"
    main_config.write_text(yaml.safe_dump(main_payload, sort_keys=False), encoding="utf-8")
    safety_config = Path("configs/experiment/fp_naa_reference_safety_v1.yaml")
    audio_root = tmp_path / "audio"
    base_cache = tmp_path / "base"
    augmentation_cache = tmp_path / "augmentation"
    base_cache.mkdir()
    augmentation_cache.mkdir()
    rows = []
    for index in range(2):
        relative = Path("fan/train") / f"clip-{index}.wav"
        path = audio_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        time = np.arange(800, dtype=np.float64) / 8000
        near = 0.1 * np.sin(2.0 * np.pi * (300.0 + 20.0 * index) * time)
        far = 0.03 * np.sin(2.0 * np.pi * (700.0 + 30.0 * index) * time)
        sf.write(path, np.column_stack((near, far)), 8000)
        rows.append(
            {
                "file_id": f"clip-{index}",
                "relative_path": relative.as_posix(),
                "machine_type": "fan",
                "section": "section_00",
                "condition": "normal",
                "dataset_split": "dev_train",
            }
        )
    pd.DataFrame(rows).to_parquet(base_cache / "index.parquet", index=False)
    metadata = {"checkpoint_sha256": checkpoint_sha}
    (base_cache / "cache.json").write_text(json.dumps(metadata), encoding="utf-8")
    augmentation_rows = []
    for index, row in enumerate(rows):
        augmentation_rows.append(
            {
                **row,
                "donor_file_id": f"clip-{1 - index}",
                "fault_seed": 100 + index,
                "requested_noise_snr_db": 0.0,
                "requested_fault_delta_level_db": -18.0,
                "heldout": True,
                "heldout_fault_family": "friction_burst",
            }
        )
    pd.DataFrame(augmentation_rows).to_parquet(augmentation_cache / "index.parquet", index=False)
    (augmentation_cache / "cache.json").write_text(json.dumps(metadata), encoding="utf-8")
    calls = 0

    def factory(*_args: object) -> _FakeFrontend:
        nonlocal calls
        calls += 1
        return _FakeFrontend()

    output = tmp_path / "safety"
    result = build_fp_naa_reference_safety_cache(
        base_cache_directory=base_cache,
        augmentation_cache_directory=augmentation_cache,
        audio_root=audio_root,
        output_directory=output,
        config_path=main_config,
        safety_config_path=safety_config,
        beats_source_directory=tmp_path / "source",
        checkpoint_path=checkpoint,
        workers=1,
        device="cpu",
        frontend_factory=factory,
    )
    assert result.clips == 2
    assert calls == 1
    index = pd.read_parquet(result.index_path)
    with np.load(output / index.loc[0, "safety_feature_file"], allow_pickle=False) as payload:
        assert {
            "leakage_low_clean_reference",
            "leakage_low_fault_reference",
            "leakage_medium_clean_reference",
            "leakage_medium_fault_reference",
            "leakage_high_clean_reference",
            "leakage_high_fault_reference",
        }.issubset(payload.files)
        metadata_payload = json.loads(str(payload["metadata_json"].item()))
        assert metadata_payload["actual_machine_to_noise_db"]["medium"] == pytest.approx(-10.0)
    silence = np.load(result.silence_reference_path, allow_pickle=False)
    assert silence.shape == (2, 8, 3)
    assert np.all(silence == 0.0)

    def forbidden(*_args: object) -> _FakeFrontend:
        raise AssertionError("completed safety cache should be reused")

    reused = build_fp_naa_reference_safety_cache(
        base_cache_directory=base_cache,
        augmentation_cache_directory=augmentation_cache,
        audio_root=audio_root,
        output_directory=output,
        config_path=main_config,
        safety_config_path=safety_config,
        beats_source_directory=tmp_path / "source",
        checkpoint_path=checkpoint,
        workers=1,
        device="cpu",
        frontend_factory=forbidden,
    )
    assert reused == result
