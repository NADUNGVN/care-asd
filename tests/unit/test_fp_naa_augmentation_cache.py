from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from care_asd.data.fp_naa_augmentation_cache import build_fp_naa_augmentation_cache


class _FakeFrontend:
    def extract(self, waveforms: np.ndarray) -> np.ndarray:
        output = np.zeros((len(waveforms), 2, 8, 3), dtype=np.float32)
        output += waveforms.mean(axis=1)[:, None, None, None]
        return output


def test_counterfactual_augmentation_cache_is_resumable(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config_payload = yaml.safe_load(Path("configs/experiment/fp_naa_v1.yaml").read_text())
    config_payload["provenance"]["checkpoint_sha256"] = checkpoint_sha
    config_payload["frontend"].update(
        {
            "sample_rate": 8000,
            "duration_seconds": 0.1,
            "embedding_dim": 3,
            "inference_batch_size": 2,
        }
    )
    config_payload["augmentation"]["heldout_fraction"] = 0.99
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(config_payload, sort_keys=False), encoding="utf-8")
    audio_root = tmp_path / "audio"
    base_cache = tmp_path / "base"
    (base_cache / "features").mkdir(parents=True)
    rows = []
    for index in range(4):
        relative = Path("fan/train") / f"clip-{index}.wav"
        audio_path = audio_root / relative
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        time = np.arange(800, dtype=np.float64) / 8000
        near = 0.1 * np.sin(2.0 * np.pi * (300 + index * 20) * time)
        far = 0.03 * np.sin(2.0 * np.pi * (700 + index * 30) * time)
        sf.write(audio_path, np.column_stack([near, far]), 8000)
        rows.append(
            {
                "file_id": f"clip-{index}",
                "relative_path": relative.as_posix(),
                "feature_file": f"features/base-{index}.npz",
                "machine_type": "fan",
                "section": "section_00",
                "domain": "source",
                "condition": "normal",
                "dataset_split": "dev_train",
            }
        )
    pd.DataFrame(rows).to_parquet(base_cache / "index.parquet", index=False)
    (base_cache / "cache.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": checkpoint_sha,
                "beats_commit": config_payload["provenance"]["beats_commit"],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "augmentation"
    calls = 0

    def factory(*_args: object) -> _FakeFrontend:
        nonlocal calls
        calls += 1
        return _FakeFrontend()

    result = build_fp_naa_augmentation_cache(
        base_cache_directory=base_cache,
        audio_root=audio_root,
        output_directory=output,
        config_path=config,
        beats_source_directory=tmp_path / "source",
        checkpoint_path=checkpoint,
        workers=1,
        device="cpu",
        frontend_factory=factory,
    )
    assert result.clips == 4
    assert result.heldout_clips >= 1
    assert calls == 1
    index = pd.read_parquet(result.index_path)
    assert (index["file_id"] != index["donor_file_id"]).all()
    with np.load(output / index.loc[0, "augmentation_file"], allow_pickle=False) as payload:
        assert {"noisy_clean", "reference", "fault_teacher", "fault_noisy"}.issubset(payload.files)
        assert "heldout_noisy_clean" in payload.files
        assert payload["noisy_clean"].dtype == np.float16
        assert payload["noisy_clean"].shape == (2, 8, 3)

    def forbidden(*_args: object) -> _FakeFrontend:
        raise AssertionError("completed cache should be reused")

    reused = build_fp_naa_augmentation_cache(
        base_cache_directory=base_cache,
        audio_root=audio_root,
        output_directory=output,
        config_path=config,
        beats_source_directory=tmp_path / "source",
        checkpoint_path=checkpoint,
        workers=1,
        device="cpu",
        frontend_factory=forbidden,
    )
    assert reused == result
