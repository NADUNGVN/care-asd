from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import yaml

from care_asd.data.beats_cache import build_beats_token_cache
from care_asd.fp_naa_config import load_fp_naa_config
from care_asd.models.beats_frontend import fixed_duration_waveform


class _FakeFrontend:
    def extract(self, waveforms: np.ndarray) -> np.ndarray:
        means = waveforms.mean(axis=1, keepdims=True)
        output = np.zeros((len(waveforms), 2, 8, 3), dtype=np.float32)
        output += means[:, None, None, :]
        return output


def _write_config(path: Path, checkpoint_sha: str) -> None:
    payload = {
        "schema_version": 1,
        "experiment_id": "test",
        "provenance": {
            "beats_repository": "https://github.com/microsoft/unilm.git",
            "beats_commit": "8" * 40,
            "checkpoint_url": "https://example.com/BEATs_iter3.pt",
            "checkpoint_sha256": checkpoint_sha,
        },
        "frontend": {
            "sample_rate": 8000,
            "duration_seconds": 0.1,
            "channels": ["near", "far"],
            "frequency_patches": 8,
            "embedding_dim": 3,
            "cache_dtype": "float16",
            "inference_batch_size": 2,
        },
        "backend": {
            "temporal_pooling": "rdp",
            "rdp_gamma": 8.0,
            "scorer": "beam",
            "cosine_distance_scale": 0.5,
            "local_density_neighbors": 1,
            "score_rescaling": "variance_minimization_train_all",
            "eps": 1.0e-12,
        },
        "adapter": {
            "hidden_dim": 8,
            "attention_heads": 2,
            "dropout": 0.1,
            "reference_dropout_probability": 0.2,
            "reference_corruption_probability": 0.3,
        },
        "objective": {
            "normal_mse_weight": 1.0,
            "fault_direction_weight": 1.0,
            "fault_magnitude_weight": 0.5,
            "reference_consistency_weight": 0.25,
            "magnitude_huber_delta": 0.1,
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "workers": 1,
            "mixed_precision": False,
            "screening_seeds": [1],
            "confirmatory_seeds": [1],
        },
        "gates": {
            "baseline_minimum_official_score": 0.6,
            "screening_minimum_official_score": 0.62,
            "screening_minimum_gain_over_c0": 0.01,
            "screening_minimum_gain_over_c1": 0.005,
            "confirmatory_minimum_ensemble_official_score": 0.63,
            "confirmatory_minimum_gain_over_c1": 0.0075,
            "confirmatory_bootstrap_ci_low_minimum": 0.0,
            "fault_delta_retention_median_minimum": 0.9,
            "fault_delta_retention_q05_minimum": 0.75,
            "screening_maximum_machine_drop": 0.02,
            "confirmatory_maximum_machine_drop": 0.01,
            "screening_positive_lomo_folds_minimum": 5,
            "confirmatory_positive_lomo_folds_minimum": 6,
            "bootstrap_iterations": 100,
        },
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_beats_cache_is_stereo_resumable_and_provenance_locked(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    rows = []
    sample_rate = 8000
    for index in range(4):
        split = "dev_train" if index < 2 else "dev_test"
        condition = "normal" if index != 3 else "anomaly"
        relative = Path("fan") / split / f"clip-{index}.wav"
        path = audio_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.full(400 + 200 * index, 0.1 * (index + 1), dtype=np.float32)
        sf.write(path, np.column_stack([samples, -samples]), sample_rate)
        rows.append(
            {
                "file_id": f"clip-{index}",
                "relative_path": relative.as_posix(),
                "machine_type": "fan",
                "section": "section_00",
                "domain": "source" if index % 2 == 0 else "target",
                "condition": condition,
                "dataset_split": split,
            }
        )
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    config = tmp_path / "config.yaml"
    _write_config(config, checkpoint_sha)
    source = tmp_path / "source"
    source.mkdir()
    output = tmp_path / "cache"
    factory_calls = 0

    def factory(*_args: object) -> _FakeFrontend:
        nonlocal factory_calls
        factory_calls += 1
        return _FakeFrontend()

    result = build_beats_token_cache(
        manifest_path=manifest,
        audio_root=audio_root,
        output_directory=output,
        config_path=config,
        beats_source_directory=source,
        checkpoint_path=checkpoint,
        workers=1,
        device="cpu",
        frontend_factory=factory,
    )
    assert result.clips == 4
    assert result.token_shape == (2, 8, 3)
    assert factory_calls == 1
    index = pd.read_parquet(result.index_path)
    with np.load(output / index.loc[0, "feature_file"], allow_pickle=False) as payload:
        assert payload["near"].dtype == np.float16
        assert payload["near"].shape == (2, 8, 3)
        assert not np.array_equal(payload["near"], payload["far"])

    def forbidden_factory(*_args: object) -> _FakeFrontend:
        raise AssertionError("completed immutable cache should be reused")

    reused = build_beats_token_cache(
        manifest_path=manifest,
        audio_root=audio_root,
        output_directory=output,
        config_path=config,
        beats_source_directory=source,
        checkpoint_path=checkpoint,
        workers=1,
        device="cpu",
        frontend_factory=forbidden_factory,
    )
    assert reused == result


def test_fixed_duration_waveform_and_checked_in_config() -> None:
    short = fixed_duration_waveform(np.ones(3), sample_rate=2, duration_seconds=2.0)
    long = fixed_duration_waveform(np.arange(6), sample_rate=2, duration_seconds=2.0)
    np.testing.assert_array_equal(short, [1.0, 1.0, 1.0, 0.0])
    np.testing.assert_array_equal(long, [0.0, 1.0, 2.0, 3.0])
    config = load_fp_naa_config(Path("configs/experiment/fp_naa_v1.yaml"))
    assert config.training.workers == 12
    assert config.gates.baseline_minimum_official_score == 0.605
