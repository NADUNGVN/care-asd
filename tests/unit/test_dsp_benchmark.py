"""Small end-to-end contract tests for the immutable DSP benchmark evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from care_asd.evaluation.dsp_benchmark import (
    run_care_development_benchmark,
    run_dsp_development_benchmark,
)


def test_dsp_benchmark_writes_scores_metrics_and_energy_audit(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    rows: list[dict[str, str]] = []
    for index, (split, domain, condition, gain) in enumerate(
        (
            ("dev_train", "source", "normal", 1.0),
            ("dev_train", "target", "normal", 1.1),
            ("dev_test", "source", "normal", 1.0),
            ("dev_test", "source", "anomaly", 2.0),
            ("dev_test", "target", "normal", 1.0),
            ("dev_test", "target", "anomaly", 2.0),
        )
    ):
        relative = f"ToyCar/{split}/{domain}_{condition}_{index:03d}.wav"
        path = audio_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.arange(128, dtype=np.float64)
        waveform = np.column_stack((gain * np.sin(samples / 5.0), 0.2 * np.cos(samples / 7.0)))
        sf.write(path, waveform, 16000)
        rows.append(
            {
                "file_id": relative,
                "relative_path": relative,
                "machine_type": "ToyCar",
                "section": "section_00",
                "domain": domain,
                "condition": condition,
                "dataset_split": split,
            }
        )
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)

    output = tmp_path / "benchmark"
    result = run_dsp_development_benchmark(
        manifest_path=manifest,
        audio_root=audio_root,
        output_directory=output,
        experiment_id="unit_dsp",
        frontends=("near", "difference"),
        workers=2,
    )

    assert result.summary_path.is_file()
    assert result.overcancellation_path.is_file()
    assert result.run_metadata_path.is_file()
    assert set((output / "scores").glob("*.csv")) == {
        output / "scores" / "near.csv",
        output / "scores" / "difference.csv",
    }
    assert len(pd.read_csv(result.overcancellation_path)) == 8
    with pytest.raises(FileExistsError):
        run_dsp_development_benchmark(
            manifest_path=manifest,
            audio_root=audio_root,
            output_directory=output,
            experiment_id="unit_dsp",
            frontends=("near",),
        )


def test_care_benchmark_uses_the_same_immutable_evidence_contract(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    rows: list[dict[str, str]] = []
    for index, (split, domain, condition, gain) in enumerate(
        (
            ("dev_train", "source", "normal", 1.0),
            ("dev_train", "target", "normal", 1.1),
            ("dev_test", "source", "normal", 1.0),
            ("dev_test", "source", "anomaly", 2.0),
            ("dev_test", "target", "normal", 1.0),
            ("dev_test", "target", "anomaly", 2.0),
        )
    ):
        relative = f"ToyCar/{split}/{domain}_{condition}_{index:03d}.wav"
        path = audio_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.arange(128, dtype=np.float64)
        waveform = np.column_stack((gain * np.sin(samples / 5.0), 0.4 * np.sin(samples / 5.0)))
        sf.write(path, waveform, 16000)
        rows.append(
            {
                "file_id": relative,
                "relative_path": relative,
                "machine_type": "ToyCar",
                "section": "section_00",
                "domain": domain,
                "condition": condition,
                "dataset_split": split,
            }
        )
    manifest = tmp_path / "manifest.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)

    result = run_care_development_benchmark(
        manifest_path=manifest,
        audio_root=audio_root,
        output_directory=tmp_path / "care_benchmark",
        experiment_id="unit_care",
        workers=2,
    )

    assert result.summary_path.is_file()
    assert (result.output_directory / "scores" / "care.csv").is_file()
    assert result.frequency_bands_path is not None and result.frequency_bands_path.is_file()
    assert (result.output_directory / "frequency_band_diagnostics.svg").is_file()
    assert pd.read_csv(result.summary_path)["frontend"].unique().tolist() == ["care"]
