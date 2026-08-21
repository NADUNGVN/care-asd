from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from care_asd.evaluation.dsp_frozen_sensitivity import run_frozen_dsp_sensitivity


def _score_rows(system: str, offset: float) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for machine in ("ToyCar", "fanEmu"):
        for domain in ("source", "target"):
            for condition, score in (("normal", 0.1), ("anomaly", 0.9 + offset)):
                rows.append(
                    {
                        "file_id": f"{machine}-{domain}-{condition}",
                        "machine_type": machine,
                        "section": "section_00",
                        "domain": domain,
                        "condition": condition,
                        "anomaly_score": score,
                        "model_id": system,
                        "experiment_id": system,
                    }
                )
    return rows


def test_frozen_sensitivity_writes_hashed_post_hoc_package(tmp_path: Path) -> None:
    sources: dict[str, str] = {}
    expected_hashes: dict[str, str] = {}
    for system, offset in (("B00", 0.0), ("B01", -0.05), ("B02", -0.02)):
        path = tmp_path / f"{system.lower()}.csv"
        pd.DataFrame(_score_rows(system, offset)).to_csv(path, index=False)
        sources[system] = path.name
        expected_hashes[system] = hashlib.sha256(path.read_bytes()).hexdigest()

    config = {
        "study_id": "test",
        "source_snapshot_commit": "frozen",
        "sources": sources,
        "expected_source_sha256": expected_hashes,
        "reference": "B00",
        "machine_groups": {"real": ["ToyCar"], "emulated": ["fanEmu"]},
        "analysis_code": ["analysis.py"],
        "output_dir": "output",
        "decisions": {"post_hoc": True, "model_tuning": "prohibited"},
    }
    (tmp_path / "analysis.py").write_text("# fixed\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    output = run_frozen_dsp_sensitivity(config_path, tmp_path)
    summary = pd.read_csv(output / "official_metric_summary.csv")
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert len(summary) == 9
    assert set(summary["scope"]) == {"all", "real", "emulated"}
    assert run["chronological_label"] == "derived post hoc from frozen predictions/artifacts"
    assert run["config"]["path"] == "config.yaml"
    assert run["pairing"]["file_ids_identical"] is True
    assert len(manifest["inputs"]) == 3
    assert len(manifest["outputs"]) == 4

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_frozen_dsp_sensitivity(config_path, tmp_path)

    reproduced = run_frozen_dsp_sensitivity(
        config_path,
        tmp_path,
        output_dir_override="reproduced",
    )
    reproduced_summary = pd.read_csv(reproduced / "official_metric_summary.csv")
    pd.testing.assert_frame_equal(summary, reproduced_summary)
