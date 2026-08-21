"""Deterministic post-hoc DSP sensitivity analysis over frozen development scores."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from care_asd.evaluation.dcase2026_metrics import calculate_dcase2026_official_metrics
from care_asd.evaluation.official_baseline import SCORE_COLUMNS


def run_frozen_dsp_sensitivity(
    config_path: str | Path,
    repo_root: str | Path,
    *,
    output_dir_override: str | Path | None = None,
) -> Path:
    """Derive official-metric and real/emulated summaries without altering frozen inputs."""
    root = Path(repo_root).resolve()
    config_file = Path(config_path).resolve()
    config = _read_config(config_file)
    configured_output = Path(str(config["output_dir"]))
    selected_output = Path(output_dir_override) if output_dir_override else configured_output
    output_dir = (root / selected_output).resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Sensitivity output must remain inside the repository: {output_dir}") from error
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite sensitivity output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_commit = str(config["source_snapshot_commit"])
    source_entries: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    source_paths: dict[str, Path] = {}
    expected_hashes = {
        str(system): str(digest)
        for system, digest in dict(config["expected_source_sha256"]).items()
    }
    for system, relative_path in dict(config["sources"]).items():
        path = (root / str(relative_path)).resolve()
        digest = _sha256(path)
        if expected_hashes.get(str(system)) != digest:
            raise ValueError(f"Frozen source hash mismatch for {system}: {path}")
        frame = pd.read_csv(path)
        _validate_score_frame(frame, system=str(system))
        frames[str(system)] = frame
        source_paths[str(system)] = path
        source_entries.append(
            {
                "system": str(system),
                "path": _portable(path, root),
                "sha256": digest,
                "rows": len(frame),
            }
        )

    reference = str(config["reference"])
    if reference not in frames:
        raise ValueError(f"Reference system is not present in sources: {reference}")
    _validate_pairing(frames, reference=reference)

    machine_groups = {
        str(name): [str(machine) for machine in machines]
        for name, machines in dict(config["machine_groups"]).items()
    }
    _validate_machine_groups(frames[reference], machine_groups)
    scopes = {"all": sorted(frames[reference]["machine_type"].unique().tolist())}
    scopes.update(machine_groups)

    summary_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    scores_by_scope: dict[str, dict[str, float]] = {}
    with tempfile.TemporaryDirectory(prefix="care_asd_dsp_sensitivity_") as temp_name:
        temp_dir = Path(temp_name)
        for scope, machines in scopes.items():
            scores_by_scope[scope] = {}
            for system, frame in frames.items():
                subset = frame.loc[frame["machine_type"].isin(machines)].copy()
                score_path = temp_dir / f"{system.lower()}_{scope}_scores.csv"
                metric_path = temp_dir / f"{system.lower()}_{scope}_metrics.json"
                subset.to_csv(score_path, index=False)
                calculate_dcase2026_official_metrics(score_path, metric_path)
                payload = json.loads(metric_path.read_text(encoding="utf-8"))
                score = float(payload["official_score"])
                scores_by_scope[scope][system] = score
                summary_rows.append(
                    {
                        "system": system,
                        "scope": scope,
                        "machine_count": len(machines),
                        "cell_count": int(payload["cell_count"]),
                        "official_score": score,
                        "official_score_percent": 100.0 * score,
                        "analysis_label": "derived_post_hoc_from_frozen_predictions_artifacts",
                    }
                )
                for group, values in dict(payload["groups"]).items():
                    machine, section = group.split("/", maxsplit=1)
                    for metric, value in dict(values).items():
                        cell_rows.append(
                            {
                                "system": system,
                                "scope": scope,
                                "machine_type": machine,
                                "section": section,
                                "metric": metric,
                                "value": float(value),
                            }
                        )

    for row in summary_rows:
        row["delta_vs_b00"] = (
            float(row["official_score"]) - scores_by_scope[str(row["scope"])][reference]
        )

    summary_path = output_dir / "official_metric_summary.csv"
    cells_path = output_dir / "official_metric_cells.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(cell_rows).to_csv(cells_path, index=False)

    code_entries = []
    for relative_path in list(config["analysis_code"]):
        path = (root / str(relative_path)).resolve()
        code_entries.append({"path": _portable(path, root), "sha256": _sha256(path)})

    config_entry = {"path": _portable(config_file, root), "sha256": _sha256(config_file)}
    run_payload: dict[str, Any] = {
        "schema_version": 1,
        "study_id": str(config["study_id"]),
        "analysis_type": "post_hoc_sensitivity",
        "chronological_label": "derived post hoc from frozen predictions/artifacts",
        "source_snapshot_commit": source_commit,
        "config": config_entry,
        "reference": reference,
        "metric_contract": (
            "DCASE2026: domain normal versus all anomalies; pooled pAUC@0.1; "
            "harmonic mean over every cell"
        ),
        "machine_groups": machine_groups,
        "pairing": {
            "file_ids_identical": True,
            "metadata_identical": True,
            "rows_per_system": len(frames[reference]),
        },
        "inputs": source_entries,
        "analysis_code": code_entries,
        "decisions": dict(config["decisions"]),
        "outputs": [
            "official_metric_summary.csv",
            "official_metric_cells.csv",
            "README.md",
            "manifest.json",
        ],
    }
    run_path = output_dir / "run.json"
    run_path.write_text(json.dumps(run_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme_path = output_dir / "README.md"
    readme_path.write_text(
        _build_readme(source_commit=source_commit, machine_groups=machine_groups),
        encoding="utf-8",
    )

    output_entries = []
    for path in (summary_path, cells_path, run_path, readme_path):
        output_entries.append(
            {"path": _portable(path, root), "sha256": _sha256(path), "bytes": path.stat().st_size}
        )
    manifest = {
        "schema_version": 1,
        "study_id": str(config["study_id"]),
        "source_snapshot_commit": source_commit,
        "config": config_entry,
        "inputs": source_entries,
        "outputs": output_entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def _read_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Sensitivity config must be a mapping")
    required = {
        "study_id",
        "source_snapshot_commit",
        "sources",
        "expected_source_sha256",
        "reference",
        "machine_groups",
        "analysis_code",
        "output_dir",
        "decisions",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Sensitivity config is missing: {', '.join(missing)}")
    return payload


def _validate_score_frame(frame: pd.DataFrame, *, system: str) -> None:
    missing = sorted(set(SCORE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"{system} scores are missing: {', '.join(missing)}")
    if frame["file_id"].duplicated().any():
        raise ValueError(f"{system} scores contain duplicate file_id values")
    if not set(frame["condition"]).issubset({"normal", "anomaly"}):
        raise ValueError(f"{system} scores contain an invalid condition")
    if not set(frame["domain"]).issubset({"source", "target"}):
        raise ValueError(f"{system} scores contain an invalid domain")


def _validate_pairing(frames: dict[str, pd.DataFrame], *, reference: str) -> None:
    metadata = ["file_id", "machine_type", "section", "domain", "condition"]
    expected = frames[reference][metadata].sort_values("file_id").reset_index(drop=True)
    for system, frame in frames.items():
        observed = frame[metadata].sort_values("file_id").reset_index(drop=True)
        if not expected.equals(observed):
            raise ValueError(f"Frozen score metadata are not paired for {reference} and {system}")


def _validate_machine_groups(frame: pd.DataFrame, groups: dict[str, list[str]]) -> None:
    observed = set(frame["machine_type"].unique())
    assigned: list[str] = [machine for machines in groups.values() for machine in machines]
    if len(assigned) != len(set(assigned)):
        raise ValueError("Machine sensitivity groups overlap")
    if set(assigned) != observed:
        missing = sorted(observed.difference(assigned))
        extra = sorted(set(assigned).difference(observed))
        raise ValueError(f"Machine groups do not partition inputs; missing={missing}, extra={extra}")


def _build_readme(*, source_commit: str, machine_groups: dict[str, list[str]]) -> str:
    real = ", ".join(machine_groups["real"])
    emulated = ", ".join(machine_groups["emulated"])
    return f"""# Frozen DSP sensitivity analysis

This directory contains a **derived post hoc analysis from frozen predictions/artifacts** at
Audit-A4 source snapshot `{source_commit}`. It performs no training, retuning, threshold change,
subset search, or evaluation-set access.

`official_metric_summary.csv` reports the exact DCASE 2026 Task 2 development harmonic score for
B00, B01, and B02. The historical paired AUC/pAUC deltas remain the frozen inferential estimands;
the harmonic score is secondary and descriptive.

The same deterministic calculation is stratified by the repository's dataset construction:
real synchronized machine types ({real}) and emulated two-channel machine types ({emulated}). With
only two real machine types and five emulated types, these strata are descriptive sensitivity
analyses, not preregistered subgroup inference.

`run.json` records the analysis label, controls, input hashes, code hashes, pairing checks, and
machine partition. `manifest.json` records input and generated-output hashes. The script refuses to
overwrite this directory.

Reproduce into a new path with:

```bash
uv run python scripts/run_dsp_frozen_sensitivity.py \\
  --config configs/research/dsp_frozen_sensitivity_v1.yaml \\
  --output-dir <new-output-directory>
```
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()
