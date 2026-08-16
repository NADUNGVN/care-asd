"""Strong frozen-BEATs C0 baseline for the FP-NAA successor track."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from care_asd.evaluation.dcase2026_metrics import (
    calculate_dcase2026_official_metrics,
    read_official_score,
)
from care_asd.evaluation.fp_naa_backend import (
    accelerated_beam_scores,
    accelerated_global_knn_scores,
    accelerated_rdp_pool,
)
from care_asd.evaluation.official_baseline import SCORE_COLUMNS
from care_asd.fp_naa_config import load_fp_naa_config

BASELINE_METHODS = (
    "global_ap",
    "freq_ap",
    "freq_ap_beam",
    "freq_rdp4_beam",
    "freq_rdp8_beam",
)


@dataclass(frozen=True)
class FPNaaBaselineResult:
    output_directory: Path
    summary_path: Path
    gate_path: Path
    gate_passed: bool
    c0_official_score: float


def run_fp_naa_baseline(
    *,
    cache_directory: str | Path,
    output_directory: str | Path,
    config_path: str | Path,
    experiment_id: str,
    device: str = "cuda",
) -> FPNaaBaselineResult:
    """Evaluate published backend increments using one immutable BEATs cache."""
    cache = Path(cache_directory).resolve()
    output = Path(output_directory).resolve()
    config_source = Path(config_path).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite baseline output: {output}")
    config = load_fp_naa_config(config_source)
    cache_metadata = json.loads((cache / "cache.json").read_text(encoding="utf-8"))
    if cache_metadata.get("checkpoint_sha256") != config.provenance.checkpoint_sha256:
        raise ValueError("BEATs cache checkpoint provenance does not match FP-NAA config")
    frame = pd.read_parquet(cache / "index.parquet")
    required = {
        "file_id",
        "feature_file",
        "machine_type",
        "section",
        "domain",
        "condition",
        "dataset_split",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"BEATs cache index is missing columns: {', '.join(missing)}")
    train = frame.loc[frame["dataset_split"] == "dev_train"].copy()
    test = frame.loc[frame["dataset_split"] == "dev_test"].copy().reset_index(drop=True)
    if train.empty or test.empty or test["file_id"].duplicated().any():
        raise ValueError("Baseline requires non-empty unique development train/test rows")
    output.mkdir(parents=True, exist_ok=True)
    method_scores = {method: np.full(len(test), np.nan, dtype=np.float64) for method in BASELINE_METHODS}
    diagnostics: dict[str, dict[str, object]] = {}
    groups = sorted(
        set(zip(test["machine_type"].astype(str), test["section"].astype(str), strict=True))
    )
    for group_number, (machine, section) in enumerate(groups, start=1):
        train_group = train.loc[
            (train["machine_type"].astype(str) == machine)
            & (train["section"].astype(str) == section)
        ]
        test_mask = (test["machine_type"].astype(str) == machine) & (
            test["section"].astype(str) == section
        )
        test_group = test.loc[test_mask]
        if len(train_group) < config.backend.local_density_neighbors + 1 or test_group.empty:
            raise ValueError(f"Insufficient cache rows for {machine}/{section}")
        train_tokens = _load_near_tokens(cache, train_group)
        test_tokens = _load_near_tokens(cache, test_group)
        train_frequency_mean = train_tokens.mean(axis=1, dtype=np.float32)
        test_frequency_mean = test_tokens.mean(axis=1, dtype=np.float32)
        train_global = train_tokens.mean(axis=(1, 2), dtype=np.float32)
        test_global = test_tokens.mean(axis=(1, 2), dtype=np.float32)
        train_frequency_flat = train_frequency_mean.reshape(len(train_group), -1)
        test_frequency_flat = test_frequency_mean.reshape(len(test_group), -1)
        rdp4_train = accelerated_rdp_pool(
            train_tokens,
            gamma=4.0,
            device=device,
            batch_size=config.frontend.inference_batch_size * 4,
            eps=config.backend.eps,
        )
        rdp4_test = accelerated_rdp_pool(
            test_tokens,
            gamma=4.0,
            device=device,
            batch_size=config.frontend.inference_batch_size * 4,
            eps=config.backend.eps,
        )
        rdp8_train = accelerated_rdp_pool(
            train_tokens,
            gamma=config.backend.rdp_gamma,
            device=device,
            batch_size=config.frontend.inference_batch_size * 4,
            eps=config.backend.eps,
        )
        rdp8_test = accelerated_rdp_pool(
            test_tokens,
            gamma=config.backend.rdp_gamma,
            device=device,
            batch_size=config.frontend.inference_batch_size * 4,
            eps=config.backend.eps,
        )
        neighbors = config.backend.local_density_neighbors
        global_scores, global_alpha = accelerated_global_knn_scores(
            test_global,
            train_global,
            neighbors=neighbors,
            device=device,
            eps=config.backend.eps,
        )
        frequency_scores, frequency_alpha = accelerated_global_knn_scores(
            test_frequency_flat,
            train_frequency_flat,
            neighbors=neighbors,
            device=device,
            eps=config.backend.eps,
        )
        mean_beam, mean_alpha = accelerated_beam_scores(
            test_frequency_mean,
            train_frequency_mean,
            neighbors=neighbors,
            device=device,
            eps=config.backend.eps,
        )
        rdp4_beam, rdp4_alpha = accelerated_beam_scores(
            rdp4_test,
            rdp4_train,
            neighbors=neighbors,
            device=device,
            eps=config.backend.eps,
        )
        rdp8_beam, rdp8_alpha = accelerated_beam_scores(
            rdp8_test,
            rdp8_train,
            neighbors=neighbors,
            device=device,
            eps=config.backend.eps,
        )
        target_indices = test.index[test_mask].to_numpy()
        values = {
            "global_ap": global_scores,
            "freq_ap": frequency_scores,
            "freq_ap_beam": mean_beam,
            "freq_rdp4_beam": rdp4_beam,
            "freq_rdp8_beam": rdp8_beam,
        }
        for method, scores in values.items():
            method_scores[method][target_indices] = scores
        diagnostics[f"{machine}/{section}"] = {
            "group_number": group_number,
            "train_clips": len(train_group),
            "test_clips": len(test_group),
            "alpha": {
                "global_ap": global_alpha,
                "freq_ap": frequency_alpha,
                "freq_ap_beam": mean_alpha.tolist(),
                "freq_rdp4_beam": rdp4_alpha.tolist(),
                "freq_rdp8_beam": rdp8_alpha.tolist(),
            },
        }
        _write_progress(output, completed=group_number, total=len(groups), group=f"{machine}/{section}")

    summary_rows: list[dict[str, object]] = []
    for method in BASELINE_METHODS:
        scores = method_scores[method]
        if not np.isfinite(scores).all():
            raise RuntimeError(f"Incomplete scores for baseline method {method}")
        method_directory = output / method
        method_directory.mkdir()
        score_frame = test[["file_id", "machine_type", "section", "domain", "condition"]].copy()
        score_frame["anomaly_score"] = scores
        score_frame["model_id"] = f"fp_naa_c0_{method}"
        score_frame["experiment_id"] = experiment_id
        score_frame = score_frame[SCORE_COLUMNS]
        score_path = method_directory / "scores.csv"
        metrics_path = method_directory / "metrics.json"
        score_frame.to_csv(score_path, index=False)
        calculate_dcase2026_official_metrics(score_path, metrics_path)
        official_score = read_official_score(metrics_path)
        summary_rows.append(
            {
                "method": method,
                "official_score": official_score,
                "official_score_percent": 100.0 * official_score,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = output / "summary.csv"
    summary.to_csv(summary_path, index=False)
    c0_score = float(
        summary.loc[summary["method"] == "freq_rdp8_beam", "official_score"].iloc[0]
    )
    gate_passed = c0_score >= config.gates.baseline_minimum_official_score
    gate = {
        "schema_version": 1,
        "gate": "G1_strong_baseline_reproduction",
        "method": "freq_rdp8_beam",
        "official_score": c0_score,
        "official_score_percent": 100.0 * c0_score,
        "minimum": config.gates.baseline_minimum_official_score,
        "minimum_percent": 100.0 * config.gates.baseline_minimum_official_score,
        "published_reference_percent": 62.02,
        "passed": gate_passed,
    }
    gate_path = output / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "cache_directory": str(cache),
                "device": device,
                "methods": list(BASELINE_METHODS),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_progress(output, completed=len(groups), total=len(groups), group="complete")
    return FPNaaBaselineResult(output, summary_path, gate_path, gate_passed, c0_score)


def _load_near_tokens(cache: Path, frame: pd.DataFrame) -> np.ndarray:
    grids: list[np.ndarray] = []
    expected: tuple[int, ...] | None = None
    for relative in frame["feature_file"].astype(str):
        path = cache / relative
        with np.load(path, allow_pickle=False) as payload:
            grid = payload["near"]
        if grid.ndim != 3 or grid.dtype != np.float16 or not np.isfinite(grid).all():
            raise ValueError(f"Invalid near-channel BEATs tokens: {path}")
        if expected is None:
            expected = grid.shape
        elif grid.shape != expected:
            raise ValueError(f"Inconsistent BEATs token shape: {path}")
        grids.append(grid)
    if not grids:
        raise ValueError("Cannot load an empty BEATs token group")
    return np.stack(grids)


def _write_progress(output: Path, *, completed: int, total: int, group: str) -> None:
    (output / "progress.env").write_text(
        f"stage=score\ncompleted_groups={completed}\ntotal_groups={total}\n"
        f"current_group={group}\nupdated_utc={datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}\n",
        encoding="utf-8",
    )

