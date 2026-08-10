"""Tests for TensorRT benchmark report validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from care_asd.deployment import validate_tensorrt_model_latency_report


def _write_report(path: Path, *, p95: float = 2.0) -> Path:
    path.write_text(
        """{
  "runner": "care_asd_trt_runner",
  "warmup_windows": 100,
  "timed_windows": 1000,
  "input_elements": 32,
  "output_elements": 8,
  "model_latency_ms": {"mean": 1.2, "p50": 1.0, "p95": """
        + str(p95)
        + ", \"p99\": 3.0}\n}\n",
        encoding="utf-8",
    )
    return path


def test_tensorrt_report_contract_round_trip(tmp_path: Path) -> None:
    report = validate_tensorrt_model_latency_report(_write_report(tmp_path / "report.json"))

    assert report.timed_windows == 1000
    assert report.model_latency_ms.p95 == 2.0


def test_tensorrt_report_rejects_invalid_percentile_order(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="p50 <= p95 <= p99"):
        validate_tensorrt_model_latency_report(_write_report(tmp_path / "report.json", p95=0.5))
