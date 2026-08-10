"""Validation for hardware-produced TensorRT benchmark reports."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelLatencyStats(BaseModel):
    """Model-only latency summaries in milliseconds."""

    model_config = ConfigDict(extra="forbid")

    mean: float = Field(ge=0.0)
    p50: float = Field(ge=0.0)
    p95: float = Field(ge=0.0)
    p99: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_percentile_order(self) -> ModelLatencyStats:
        if not self.p50 <= self.p95 <= self.p99:
            raise ValueError("latency percentiles must satisfy p50 <= p95 <= p99")
        return self


class TensorRTModelLatencyReport(BaseModel):
    """Stable JSON report produced by ``care_asd_trt_runner``."""

    model_config = ConfigDict(extra="forbid")

    runner: Literal["care_asd_trt_runner"]
    warmup_windows: int = Field(ge=0)
    timed_windows: int = Field(gt=0)
    input_elements: int = Field(gt=0)
    output_elements: int = Field(gt=0)
    model_latency_ms: ModelLatencyStats


def validate_tensorrt_model_latency_report(path: str | Path) -> TensorRTModelLatencyReport:
    """Parse untrusted runner JSON against the public benchmark report contract."""
    report_path = Path(path)
    if not report_path.is_file():
        raise FileNotFoundError(f"TensorRT benchmark report not found: {report_path}")
    return TensorRTModelLatencyReport.model_validate_json(report_path.read_text(encoding="utf-8"))
