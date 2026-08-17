"""Conda-native FP-NAA job controller for SERVER-02.

The public contract is exposed through ``care-asd fp-naa job``. This module never
invokes a shell: every child process receives an argument vector, job state is
stored as atomic JSON, and detached jobs use the active Conda interpreter.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from care_asd.beats_contract import BEATS_COMMIT, BEATS_ITER3_SHA256, BEATS_REPOSITORY

SCHEMA_VERSION = 1
EXPECTED_CONDA_ENV = "care-asd-fp-naa"
EXPECTED_TORCH = "2.6.0+cu118"
EXPECTED_TORCHAUDIO = "2.6.0+cu118"
EXPECTED_CUDA = "11.8"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
BRANCH = "research/fp-naa"
CHECKPOINT_URL = (
    "https://huggingface.co/lpepino/beats_ckpts/resolve/"
    "a2ddb6b0411c39942ae144a6414872e14e5a4329/BEATs_iter3.pt"
)


class JobStage(StrEnum):
    """Stable public stage identifiers."""

    C0 = "c0"
    SCREENING = "screening"
    LOMO = "lomo"
    CONFIRMATORY = "confirmatory"
    CONFIRMATORY_LOMO = "confirmatory-lomo"
    REFERENCE_SAFETY = "reference-safety"


class JobStatus(StrEnum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"


class JobError(RuntimeError):
    """Boundary error with a stable machine-readable representation."""

    def __init__(self, code: str, message: str, details: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            payload["details"] = self.details
        return {"error": payload}


@dataclass(frozen=True)
class FPNAAJobContext:
    """Validated filesystem and execution boundary for one server checkout."""

    repo_root: Path
    data_root: Path
    branch: str = BRANCH

    @classmethod
    def from_environment(
        cls,
        *,
        repo_root: str | Path | None = None,
        data_root: str | Path | None = None,
    ) -> FPNAAJobContext:
        default_repo = (
            Path.cwd()
            if (Path.cwd() / "pyproject.toml").is_file()
            else Path.home() / "Dung_TDTU" / "CARE_ASD"
        )
        repo = Path(repo_root or os.environ.get("CARE_ASD_REPO_DIR") or default_repo).resolve()
        data = Path(
            data_root
            or os.environ.get("CARE_ASD_DATA_ROOT")
            or Path.home() / "Dung_TDTU" / "data" / "CARE_ASD"
        ).resolve()
        if not (repo / "pyproject.toml").is_file():
            raise JobError("INVALID_REPOSITORY", "CARE-ASD repository was not found", str(repo))
        return cls(repo_root=repo, data_root=data)

    @property
    def control_root(self) -> Path:
        return self.repo_root / "outputs" / "fp_naa" / "job_control"

    @property
    def jobs_root(self) -> Path:
        return self.repo_root / "outputs" / "fp_naa" / "jobs"

    @property
    def reports_root(self) -> Path:
        return self.repo_root / "reports" / "fp_naa"


@dataclass(frozen=True)
class JobState:
    schema_version: int
    run_id: str
    stage: str
    status: str
    current_step: str
    task_status: int
    workers: int
    pid: int | None
    source_git_sha: str
    created_utc: str
    updated_utc: str
    log: str
    report: str
    gate_passed: bool | None = None
    push_status: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JobState:
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise JobError("STATE_SCHEMA_MISMATCH", "Unsupported FP-NAA job state", payload)
        return cls(**payload)


@dataclass(frozen=True)
class StageResult:
    gate_path: Path | None
    gate_passed: bool
    summary_path: Path | None
    extra_paths: tuple[Path, ...] = ()


_STAGE_PREFIX = {
    JobStage.C0: "server02_fp_naa_c0",
    JobStage.SCREENING: "server02_fp_naa_screening",
    JobStage.LOMO: "server02_fp_naa_lomo",
    JobStage.CONFIRMATORY: "server02_fp_naa_confirmatory",
    JobStage.CONFIRMATORY_LOMO: "server02_fp_naa_confirmatory_lomo",
    JobStage.REFERENCE_SAFETY: "server02_fp_naa_reference_safety",
}

_GATE_CONTRACT = {
    JobStage.C0: ("c0_baseline", "passed"),
    JobStage.SCREENING: ("screening", "core_screening"),
    JobStage.LOMO: ("lomo", "passed"),
    JobStage.CONFIRMATORY: ("confirmatory", "core_confirmatory"),
    JobStage.CONFIRMATORY_LOMO: ("confirmatory_lomo", "passed"),
    JobStage.REFERENCE_SAFETY: ("reference_safety", "passed"),
}


def fp_naa_runtime_check(*, run_convolution: bool = True) -> dict[str, Any]:
    """Validate the exact Conda/PyTorch boundary used by every GPU job."""
    removed_dynamic_library_variables = []
    for variable in ("LD_LIBRARY_PATH", "LD_PRELOAD"):
        if os.environ.pop(variable, None) is not None:
            removed_dynamic_library_variables.append(variable)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    prefix_name = Path(sys.prefix).name
    conda_name = os.environ.get("CONDA_DEFAULT_ENV")
    if prefix_name != EXPECTED_CONDA_ENV and conda_name != EXPECTED_CONDA_ENV:
        raise JobError(
            "WRONG_CONDA_ENV",
            f"Run this command inside Conda environment {EXPECTED_CONDA_ENV}",
            {"sys_prefix": sys.prefix, "conda_default_env": conda_name},
        )
    try:
        import torch
        import torchaudio
    except (ImportError, OSError) as exc:
        raise JobError(
            "TORCH_IMPORT_FAILED", "Torch runtime could not be loaded", str(exc)
        ) from exc
    installed = _installed_package_names()
    cu13 = sorted(name for name in installed if "cu13" in name)
    checks = {
        "torch": str(torch.__version__),
        "torchaudio": str(torchaudio.__version__),
        "cuda": str(torch.version.cuda),
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cu13_packages": cu13,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "removed_dynamic_library_variables": removed_dynamic_library_variables,
    }
    if checks["torch"] != EXPECTED_TORCH:
        raise JobError("TORCH_VERSION_MISMATCH", "Unexpected Torch version", checks)
    if checks["torchaudio"] != EXPECTED_TORCHAUDIO:
        raise JobError("TORCHAUDIO_VERSION_MISMATCH", "Unexpected torchaudio version", checks)
    if checks["cuda"] != EXPECTED_CUDA or not checks["cuda_available"] or cu13:
        raise JobError("CUDA_RUNTIME_MISMATCH", "Conda CUDA runtime is not frozen", checks)
    checks["gpu"] = torch.cuda.get_device_name(0)
    if run_convolution:
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.use_deterministic_algorithms(True, warn_only=False)
            layer = torch.nn.Conv2d(1, 4, 3, padding=1).cuda()
            sample = torch.randn(2, 1, 32, 32, device="cuda")
            output = layer(sample)
            attention = torch.nn.MultiheadAttention(32, 4, batch_first=True).cuda()
            sequence = torch.randn(2, 8, 32, device="cuda", requires_grad=True)
            attended, _ = attention(sequence, sequence, sequence, need_weights=False)
            attended.square().mean().backward()
            torch.cuda.synchronize()
        except RuntimeError as exc:
            raise JobError(
                "DETERMINISTIC_GPU_PROBE_FAILED",
                "Deterministic convolution/attention probe failed",
                str(exc),
            ) from exc
        if tuple(output.shape) != (2, 4, 32, 32):
            raise JobError("CUDNN_PROBE_FAILED", "Unexpected convolution output shape")
        checks["convolution_probe"] = "passed"
        checks["deterministic_attention_backward_probe"] = "passed"
        checks["deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
        checks["deterministic_warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
        checks["flash_sdp_enabled"] = torch.backends.cuda.flash_sdp_enabled()
        checks["memory_efficient_sdp_enabled"] = torch.backends.cuda.mem_efficient_sdp_enabled()
        checks["math_sdp_enabled"] = torch.backends.cuda.math_sdp_enabled()
        try:
            from care_asd.evaluation.fp_naa_candidate import _primary_safe_backward
            from care_asd.fp_naa_config import FPObjectiveConfig
            from care_asd.models.fp_naa_adapter import rdp_salient_contraction_projection
            from care_asd.models.fp_naa_objective import fp_naa_loss

            objective = FPObjectiveConfig(
                normal_mse_weight=1.0,
                fault_direction_weight=0.25,
                fault_magnitude_weight=1.0,
                fault_separation_weight=2.0,
                reference_consistency_weight=0.0,
                magnitude_huber_delta=0.05,
                fault_loss_mode="tail_constrained",
                direction_cosine_floor=0.5,
                gain_lower_bound=1.05,
                gain_upper_bound=1.20,
                tail_fraction=0.10,
                score_gain_lower_bound=1.05,
                score_patch_fraction=0.20,
                primary_safe_gradient_projection=True,
            )
            clean = torch.ones(4, 2, 2, 8, device="cuda")
            delta = torch.zeros_like(clean)
            delta[..., 0] = 0.01
            delta[..., 1] = -0.01
            teacher_fault = clean + delta
            erased = fp_naa_loss(
                objective="c2_fault_preserving",
                student_clean=clean,
                teacher_clean=clean,
                student_fault=clean,
                teacher_fault=teacher_fault,
                config=objective,
            )
            safe = fp_naa_loss(
                objective="c2_fault_preserving",
                student_clean=clean,
                teacher_clean=clean,
                student_fault=clean + 1.10 * delta,
                teacher_fault=teacher_fault,
                config=objective,
            )
            if not bool(torch.isfinite(safe.total)) or not bool(safe.total < erased.total):
                raise RuntimeError("Tail-safe objective did not prefer the bounded-gain sample")
            parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0], device="cuda"))
            cosine, conflict = _primary_safe_backward(
                primary=parameter[0].square(),
                auxiliary=-parameter[0] + parameter[1],
                parameters=[parameter],
                auxiliary_scale=1.0,
            )
            expected = torch.tensor([2.0, 1.0], device="cuda")
            if parameter.grad is None or not torch.allclose(parameter.grad, expected):
                raise RuntimeError("Primary-safe gradient projection returned an invalid update")
            if not bool(cosine < 0.0) or not bool(conflict == 1.0):
                raise RuntimeError("Primary-safe gradient conflict was not detected")
            reference = torch.zeros(2, 4, 2, 8, device="cuda")
            target = torch.stack(
                [torch.full((2, 2, 8), float(index + 1), device="cuda") for index in range(4)],
                dim=1,
            )
            raw_correction = -target
            projected = rdp_salient_contraction_projection(
                correction=raw_correction,
                target=target,
                reference=reference,
                protected_fraction=0.50,
                maximum_contraction=0.10,
            )
            discrepancy = target - reference
            contraction = (projected * discrepancy).sum(dim=(2, 3))
            norm_sq = discrepancy.square().sum(dim=(2, 3))
            if not torch.all(contraction[:, -2:] >= -0.10001 * norm_sq[:, -2:]):
                raise RuntimeError("RDP-salient projection violated its contraction bound")
            if not torch.allclose(projected[:, :2], raw_correction[:, :2]):
                raise RuntimeError("RDP-salient projection changed an unprotected temporal row")
            torch.cuda.synchronize()
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise JobError(
                "TAIL_SAFE_METHOD_PROBE_FAILED",
                "FP-NAA v2 tail-loss/gradient probe failed",
                str(exc),
            ) from exc
        checks["tail_safe_objective_probe"] = "passed"
        checks["primary_safe_gradient_probe"] = "passed"
        checks["rdp_salient_projection_probe"] = "passed"
    return {"schema_version": SCHEMA_VERSION, "runtime": checks}


def start_fp_naa_job(
    stage: JobStage,
    *,
    workers: int = 12,
    context: FPNAAJobContext | None = None,
) -> dict[str, Any]:
    """Atomically validate and detach one stage using the current Conda interpreter."""
    ctx = context or FPNAAJobContext.from_environment()
    with _launch_lock(ctx):
        return _start_fp_naa_job_locked(stage, workers=workers, context=ctx)


def _start_fp_naa_job_locked(
    stage: JobStage,
    *,
    workers: int,
    context: FPNAAJobContext,
) -> dict[str, Any]:
    """Start one stage while the caller owns the atomic launch lock."""
    ctx = context
    if not 1 <= workers <= 12:
        raise JobError("INVALID_WORKER_COUNT", "workers must be in [1, 12]", workers)
    _assert_clean_branch(ctx)
    active = _active_state(ctx)
    if active is not None:
        raise JobError("JOB_ALREADY_RUNNING", "An FP-NAA job is already running", asdict(active))
    fp_naa_runtime_check()
    _validate_stage_prerequisites(ctx, stage)

    now = _utc_now()
    run_id = f"{_STAGE_PREFIX[stage]}_{now.replace('-', '').replace(':', '')}"
    run_id = run_id.replace("T", "T").replace("Z", "Z")
    job_dir = ctx.jobs_root / run_id
    report_dir = ctx.reports_root / run_id
    log_path = job_dir / "job.log"
    state_path = job_dir / "state.json"
    job_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    source_sha = _git_output(ctx, "rev-parse", "HEAD")
    state = JobState(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        stage=stage.value,
        status=JobStatus.STARTING.value,
        current_step="launch",
        task_status=99,
        workers=workers,
        pid=None,
        source_git_sha=source_sha,
        created_utc=now,
        updated_utc=now,
        log=str(log_path.relative_to(ctx.repo_root)),
        report=str(
            (ctx.repo_root / "reports" / "server" / f"{run_id}.md").relative_to(ctx.repo_root)
        ),
    )
    _write_state(state_path, state)
    _write_latest(ctx, state)
    command = [
        sys.executable,
        "-m",
        "care_asd.cli",
        "fp-naa",
        "job",
        "run-internal",
        "--stage",
        stage.value,
        "--run-id",
        run_id,
        "--workers",
        str(workers),
    ]
    clean_env = _subprocess_environment()
    try:
        with log_path.open("ab", buffering=0) as log_handle:
            process = subprocess.Popen(
                command,
                cwd=ctx.repo_root,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
                env=clean_env,
            )
    except OSError as exc:
        failed = replace(
            state,
            status=JobStatus.FAILED.value,
            current_step="launch",
            task_status=1,
            updated_utc=_utc_now(),
        )
        _write_state(state_path, failed)
        _write_latest(ctx, failed)
        raise JobError(
            "JOB_LAUNCH_FAILED", "Could not launch detached Python job", str(exc)
        ) from exc
    running = replace(
        state,
        status=JobStatus.RUNNING.value,
        current_step="startup",
        pid=process.pid,
        updated_utc=_utc_now(),
    )
    _write_state(state_path, running)
    _write_latest(ctx, running)
    return {
        "schema_version": SCHEMA_VERSION,
        "job": asdict(running),
        "message": "FP-NAA job started in the background",
    }


def execute_fp_naa_job(
    stage: JobStage,
    run_id: str,
    *,
    workers: int,
    context: FPNAAJobContext | None = None,
) -> int:
    """Internal foreground worker invoked only by ``start_fp_naa_job``."""
    ctx = context or FPNAAJobContext.from_environment()
    if not run_id.startswith(f"{_STAGE_PREFIX[stage]}_"):
        raise JobError("INVALID_RUN_ID", "Run ID does not match the requested stage", run_id)
    job_dir = ctx.jobs_root / run_id
    state_path = job_dir / "state.json"
    if not state_path.is_file():
        raise JobError("STATE_NOT_FOUND", "Job state does not exist", str(state_path))
    state = _wait_for_parent_launch_state(state_path)
    state = replace(
        state,
        status=JobStatus.RUNNING.value,
        current_step="runtime-check",
        pid=os.getpid(),
        updated_utc=_utc_now(),
    )
    _write_state(state_path, state)
    print(json.dumps({"event": "job_started", "run_id": run_id, "stage": stage.value}), flush=True)

    task_status = 0
    result: StageResult | None = None
    error_payload: dict[str, Any] | None = None
    try:
        fp_naa_runtime_check()
        result = _execute_stage(ctx, state_path, stage, run_id, workers)
    except JobError as exc:
        task_status = 2
        error_payload = exc.as_dict()
        print(json.dumps(error_payload), flush=True)
    except subprocess.CalledProcessError as exc:
        task_status = int(exc.returncode or 1)
        error_payload = JobError(
            "EXTERNAL_COMMAND_FAILED",
            "An experiment subprocess failed",
            {"command": exc.cmd, "returncode": exc.returncode},
        ).as_dict()
        print(json.dumps(error_payload), flush=True)
    except Exception as exc:  # pragma: no cover - final server safety boundary
        task_status = 1
        error_payload = JobError("UNEXPECTED_JOB_FAILURE", str(exc)).as_dict()
        print(json.dumps(error_payload), flush=True)

    gate_passed = result.gate_passed if result is not None else False
    final_status = JobStatus.DONE if task_status == 0 else JobStatus.FAILED
    final = replace(
        _read_state(state_path),
        status=final_status.value,
        current_step="report",
        task_status=task_status,
        gate_passed=gate_passed,
        updated_utc=_utc_now(),
    )
    _write_state(state_path, final)
    report_path = _write_server_report(ctx, final, result, error_payload)
    push_status = _publish_report(ctx, run_id, ctx.reports_root / run_id, report_path)
    final = replace(final, current_step="complete", push_status=push_status, updated_utc=_utc_now())
    _write_state(state_path, final)
    _write_latest(ctx, final)
    print(
        json.dumps(
            {
                "event": "job_complete",
                "run_id": run_id,
                "task_status": task_status,
                "gate_passed": gate_passed,
                "push_status": push_status,
            }
        ),
        flush=True,
    )
    return task_status


def fp_naa_job_status(
    *,
    run_id: str | None = None,
    context: FPNAAJobContext | None = None,
    log_lines: int = 30,
) -> dict[str, Any]:
    """Return one stable status document without sleeping or polling."""
    ctx = context or FPNAAJobContext.from_environment()
    state = _state_by_id(ctx, run_id) if run_id else _latest_state(ctx)
    alive = bool(state.pid and _process_alive(state.pid))
    log_path = ctx.repo_root / state.log
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "job": asdict(state),
        "process": {"is_alive": alive},
        "progress": _progress_payload(ctx, state),
        "gpu_processes": _gpu_processes(ctx),
        "log_tail": _tail(log_path, log_lines),
    }
    gate_path = _stage_gate_path(ctx.reports_root / state.run_id, JobStage(state.stage))
    if gate_path.is_file():
        payload["gate"] = _read_json(gate_path)
    if state.status in {JobStatus.RUNNING.value, JobStatus.STARTING.value} and not alive:
        payload["warning"] = {
            "code": "PROCESS_NOT_ALIVE",
            "message": "State says the job is active, but its PID is not running",
        }
    return payload


def list_fp_naa_jobs(*, context: FPNAAJobContext | None = None, limit: int = 10) -> dict[str, Any]:
    ctx = context or FPNAAJobContext.from_environment()
    if not 1 <= limit <= 100:
        raise JobError("INVALID_LIMIT", "limit must be in [1, 100]", limit)
    states = sorted(
        (_read_state(path) for path in ctx.jobs_root.glob("*/state.json")),
        key=lambda item: item.created_utc,
        reverse=True,
    )[:limit]
    return {"schema_version": SCHEMA_VERSION, "jobs": [asdict(state) for state in states]}


def continue_fp_naa_job(
    *, workers: int = 12, context: FPNAAJobContext | None = None
) -> dict[str, Any]:
    """Report an active/failed job or start the next gate-eligible stage."""
    ctx = context or FPNAAJobContext.from_environment()
    active = _active_state(ctx)
    if active is not None:
        return fp_naa_job_status(run_id=active.run_id, context=ctx)
    try:
        latest = _latest_state(ctx)
    except JobError as exc:
        if exc.code != "NO_JOBS":
            raise
    else:
        if latest.status == JobStatus.FAILED.value:
            raise JobError(
                "LATEST_JOB_FAILED",
                "Automatic continuation is blocked; inspect status and retry an explicit stage",
                asdict(latest),
            )
    next_stage = _next_stage(ctx)
    return start_fp_naa_job(next_stage, workers=workers, context=ctx)


def _execute_stage(
    ctx: FPNAAJobContext,
    state_path: Path,
    stage: JobStage,
    run_id: str,
    workers: int,
) -> StageResult:
    handlers = {
        JobStage.C0: _run_c0,
        JobStage.SCREENING: _run_screening,
        JobStage.LOMO: _run_lomo,
        JobStage.CONFIRMATORY: _run_confirmatory,
        JobStage.CONFIRMATORY_LOMO: _run_confirmatory_lomo,
        JobStage.REFERENCE_SAFETY: _run_reference_safety,
    }
    return handlers[stage](ctx, state_path, run_id, workers)


def _run_c0(ctx: FPNAAJobContext, state_path: Path, run_id: str, workers: int) -> StageResult:
    paths = _common_paths(ctx)
    report = ctx.reports_root / run_id / "c0_baseline"
    _set_step(state_path, "assets")
    beats_source, checkpoint = _ensure_beats_assets(ctx)
    _set_step(state_path, "cache")
    _call_cli(
        ctx,
        "fp-naa",
        "cache-beats",
        "--manifest",
        str(paths["manifest"]),
        "--audio-root",
        str(paths["audio_root"]),
        "--output-dir",
        str(paths["base_cache"]),
        "--config",
        str(paths["config"]),
        "--beats-source",
        str(beats_source),
        "--checkpoint",
        str(checkpoint),
        "--workers",
        str(workers),
        "--device",
        "cuda",
    )
    _set_step(state_path, "baseline")
    _call_cli(
        ctx,
        "fp-naa",
        "baseline-dev",
        "--cache-dir",
        str(paths["base_cache"]),
        "--output-dir",
        str(report),
        "--config",
        str(paths["config"]),
        "--experiment-id",
        run_id,
        "--device",
        "cuda",
    )
    return _stage_result(report / "gate.json", "passed", report / "summary.csv")


def _run_screening(
    ctx: FPNAAJobContext, state_path: Path, run_id: str, workers: int
) -> StageResult:
    paths = _common_paths(ctx)
    c0_gate = _require_gate(ctx, JobStage.C0)
    c0_scores = c0_gate.parent / "freq_rdp8_beam" / "scores.csv"
    report = ctx.reports_root / run_id / "screening"
    checkpoints = ctx.data_root / "fp_naa" / "checkpoints" / run_id
    _set_step(state_path, "assets")
    beats_source, checkpoint = _ensure_beats_assets(ctx)
    _run_pytest(ctx, "tests/unit/test_fp_naa_adapter.py", "tests/unit/test_fp_naa_candidate.py")
    _set_step(state_path, "augmentation-cache")
    _call_cli(
        ctx,
        "fp-naa",
        "cache-augmentation",
        "--base-cache-dir",
        str(paths["base_cache"]),
        "--audio-root",
        str(paths["audio_root"]),
        "--output-dir",
        str(paths["augmentation_cache"]),
        "--config",
        str(paths["config"]),
        "--beats-source",
        str(beats_source),
        "--checkpoint",
        str(checkpoint),
        "--workers",
        str(workers),
        "--device",
        "cuda",
    )
    _set_step(state_path, "screening")
    _call_cli(
        ctx,
        "fp-naa",
        "screen-dev",
        "--base-cache-dir",
        str(paths["base_cache"]),
        "--augmentation-cache-dir",
        str(paths["augmentation_cache"]),
        "--c0-scores",
        str(c0_scores),
        "--output-dir",
        str(report),
        "--checkpoint-dir",
        str(checkpoints),
        "--config",
        str(paths["config"]),
        "--experiment-id",
        run_id,
        "--device",
        "cuda",
        "--preload-workers",
        str(workers),
    )
    return _stage_result(report / "gate.json", "core_screening", report / "screening_summary.csv")


def _run_lomo(ctx: FPNAAJobContext, state_path: Path, run_id: str, workers: int) -> StageResult:
    paths = _common_paths(ctx)
    screen_gate = _require_gate(ctx, JobStage.SCREENING)
    report = ctx.reports_root / run_id / "lomo"
    checkpoints = ctx.data_root / "fp_naa" / "checkpoints" / run_id
    _run_pytest(ctx, "tests/unit/test_fp_naa_adapter.py", "tests/unit/test_fp_naa_candidate.py")
    _set_step(state_path, "lomo")
    _call_cli(
        ctx,
        "fp-naa",
        "lomo-dev",
        "--base-cache-dir",
        str(paths["base_cache"]),
        "--augmentation-cache-dir",
        str(paths["augmentation_cache"]),
        "--screening-dir",
        str(screen_gate.parent),
        "--output-dir",
        str(report),
        "--checkpoint-dir",
        str(checkpoints),
        "--config",
        str(paths["config"]),
        "--experiment-id",
        run_id,
        "--device",
        "cuda",
        "--preload-workers",
        str(workers),
    )
    return _stage_result(report / "gate.json", "passed", report / "lomo_fold_means.csv")


def _run_confirmatory(
    ctx: FPNAAJobContext, state_path: Path, run_id: str, workers: int
) -> StageResult:
    paths = _common_paths(ctx)
    c0_gate = _require_gate(ctx, JobStage.C0)
    screen_gate = _require_gate(ctx, JobStage.SCREENING)
    lomo_gate = _require_gate(ctx, JobStage.LOMO)
    c0_scores = c0_gate.parent / "freq_rdp8_beam" / "scores.csv"
    _verify_confirmatory_inputs(screen_gate, lomo_gate, c0_scores)
    report = ctx.reports_root / run_id / "confirmatory"
    checkpoints = ctx.data_root / "fp_naa" / "checkpoints" / run_id
    _run_pytest(
        ctx,
        "tests/unit/test_fp_naa_adapter.py",
        "tests/unit/test_fp_naa_candidate.py",
        "tests/unit/test_fp_naa_confirmatory.py",
        "tests/unit/test_fp_naa_statistics.py",
    )
    _set_step(state_path, "confirmatory")
    _call_cli(
        ctx,
        "fp-naa",
        "confirm-dev",
        "--base-cache-dir",
        str(paths["base_cache"]),
        "--augmentation-cache-dir",
        str(paths["augmentation_cache"]),
        "--c0-scores",
        str(c0_scores),
        "--screening-dir",
        str(screen_gate.parent),
        "--lomo-dir",
        str(lomo_gate.parent),
        "--output-dir",
        str(report),
        "--checkpoint-dir",
        str(checkpoints),
        "--config",
        str(paths["config"]),
        "--experiment-id",
        run_id,
        "--device",
        "cuda",
        "--preload-workers",
        str(workers),
    )
    return _stage_result(
        report / "gate.json",
        "core_confirmatory",
        report / "confirmatory_summary.csv",
        report / "exact_paired_bootstrap_c2_vs_c1.json",
    )


def _run_confirmatory_lomo(
    ctx: FPNAAJobContext, state_path: Path, run_id: str, workers: int
) -> StageResult:
    paths = _common_paths(ctx)
    lomo_gate = _require_gate(ctx, JobStage.LOMO)
    confirm_gate = _require_gate(ctx, JobStage.CONFIRMATORY)
    confirm_contract = _read_json(confirm_gate.parent / "contract.json")
    if confirm_contract.get("lomo_gate_sha256") != _sha256(lomo_gate):
        raise JobError("ARTIFACT_MISMATCH", "Confirmatory/LOMO artifact mismatch")
    report = ctx.reports_root / run_id / "confirmatory_lomo"
    checkpoints = ctx.data_root / "fp_naa" / "checkpoints" / run_id
    _run_pytest(
        ctx,
        "tests/unit/test_fp_naa_adapter.py",
        "tests/unit/test_fp_naa_candidate.py",
        "tests/unit/test_fp_naa_confirmatory.py",
    )
    _set_step(state_path, "confirmatory-lomo")
    _call_cli(
        ctx,
        "fp-naa",
        "confirm-lomo-dev",
        "--base-cache-dir",
        str(paths["base_cache"]),
        "--augmentation-cache-dir",
        str(paths["augmentation_cache"]),
        "--screening-lomo-dir",
        str(lomo_gate.parent),
        "--confirmatory-dir",
        str(confirm_gate.parent),
        "--output-dir",
        str(report),
        "--checkpoint-dir",
        str(checkpoints),
        "--config",
        str(paths["config"]),
        "--experiment-id",
        run_id,
        "--device",
        "cuda",
        "--preload-workers",
        str(workers),
    )
    return _stage_result(
        report / "gate.json", "passed", report / "confirmatory_lomo_fold_means.csv"
    )


def _run_reference_safety(
    ctx: FPNAAJobContext, state_path: Path, run_id: str, workers: int
) -> StageResult:
    paths = _common_paths(ctx)
    screen_gate = _require_gate(ctx, JobStage.SCREENING)
    lomo_gate = _require_gate(ctx, JobStage.LOMO)
    confirm_gate = _require_gate(ctx, JobStage.CONFIRMATORY)
    confirm_lomo_gate = _require_gate(ctx, JobStage.CONFIRMATORY_LOMO)
    _verify_safety_inputs(screen_gate, lomo_gate, confirm_gate, confirm_lomo_gate)
    beats_source, checkpoint = _ensure_beats_assets(ctx)
    report = ctx.reports_root / run_id / "reference_safety"
    safety_cache = (
        ctx.data_root / "fp_naa" / "reference_safety_cache" / "dev" / "waveform_fp32infer_v2"
    )
    screen_run = screen_gate.parents[1].name
    confirm_run = confirm_gate.parents[1].name
    screen_checkpoints = ctx.data_root / "fp_naa" / "checkpoints" / screen_run
    confirm_checkpoints = ctx.data_root / "fp_naa" / "checkpoints" / confirm_run
    _run_pytest(
        ctx,
        "tests/unit/test_fp_naa_reference_safety_config.py",
        "tests/unit/test_fp_naa_reference_safety_cache.py",
        "tests/unit/test_fp_naa_reference_safety.py",
        "tests/unit/test_fp_naa_adapter.py",
    )
    _set_step(state_path, "reference-safety-cache")
    _call_cli(
        ctx,
        "fp-naa",
        "cache-reference-safety",
        "--base-cache-dir",
        str(paths["base_cache"]),
        "--augmentation-cache-dir",
        str(paths["augmentation_cache"]),
        "--audio-root",
        str(paths["audio_root"]),
        "--output-dir",
        str(safety_cache),
        "--config",
        str(paths["config"]),
        "--safety-config",
        str(paths["safety_config"]),
        "--beats-source",
        str(beats_source),
        "--checkpoint",
        str(checkpoint),
        "--workers",
        str(workers),
        "--device",
        "cuda",
    )
    _set_step(state_path, "reference-safety-evaluation")
    _call_cli(
        ctx,
        "fp-naa",
        "reference-safety-dev",
        "--base-cache-dir",
        str(paths["base_cache"]),
        "--augmentation-cache-dir",
        str(paths["augmentation_cache"]),
        "--safety-cache-dir",
        str(safety_cache),
        "--screening-checkpoint-dir",
        str(screen_checkpoints),
        "--confirmatory-checkpoint-dir",
        str(confirm_checkpoints),
        "--confirmatory-dir",
        str(confirm_gate.parent),
        "--confirmatory-lomo-dir",
        str(confirm_lomo_gate.parent),
        "--output-dir",
        str(report),
        "--config",
        str(paths["config"]),
        "--safety-config",
        str(paths["safety_config"]),
        "--experiment-id",
        run_id,
        "--device",
        "cuda",
        "--preload-workers",
        str(workers),
    )
    return _stage_result(report / "gate.json", "passed", report / "reference_safety_summary.csv")


def _common_paths(ctx: FPNAAJobContext) -> dict[str, Path]:
    return {
        "manifest": ctx.repo_root / "data" / "manifests" / "dcase2026_dev.parquet",
        "audio_root": ctx.data_root / "raw" / "dcase2026" / "dev" / "extracted",
        "config": ctx.repo_root / "configs" / "experiment" / "fp_naa_v3.yaml",
        "safety_config": ctx.repo_root
        / "configs"
        / "experiment"
        / "fp_naa_reference_safety_v1.yaml",
        "base_cache": ctx.data_root
        / "fp_naa"
        / "beats_cache"
        / "dev"
        / "beats_iter3_stereo_10s_fp32infer_v2",
        "augmentation_cache": ctx.data_root
        / "fp_naa"
        / "augmentation_cache"
        / "dev"
        / "counterfactual_fp32infer_v3",
    }


def _validate_stage_prerequisites(ctx: FPNAAJobContext, stage: JobStage) -> None:
    paths = _common_paths(ctx)
    if not paths["manifest"].is_file():
        raise JobError("MANIFEST_NOT_FOUND", "DCASE development manifest is missing")
    if not paths["audio_root"].is_dir():
        raise JobError("AUDIO_ROOT_NOT_FOUND", "DCASE development audio root is missing")
    prerequisites = {
        JobStage.C0: (),
        JobStage.SCREENING: (JobStage.C0,),
        JobStage.LOMO: (JobStage.SCREENING,),
        JobStage.CONFIRMATORY: (JobStage.C0, JobStage.SCREENING, JobStage.LOMO),
        JobStage.CONFIRMATORY_LOMO: (JobStage.LOMO, JobStage.CONFIRMATORY),
        JobStage.REFERENCE_SAFETY: (
            JobStage.SCREENING,
            JobStage.LOMO,
            JobStage.CONFIRMATORY,
            JobStage.CONFIRMATORY_LOMO,
        ),
    }
    for prerequisite in prerequisites[stage]:
        _require_gate(ctx, prerequisite)


def _next_stage(ctx: FPNAAJobContext) -> JobStage:
    for stage in JobStage:
        gate = _latest_gate(ctx, stage)
        if gate is None:
            return stage
        _, key = _GATE_CONTRACT[stage]
        payload = _read_json(gate)
        if payload.get(key) is not True:
            raise JobError(
                "GATE_BLOCKED",
                f"Latest {stage.value} gate did not pass",
                {"gate": str(gate.relative_to(ctx.repo_root)), "result": payload},
            )
    raise JobError("PIPELINE_COMPLETE", "All frozen FP-NAA stages have completed")


def _require_gate(ctx: FPNAAJobContext, stage: JobStage) -> Path:
    gate = _latest_gate(ctx, stage)
    if gate is None:
        raise JobError("PREREQUISITE_GATE_MISSING", f"No {stage.value} gate was found")
    _, key = _GATE_CONTRACT[stage]
    payload = _read_json(gate)
    if payload.get(key) is not True:
        raise JobError(
            "PREREQUISITE_GATE_FAILED",
            f"Latest {stage.value} gate did not pass",
            {"gate": str(gate.relative_to(ctx.repo_root)), "result": payload},
        )
    return gate


def _latest_gate(ctx: FPNAAJobContext, stage: JobStage) -> Path | None:
    subdir, _ = _GATE_CONTRACT[stage]
    candidates = list(ctx.reports_root.glob(f"*/{subdir}/gate.json"))
    return (
        max(candidates, key=lambda path: (path.parents[1].name, path.stat().st_mtime_ns))
        if candidates
        else None
    )


def _stage_gate_path(run_root: Path, stage: JobStage) -> Path:
    subdir, _ = _GATE_CONTRACT[stage]
    return run_root / subdir / "gate.json"


def _stage_result(gate: Path, key: str, summary: Path, *extra: Path) -> StageResult:
    payload = _read_json(gate)
    return StageResult(
        gate_path=gate,
        gate_passed=payload.get(key) is True,
        summary_path=summary if summary.is_file() else None,
        extra_paths=tuple(path for path in extra if path.is_file()),
    )


def _ensure_beats_assets(ctx: FPNAAJobContext) -> tuple[Path, Path]:
    external = ctx.data_root / "external" / "fp_naa"
    source_root = external / f"unilm_{BEATS_COMMIT[:12]}"
    source = source_root / "beats"
    checkpoint = external / "BEATs_iter3.pt"
    external.mkdir(parents=True, exist_ok=True)
    if not (source_root / ".git").is_dir():
        temporary = Path(tempfile.mkdtemp(prefix="unilm_", dir=external))
        try:
            _run_external(
                [
                    "git",
                    "clone",
                    "--filter=blob:none",
                    "--no-checkout",
                    BEATS_REPOSITORY,
                    str(temporary),
                ],
                cwd=ctx.repo_root,
            )
            _run_external(["git", "sparse-checkout", "set", "beats"], cwd=temporary)
            _run_external(["git", "checkout", "--detach", BEATS_COMMIT], cwd=temporary)
            os.replace(temporary, source_root)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    actual_commit = _external_output(["git", "rev-parse", "HEAD"], cwd=source_root)
    if actual_commit != BEATS_COMMIT:
        raise JobError(
            "BEATS_SOURCE_MISMATCH", "Pinned BEATs source commit mismatch", actual_commit
        )
    if not checkpoint.is_file() or _sha256(checkpoint) != BEATS_ITER3_SHA256:
        _download_checkpoint(checkpoint)
    if _sha256(checkpoint) != BEATS_ITER3_SHA256:
        raise JobError("BEATS_CHECKPOINT_MISMATCH", "BEATs checkpoint SHA-256 mismatch")
    return source, checkpoint


def _download_checkpoint(destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    start = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(CHECKPOINT_URL, headers={"User-Agent": "care-asd/0.1"})
    if start:
        request.add_header("Range", f"bytes={start}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        append = start > 0 and getattr(response, "status", 200) == 206
        mode = "ab" if append else "wb"
        downloaded = start if append else 0
        with partial.open(mode) as handle:
            while block := response.read(1024 * 1024):
                handle.write(block)
                downloaded += len(block)
                if downloaded % (64 * 1024 * 1024) < len(block):
                    print(
                        json.dumps({"event": "checkpoint_download", "bytes": downloaded}),
                        flush=True,
                    )
    if _sha256(partial) != BEATS_ITER3_SHA256:
        raise JobError("BEATS_CHECKPOINT_MISMATCH", "Downloaded checkpoint SHA-256 mismatch")
    os.replace(partial, destination)


def _verify_confirmatory_inputs(screen_gate: Path, lomo_gate: Path, c0_scores: Path) -> None:
    screen = _read_json(screen_gate.parent / "contract.json")
    lomo = _read_json(lomo_gate.parent / "contract.json")
    if screen.get("c0_scores_sha256") != _sha256(c0_scores):
        raise JobError("ARTIFACT_MISMATCH", "Screening/C0 artifact mismatch")
    if lomo.get("screening_gate_sha256") != _sha256(screen_gate):
        raise JobError("ARTIFACT_MISMATCH", "LOMO/screening artifact mismatch")


def _verify_safety_inputs(
    screen_gate: Path, lomo_gate: Path, confirm_gate: Path, confirm_lomo_gate: Path
) -> None:
    confirm = _read_json(confirm_gate.parent / "contract.json")
    confirm_lomo = _read_json(confirm_lomo_gate.parent / "contract.json")
    checks = (
        (confirm.get("screening_gate_sha256"), _sha256(screen_gate), "confirmatory/screening"),
        (confirm.get("lomo_gate_sha256"), _sha256(lomo_gate), "confirmatory/LOMO"),
        (
            confirm_lomo.get("confirmatory_gate_sha256"),
            _sha256(confirm_gate),
            "confirmatory-LOMO/core",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise JobError("ARTIFACT_MISMATCH", f"{label} artifact mismatch")


def _call_cli(ctx: FPNAAJobContext, *arguments: str) -> None:
    _run_external([sys.executable, "-m", "care_asd.cli", *arguments], cwd=ctx.repo_root)


def _run_pytest(ctx: FPNAAJobContext, *tests: str) -> None:
    _run_external([sys.executable, "-m", "pytest", *tests, "-q"], cwd=ctx.repo_root)


def _run_external(command: list[str], *, cwd: Path) -> None:
    print(json.dumps({"event": "command", "argv": command}), flush=True)
    clean_env = _subprocess_environment()
    subprocess.run(command, cwd=cwd, env=clean_env, check=True)


def _external_output(command: list[str], *, cwd: Path) -> str:
    clean_env = _subprocess_environment()
    return subprocess.check_output(command, cwd=cwd, env=clean_env, text=True).strip()


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment.pop("LD_PRELOAD", None)
    environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    environment.setdefault("PYTHONHASHSEED", "0")
    return environment


def _assert_clean_branch(ctx: FPNAAJobContext) -> None:
    branch = _git_output(ctx, "branch", "--show-current")
    if branch != ctx.branch:
        raise JobError(
            "WRONG_GIT_BRANCH", f"Expected branch {ctx.branch}", {"actual_branch": branch}
        )
    status = _git_output(ctx, "status", "--porcelain")
    if status:
        raise JobError("DIRTY_WORKTREE", "Server worktree is not clean", status.splitlines())


def _git_output(ctx: FPNAAJobContext, *arguments: str) -> str:
    return _external_output(["git", *arguments], cwd=ctx.repo_root)


def _publish_report(ctx: FPNAAJobContext, run_id: str, report_dir: Path, report: Path) -> int:
    try:
        _run_external(["git", "add", str(report_dir), str(report)], cwd=ctx.repo_root)
        _run_external(["git", "commit", "-m", f"report: add {run_id}"], cwd=ctx.repo_root)
        _run_external(["git", "pull", "--rebase", "origin", ctx.branch], cwd=ctx.repo_root)
        _run_external(["git", "push", "origin", f"HEAD:{ctx.branch}"], cwd=ctx.repo_root)
    except subprocess.CalledProcessError as exc:
        print(
            json.dumps({"error": {"code": "REPORT_PUSH_FAILED", "returncode": exc.returncode}}),
            flush=True,
        )
        return int(exc.returncode or 1)
    return 0


def _write_server_report(
    ctx: FPNAAJobContext,
    state: JobState,
    result: StageResult | None,
    error: dict[str, Any] | None,
) -> Path:
    report = ctx.repo_root / state.report
    report.parent.mkdir(parents=True, exist_ok=True)
    log_path = ctx.repo_root / state.log
    sections = [
        f"# FP-NAA {state.stage} report",
        "",
        f"run_id={state.run_id}",
        f"source_git_sha={state.source_git_sha}",
        f"conda_environment={EXPECTED_CONDA_ENV}",
        f"workers={state.workers}",
        f"task_status={state.task_status}",
        f"gate_passed={str(state.gate_passed).lower()}",
        "",
    ]
    if error is not None:
        sections.extend(["## Error", "", "```json", json.dumps(error, indent=2), "```", ""])
    if result is not None and result.gate_path is not None and result.gate_path.is_file():
        sections.extend(
            ["## Gate", "", "```json", result.gate_path.read_text(encoding="utf-8"), "```", ""]
        )
    if result is not None and result.summary_path is not None:
        sections.extend(
            [
                "## Summary",
                "",
                "```csv",
                result.summary_path.read_text(encoding="utf-8"),
                "```",
                "",
            ]
        )
    for extra in result.extra_paths if result is not None else ():
        sections.extend(
            [
                f"## {extra.name}",
                "",
                "```text",
                extra.read_text(encoding="utf-8"),
                "```",
                "",
            ]
        )
    sections.extend(["## Log tail", "", "```text", *_tail(log_path, 100), "```", ""])
    report.write_text("\n".join(sections), encoding="utf-8")
    return report


def _progress_payload(ctx: FPNAAJobContext, state: JobState) -> dict[str, Any] | None:
    paths = _common_paths(ctx)
    candidates = [
        paths["base_cache"] / "progress.env",
        paths["augmentation_cache"] / "progress.env",
        ctx.reports_root / state.run_id / _GATE_CONTRACT[JobStage(state.stage)][0] / "progress.env",
    ]
    existing = [path for path in candidates if path.is_file()]
    if not existing:
        return None
    latest = max(existing, key=lambda path: path.stat().st_mtime_ns)
    values: dict[str, Any] = {"path": str(latest)}
    for line in latest.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def _gpu_processes(ctx: FPNAAJobContext) -> list[str]:
    try:
        output = _external_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            cwd=ctx.repo_root,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return output.splitlines() if output else []


def _active_state(ctx: FPNAAJobContext) -> JobState | None:
    states = sorted(
        (_read_state(path) for path in ctx.jobs_root.glob("*/state.json")),
        key=lambda item: item.updated_utc,
        reverse=True,
    )
    for state in states:
        recently_started = state.pid is None and _age_seconds(state.updated_utc) < 60.0
        if state.status in {JobStatus.STARTING.value, JobStatus.RUNNING.value} and (
            recently_started or (state.pid is not None and _process_alive(state.pid))
        ):
            return state
    return None


@contextmanager
def _launch_lock(ctx: FPNAAJobContext) -> Iterator[None]:
    """Serialize start requests without relying on a shell or process-name search."""
    path = ctx.control_root / "launch.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            age = time.time() - path.stat().st_mtime
            try:
                owner = _read_json(path)
            except JobError:
                owner = {}
            owner_pid = owner.get("pid")
            owner_alive = isinstance(owner_pid, int) and _process_alive(owner_pid)
            if owner_alive or age < 60.0:
                raise JobError(
                    "JOB_START_LOCKED",
                    "Another FP-NAA start request is already in progress",
                    {"lock": str(path), "owner_pid": owner_pid},
                ) from exc
            path.unlink(missing_ok=True)
            continue
        else:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"pid": os.getpid(), "created_utc": _utc_now()}, handle)
                handle.write("\n")
            acquired = True
            break
    if not acquired:
        raise JobError("JOB_START_LOCKED", "Could not acquire the FP-NAA launch lock")
    try:
        yield
    finally:
        try:
            owner = _read_json(path)
        except JobError:
            owner = {}
        if owner.get("pid") == os.getpid():
            path.unlink(missing_ok=True)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _state_by_id(ctx: FPNAAJobContext, run_id: str) -> JobState:
    path = ctx.jobs_root / run_id / "state.json"
    if not path.is_file():
        raise JobError("JOB_NOT_FOUND", "FP-NAA job was not found", run_id)
    return _read_state(path)


def _latest_state(ctx: FPNAAJobContext) -> JobState:
    pointer = ctx.control_root / "latest.json"
    if not pointer.is_file():
        raise JobError("NO_JOBS", "No Python-managed FP-NAA job has been started")
    payload = _read_json(pointer)
    run_id = payload.get("run_id")
    if not isinstance(run_id, str):
        raise JobError("INVALID_LATEST_POINTER", "Latest FP-NAA job pointer is invalid")
    return _state_by_id(ctx, run_id)


def _write_latest(ctx: FPNAAJobContext, state: JobState) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": state.run_id,
        "stage": state.stage,
        "updated_utc": state.updated_utc,
    }
    _atomic_json(ctx.control_root / "latest.json", payload)
    _atomic_json(ctx.control_root / f"latest_{state.stage}.json", payload)


def _set_step(state_path: Path, step: str) -> None:
    state = _read_state(state_path)
    _write_state(state_path, replace(state, current_step=step, updated_utc=_utc_now()))
    print(json.dumps({"event": "stage", "step": step}), flush=True)


def _wait_for_parent_launch_state(state_path: Path) -> JobState:
    """Avoid a parent/child atomic-state race during process detachment."""
    for _ in range(100):
        state = _read_state(state_path)
        if state.pid == os.getpid():
            return state
        time.sleep(0.05)
    raise JobError("LAUNCH_STATE_TIMEOUT", "Parent did not publish the detached child PID")


def _write_state(path: Path, state: JobState) -> None:
    _atomic_json(path, asdict(state))


def _read_state(path: Path) -> JobState:
    return JobState.from_dict(_read_json(path))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JobError("INVALID_JSON_ARTIFACT", "Could not read JSON artifact", str(path)) from exc
    if not isinstance(payload, dict):
        raise JobError("INVALID_JSON_ARTIFACT", "Expected a JSON object", str(path))
    return payload


def _tail(path: Path, lines: int) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_package_names() -> set[str]:
    import importlib.metadata as metadata

    return {str(distribution.metadata["Name"]).lower() for distribution in metadata.distributions()}


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _age_seconds(timestamp: str) -> float:
    try:
        updated = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return float("inf")
    return max(0.0, (datetime.now(UTC) - updated).total_seconds())
