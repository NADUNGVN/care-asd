from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import cast

import pytest

from care_asd.server.fp_naa_jobs import (
    CUBLAS_WORKSPACE_CONFIG,
    SCHEMA_VERSION,
    FPNAAJobContext,
    JobError,
    JobStage,
    JobState,
    JobStatus,
    _atomic_json,
    _common_paths,
    _launch_lock,
    _next_stage,
    _write_latest,
    _write_state,
    continue_fp_naa_job,
    fp_naa_job_status,
    fp_naa_runtime_check,
    start_fp_naa_job,
)


def _context(tmp_path: Path) -> FPNAAJobContext:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
    return FPNAAJobContext(repo_root=repo, data_root=data)


def _state(context: FPNAAJobContext, *, status: JobStatus = JobStatus.DONE) -> JobState:
    run_id = "server02_fp_naa_c0_20260817T000000Z"
    return JobState(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        stage=JobStage.C0.value,
        status=status.value,
        current_step="complete",
        task_status=0 if status is JobStatus.DONE else 1,
        workers=12,
        pid=os.getpid() if status is JobStatus.RUNNING else None,
        source_git_sha="a" * 40,
        created_utc="2026-08-17T00:00:00Z",
        updated_utc="2026-08-17T00:01:00Z",
        log=f"outputs/fp_naa/jobs/{run_id}/job.log",
        report=f"reports/server/{run_id}.md",
        gate_passed=False,
        push_status=0,
    )


def _write_gate(context: FPNAAJobContext, run: str, subdir: str, **values: object) -> Path:
    path = context.reports_root / run / subdir / "gate.json"
    _atomic_json(path, dict(values))
    return path


def test_next_stage_follows_frozen_gate_order(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert _next_stage(context) is JobStage.C0
    _write_gate(context, "c0", "c0_baseline", passed=True)
    assert _next_stage(context) is JobStage.SCREENING
    _write_gate(context, "screen", "screening", core_screening=True)
    assert _next_stage(context) is JobStage.LOMO
    _write_gate(context, "lomo", "lomo", passed=True)
    assert _next_stage(context) is JobStage.CONFIRMATORY
    _write_gate(context, "confirm", "confirmatory", core_confirmatory=True)
    assert _next_stage(context) is JobStage.CONFIRMATORY_LOMO
    _write_gate(context, "confirm-lomo", "confirmatory_lomo", passed=True)
    assert _next_stage(context) is JobStage.REFERENCE_SAFETY


def test_next_stage_stops_on_failed_gate(tmp_path: Path) -> None:
    context = _context(tmp_path)
    gate = _write_gate(context, "c0", "c0_baseline", passed=False, score=0.5)
    with pytest.raises(JobError, match="did not pass") as captured:
        _next_stage(context)
    assert captured.value.code == "GATE_BLOCKED"
    assert captured.value.details["gate"] == str(gate.relative_to(context.repo_root))


def test_status_is_structured_and_does_not_start_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    state = _state(context, status=JobStatus.RUNNING)
    state_path = context.jobs_root / state.run_id / "state.json"
    _write_state(state_path, state)
    _write_latest(context, state)
    log_path = context.repo_root / state.log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")
    monkeypatch.setattr(
        "care_asd.server.fp_naa_jobs._gpu_processes", lambda _context: ["123, python, 512 MiB"]
    )
    result = fp_naa_job_status(context=context, log_lines=2)
    assert result["job"] == asdict(state)
    assert result["process"] == {"is_alive": True}
    assert result["log_tail"] == ["two", "three"]
    assert result["gpu_processes"] == ["123, python, 512 MiB"]


def test_continue_refuses_automatic_retry_after_failure(tmp_path: Path) -> None:
    context = _context(tmp_path)
    state = _state(context, status=JobStatus.FAILED)
    _write_state(context.jobs_root / state.run_id / "state.json", state)
    _write_latest(context, state)
    with pytest.raises(JobError, match="explicit stage") as captured:
        continue_fp_naa_job(context=context)
    assert captured.value.code == "LATEST_JOB_FAILED"


def test_runtime_check_rejects_wrong_conda_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "wrong-environment"))
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)
    with pytest.raises(JobError) as captured:
        fp_naa_runtime_check(run_convolution=False)
    assert captured.value.code == "WRONG_CONDA_ENV"


def test_start_uses_detached_current_python_without_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    manifest = context.repo_root / "data" / "manifests" / "dcase2026_dev.parquet"
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(b"manifest")
    audio = context.data_root / "raw" / "dcase2026" / "dev" / "extracted"
    audio.mkdir(parents=True)
    monkeypatch.setattr(
        "care_asd.server.fp_naa_jobs._git_output",
        lambda _context, *arguments: (
            ("research/fp-naa" if arguments == ("branch", "--show-current") else "")
            if arguments != ("rev-parse", "HEAD")
            else "b" * 40
        ),
    )
    monkeypatch.setattr("care_asd.server.fp_naa_jobs.fp_naa_runtime_check", lambda: {})
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 4321

    def fake_popen(command: list[str], **kwargs: object) -> FakeProcess:
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr("care_asd.server.fp_naa_jobs.subprocess.Popen", fake_popen)
    result = start_fp_naa_job(JobStage.C0, workers=12, context=context)
    command = cast(list[str], captured["command"])
    kwargs = cast(dict[str, object], captured["kwargs"])
    assert command[:3] == [sys.executable, "-m", "care_asd.cli"]
    assert command[3:6] == ["fp-naa", "job", "run-internal"]
    assert kwargs["start_new_session"] is True
    assert "shell" not in kwargs
    environment = cast(dict[str, str], kwargs["env"])
    assert environment["CUBLAS_WORKSPACE_CONFIG"] == CUBLAS_WORKSPACE_CONFIG
    assert "LD_LIBRARY_PATH" not in environment
    assert "LD_PRELOAD" not in environment
    assert result["job"]["pid"] == 4321


def test_launch_lock_rejects_concurrent_start_request(tmp_path: Path) -> None:
    context = _context(tmp_path)

    with (
        _launch_lock(context),
        pytest.raises(JobError) as captured,
        _launch_lock(context),
    ):
        pytest.fail("A concurrent start request acquired the same lock")

    assert captured.value.code == "JOB_START_LOCKED"
    assert not (context.control_root / "launch.lock").exists()


def test_fp_naa_shell_wrappers_are_not_part_of_the_architecture() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert list((repository / "scripts" / "server").glob("*fp_naa*.sh")) == []


def test_server_pipeline_uses_the_versioned_equivariant_config(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert _common_paths(context)["config"].name == "fp_naa_v5.yaml"


def test_state_schema_is_validated() -> None:
    payload = asdict(_state(FPNAAJobContext(Path("repo"), Path("data"))))
    payload["schema_version"] = 999
    with pytest.raises(JobError) as captured:
        JobState.from_dict(payload)
    assert captured.value.code == "STATE_SCHEMA_MISMATCH"
