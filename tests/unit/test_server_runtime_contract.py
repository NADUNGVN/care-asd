from __future__ import annotations

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SERVER_SCRIPTS = REPOSITORY_ROOT / "scripts" / "server"


def test_reference_safety_jobs_never_resolve_the_environment() -> None:
    for name in (
        "run_phase10_reference_safety.sh",
        "run_phase11_reference_safety.sh",
        "run_phase12_reference_safety.sh",
    ):
        script = (SERVER_SCRIPTS / name).read_text(encoding="utf-8")
        assert "uv run --no-sync care-asd" in script
        assert "uv run --extra" not in script


def test_gpu_phases_enforce_the_verified_server_runtime() -> None:
    check = (SERVER_SCRIPTS / "check_gpu_runtime.sh").read_text(encoding="utf-8")
    assert 'expected="2.6.0+cu118"' in check
    assert 'runtime == "11.8"' in check
    assert "torch.cuda.is_available()" in check

    for name in (
        "start_phase11_reference_safety.sh",
        "start_phase12_reference_safety.sh",
    ):
        launcher = (SERVER_SCRIPTS / name).read_text(encoding="utf-8")
        assert "bash scripts/server/check_gpu_runtime.sh" in launcher
