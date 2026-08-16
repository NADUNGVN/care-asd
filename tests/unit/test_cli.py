"""CLI smoke tests — no dataset dependency."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from care_asd import __version__
from care_asd.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "CARE-ASD" in result.stdout or "care-asd" in result.stdout.lower()


def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_env_report() -> None:
    result = runner.invoke(app, ["env-report"])
    assert result.exit_code == 0
    assert "python_version" in result.stdout


def test_env_report_to_file(tmp_path: Path) -> None:
    out = tmp_path / "env.json"
    result = runner.invoke(app, ["env-report", "-o", str(out)])
    assert result.exit_code == 0
    assert out.exists()
    assert "python_version" in out.read_text(encoding="utf-8")


def test_config_show_default() -> None:
    result = runner.invoke(app, ["config-show"])
    assert result.exit_code == 0


def test_config_show_hash() -> None:
    result = runner.invoke(app, ["config-show", "--hash"])
    assert result.exit_code == 0
    digest = result.stdout.strip()
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_config_show_with_project_yaml() -> None:
    root = Path(__file__).resolve().parents[2]
    yaml_path = root / "configs" / "experiment" / "default.yaml"
    result = runner.invoke(app, ["config-show", "--config", str(yaml_path)])
    assert result.exit_code == 0


def test_seed_check_dry_run() -> None:
    result = runner.invoke(app, ["seed-check", "--seed", "1", "--dry-run"])
    assert result.exit_code == 0


def test_seed_check_applies() -> None:
    result = runner.invoke(app, ["seed-check", "--seed", "42"])
    assert result.exit_code == 0
    assert "seed=42" in result.stdout or "seed=42" in result.stderr


def test_data_download_dry_run() -> None:
    result = runner.invoke(app, ["data", "download", "--split", "dev", "--dry-run"])
    assert result.exit_code == 0
    combined = (result.stdout + result.stderr).lower()
    assert "would download" in combined


def test_train_stub() -> None:
    result = runner.invoke(app, ["train", "--dry-run"])
    assert result.exit_code == 0


def test_phase7_commands_are_exposed_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "care-residual-alignment-dev" in result.stdout

    result = runner.invoke(app, ["data", "--help"])
    assert result.exit_code == 0
    assert "cache-care-residual-vectors" in result.stdout


def test_phase8_analysis_command_is_exposed_in_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "care-residual-analysis-dev" in result.stdout


def test_phase9_fusion_commands_are_exposed_in_help() -> None:
    assert "gated-fusion-dev" in runner.invoke(app, ["--help"]).stdout
    assert "build-reliability-index" in runner.invoke(app, ["data", "--help"]).stdout


def test_ap_care_g1_dry_run_is_exposed_and_side_effect_free(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    output = tmp_path / "must-not-exist"
    result = runner.invoke(
        app,
        [
            "ap-care",
            "simulate",
            "--config",
            str(root / "configs" / "experiment" / "ap_care_v2.yaml"),
            "--output-dir",
            str(output),
            "--cases",
            "32",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert '"cases": 32' in result.stdout
    assert not output.exists()
