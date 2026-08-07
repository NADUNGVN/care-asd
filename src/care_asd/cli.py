"""CARE-ASD command-line interface.

Phase 0 provides the CLI skeleton, environment reporting, and config utilities.
Dataset, training, and evaluation subcommands are stubs that document upcoming
phases and support ``--dry-run`` / ``--config`` conventions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from care_asd import __version__
from care_asd.config import config_hash, default_config, load_config, validate_config
from care_asd.logging_utils import setup_logging
from care_asd.reproducibility import collect_environment_report, set_seed

app = typer.Typer(
    name="care-asd",
    help=(
        "CARE-ASD: Causal Acoustic-Path and Reliability-Calibrated "
        "Anomalous Sound Detection for Unseen Machines at the Edge."
    ),
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Dataset download, extraction, manifest, and validation.")
app.add_typer(data_app, name="data")

console = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"care-asd {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable DEBUG logging."),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Only show warnings and errors."),
    ] = False,
) -> None:
    """CARE-ASD research toolkit."""
    if quiet:
        level = "WARNING"
    elif verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    setup_logging(level)


@app.command("env-report")
def env_report(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write JSON report to this path."),
    ] = None,
) -> None:
    """Print a reproducibility environment report."""
    report = collect_environment_report()
    text = report.to_json()
    if output is not None:
        if output.exists():
            console.print(f"[red]Refusing to overwrite existing file:[/red] {output}")
            raise typer.Exit(code=1)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
        console.print(f"[green]Wrote environment report to[/green] {output}")
    else:
        typer.echo(text)


@app.command("config-show")
def config_show(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to YAML config file."),
    ] = None,
    override: Annotated[
        list[str] | None,
        typer.Option("--override", help="Dot-list override, e.g. experiment.seed=1"),
    ] = None,
    show_hash: Annotated[
        bool,
        typer.Option("--hash", help="Print config SHA-256 hash only."),
    ] = False,
) -> None:
    """Load, validate, and display configuration."""
    cfg = load_config(config, overrides=override)
    if show_hash:
        typer.echo(config_hash(cfg))
        return

    validated = validate_config(cfg)
    table = Table(title="CARE-ASD Configuration", show_header=True)
    table.add_column("Section", style="cyan")
    table.add_column("Key values", style="white")

    data = validated.model_dump(by_alias=True)
    for section, values in data.items():
        if isinstance(values, dict):
            summary = ", ".join(f"{k}={v!r}" for k, v in values.items())
        else:
            summary = repr(values)
        table.add_row(section, summary)

    console.print(table)
    console.print(f"[dim]config_hash={config_hash(cfg)}[/dim]")


@app.command("config-init")
def config_init(
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Where to write the default config."),
    ] = Path("configs/experiment/default.yaml"),
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite if the file exists."),
    ] = False,
) -> None:
    """Write a default configuration YAML file."""
    if output.exists() and not force:
        console.print(f"[red]File exists (use --force to overwrite):[/red] {output}")
        raise typer.Exit(code=1)

    cfg = default_config()
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output)
    console.print(f"[green]Wrote default config to[/green] {output}")
    console.print(f"[dim]config_hash={config_hash(cfg)}[/dim]")


@app.command("seed-check")
def seed_check(
    seed: Annotated[int, typer.Option("--seed", help="Seed value to apply.")] = 42,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate only; do not mutate RNG state."),
    ] = False,
) -> None:
    """Apply a seed and print a short reproducibility self-check."""
    if seed < 0:
        console.print("[red]Seed must be non-negative.[/red]")
        raise typer.Exit(code=1)

    if dry_run:
        console.print(f"[yellow]dry-run:[/yellow] would set seed={seed}")
        raise typer.Exit(code=0)

    set_seed(seed)
    sample = [__import__("random").random() for _ in range(3)]
    console.print(f"seed={seed}")
    console.print(f"random_samples={sample}")


@data_app.command("download")
def data_download(
    split: Annotated[
        str,
        typer.Option("--split", help="dev | additional | evaluation"),
    ] = "dev",
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Config path."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show plan without downloading."),
    ] = False,
    accept_eval_policy: Annotated[
        bool,
        typer.Option(
            "--accept-eval-policy",
            help="Required for evaluation split (Phase 1).",
        ),
    ] = False,
) -> None:
    """Download DCASE 2026 Task 2 data from Zenodo (Phase 1)."""
    extra = f"accept_eval_policy={accept_eval_policy}, config={config}"
    _phase_stub(phase=1, action=f"data download --split {split}", dry_run=dry_run, extra=extra)


@data_app.command("extract")
def data_extract(
    split: Annotated[str, typer.Option("--split")] = "dev",
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Safely extract downloaded archives (Phase 1)."""
    _phase_stub(phase=1, action=f"data extract --split {split}", dry_run=dry_run, extra=str(config))


@data_app.command("manifest")
def data_manifest(
    split: Annotated[str, typer.Option("--split")] = "dev",
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Build a deterministic dataset manifest (Phase 1)."""
    _phase_stub(
        phase=1,
        action=f"data manifest --split {split}",
        dry_run=dry_run,
        extra=str(config),
    )


@data_app.command("validate")
def data_validate(
    split: Annotated[str, typer.Option("--split")] = "dev",
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Audit dataset integrity and stereo layout (Phase 1)."""
    _phase_stub(
        phase=1,
        action=f"data validate --split {split}",
        dry_run=dry_run,
        extra=str(config),
    )


@app.command("train")
def train(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Experiment config YAML."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Train CARE-ASD models (Phases 4-5)."""
    _phase_stub(phase=5, action="train", dry_run=dry_run, extra=str(config))


@app.command("evaluate")
def evaluate(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Experiment config YAML."),
    ] = None,
    freeze_file: Annotated[
        Path | None,
        typer.Option("--freeze-file", help="Required for official evaluation (Phase 9)."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Evaluate models and write score files (Phases 2+)."""
    _phase_stub(
        phase=2,
        action="evaluate",
        dry_run=dry_run,
        extra=f"config={config}, freeze_file={freeze_file}",
    )


def _phase_stub(
    *,
    phase: int,
    action: str,
    dry_run: bool,
    extra: str | None = None,
) -> None:
    """Shared message for not-yet-implemented phase commands."""
    msg = (
        f"[yellow]Phase {phase} not implemented yet.[/yellow]\n"
        f"Requested: [bold]{action}[/bold]\n"
        "See CARE_ASD_CODEX_IMPLEMENTATION_PLAN.md for the phase specification."
    )
    if extra and extra != "None":
        msg += f"\nOptions: {extra}"
    if dry_run:
        msg += "\n[dim]--dry-run: no side effects.[/dim]"
    console.print(msg)
    raise typer.Exit(code=0)


def run() -> None:
    """Entry point for ``python -m care_asd.cli``."""
    app()


if __name__ == "__main__":
    app()
