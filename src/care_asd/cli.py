"""CARE-ASD command-line interface.

Phase 0 provides the CLI skeleton, environment reporting, and config utilities.
Dataset, training, and evaluation subcommands are stubs that document upcoming
phases and support ``--dry-run`` / ``--config`` conventions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated, cast

import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.table import Table

from care_asd import __version__
from care_asd.config import config_hash, default_config, load_config, validate_config
from care_asd.data import (
    audit_dcase2026_manifest,
    build_care_residual_vector_cache,
    build_dcase2026_manifest,
    build_neural_feature_cache,
    build_official_vector_cache,
    dcase2026_manifest_path,
    download_dcase2026_split,
    extract_dcase2026_split,
    normalize_split,
)
from care_asd.deployment import validate_deployment_bundle, validate_tensorrt_model_latency_report
from care_asd.evaluation import (
    ScoreMode,
    calculate_development_auc_metrics,
    normalize_official_development_scores,
    run_care_development_benchmark,
    run_dsp_development_benchmark,
    write_paired_bootstrap_comparison,
    write_seed_ensemble_scores,
)
from care_asd.logging_utils import setup_logging
from care_asd.models import (
    OFFICIAL_BASELINE_COMMIT,
    OFFICIAL_BASELINE_REPOSITORY,
    OFFICIAL_EVALUATOR_COMMIT,
    OFFICIAL_EVALUATOR_REPOSITORY,
    BaselineMode,
    checkout_pinned_reference,
    run_official_development_baseline,
    stage_official_development_data,
)
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
baseline_app = typer.Typer(help="Pinned official DCASE 2026 baseline reproduction.")
app.add_typer(baseline_app, name="baseline")
dsp_app = typer.Typer(help="Phase 3 deterministic stereo DSP controls.")
app.add_typer(dsp_app, name="dsp")

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


@app.command("bundle-validate")
def bundle_validate(
    bundle: Annotated[
        Path,
        typer.Argument(help="Path to a CARE-ASD deployment bundle directory."),
    ],
) -> None:
    """Validate deployment artifact hashes and scorer/calibration contracts."""
    try:
        manifest = validate_deployment_bundle(bundle)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Invalid deployment bundle:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]Deployment bundle valid.[/green] "
        f"schema={manifest.schema_version} config_hash={manifest.config_hash}"
    )


@app.command("benchmark-report-validate")
def benchmark_report_validate(
    report: Annotated[
        Path,
        typer.Argument(help="Path to care_asd_trt_runner JSON report."),
    ],
) -> None:
    """Validate a hardware-produced TensorRT model-latency report."""
    try:
        parsed = validate_tensorrt_model_latency_report(report)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Invalid TensorRT benchmark report:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]TensorRT benchmark report valid.[/green] "
        f"p50={parsed.model_latency_ms.p50:.3f} ms "
        f"p95={parsed.model_latency_ms.p95:.3f} ms"
    )


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
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Dataset root; defaults to data.root in config."),
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
    """Download and checksum-verify DCASE 2026 Task 2 archives from Zenodo."""
    try:
        normalized = normalize_split(split)
        root = _data_root(config, data_root)
        if dry_run:
            console.print(
                f"[yellow]dry-run:[/yellow] would download split={normalized} to {root} "
                f"(accept_eval_policy={accept_eval_policy})"
            )
            return
        archives = download_dcase2026_split(
            root,
            normalized,
            accept_evaluation_policy=accept_eval_policy,
        )
    except (FileNotFoundError, PermissionError, ValueError, OSError) as exc:
        console.print(f"[red]Dataset download failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    new_count = sum(item.downloaded for item in archives)
    console.print(
        f"[green]Verified {len(archives)} archive(s).[/green] "
        f"downloaded={new_count}, reused={len(archives) - new_count}"
    )


@data_app.command("extract")
def data_extract(
    split: Annotated[str, typer.Option("--split")] = "dev",
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Dataset root; defaults to data.root in config."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Safely extract downloaded archives without overwriting files."""
    try:
        normalized = normalize_split(split)
        root = _data_root(config, data_root)
        if dry_run:
            console.print(
                f"[yellow]dry-run:[/yellow] would extract split={normalized} under {root}"
            )
            return
        summaries = extract_dcase2026_split(root, normalized)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        console.print(f"[red]Dataset extraction failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    extracted = sum(item.extracted_files for item in summaries)
    reused = sum(item.reused_files for item in summaries)
    console.print(
        f"[green]Extraction complete.[/green] archives={len(summaries)}, "
        f"new_files={extracted}, reused_files={reused}"
    )


@data_app.command("manifest")
def data_manifest(
    split: Annotated[str, typer.Option("--split")] = "dev",
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Dataset root; defaults to data.root in config."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Build a deterministic immutable Parquet manifest from extracted audio."""
    try:
        normalized = normalize_split(split)
        root = _data_root(config, data_root)
        target = dcase2026_manifest_path(root, normalized)
        if dry_run:
            console.print(f"[yellow]dry-run:[/yellow] would write manifest to {target}")
            return
        manifest = build_dcase2026_manifest(root, normalized)
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as exc:
        console.print(f"[red]Manifest build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Wrote immutable manifest:[/green] {manifest}")


@data_app.command("validate")
def data_validate(
    split: Annotated[str, typer.Option("--split")] = "dev",
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    data_root: Annotated[
        Path | None,
        typer.Option("--data-root", help="Dataset root; defaults to data.root in config."),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Audit manifest integrity and enforce the DCASE stereo-audio contract."""
    try:
        normalized = normalize_split(split)
        root = _data_root(config, data_root)
        manifest = dcase2026_manifest_path(root, normalized)
        if dry_run:
            console.print(f"[yellow]dry-run:[/yellow] would audit {manifest}")
            return
        audit = audit_dcase2026_manifest(manifest)
    except (FileNotFoundError, ValueError, OSError) as exc:
        console.print(f"[red]Dataset validation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]Dataset audit passed.[/green] "
        f"clips={audit.clips}, stereo={audit.stereo_clips}, "
        f"sample_rates={list(audit.sample_rates)}, conditions={list(audit.conditions)}, "
        f"domains={list(audit.domains)}"
    )


@data_app.command("cache-neural")
def data_cache_neural(
    manifest: Annotated[Path, typer.Option("--manifest")],
    audio_root: Annotated[Path, typer.Option("--audio-root")],
    output_directory: Annotated[Path, typer.Option("--output-dir")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
) -> None:
    """Build one reusable local near/far/CARE feature cache for every MVP ablation."""
    try:
        cfg = validate_config(load_config(config))
        result = build_neural_feature_cache(
            manifest_path=manifest,
            audio_root=audio_root,
            output_directory=output_directory,
            signal=cfg.signal,
            frontend=cfg.frontend,
            features=cfg.features,
            workers=workers,
            limit=limit,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]Neural cache build failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Neural cache complete.[/green] clips={result.clips} index={result.index_path}"
    )


@data_app.command("cache-official-vectors")
def data_cache_official_vectors(
    manifest: Annotated[Path, typer.Option("--manifest")],
    audio_root: Annotated[Path, typer.Option("--audio-root")],
    output_directory: Annotated[Path, typer.Option("--output-dir")],
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
) -> None:
    """Build exact channel-0 five-frame log-Mel vectors for official alignment."""
    try:
        result = build_official_vector_cache(
            manifest_path=manifest,
            audio_root=audio_root,
            output_directory=output_directory,
            workers=workers,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Official vector cache failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]Official vector cache complete.[/green] "
        f"clips={result.clips} index={result.index_path}"
    )


@data_app.command("cache-care-residual-vectors")
def data_cache_care_residual_vectors(
    manifest: Annotated[Path, typer.Option("--manifest")],
    audio_root: Annotated[Path, typer.Option("--audio-root")],
    output_directory: Annotated[Path, typer.Option("--output-dir")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    workers: Annotated[int, typer.Option("--workers", min=1)] = 1,
) -> None:
    """Build CARE residual vectors using the otherwise unchanged official stack."""
    try:
        cfg = validate_config(load_config(config))
        result = build_care_residual_vector_cache(
            manifest_path=manifest,
            audio_root=audio_root,
            output_directory=output_directory,
            signal=cfg.signal,
            frontend=cfg.frontend,
            workers=workers,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]CARE residual vector cache failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]CARE residual vector cache complete.[/green] "
        f"clips={result.clips} index={result.index_path}"
    )


@baseline_app.command("checkout")
def baseline_checkout(
    baseline_dir: Annotated[
        Path, typer.Option("--baseline-dir", help="External official baseline checkout.")
    ] = Path("external/dcase2026_task2_baseline_ae"),
    evaluator_dir: Annotated[
        Path, typer.Option("--evaluator-dir", help="External official evaluator checkout.")
    ] = Path("external/dcase2026_task2_evaluator"),
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Clone and pin external official baseline/evaluator references."""
    if dry_run:
        console.print(
            f"[yellow]dry-run:[/yellow] would pin baseline={baseline_dir} at "
            f"{OFFICIAL_BASELINE_COMMIT} and evaluator={evaluator_dir} at {OFFICIAL_EVALUATOR_COMMIT}"
        )
        return
    try:
        baseline = checkout_pinned_reference(
            baseline_dir,
            repository=OFFICIAL_BASELINE_REPOSITORY,
            commit=OFFICIAL_BASELINE_COMMIT,
            required_file="01_train_2026t2.sh",
        )
        evaluator = checkout_pinned_reference(
            evaluator_dir,
            repository=OFFICIAL_EVALUATOR_REPOSITORY,
            commit=OFFICIAL_EVALUATOR_COMMIT,
            required_file="dcase2026_task2_evaluator.py",
        )
    except (FileNotFoundError, OSError, ValueError, subprocess.CalledProcessError) as exc:
        console.print(f"[red]Official reference checkout failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Pinned baseline:[/green] {baseline.directory} @ {baseline.commit}")
    console.print(f"[green]Pinned evaluator:[/green] {evaluator.directory} @ {evaluator.commit}")


@baseline_app.command("stage-dev")
def baseline_stage_dev(
    baseline_dir: Annotated[Path, typer.Option("--baseline-dir")],
    data_root: Annotated[Path, typer.Option("--data-root")],
    manifest: Annotated[Path, typer.Option("--manifest")],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Create no-copy symlinks from audited development audio to the official layout."""
    if dry_run:
        console.print(f"[yellow]dry-run:[/yellow] would stage {manifest} into {baseline_dir}")
        return
    try:
        result = stage_official_development_data(
            baseline_directory=baseline_dir, data_root=data_root, manifest_path=manifest
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]Official baseline staging failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Staged {len(result.machine_types)} machine type(s) without copying WAVs.[/green] "
        f"created_links={result.created_links}, reused_links={result.reused_links}"
    )


@baseline_app.command("run-dev")
def baseline_run_dev(
    baseline_dir: Annotated[Path, typer.Option("--baseline-dir")],
    official_python: Annotated[
        Path, typer.Option("--official-python", help="Python in the isolated official environment.")
    ],
    mode: Annotated[str, typer.Option("--mode", help="mse | mahala | all")] = "all",
    skip_training: Annotated[
        bool,
        typer.Option("--skip-training", help="Score from existing official checkpoints only."),
    ] = False,
    log: Annotated[Path, typer.Option("--log", help="New immutable official run log.")] = Path(
        "outputs/baseline/official_dev.log"
    ),
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Run the unmodified official training and selected development score mode(s)."""
    if mode not in {"mse", "mahala", "all"}:
        console.print("[red]mode must be one of: mse, mahala, all[/red]")
        raise typer.Exit(code=1)
    if skip_training and mode == "all":
        console.print("[red]--skip-training requires --mode=mse or --mode=mahala[/red]")
        raise typer.Exit(code=1)
    if dry_run:
        console.print(
            f"[yellow]dry-run:[/yellow] would run official baseline mode={mode} "
            f"skip_training={skip_training} log={log}"
        )
        return
    try:
        run_official_development_baseline(
            baseline_directory=baseline_dir,
            official_python=official_python,
            mode=cast("BaselineMode", mode),
            log_path=log,
            skip_training=skip_training,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        ValueError,
        subprocess.CalledProcessError,
    ) as exc:
        console.print(f"[red]Official baseline run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Official baseline completed; immutable log:[/green] {log}")


@baseline_app.command("normalize")
def baseline_normalize(
    official_scores: Annotated[
        Path,
        typer.Option(
            "--official-scores", help="Official results directory containing anomaly_score CSVs."
        ),
    ],
    manifest: Annotated[Path, typer.Option("--manifest")],
    score_mode: Annotated[str, typer.Option("--score-mode", help="mse | mahala")],
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Normalize official CSV scores into the CARE-ASD score schema."""
    if score_mode not in {"mse", "mahala"}:
        console.print("[red]score-mode must be one of: mse, mahala[/red]")
        raise typer.Exit(code=1)
    try:
        result = normalize_official_development_scores(
            official_score_directory=official_scores,
            manifest_path=manifest,
            score_mode=cast("ScoreMode", score_mode),
            experiment_id=experiment_id,
            output_path=output,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]Official score normalization failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Wrote normalized official scores:[/green] {result}")


@baseline_app.command("metrics")
def baseline_metrics(
    scores: Annotated[Path, typer.Option("--scores")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Recompute deterministic development AUC/pAUC from normalized scores."""
    try:
        result = calculate_development_auc_metrics(scores, output)
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]Development metric computation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Wrote development metrics:[/green] {result}")


@dsp_app.command("benchmark-dev")
def dsp_benchmark_dev(
    manifest: Annotated[
        Path, typer.Option("--manifest", help="DCASE development Parquet manifest.")
    ],
    audio_root: Annotated[
        Path,
        typer.Option(
            "--audio-root",
            help="Extracted DCASE dev audio directory that contains the manifest paths.",
        ),
    ],
    output_directory: Annotated[
        Path,
        typer.Option("--output-dir", help="New immutable directory for all benchmark evidence."),
    ],
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    frontends: Annotated[
        str,
        typer.Option("--frontends", help="Comma list or 'all' (the default)."),
    ] = "all",
    workers: Annotated[
        int,
        typer.Option(
            "--workers", min=1, help="CPU processes; use 1 for strictly serial execution."
        ),
    ] = 1,
) -> None:
    """Run all selected Phase 3 DSP controls with one fixed reference scorer."""
    try:
        from care_asd.signal.dsp_baselines import available_dsp_frontends

        selected = (
            available_dsp_frontends()
            if frontends == "all"
            else tuple(item.strip() for item in frontends.split(",") if item.strip())
        )
        result = run_dsp_development_benchmark(
            manifest_path=manifest,
            audio_root=audio_root,
            output_directory=output_directory,
            experiment_id=experiment_id,
            frontends=selected,
            workers=workers,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]DSP benchmark failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]DSP development benchmark completed.[/green] "
        f"summary={result.summary_path} overcancellation={result.overcancellation_path}"
    )


@app.command("care-benchmark-dev")
def care_benchmark_dev(
    manifest: Annotated[
        Path, typer.Option("--manifest", help="DCASE development Parquet manifest.")
    ],
    audio_root: Annotated[
        Path,
        typer.Option("--audio-root", help="Extracted DCASE audio directory."),
    ],
    output_directory: Annotated[
        Path,
        typer.Option("--output-dir", help="New immutable directory for CARE evidence."),
    ],
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
    workers: Annotated[
        int,
        typer.Option(
            "--workers", min=1, help="CPU processes; use 1 for strictly serial execution."
        ),
    ] = 1,
) -> None:
    """Benchmark the Safe CARE front-end with the fixed Phase 3 scorer."""
    try:
        result = run_care_development_benchmark(
            manifest_path=manifest,
            audio_root=audio_root,
            output_directory=output_directory,
            experiment_id=experiment_id,
            workers=workers,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]CARE benchmark failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]CARE development benchmark completed.[/green] "
        f"summary={result.summary_path} frequency_bands={result.frequency_bands_path}"
    )


@app.command("mvp-neural-dev")
def mvp_neural_dev(
    cache_directory: Annotated[Path, typer.Option("--cache-dir")],
    output_directory: Annotated[Path, typer.Option("--output-dir")],
    checkpoint_directory: Annotated[Path, typer.Option("--checkpoint-dir")],
    ablation: Annotated[str, typer.Option("--ablation")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    epochs: Annotated[int | None, typer.Option("--epochs", min=1)] = None,
) -> None:
    """Train one normal-only GPU MVP ablation and score all dev test clips."""
    try:
        from care_asd.evaluation.mvp_neural import (
            available_mvp_ablations,
            run_mvp_neural_development,
        )

        if ablation not in available_mvp_ablations():
            raise ValueError(f"ablation must be one of: {', '.join(available_mvp_ablations())}")
        result = run_mvp_neural_development(
            cache_directory=cache_directory,
            output_directory=output_directory,
            checkpoint_directory=checkpoint_directory,
            config=validate_config(load_config(config)),
            ablation=ablation,
            epochs=epochs,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]MVP neural run failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]MVP neural development run completed.[/green] "
        f"scores={result.score_path} summary={result.summary_path}"
    )


@app.command("mvp-neural-screening-dev")
def mvp_neural_screening_dev(
    cache_directory: Annotated[Path, typer.Option("--cache-dir")],
    output_directory: Annotated[Path, typer.Option("--output-dir")],
    checkpoint_directory: Annotated[Path, typer.Option("--checkpoint-dir")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    preload_workers: Annotated[int, typer.Option("--preload-workers", min=1)] = 16,
) -> None:
    """Screen all three fixed GPU views after one shared in-memory cache preload."""
    try:
        from care_asd.evaluation.mvp_neural import run_mvp_neural_screening_development

        result = run_mvp_neural_screening_development(
            cache_directory=cache_directory,
            output_directory=output_directory,
            checkpoint_directory=checkpoint_directory,
            config=validate_config(load_config(config)),
            preload_workers=preload_workers,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]MVP neural screening failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]MVP neural screening completed.[/green] " f"summary={result.summary_path}"
    )


@app.command("mvp-neural-replication-dev")
def mvp_neural_replication_dev(
    cache_directory: Annotated[Path, typer.Option("--cache-dir")],
    output_directory: Annotated[Path, typer.Option("--output-dir")],
    checkpoint_directory: Annotated[Path, typer.Option("--checkpoint-dir")],
    seeds: Annotated[str, typer.Option("--seeds")] = "42,2026",
    ablations: Annotated[str, typer.Option("--ablations")] = "a00_near,a02_care_multiview",
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
    preload_workers: Annotated[int, typer.Option("--preload-workers", min=1)] = 16,
) -> None:
    """Replicate selected fixed GPU views across seeds after one RAM preload."""
    try:
        from care_asd.evaluation.mvp_neural import run_mvp_neural_replication_development

        seed_values = tuple(int(value.strip()) for value in seeds.split(",") if value.strip())
        ablation_values = tuple(value.strip() for value in ablations.split(",") if value.strip())
        result = run_mvp_neural_replication_development(
            cache_directory=cache_directory,
            output_directory=output_directory,
            checkpoint_directory=checkpoint_directory,
            config=validate_config(load_config(config)),
            seeds=seed_values,
            ablations=ablation_values,  # type: ignore[arg-type]
            preload_workers=preload_workers,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]MVP neural replication failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]MVP neural replication completed.[/green] " f"summary={result.summary_path}"
    )


@app.command("official-alignment-dev")
def official_alignment_dev(
    cache_directory: Annotated[Path, typer.Option("--cache-dir")],
    output_directory: Annotated[Path, typer.Option("--output-dir")],
    checkpoint_directory: Annotated[Path, typer.Option("--checkpoint-dir")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Run the internal architecture-level reproduction of official MSE scoring."""
    try:
        from care_asd.evaluation.official_alignment import run_official_alignment_development

        result = run_official_alignment_development(
            cache_directory=cache_directory,
            output_directory=output_directory,
            checkpoint_directory=checkpoint_directory,
            config=validate_config(load_config(config)),
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]Official alignment failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]Official alignment completed.[/green] " f"summary={result.summary_path}"
    )


@app.command("care-residual-alignment-dev")
def care_residual_alignment_dev(
    cache_directory: Annotated[Path, typer.Option("--cache-dir")],
    output_directory: Annotated[Path, typer.Option("--output-dir")],
    checkpoint_directory: Annotated[Path, typer.Option("--checkpoint-dir")],
    config: Annotated[Path | None, typer.Option("--config", "-c")] = None,
) -> None:
    """Run Phase 7 CARE residual-only test under the locked official AE protocol."""
    try:
        from care_asd.evaluation.official_alignment import (
            run_care_residual_alignment_development,
        )

        result = run_care_residual_alignment_development(
            cache_directory=cache_directory,
            output_directory=output_directory,
            checkpoint_directory=checkpoint_directory,
            config=validate_config(load_config(config)),
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as exc:
        console.print(f"[red]CARE residual alignment failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]CARE residual alignment completed.[/green] " f"summary={result.summary_path}"
    )


@app.command("care-residual-analysis-dev")
def care_residual_analysis_dev(
    near_cache_directory: Annotated[Path, typer.Option("--near-cache-dir")],
    residual_cache_directory: Annotated[Path, typer.Option("--residual-cache-dir")],
    reference_scores: Annotated[Path, typer.Option("--reference-scores")],
    candidate_scores: Annotated[Path, typer.Option("--candidate-scores")],
    output_directory: Annotated[Path, typer.Option("--output-dir")],
) -> None:
    """Explain frozen B00/B01 changes without selecting another model."""
    try:
        from care_asd.evaluation.care_residual_analysis import analyze_care_residual_development

        result = analyze_care_residual_development(
            near_cache_directory=near_cache_directory,
            residual_cache_directory=residual_cache_directory,
            reference_scores=reference_scores,
            candidate_scores=candidate_scores,
            output_directory=output_directory,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]CARE residual analysis failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(
        "[green]CARE residual analysis completed.[/green] " f"report={result.report_path}"
    )


@app.command("mvp-ensemble")
def mvp_ensemble(
    scores: Annotated[list[Path], typer.Option("--scores")],
    output: Annotated[Path, typer.Option("--output")],
    model_id: Annotated[str, typer.Option("--model-id")],
    experiment_id: Annotated[str, typer.Option("--experiment-id")],
) -> None:
    """Average matched per-file scores from independent seeds before bootstrap."""
    try:
        result = write_seed_ensemble_scores(
            score_paths=scores,
            output_path=output,
            model_id=model_id,
            experiment_id=experiment_id,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]MVP score ensemble failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]MVP score ensemble complete:[/green] {result}")


@app.command("mvp-bootstrap")
def mvp_bootstrap(
    reference_scores: Annotated[Path, typer.Option("--reference-scores")],
    candidate_scores: Annotated[Path, typer.Option("--candidate-scores")],
    output: Annotated[Path, typer.Option("--output")],
    iterations: Annotated[int, typer.Option("--iterations", min=100)] = 2000,
    seed: Annotated[int, typer.Option("--seed", min=0)] = 2026,
) -> None:
    """Compute paired stratified bootstrap deltas after final multi-seed selection."""
    try:
        result = write_paired_bootstrap_comparison(
            reference_scores=reference_scores,
            candidate_scores=candidate_scores,
            output_path=output,
            iterations=iterations,
            seed=seed,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as exc:
        console.print(f"[red]MVP bootstrap failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Paired bootstrap complete:[/green] {result}")


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


def _data_root(config_path: Path | None, data_root: Path | None) -> Path:
    """Resolve CLI data storage without mutating the experiment configuration."""
    if data_root is not None:
        return data_root
    config = validate_config(load_config(config_path))
    return Path(config.data.root)


def run() -> None:
    """Entry point for ``python -m care_asd.cli``."""
    app()


if __name__ == "__main__":
    app()
