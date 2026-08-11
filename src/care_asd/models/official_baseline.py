"""Pinned external execution support for the official DCASE 2026 AE baseline."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

OFFICIAL_BASELINE_REPOSITORY = "https://github.com/nttcslab/dcase2023_task2_baseline_ae.git"
OFFICIAL_BASELINE_COMMIT = "f44242ec1f78f6cc34f53f43fb88be1ce5d13d47"
OFFICIAL_EVALUATOR_REPOSITORY = "https://github.com/nttcslab/dcase2026_task2_evaluator.git"
OFFICIAL_EVALUATOR_COMMIT = "f6a94a2b5e614a9626c9d1ccff6df0705e6aaa75"
BaselineMode = Literal["mse", "mahala", "all"]


@dataclass(frozen=True)
class BaselineReference:
    """A verified immutable checkout of an official repository."""

    directory: Path
    repository: str
    commit: str


@dataclass(frozen=True)
class BaselineStagingSummary:
    """Symlink-only data staging outcome for the official baseline layout."""

    machine_types: tuple[str, ...]
    created_links: int
    reused_links: int


def checkout_pinned_reference(
    directory: str | Path,
    *,
    repository: str,
    commit: str,
    required_file: str,
) -> BaselineReference:
    """Clone a reference once and hard-fail if an existing checkout differs."""
    root = Path(directory)
    if root.exists():
        return verify_pinned_reference(
            root, repository=repository, commit=commit, required_file=required_file
        )

    root.parent.mkdir(parents=True, exist_ok=True)
    _run_git(["clone", repository, str(root)])
    _run_git(["-C", str(root), "checkout", "--detach", commit])
    return verify_pinned_reference(
        root, repository=repository, commit=commit, required_file=required_file
    )


def verify_pinned_reference(
    directory: str | Path,
    *,
    repository: str,
    commit: str,
    required_file: str,
) -> BaselineReference:
    """Verify the exact revision and minimal file contract of a checkout."""
    root = Path(directory)
    if not (root / ".git").exists():
        raise FileNotFoundError(f"Official reference is not a Git checkout: {root}")
    if not (root / required_file).is_file():
        raise FileNotFoundError(f"Official reference is missing {required_file}: {root}")
    actual = _run_git(["-C", str(root), "rev-parse", "HEAD"]).stdout.strip()
    if actual != commit:
        raise ValueError(f"Pinned reference mismatch: expected {commit}, found {actual}")
    remote = _run_git(["-C", str(root), "remote", "get-url", "origin"]).stdout.strip()
    if remote.rstrip("/") != repository.rstrip("/"):
        raise ValueError(f"Reference origin mismatch: expected {repository}, found {remote}")
    return BaselineReference(directory=root, repository=repository, commit=commit)


def stage_official_development_data(
    *,
    baseline_directory: str | Path,
    data_root: str | Path,
    manifest_path: str | Path,
) -> BaselineStagingSummary:
    """Expose audited data through symlinks in the official baseline layout.

    No WAV is copied and no existing target directory is replaced.  The official
    baseline receives the original two-channel WAV files unchanged.
    """
    baseline = verify_pinned_reference(
        baseline_directory,
        repository=OFFICIAL_BASELINE_REPOSITORY,
        commit=OFFICIAL_BASELINE_COMMIT,
        required_file="01_train_2026t2.sh",
    )
    frame = pd.read_parquet(manifest_path)
    required = {"machine_type", "dataset_split", "condition", "relative_path"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {', '.join(missing)}")
    train = frame.loc[frame["dataset_split"] == "dev_train"]
    test = frame.loc[frame["dataset_split"] == "dev_test"]
    if train.empty or test.empty:
        raise ValueError("Development manifest must contain both dev_train and dev_test clips")
    if not (train["condition"] == "normal").all():
        raise ValueError("Official baseline training input must be normal-only")

    extracted = Path(data_root) / "raw" / "dcase2026" / "dev" / "extracted"
    target_root = baseline.directory / "data" / "dcase2026t2" / "dev_data" / "raw"
    created = 0
    reused = 0
    machine_types = tuple(sorted(str(value) for value in frame["machine_type"].unique()))
    for machine_type in machine_types:
        for directory_name, subset in (("train", train), ("test", test)):
            selected = subset.loc[subset["machine_type"] == machine_type]
            if selected.empty:
                raise ValueError(f"Missing {directory_name} samples for {machine_type}")
            source = extracted / machine_type / directory_name
            if not source.is_dir():
                raise FileNotFoundError(f"Audited source directory is missing: {source}")
            target = target_root / machine_type / directory_name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise FileExistsError(f"Existing link targets a different directory: {target}")
                reused += 1
            elif target.exists():
                raise FileExistsError(
                    f"Refusing to replace existing baseline data directory: {target}"
                )
            else:
                target.symlink_to(source, target_is_directory=True)
                created += 1
    return BaselineStagingSummary(
        machine_types=machine_types, created_links=created, reused_links=reused
    )


def run_official_development_baseline(
    *,
    baseline_directory: str | Path,
    official_python: str | Path,
    mode: BaselineMode,
    log_path: str | Path,
) -> None:
    """Run unmodified official scripts with the supplied isolated Python runtime."""
    baseline = verify_pinned_reference(
        baseline_directory,
        repository=OFFICIAL_BASELINE_REPOSITORY,
        commit=OFFICIAL_BASELINE_COMMIT,
        required_file="01_train_2026t2.sh",
    )
    python = Path(official_python)
    if not python.is_file():
        raise FileNotFoundError(f"Official baseline Python executable not found: {python}")
    scripts = ["01_train_2026t2.sh"]
    if mode in {"mse", "all"}:
        scripts.append("02a_test_2026t2.sh")
    if mode in {"mahala", "all"}:
        scripts.append("02b_test_2026t2.sh")
    output = Path(log_path)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite baseline log: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.pop("LD_LIBRARY_PATH", None)
    environment["PATH"] = str(python.parent) + os.pathsep + environment.get("PATH", "")
    with output.open("x", encoding="utf-8") as log:
        for script in scripts:
            subprocess.run(
                ["bash", script, "-d"],
                cwd=baseline.directory,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=True,
            )


def _run_git(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    command = ["git", *arguments]
    environment = os.environ.copy()
    # Conda's libffi can shadow Ubuntu's libffi and break git-remote-https.
    # Keep the official checkout independent from the caller's active Conda env.
    environment.pop("LD_LIBRARY_PATH", None)
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=True,
            env=environment,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or "no Git diagnostic was produced"
        raise ValueError(f"Git reference command failed ({' '.join(command)}): {detail}") from exc
