"""Reproducibility utilities: seeds, environment reports, experiment provenance."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EnvironmentReport:
    """Snapshot of the runtime environment for experiment provenance."""

    python_version: str
    platform: str
    platform_release: str
    machine: str
    processor: str
    cwd: str
    git_commit: str | None
    git_dirty: bool | None
    package_versions: dict[str, str]
    cuda_available: bool | None
    cuda_version: str | None
    torch_version: str | None
    hostname: str
    timestamp_utc: str
    env_vars_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize as JSON string."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass
class ExperimentProvenance:
    """Full provenance record for a single experiment run."""

    experiment_id: str
    seed: int
    config_hash: str
    manifest_hash: str | None
    git_commit: str | None
    environment: EnvironmentReport
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to a plain dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "seed": self.seed,
            "config_hash": self.config_hash,
            "manifest_hash": self.manifest_hash,
            "git_commit": self.git_commit,
            "environment": self.environment.to_dict(),
            "created_at_utc": self.created_at_utc,
            "notes": self.notes,
        }

    def save(self, path: str | Path) -> Path:
        """Write provenance JSON to disk without overwriting existing files."""
        out = Path(path)
        if out.exists():
            raise FileExistsError(f"Provenance file already exists (refusing to overwrite): {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return out


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch if available.

    Parameters
    ----------
    seed:
        Integer seed shared across libraries.
    deterministic:
        If True and torch is available, enable deterministic algorithms where possible.
    """
    if seed < 0:
        raise ValueError(f"Seed must be non-negative, got {seed}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Best-effort on older torch builds / unsupported ops
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def get_git_commit(repo_root: str | Path | None = None) -> str | None:
    """Return the current git HEAD SHA, or None if unavailable."""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def is_git_dirty(repo_root: str | Path | None = None) -> bool | None:
    """Return True if the working tree has uncommitted changes, or None if unknown."""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            cwd=cwd,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return bool(result.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return None


def _safe_package_version(name: str) -> str | None:
    """Best-effort package version lookup."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def _collect_package_versions() -> dict[str, str]:
    """Collect versions of key CARE-ASD dependencies."""
    names = [
        "care-asd",
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "torchaudio",
        "omegaconf",
        "pydantic",
        "soundfile",
        "scikit-learn",
        "pyarrow",
    ]
    versions: dict[str, str] = {}
    for name in names:
        ver = _safe_package_version(name)
        if ver is not None:
            versions[name] = ver
    return versions


def _env_vars_hash(keys: list[str] | None = None) -> str:
    """Hash selected environment variables (names only presence + non-secret values)."""
    interesting = keys or [
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "PYTHONHASHSEED",
        "CARE_ASD_DATA_ROOT",
    ]
    payload = {k: os.environ.get(k, "") for k in sorted(interesting)}
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def collect_environment_report(repo_root: str | Path | None = None) -> EnvironmentReport:
    """Build a full environment report for provenance logging."""
    torch_version: str | None = None
    cuda_available: bool | None = None
    cuda_version: str | None = None

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            cuda_version = getattr(torch.version, "cuda", None)
    except ImportError:
        pass

    return EnvironmentReport(
        python_version=sys.version.replace("\n", " "),
        platform=platform.system(),
        platform_release=platform.release(),
        machine=platform.machine(),
        processor=platform.processor() or "unknown",
        cwd=str(Path.cwd()),
        git_commit=get_git_commit(repo_root),
        git_dirty=is_git_dirty(repo_root),
        package_versions=_collect_package_versions(),
        cuda_available=cuda_available,
        cuda_version=cuda_version,
        torch_version=torch_version,
        hostname=platform.node(),
        timestamp_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        env_vars_hash=_env_vars_hash(),
    )


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file in chunks."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def content_hash(data: bytes | str) -> str:
    """SHA-256 of in-memory content."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()
