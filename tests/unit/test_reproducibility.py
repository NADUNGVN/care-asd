"""Tests for seed control, hashing, and environment provenance."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from care_asd.reproducibility import (
    ExperimentProvenance,
    collect_environment_report,
    content_hash,
    file_sha256,
    get_git_commit,
    set_seed,
)


def test_set_seed_reproducible_python_random() -> None:
    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_set_seed_different_seeds_differ() -> None:
    set_seed(1)
    a = [random.random() for _ in range(5)]
    set_seed(2)
    b = [random.random() for _ in range(5)]
    assert a != b


def test_set_seed_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        set_seed(-1)


def test_set_seed_numpy_if_available() -> None:
    np = pytest.importorskip("numpy")
    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    assert (a == b).all()


def test_set_seed_torch_if_available() -> None:
    torch = pytest.importorskip("torch")
    set_seed(7)
    a = torch.rand(4)
    set_seed(7)
    b = torch.rand(4)
    assert torch.allclose(a, b)


def test_content_hash_stable() -> None:
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")
    assert content_hash(b"abc") == content_hash("abc")


def test_file_sha256(tmp_path: Path) -> None:
    p = tmp_path / "blob.bin"
    p.write_bytes(b"care-asd-fixture")
    h1 = file_sha256(p)
    h2 = file_sha256(p)
    assert h1 == h2
    assert len(h1) == 64


def test_environment_report_structure() -> None:
    report = collect_environment_report()
    d = report.to_dict()
    assert "python_version" in d
    assert "package_versions" in d
    assert "timestamp_utc" in d
    assert isinstance(d["package_versions"], dict)
    # JSON serializable
    json.loads(report.to_json())


def test_provenance_save_refuses_overwrite(tmp_path: Path) -> None:
    report = collect_environment_report()
    prov = ExperimentProvenance(
        experiment_id="exp_test",
        seed=42,
        config_hash="abc",
        manifest_hash=None,
        git_commit=get_git_commit(),
        environment=report,
    )
    path = tmp_path / "prov.json"
    prov.save(path)
    assert path.exists()
    with pytest.raises(FileExistsError):
        prov.save(path)


def test_git_commit_is_string_or_none() -> None:
    commit = get_git_commit()
    assert commit is None or (isinstance(commit, str) and len(commit) >= 7)
