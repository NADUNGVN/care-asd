"""Tests for configuration loading, validation, and hashing."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from care_asd.config import (
    CareASDConfig,
    config_hash,
    config_to_dict,
    default_config,
    load_config,
    save_config,
    validate_config,
)


def test_default_config_validates() -> None:
    cfg = default_config()
    validated = validate_config(cfg)
    assert isinstance(validated, CareASDConfig)
    assert validated.experiment.seed == 42
    assert validated.data.channel_map.near == 0
    assert validated.data.channel_map.far == 1


def test_config_hash_is_stable() -> None:
    cfg_a = default_config()
    cfg_b = default_config()
    assert config_hash(cfg_a) == config_hash(cfg_b)
    assert len(config_hash(cfg_a)) == 64


def test_config_hash_changes_with_seed() -> None:
    cfg_a = default_config()
    cfg_b = OmegaConf.merge(default_config(), OmegaConf.from_dotlist(["experiment.seed=99"]))
    assert config_hash(cfg_a) != config_hash(cfg_b)  # type: ignore[arg-type]


def test_load_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text(
        "experiment:\n  id: test_run\n  seed: 7\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    validated = validate_config(cfg)
    assert validated.experiment.id == "test_run"
    assert validated.experiment.seed == 7
    # Unspecified fields keep defaults
    assert validated.features.n_mels == 128


def test_load_config_with_overrides() -> None:
    cfg = load_config(None, overrides=["experiment.seed=123", "model.embedding_dim=64"])
    validated = validate_config(cfg)
    assert validated.experiment.seed == 123
    assert validated.model.embedding_dim == 64


def test_load_config_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_config(Path("does_not_exist_care_asd.yaml"))


def test_invalid_config_rejected() -> None:
    bad = {"experiment": {"seed": "not-an-int"}}
    with pytest.raises(ValidationError):
        CareASDConfig.model_validate(bad)


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        CareASDConfig.model_validate({"experiment": {"id": "x", "unknown_field": 1}})


def test_save_and_reload_roundtrip(tmp_path: Path) -> None:
    cfg = default_config()
    path = tmp_path / "out" / "cfg.yaml"
    save_config(cfg, path)
    assert path.exists()
    reloaded = load_config(path)
    assert config_hash(cfg) == config_hash(reloaded)


def test_config_to_dict_serializable() -> None:
    d = config_to_dict(default_config())
    assert isinstance(d, dict)
    assert "experiment" in d
    assert d["scoring"]["target_adaptation"]["lambda"] == 0.2


def test_project_default_yaml_loads() -> None:
    root = Path(__file__).resolve().parents[2]
    yaml_path = root / "configs" / "experiment" / "default.yaml"
    if not yaml_path.exists():
        pytest.skip("project default.yaml not present")
    cfg = load_config(yaml_path)
    validate_config(cfg)


def test_channel_map_defaults_match_dcase() -> None:
    cfg = validate_config(default_config())
    assert cfg.data.channel_map.near == 0
    assert cfg.data.channel_map.far == 1
