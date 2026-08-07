"""Tests for logging setup."""

from __future__ import annotations

import logging
from pathlib import Path

from care_asd.logging_utils import get_logger, setup_logging


def test_setup_logging_returns_logger() -> None:
    logger = setup_logging("INFO", force=True)
    assert isinstance(logger, logging.Logger)
    assert logger.name == "care_asd"


def test_setup_logging_file_handler(tmp_path: Path) -> None:
    log_path = tmp_path / "logs" / "test.log"
    logger = setup_logging("DEBUG", log_file=log_path, force=True)
    logger.info("hello care-asd")
    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "hello care-asd" in text


def test_get_logger_nested_name() -> None:
    logger = get_logger("config")
    assert logger.name == "care_asd.config"
