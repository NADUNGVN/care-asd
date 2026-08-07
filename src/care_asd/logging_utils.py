"""Structured logging setup for CARE-ASD."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

_CONFIGURED = False


def setup_logging(
    level: LogLevel | str = "INFO",
    *,
    log_file: str | Path | None = None,
    name: str = "care_asd",
    force: bool = False,
) -> logging.Logger:
    """Configure root CARE-ASD logger with console (and optional file) handlers.

    Parameters
    ----------
    level:
        Logging level name.
    log_file:
        Optional path for a file handler. Parent dirs are created.
    name:
        Logger name (default package logger).
    force:
        Reconfigure even if already set up.
    """
    global _CONFIGURED

    logger = logging.getLogger(name)
    numeric_level = getattr(logging, str(level).upper(), logging.INFO)

    if _CONFIGURED and not force:
        logger.setLevel(numeric_level)
        return logger

    logger.handlers.clear()
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _CONFIGURED = True
    return logger


def get_logger(name: str = "care_asd") -> logging.Logger:
    """Return a named logger under the CARE-ASD hierarchy."""
    if name == "care_asd" or name.startswith("care_asd."):
        return logging.getLogger(name)
    return logging.getLogger(f"care_asd.{name}")
