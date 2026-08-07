.PHONY: help sync test lint typecheck format check cli clean pre-commit

help:
	@echo "CARE-ASD development targets"
	@echo "  make sync        Install dependencies with uv"
	@echo "  make test        Run unit tests"
	@echo "  make lint        Run ruff check"
	@echo "  make typecheck   Run mypy"
	@echo "  make format      Format with ruff"
	@echo "  make check       lint + typecheck + test"
	@echo "  make cli         Show CLI help"
	@echo "  make pre-commit  Install pre-commit hooks"
	@echo "  make clean       Remove caches and build artifacts"

sync:
	uv sync --extra dev --extra torch

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

format:
	uv run ruff format .
	uv run ruff check --fix .

check: lint typecheck test

cli:
	uv run care-asd --help

pre-commit:
	uv run pre-commit install

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
