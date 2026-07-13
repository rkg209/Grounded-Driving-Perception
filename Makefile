.PHONY: help install test lint fmt smoke fixtures

help:
	@echo "install   uv sync (create venv, resolve deps)"
	@echo "test      run pytest"
	@echo "lint      ruff check"
	@echo "fmt       ruff format + fix"
	@echo "smoke     end-to-end smoke test (must exit 0)"
	@echo "fixtures  regenerate the synthetic mini-BDD fixtures"

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

fmt:
	uv run ruff format . && uv run ruff check --fix .

smoke:
	bash scripts/smoke.sh

fixtures:
	uv run python scripts/make_fixtures.py
