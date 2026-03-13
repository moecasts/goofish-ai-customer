.PHONY: install install-dev install-deps install-hooks sync-requirements format check help test test-cov test-parallel test-failed test-debug clean-cov run

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: install-dev install-hooks  ## Install all dependencies and setup hooks
	@echo "✅ Installation complete!"

install-dev:  ## Install development dependencies
	uv pip install -e ".[dev]"
	uv run playwright install chromium

install-deps:  ## Install Python and Node dependencies (legacy)
	uv pip sync requirements.txt
	uv run playwright install chromium

install-hooks:  ## Install git hooks
	uv run lefthook install

sync-requirements:  ## Generate requirements.txt from pyproject.toml
	uv pip compile pyproject.toml -o requirements.txt

format:  ## Format code with ruff
	uv run ruff format .

check:  ## Check code with ruff
	uv run ruff check .

test:  ## Run all tests
	uv run pytest tests/ -v

test-cov:  ## Run tests with HTML coverage report
	uv run pytest tests/ -v --cov=. --cov-report=html:htmlcov --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/index.html"

test-parallel:  ## Run tests in parallel
	uv run pytest tests/ -v -n auto

test-failed:  ## Run only failed tests from last run
	uv run pytest tests/ -v --lf

test-debug:  ## Run tests in debug mode (stop on first failure)
	uv run pytest tests/ -v -x

clean-cov:  ## Clean coverage files
	@rm -rf htmlcov/ .coverage
	@echo "🧹 Coverage files cleaned"

run:  ## Run the application
	uv run python main.py
