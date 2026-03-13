.PHONY: install install-deps install-hooks sync-requirements format check help

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: install-deps install-hooks  ## Install all dependencies and setup hooks
	@echo "✅ Installation complete!"

install-deps:  ## Install Python and Node dependencies
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

test:  ## Run tests
	uv run pytest tests/ -v

run:  ## Run the application
	uv run python main.py
