#!/usr/bin/env bash
set -euo pipefail

# Run Python fixture generator via uv from the repo root.

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is not installed or not on PATH." >&2
  echo "Install uv: https://github.com/astral-sh/uv" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/reference"

# Ensure deps are installed for the reference project
uv sync

# Run inside the Python project so uv discovers pyproject.toml
uv run python gen_fixtures.py --output-dir "$SCRIPT_DIR/fixtures" --seed 42

echo "Generating tokenizer fixtures"
uv run python gen_tokenizer_fixtures.py --output-dir "$SCRIPT_DIR/fixtures" --seed 42