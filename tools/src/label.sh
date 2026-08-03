#!/usr/bin/env bash
set -euo pipefail

if ! uv pip show label-studio >/dev/null 2>&1; then
    echo "Error: Current uv environment has not installed label-studio" >&2
    echo "Please run the following command to install:" >&2
    echo "  uv pip install label-studio" >&2
    echo "Or if using uv project management:" >&2
    echo "  uv add label-studio" >&2
    exit 1
fi

uv run label-studio start