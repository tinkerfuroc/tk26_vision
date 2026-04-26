#!/usr/bin/env bash
# Wrapper: sources the vision venv and runs download_models.py.
# The venv carries ultralytics / torch / torchvision / mediapipe, which the
# system python3 does not have.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../.venv-vision-main"

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "error: venv not found at $VENV" >&2
    echo "       create it per src/tk26_vision/CLAUDE.md §Environment" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
exec python3 "$SCRIPT_DIR/download_models.py" "$@"
