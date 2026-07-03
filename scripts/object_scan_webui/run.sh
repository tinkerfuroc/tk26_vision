#!/usr/bin/env bash
# Launch the object_scan tuning WebUI under .venv-vision-main.
# Usage: ./run.sh [--host 0.0.0.0] [--port 8000]
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
# .venv-vision-main lives at src/tk26_vision/.venv-vision-main
VENV="$HERE/../../.venv-vision-main"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "venv python not found at $PY — falling back to python3"
  PY="python3"
fi
echo "using: $PY"
exec "$PY" "$HERE/server.py" "$@"
