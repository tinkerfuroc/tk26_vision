#!/usr/bin/env bash
# Rewrite colcon-installed entry-point shebangs under install/<pkg>/lib/<pkg>/
# so they invoke the tk26_vision venv python. Idempotent — safe to re-run.
#
# Why: colcon spawns `setup.py install` with its own Python (/usr/bin/python3
# when using system colcon), which becomes the baked shebang. Those scripts
# can't then import packages that live only in the venv (openai, dotenv,
# ultralytics, pyserial, ...). This patch rewrites the shebangs so `ros2 run`
# invokes the venv python instead.
#
# Usage:
#   ./scripts/fix_venv_shebangs.sh
#   WS_ROOT=/path VENV_PY=/path/to/venv/bin/python ./fix_venv_shebangs.sh
#   PACKAGES="pan_tilt kimi_api" ./fix_venv_shebangs.sh

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
FALLBACK_VENV_PY="$(cd "$REPO_ROOT/../.." && pwd)/src/tk26_vision/.venv-vision-main/bin/python"

WS_ROOT="${WS_ROOT:-$REPO_ROOT}"
VENV_PY="${VENV_PY:-$WS_ROOT/.venv-vision-main/bin/python}"
if [ ! -x "$VENV_PY" ] && [ -x "$FALLBACK_VENV_PY" ]; then
    VENV_PY="$FALLBACK_VENV_PY"
fi

# Default package list; override via `PACKAGES="a b c" ./fix_venv_shebangs.sh`
DEFAULT_PACKAGES=(
    object_detection_new
    object_detection_generalist
    vision_util
    pan_tilt
    kimi_api
    vision_track
    tk_vision_specialized
    restaurant_nav_test_web
    handeye_calib
)
if [ -n "${PACKAGES:-}" ]; then
    # shellcheck disable=SC2206
    PKGS=(${PACKAGES})
else
    PKGS=("${DEFAULT_PACKAGES[@]}")
fi

if [ ! -x "$VENV_PY" ]; then
    echo "error: venv python not executable: $VENV_PY" >&2
    exit 1
fi

count=0
skipped=0
for pkg in "${PKGS[@]}"; do
    dir="$WS_ROOT/install/$pkg/lib/$pkg"
    if [ ! -d "$dir" ]; then
        continue
    fi
    for script in "$dir"/*; do
        [ -f "$script" ] || continue
        first=$(head -1 "$script" 2>/dev/null || true)
        case "$first" in
            \#\!*python*)
                if [ "$first" = "#!$VENV_PY" ]; then
                    skipped=$((skipped + 1))
                else
                    sed -i "1s|^#!.*|#!$VENV_PY|" "$script"
                    echo "patched: $script"
                    count=$((count + 1))
                fi
                ;;
        esac
    done
done

echo "done — $count script(s) updated, $skipped already correct"
