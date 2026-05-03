#!/usr/bin/env bash
# Build the monocular_depth ROS2 package under the dedicated .venv-da3 env.
#
# Why this exists: depth_anything_3 pins numpy<2 and the shared
# .venv-vision-main has numpy 2.2.6 (torch 2.11 / scipy / ultralytics need
# the 2.x ABI). Installing DA3 into the shared venv would cascade-break
# every other vision node, so monocular_depth lives in its own venv.
#
# Forwards extra args to colcon, e.g.:
#   ./scripts/build_monocular_depth.sh
#   ./scripts/build_monocular_depth.sh --packages-up-to monocular_depth
#
# After colcon, this script rewrites the entry-point shebang to point at
# .venv-da3's python so `ros2 run monocular_depth monocular_depth_pc`
# resolves depth_anything_3.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
FALLBACK_VENV="$(cd "$REPO_ROOT/../.." && pwd)/src/tk26_vision/.venv-da3"

WS_ROOT="${WS_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"
VENV="${VENV:-$REPO_ROOT/.venv-da3}"
if [ ! -f "$VENV/bin/activate" ] && [ -f "$FALLBACK_VENV/bin/activate" ]; then
    VENV="$FALLBACK_VENV"
fi
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "error: .venv-da3 activate not found: $VENV/bin/activate" >&2
    echo "       provision it first — see src/monocular_depth/README.md" >&2
    exit 1
fi
if [ ! -f "$ROS_SETUP" ]; then
    echo "error: ROS setup not found: $ROS_SETUP" >&2
    exit 1
fi

set +u
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# shellcheck disable=SC1091
source "$ROS_SETUP"
set -u
export ROS2_PTH_WARNED=1

cd "$WS_ROOT"

# Default to a single-package build to avoid cross-venv accidents — caller
# can override with --packages-select, --packages-up-to, etc.
if [ "$#" -eq 0 ]; then
    set -- --packages-select monocular_depth
fi

colcon build --symlink-install "$@"

# Re-shebang the entry-point script(s) installed for monocular_depth.
ENTRY_DIR="$WS_ROOT/install/monocular_depth/lib/monocular_depth"
TARGET_PY="$VENV/bin/python3"
if [ -d "$ENTRY_DIR" ]; then
    for script in "$ENTRY_DIR"/*; do
        [ -f "$script" ] || continue
        first_line="$(head -n 1 -- "$script" 2>/dev/null || true)"
        case "$first_line" in
            "#!"*python*)
                if [ "$first_line" != "#!$TARGET_PY" ]; then
                    sed -i "1c#!$TARGET_PY" "$script"
                    echo "patched: $script -> $TARGET_PY"
                fi
                ;;
        esac
    done
fi

echo "monocular_depth build complete (venv: $VENV)"
