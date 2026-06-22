#!/usr/bin/env bash
# Build the foundation_stereo ROS2 package under .venv-fs.
#
# Mirrors src/tk26_vision/scripts/build_monocular_depth.sh. Defaults to a
# single-package build to avoid cross-venv accidents; pass any extra
# --packages-select / --packages-up-to args to override.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
WS_ROOT="${WS_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"
VENV="${VENV:-$REPO_ROOT/.venv-fs}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "error: .venv-fs activate not found: $VENV/bin/activate" >&2
    echo "       provision it first — see src/foundation_stereo/README.md" >&2
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

if [ "$#" -eq 0 ]; then
    set -- --packages-select foundation_stereo
fi

colcon build --symlink-install "$@"

ENTRY_DIR="$WS_ROOT/install/foundation_stereo/lib/foundation_stereo"
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

echo "foundation_stereo build complete (venv: $VENV)"
