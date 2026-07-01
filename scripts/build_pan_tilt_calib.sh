#!/usr/bin/env bash
# Build the pan_tilt ROS2 package under .venv-calib (the pan-tilt extrinsic
# calibration venv) and re-shebang its calibration-workflow entry points so
# `ros2 run pan_tilt <calib node>` picks up the venv python.
#
# Why a dedicated venv? `calibrate_web` / `calibrate_collect` /
# `pan_tilt.calibration.*` need `cv2.aruco` (ChArUco detection), which only
# ships in `opencv-contrib-python` (the system/user `opencv-python` has no
# `aruco` module). `.venv-calib` holds opencv-contrib-python>=4.7 (the new
# ArucoDetector/CharucoDetector API the code targets) plus scipy / fastapi /
# uvicorn / aiofiles / pyserial — fully isolated from ~/.local.
#
# Provision the venv once (see src/tk26_vision/CLAUDE.md § pan_tilt calib venv):
#   cd src/tk26_vision
#   python3.10 -m venv .venv-calib --system-site-packages --symlinks
#   source .venv-calib/bin/activate && python -m pip install --upgrade pip wheel
#   python -m pip install --ignore-installed \
#       numpy==1.23.4 scipy==1.12.0 opencv-contrib-python==4.10.0.84 \
#       PyYAML fastapi 'uvicorn[standard]' aiofiles pyserial
#
# Mirrors src/tk26_vision/scripts/build_foundation_stereo.sh. Defaults to a
# single-package build to avoid cross-venv accidents; pass extra
# --packages-select / --packages-up-to args to override.
#
# NOTE: `follow_head` is intentionally NOT re-shebanged here — it needs
# ultralytics + torch, which this lean calibration venv does not carry. Leave
# it on the main vision venv / system python.

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
WS_ROOT="${WS_ROOT:-$(cd "$REPO_ROOT/../.." && pwd)}"
VENV="${VENV:-$REPO_ROOT/.venv-calib}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

# Interpreter for follow_head (ultralytics + torch). Override to point at the
# main vision venv once it exists, e.g.
#   FOLLOW_HEAD_PY=$REPO_ROOT/.venv-vision-main/bin/python3
FOLLOW_HEAD_PY="${FOLLOW_HEAD_PY:-/usr/bin/python3}"

# Calibration-workflow entry points that should run under .venv-calib.
# (follow_head deliberately excluded — see header note.)
CALIB_ENTRYPOINTS=(calibrate_web calibrate_collect controller state_publisher)

if [ ! -f "$VENV/bin/activate" ]; then
    echo "error: .venv-calib activate not found: $VENV/bin/activate" >&2
    echo "       provision it first — see the header of this script /" >&2
    echo "       src/tk26_vision/CLAUDE.md § pan_tilt calibration venv." >&2
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
    set -- --packages-select pan_tilt
fi

colcon build --symlink-install "$@"

ENTRY_DIR="$WS_ROOT/install/pan_tilt/lib/pan_tilt"
TARGET_PY="$VENV/bin/python3"

# Pin each entry point to its intended interpreter. colcon, having run under
# the active venv, would otherwise stamp *every* console script (follow_head
# included) with the venv python — so we re-pin deterministically here.
pin_shebang() {  # $1 = script path, $2 = interpreter
    local script="$1" interp="$2" first_line
    [ -f "$script" ] || return 0
    first_line="$(head -n 1 -- "$script" 2>/dev/null || true)"
    case "$first_line" in
        "#!"*python*)
            if [ "$first_line" != "#!$interp" ]; then
                sed -i "1c#!$interp" "$script"
                echo "patched: $script -> $interp"
            else
                echo "ok:      $script (already $interp)"
            fi
            ;;
    esac
}

if [ -d "$ENTRY_DIR" ]; then
    for name in "${CALIB_ENTRYPOINTS[@]}"; do
        pin_shebang "$ENTRY_DIR/$name" "$TARGET_PY"
    done
    pin_shebang "$ENTRY_DIR/follow_head" "$FOLLOW_HEAD_PY"
fi

echo "pan_tilt calibration build complete (venv: $VENV)"
echo "  .venv-calib: ${CALIB_ENTRYPOINTS[*]}"
echo "  follow_head pinned to: $FOLLOW_HEAD_PY (needs ultralytics/torch)."
