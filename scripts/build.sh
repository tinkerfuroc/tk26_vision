#!/usr/bin/env bash
# Build tk26_vision packages: source venv + ROS, run colcon, fix shebangs.
#
# Forwards all arguments to colcon, e.g.:
#   ./scripts/build.sh
#   ./scripts/build.sh --packages-select pan_tilt kimi_api
#   ./scripts/build.sh --packages-up-to object_detection_new
#
# Env overrides:
#   WS_ROOT   default: when this repo is embedded at <ws>/src/tk26_vision and
#             <ws>/tkbuild exists, build.sh DELEGATES to `tkbuild tk26_vision`
#             (artifacts land in the canonical <ws>/install tree); otherwise
#             repo root (works for the standalone worktree). Set WS_ROOT to
#             any other path (e.g. WS_ROOT=$PWD) to force the in-repo build.
#   ROS_SETUP default: /opt/ros/humble/setup.bash (unused when delegating —
#             tkbuild sources ROS itself)

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
FALLBACK_VENV="$(cd "$REPO_ROOT/../.." && pwd)/src/tk26_vision/.venv-vision-main"

# When this repo sits embedded at <ws>/src/tk26_vision inside a tkbuild-managed
# workspace, an in-repo colcon run would create a second src/tk26_vision/
# {build,install} tree that tkbuild never updates — it goes permanently stale
# and reads as "the install" to anyone standing in the repo (2026-07-02
# kimi_api incident). Delegate to the outer tkbuild instead: artifacts land in
# the single canonical <ws>/install tree, and tkbuild's per-package venv
# handling (foundation_stereo/.venv-fs, monocular_depth/.venv-da3) applies.
# Escape hatch: set WS_ROOT to any other path to force the in-repo build.
PARENT_DIR="$(dirname "$REPO_ROOT")"
OUTER_WS="$(dirname "$PARENT_DIR")"
if [ "$(basename "$PARENT_DIR")" = "src" ] && [ -x "$OUTER_WS/tkbuild" ] \
    && [ "${WS_ROOT:-$OUTER_WS}" = "$OUTER_WS" ]; then
    echo "build.sh: embedded in $OUTER_WS — delegating to: tkbuild tk26_vision $*"
    cd "$OUTER_WS"
    "$OUTER_WS/tkbuild" tk26_vision "$@"
    # setuptools never removes dropped console scripts from an existing
    # install tree; prune the retired pan_tilt legacy entrypoint here too.
    rm -f "$OUTER_WS/install/pan_tilt/lib/pan_tilt/ctrl"
    exit 0
fi

WS_ROOT="${WS_ROOT:-$REPO_ROOT}"
VENV="${VENV:-$WS_ROOT/.venv-vision-main}"
if [ ! -f "$VENV/bin/activate" ] && [ -f "$FALLBACK_VENV/bin/activate" ]; then
    VENV="$FALLBACK_VENV"
fi
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"

if [ ! -f "$VENV/bin/activate" ]; then
    echo "error: venv activate not found: $VENV/bin/activate" >&2
    exit 1
fi
if [ ! -f "$ROS_SETUP" ]; then
    echo "error: ROS setup not found: $ROS_SETUP" >&2
    exit 1
fi

# ROS setup scripts reference unset vars (AMENT_TRACE_SETUP_FILES etc.),
# which would trip `set -u`. Disable nounset for the sourcing block only.
set +u
# shellcheck disable=SC1091
source "$VENV/bin/activate"
# shellcheck disable=SC1091
source "$ROS_SETUP"
set -u
export ROS2_PTH_WARNED=1

cd "$WS_ROOT"
colcon build --symlink-install "$@"

# `setuptools` will not remove dropped console scripts from an existing install
# tree, so prune the retired pan_tilt legacy entrypoint after rebuilds.
rm -f "$WS_ROOT/install/pan_tilt/lib/pan_tilt/ctrl"

"$SCRIPTS_DIR/fix_venv_shebangs.sh"
