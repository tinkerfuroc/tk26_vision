#!/usr/bin/env bash
# Build tk26_vision packages: source venv + ROS, run colcon, fix shebangs.
#
# Forwards all arguments to colcon, e.g.:
#   ./src/tk26_vision/scripts/build.sh
#   ./src/tk26_vision/scripts/build.sh --packages-select pan_tilt kimi_api
#   ./src/tk26_vision/scripts/build.sh --packages-up-to object_detection_new
#
# Env overrides:
#   WS_ROOT   default: $HOME/tk25_ws
#   ROS_SETUP default: /opt/ros/humble/setup.bash

set -euo pipefail

WS_ROOT="${WS_ROOT:-$HOME/tk25_ws}"
VENV="$WS_ROOT/src/tk26_vision/.venv-vision-main"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
SCRIPTS_DIR="$WS_ROOT/src/tk26_vision/scripts"

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

"$SCRIPTS_DIR/fix_venv_shebangs.sh"
