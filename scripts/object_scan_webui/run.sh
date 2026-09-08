#!/usr/bin/env bash
# Launch the object_scan tuning WebUI under .venv-vision-main.
# Usage: ./run.sh [--host 0.0.0.0] [--port 8000]
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HERE/../../.venv-vision-main"          # src/tk26_vision/.venv-vision-main
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "venv python not found at $PY — falling back to python3"
  PY="python3"
fi

# Source ROS + the workspace install so the "Capture from robot camera"
# (rclpy subprocess) has a working ROS runtime. Best-effort; the webcam /
# upload / scan paths work without ROS.
source /opt/ros/humble/setup.bash 2>/dev/null || true
WS="$HERE/../../../.."                          # tk25_ws root
[ -f "$WS/install/setup.bash" ] && source "$WS/install/setup.bash" 2>/dev/null || true

echo "using: $PY (ROS_DISTRO=${ROS_DISTRO:-unset})"
exec "$PY" "$HERE/server.py" "$@"
