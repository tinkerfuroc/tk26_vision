#!/usr/bin/env bash
# T1 (startup, no cameras): verify the camera server's three public services,
# status heartbeat, bounded captured_after handling, and process liveness.
#
# This is intentionally a manual/operator check.  It assumes the workspace has
# already been built and sourced, but does not require camera hardware.
set -o pipefail

SETUP="/home/tinker/tk25_ws/install/setup.bash"
if [[ ! -r "$SETUP" ]]; then
  echo "[camera_server_t1] FAIL: missing installed workspace: $SETUP" >&2
  exit 2
fi
# Colcon's generated setup scripts assume these are unset when entering a new
# workspace.  Also keep nounset disabled while they probe optional variables.
unset AMENT_CURRENT_PREFIX COLCON_CURRENT_PREFIX
export COLCON_TRACE="${COLCON_TRACE-}"
set +u
source "$SETUP"
set -u

export ROS_LOG_DIR="${ROS_LOG_DIR:-/tmp/camera_server_t1_logs}"
mkdir -p "$ROS_LOG_DIR"

FAIL=0
note() { echo "[camera_server_t1] $*"; }

check_status() {
  local name="$1" expected="$2" output="$3"
  if grep -Eq "status(:|=)[[:space:]]*${expected}([^0-9]|$)" <<<"$output"; then
    note "PASS: $name"
  else
    note "FAIL: $name — expected status ${expected}; output was: $output"
    FAIL=1
  fi
}

ros2 run camera_server camera_server_node --ros-args -r __node:=t1_camera_server &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Give discovery and service creation a moment, while keeping the failure path
# bounded if the executable cannot start.
sleep 3

OUT=$(timeout 10 ros2 service call /t1_camera_server/get_snapshot \
  tinker_vision_msgs_26/srv/GetCameraSnapshot '{}' 2>&1 || true)
check_status "snapshot returns NO_DATA" 1 "$OUT"

OUT=$(timeout 10 ros2 service call /t1_camera_server/get_point_cloud \
  tinker_vision_msgs_26/srv/GetCameraPointCloud '{include_color: false}' 2>&1 || true)
check_status "point cloud returns NO_DATA" 1 "$OUT"

OUT=$(timeout 10 ros2 service call /t1_camera_server/get_transform \
  tinker_vision_msgs_26/srv/GetTransform \
  '{target_frame: base_link, source_frame: map, timeout_sec: 0.2}' 2>&1 || true)
check_status "transform returns UNAVAILABLE" 1 "$OUT"

OUT=$(timeout 10 ros2 topic echo --no-daemon --once /t1_camera_server/status 2>&1 || true)
if grep -q "pair_seq" <<<"$OUT"; then
  note "PASS: status heartbeat publishes"
else
  note "FAIL: status heartbeat did not publish: $OUT"
  FAIL=1
fi
if grep -q "last_pair_received_at" <<<"$OUT"; then
  note "PASS: status heartbeat includes receive-time diagnostics"
else
  note "FAIL: status heartbeat missing last_pair_received_at: $OUT"
  FAIL=1
fi
if grep -Eq "color_age_sec:[[:space:]]*-1(\.0+)?" <<<"$OUT"; then
  note "PASS: empty-store color age is -1"
else
  note "FAIL: expected empty-store color_age_sec=-1: $OUT"
  FAIL=1
fi

# A far-future boundary must not return immediately with a stale/old pair and
# must remain bounded by the server's default wait timeout.
START=$SECONDS
OUT=$(timeout 15 ros2 service call /t1_camera_server/get_snapshot \
  tinker_vision_msgs_26/srv/GetCameraSnapshot \
  '{captured_after: {sec: 2000000000}}' 2>&1 || true)
ELAPSED=$((SECONDS - START))
check_status "far-future captured_after returns NO_DATA" 1 "$OUT"
if (( ELAPSED <= 6 )); then
  note "PASS: captured_after wait bounded (${ELAPSED}s)"
else
  note "FAIL: captured_after wait took ${ELAPSED}s"
  FAIL=1
fi

if kill -0 "$SERVER_PID" 2>/dev/null; then
  note "PASS: camera server remains alive"
else
  note "FAIL: camera server exited unexpectedly"
  FAIL=1
fi

exit "$FAIL"
