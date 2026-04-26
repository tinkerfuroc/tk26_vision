#!/usr/bin/env bash
# T3 — cross-node interaction tests. Cameras + servo expected live.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$HERE/lib.sh"
source_envs

SERVO_DEVICE="${SERVO_DEVICE:-/dev/ttyUSB1}"

section "T3.1 — feature_matching ↔ yolo_seg_default_node"
if have_api_key; then
    start_node yolo_default_t3 object_detection_new yolo_seg_default_node
    wait_for_service /object_detection 20 || fail "T3.1" "yolo default didn't advertise"
    start_node feature_matching_t3 kimi_api feature_matching
    if wait_for_service /feature_matching_service 15; then
        out="$LOG_DIR/T3.1.svcout"
        timeout 45 ros2 service call /feature_matching_service tinker_vision_msgs_26/srv/FeatureMatching \
            "{camera: 'orbbec', features: ['red bottle'], max_distance: 2.0, target_frame: 'base_link'}" 2>&1 | head -c 4000 >"$out" || true
        if grep -qE 'Traceback' "$out"; then
            fail "T3.1" "traceback in response"
        elif grep -qE 'status=[01]' "$out" && grep -qE 'centroids=' "$out"; then
            st=$(grep -oE 'status=[01]' "$out" | head -1)
            pass "T3.1 matching node talked to detection node ($st)"
        else
            fail "T3.1" "head: $(head -c 400 "$out")"
        fi
    else
        fail "T3.1" "feature_matching service not advertised"
    fi
    stop_all_nodes
else
    skip "T3.1" "no valid OPENROUTER_API_KEY"
fi

section "T3.2 — spot_on_shelf ↔ yolo_seg_node"
start_node yolo_custom_t3 object_detection_new yolo_seg_node
wait_for_service /object_detection_yolo 20 || fail "T3.2" "yolo custom didn't advertise"
start_node spot_on_shelf_t3 tk_vision_specialized spot_on_shelf_server
if wait_for_action /spot_on_shelf 15; then
    out="$LOG_DIR/T3.2.actout"
    goal='{shelf_left: {header: {frame_id: "map"}, pose: {position: {x: 1.0, y: 0.3, z: 0.5}, orientation: {w: 1.0}}}, shelf_right: {header: {frame_id: "map"}, pose: {position: {x: 1.0, y: -0.3, z: 0.5}, orientation: {w: 1.0}}}, shelf_heights: [0.3, 0.6], item_ids: ["bottle"]}'
    timeout 60 ros2 action send_goal /spot_on_shelf tinker_vision_msgs_26/action/SpotOnShelf "$goal" >"$out" 2>&1 || true
    if grep -qE 'Goal finished|Result:' "$out"; then
        pass "T3.2 action terminated (no client hang)"
    else
        fail "T3.2" "head $out:\n$(head -20 "$out")"
    fi
else
    fail "T3.2" "spot_on_shelf action not advertised"
fi
stop_all_nodes

section "T3.4 — feature_matching ↔ generalist_node"
if have_api_key; then
    start_node generalist_t3 object_detection_generalist generalist_node
    wait_for_service /object_detection_generalist 30 || fail "T3.4" "generalist did not advertise"
    # feature_matching's default detection_service is 'object_detection_generalist' post-migration.
    start_node feature_matching_t3.4 kimi_api feature_matching
    if wait_for_service /feature_matching_service 15; then
        out="$LOG_DIR/T3.4.svcout"
        timeout 90 ros2 service call /feature_matching_service tinker_vision_msgs_26/srv/FeatureMatching \
            "{camera: 'realsense', features: ['person'], max_distance: 3.0, target_frame: 'base_link'}" 2>&1 \
            | head -c 4000 >"$out" || true
        if grep -qE 'Traceback' "$out"; then
            fail "T3.4" "traceback in response"
        elif grep -qE 'status=[01]' "$out" && grep -qE 'centroids=' "$out"; then
            st=$(grep -oE 'status=[01]' "$out" | head -1)
            pass "T3.4 matching node talked to generalist ($st)"
        else
            fail "T3.4" "head: $(head -c 400 "$out")"
        fi
    else
        fail "T3.4" "feature_matching_service not advertised"
    fi
    stop_all_nodes
else
    skip "T3.4" "no valid OPENROUTER_API_KEY"
fi

section "T3.3 — pan_tilt stack TF + follow_head"
if [ -c "$SERVO_DEVICE" ]; then
    start_launch ctrl_t3 pan_tilt pan_tilt.launch.py "device:=$SERVO_DEVICE"
    sleep 3
    tf_ok=0
    if timeout 5 ros2 run tf2_ros tf2_echo base_link head_camera_link >"$LOG_DIR/T3.3.tf" 2>&1 & then
        tf_pid=$!; sleep 3; kill "$tf_pid" 2>/dev/null || true; wait "$tf_pid" 2>/dev/null || true
    fi
    grep -qE 'Translation|At time' "$LOG_DIR/T3.3.tf" && tf_ok=1
    if [ "$tf_ok" -eq 1 ]; then
        pass "T3.3 TF base_link→head_camera_link resolves"
    else
        fail "T3.3" "TF not present"
    fi
    start_node follow_head_t3 pan_tilt follow_head
    sleep 10
    # No error spam in either log
    if assert_log_nogrep "$LOG_DIR/ctrl_t3.log" 'Traceback|ERROR' && \
       assert_log_nogrep "$LOG_DIR/follow_head_t3.log" 'Traceback'; then
        pass "T3.3 no error spam after 10 s with empty scene"
    else
        fail "T3.3" "errors in logs (see ctrl_t3.log / follow_head_t3.log)"
    fi
    stop_all_nodes
else
    skip "T3.3" "$SERVO_DEVICE not present"
fi

summary
