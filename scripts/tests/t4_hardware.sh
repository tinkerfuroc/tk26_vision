#!/usr/bin/env bash
# T4 — hardware-in-the-loop cases. Invoke subcommands individually:
#   t4_hardware.sh servo_motion
#   t4_hardware.sh shelf_scene
#   t4_hardware.sh person
#   t4_hardware.sh all        (default)
# Requires: servo at /dev/ttyUSB1, live cameras. Shelf/person cases need staged scenes.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$HERE/lib.sh"
source_envs

SERVO_DEVICE="${SERVO_DEVICE:-/dev/ttyUSB1}"

t4_servo_motion() {
    section "T4.1 — servo commanded motion"
    if [ ! -c "$SERVO_DEVICE" ]; then skip "T4.1" "$SERVO_DEVICE not present"; return; fi
    start_node ctrl_t4.1 pan_tilt ctrl --ros-args -p "device:=$SERVO_DEVICE"
    sleep 3
    printf '  -> commanding pan=0.3, tilt=0.0. Watch the servo move.\n'
    ros2 topic pub --once /pan_tilt_ctrl_modify tinker_vision_msgs/msg/PanTiltCtrl '{pan: 0.3, tilt: 0.0}' >/dev/null 2>&1 || true
    sleep 2
    timeout 3 ros2 run tf2_ros tf2_echo base_link camera_link >"$LOG_DIR/T4.1_after.tf" 2>&1 || true
    printf '  -> returning to home (pan=0.0, tilt=0.0)\n'
    ros2 topic pub --once /pan_tilt_ctrl_modify tinker_vision_msgs/msg/PanTiltCtrl '{pan: 0.0, tilt: 0.0}' >/dev/null 2>&1 || true
    sleep 2
    if grep -qE 'Translation' "$LOG_DIR/T4.1_after.tf"; then
        pass "T4.1 (visual inspection required: did the servo physically move?)"
    else
        fail "T4.1" "TF not resolving after command"
    fi
    stop_all_nodes
}

t4_servo_tracking() {
    section "T4.2 — servo tracking (visual)"
    if [ ! -c "$SERVO_DEVICE" ]; then skip "T4.2" "$SERVO_DEVICE not present"; return; fi
    start_node ctrl_t4.2 pan_tilt ctrl --ros-args -p "device:=$SERVO_DEVICE"
    sleep 2
    start_node follow_head_t4.2 pan_tilt follow_head
    wait_for_action /follow_head_action 15 || { fail "T4.2" "follow_head_action missing"; stop_all_nodes; return; }
    printf '  -> wave a hand in front of the orbbec camera for ~15 s now.\n'
    ros2 action send_goal /follow_head_action tinker_vision_msgs/action/FollowHeadAction '{start_following: true}' --feedback >"$LOG_DIR/T4.2.actout" 2>&1 &
    ap=$!
    sleep 15
    kill "$ap" 2>/dev/null || true; wait "$ap" 2>/dev/null || true
    if grep -qE 'Feedback:' "$LOG_DIR/T4.2.actout" && assert_log_nogrep "$LOG_DIR/follow_head_t4.2.log" 'Traceback'; then
        pass "T4.2 (visual inspection required: did the head track?)"
    else
        fail "T4.2" "no feedback or traceback present"
    fi
    stop_all_nodes
}

t4_shelf_scene() {
    section "T4.3 — populated shelf"
    printf '  -> stage 2–3 objects at two heights in front of orbbec, then press enter.\n'
    read -r _ || true
    start_node yolo_custom_t4.3 object_detection_new yolo_seg_node
    wait_for_service /object_detection_yolo 20 || { fail "T4.3" "yolo didn't advertise"; stop_all_nodes; return; }
    start_node spot_on_shelf_t4.3 tk_vision_specialized spot_on_shelf_server
    wait_for_action /spot_on_shelf 15 || { fail "T4.3" "spot_on_shelf didn't advertise"; stop_all_nodes; return; }
    out="$LOG_DIR/T4.3.actout"
    goal='{shelf_left: {header: {frame_id: "base_link"}, pose: {position: {x: 0.8, y: 0.3, z: 0.5}, orientation: {w: 1.0}}}, shelf_right: {header: {frame_id: "base_link"}, pose: {position: {x: 0.8, y: -0.3, z: 0.5}, orientation: {w: 1.0}}}, shelf_heights: [0.3, 0.6], item_ids: ["bottle", "cup"]}'
    timeout 60 ros2 action send_goal /spot_on_shelf tinker_vision_msgs_26/action/SpotOnShelf "$goal" >"$out" 2>&1 || true
    if grep -qE 'item_height_grids=\[.+\]|item_height_grids: \[.+\]' "$out"; then
        pass "T4.3 non-empty grids (verify values match your staging)"
    else
        fail "T4.3" "empty grids or no result: $(head -30 "$out")"
    fi
    stop_all_nodes
}

t4_person() {
    section "T4.4 — person tracking"
    printf '  -> operator: walk into the orbbec view in ~5 s.\n'
    sleep 5
    start_node person_track_t4.4 vision_track person_track_server
    wait_for_action /track_person 30 || { fail "T4.4" "track_person missing"; stop_all_nodes; return; }
    out="$LOG_DIR/T4.4.actout"
    timeout 30 ros2 action send_goal -f /track_person tinker_vision_msgs_26/action/TrackPerson \
        "{target_frame: 'map', return_rgb_img: false, return_depth_img: false, return_segment: false, debug: false}" >"$out" 2>&1 &
    ap=$!
    sleep 25
    printf '  -> now occlude briefly and re-emerge\n'
    sleep 5
    kill "$ap" 2>/dev/null || true; wait "$ap" 2>/dev/null || true
    if grep -qE 'target_lost=False|target_lost: false' "$out"; then
        pass "T4.4 target acquired (ReID: inspect target_track_id stability in $out)"
    else
        fail "T4.4" "target never acquired: $(head -30 "$out")"
    fi
    stop_all_nodes
}

case "${1:-all}" in
    servo_motion) t4_servo_motion ;;
    servo_tracking) t4_servo_tracking ;;
    shelf_scene) t4_shelf_scene ;;
    person) t4_person ;;
    all) t4_servo_motion; t4_servo_tracking; t4_shelf_scene; t4_person ;;
    *) printf 'usage: %s {servo_motion|servo_tracking|shelf_scene|person|all}\n' "$0"; exit 2 ;;
esac

summary
