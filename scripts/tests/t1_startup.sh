#!/usr/bin/env bash
# T1 — node startup. Each node is launched in the background; after a
# bounded wait we check its advertised services/actions, then SIGTERM.
# Cameras not required.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$HERE/lib.sh"
source_envs

SERVO_DEVICE="${SERVO_DEVICE:-/dev/ttyUSB1}"

# Usage: t1_check <tag> <expect_kind:service|action> <name> <pkg> <entry> [ros2 run args...]
t1_check() {
    local tag="$1" kind="$2" name="$3" pkg="$4" entry="$5"
    shift 5
    start_node "$tag" "$pkg" "$entry" "$@"
    local ok=0
    if [ "$kind" = service ]; then
        wait_for_service "$name" 10 && ok=1
    else
        wait_for_action "$name" 10 && ok=1
    fi
    if [ "$ok" -eq 1 ]; then
        pass "$tag advertises $kind $name"
    else
        # Show the tail of the log to help debug
        local tail_log
        tail_log=$(tail -5 "$LAST_LOG" 2>/dev/null | tr '\n' '|')
        fail "$tag" "$kind $name not found (log tail: $tail_log)"
    fi
    stop_all_nodes
}

# Usage: t1_check_multi <tag> <pkg> <entry> — verify multiple services+actions after one startup
t1_check_multi() {
    local tag="$1" pkg="$2" entry="$3"
    shift 3
    # Remaining args are "s:<srv>" or "a:<act>" expected, up to a "--" sentinel; then ros2 run args.
    local expected=() ros_args=()
    local seen_sep=0
    for a in "$@"; do
        if [ "$a" = "--" ]; then seen_sep=1; continue; fi
        if [ "$seen_sep" -eq 0 ]; then expected+=("$a"); else ros_args+=("$a"); fi
    done
    start_node "$tag" "$pkg" "$entry" "${ros_args[@]:-}"
    local any_bad=0
    for e in "${expected[@]}"; do
        local kind="${e%%:*}" name="${e##*:}"
        local ok=0
        if [ "$kind" = s ]; then wait_for_service "$name" 10 && ok=1
        else wait_for_action "$name" 10 && ok=1; fi
        if [ "$ok" -eq 1 ]; then
            pass "$tag advertises ${kind/s/service}${kind/a/action} $name"
        else
            local tail_log; tail_log=$(tail -5 "$LAST_LOG" 2>/dev/null | tr '\n' '|')
            fail "$tag" "$kind:$name not found (log tail: $tail_log)"
            any_bad=1
        fi
    done
    stop_all_nodes
}

section "T1.1 — yolo_seg_node advertises /object_detection_yolo"
t1_check T1.1 service /object_detection_yolo object_detection_new yolo_seg_node

section "T1.2 — yolo_seg_default_node advertises /object_detection"
t1_check T1.2 service /object_detection object_detection_new yolo_seg_default_node

section "T1.3 — door_detection advertises /door_detection_srv"
t1_check T1.3 service /door_detection_srv vision_util door_detection

section "T1.4 — get_point_cloud advertises /get_point_cloud_service"
t1_check T1.4 service /get_point_cloud_service vision_util get_point_cloud

section "T1.5 — pan_tilt low-level stack positive + negative cases"
# Positive: real device
if [ -c "$SERVO_DEVICE" ]; then
    start_launch T1.5_pos pan_tilt pan_tilt.launch.py "device:=$SERVO_DEVICE"
    sleep 3
    if kill -0 "${NODE_PIDS[-1]}" 2>/dev/null; then
        # Check TF chain appears
        if timeout 5 ros2 run tf2_ros tf2_echo base_link head_camera_link >"$LOG_DIR/t1.5_tf.log" 2>&1 &
        then
            tf_pid=$!; sleep 3; kill "$tf_pid" 2>/dev/null || true; wait "$tf_pid" 2>/dev/null || true
        fi
        if grep -qE 'Translation|At time' "$LOG_DIR/t1.5_tf.log"; then
            pass "T1.5 positive: stack alive on $SERVO_DEVICE, TF base_link→head_camera_link resolves"
        else
            fail "T1.5 positive" "TF chain not published (see $LOG_DIR/t1.5_tf.log)"
        fi
    else
        fail "T1.5 positive" "stack died within 3 s on $SERVO_DEVICE (log: $LAST_LOG)"
    fi
    stop_all_nodes
else
    skip "T1.5 positive" "$SERVO_DEVICE not present"
fi

# Negative: unplugged path
start_node T1.5_neg pan_tilt controller --ros-args -p device:=/dev/ttyUSB_nonexistent
sleep 3
if grep -qE 'SerialException|could not open port' "$LAST_LOG"; then
    pass "T1.5 negative: clean SerialException on missing device"
else
    fail "T1.5 negative" "expected SerialException; log tail: $(tail -5 "$LAST_LOG" | tr '\n' '|')"
fi
stop_all_nodes

section "T1.6 — follow_head advertises service + action"
t1_check_multi T1.6 pan_tilt follow_head s:/follow_head_service a:/follow_head_action --

section "T1.7 — feature_recognition: negative + positive"
# Negative: no key (and no .env reachable)
start_node T1.7_neg kimi_api feature_recognition
sleep 4
if grep -qE 'OPENROUTER_API_KEY is not set|RuntimeError' "$LAST_LOG"; then
    pass "T1.7 negative: RuntimeError about missing key"
else
    # If user has a real .env in workspace, the node may actually start — that's still acceptable as long as it doesn't crash.
    if wait_for_service /feature_extraction_service 3; then
        skip "T1.7 negative" "workspace .env provided key; cannot exercise negative path here"
    else
        fail "T1.7 negative" "no service and no RuntimeError (log tail: $(tail -5 "$LAST_LOG" | tr '\n' '|'))"
    fi
fi
stop_all_nodes

# Positive: smoke key
OPENROUTER_API_KEY=smoke start_node T1.7_pos kimi_api feature_recognition
t1_services=( /feature_extraction_service /seat_recommend_service )
any_bad=0
for s in "${t1_services[@]}"; do
    if ! wait_for_service "$s" 10; then fail "T1.7 positive" "$s not found"; any_bad=1; fi
done
[ "$any_bad" -eq 0 ] && pass "T1.7 positive: feature_extraction + seat_recommend services"
stop_all_nodes

section "T1.8 — feature_matching: negative + positive"
start_node T1.8_neg kimi_api feature_matching
sleep 4
if grep -qE 'OPENROUTER_API_KEY is not set|RuntimeError' "$LAST_LOG"; then
    pass "T1.8 negative"
elif wait_for_service /feature_matching_service 2; then
    skip "T1.8 negative" "workspace .env provided key"
else
    fail "T1.8 negative" "no RuntimeError, no service"
fi
stop_all_nodes
OPENROUTER_API_KEY=smoke start_node T1.8_pos kimi_api feature_matching
if wait_for_service /feature_matching_service 10; then
    pass "T1.8 positive"
else
    fail "T1.8 positive" "service not advertised"
fi
stop_all_nodes

section "T1.9 — grocery_categorize: negative + positive"
start_node T1.9_neg kimi_api grocery_categorize
sleep 4
if grep -qE 'OPENROUTER_API_KEY is not set|RuntimeError' "$LAST_LOG"; then
    pass "T1.9 negative"
elif wait_for_action /grocery_categorize 2; then
    skip "T1.9 negative" "workspace .env provided key"
else
    fail "T1.9 negative" "no RuntimeError, no action"
fi
stop_all_nodes
OPENROUTER_API_KEY=smoke start_node T1.9_pos kimi_api grocery_categorize
if wait_for_action /grocery_categorize 10; then
    pass "T1.9 positive"
else
    fail "T1.9 positive" "action not advertised"
fi
stop_all_nodes

section "T1.10 — spot_on_shelf_server advertises /spot_on_shelf action"
t1_check T1.10 action /spot_on_shelf tk_vision_specialized spot_on_shelf_server

section "T1.11 — person_track_server advertises /track_person action"
# enable_reid:=False keeps startup under 5 s on cold load
t1_check T1.11 action /track_person vision_track person_track_server --ros-args -p enable_reid:=false

section "T1.12 — generalist_node: no-key + with-key both advertise"
# Unlike kimi_api, the generalist checks the key lazily on the VLM branch —
# it MUST advertise cleanly even without a key so out-of-vocab fallback fails
# gracefully at call time instead of killing the node at startup.
start_node T1.12_nokey object_detection_generalist generalist_node
if wait_for_service /object_detection_generalist 20; then
    pass "T1.12 no-key: service advertised (key is checked lazily)"
    # Sanity: log must not contain a traceback in the first seconds
    if grep -qE 'Traceback' "$LAST_LOG"; then
        fail "T1.12 no-key traceback" "$(tail -5 "$LAST_LOG" | tr '\n' '|')"
    fi
else
    fail "T1.12 no-key" "service not advertised"
fi
stop_all_nodes

OPENROUTER_API_KEY=smoke start_node T1.12_withkey object_detection_generalist generalist_node
if wait_for_service /object_detection_generalist 20; then
    pass "T1.12 with-key: service advertised"
else
    fail "T1.12 with-key" "service not advertised"
fi
stop_all_nodes
# Negative-path VLM call (out-of-vocab + fallback under no key → OPENROUTER_API_KEY
# error) requires live cameras to get past the recent-frame wait, so it lives in T2.

section "T1.fs — foundation_stereo advertises srv + action"
t1_check_multi T1.fs foundation_stereo foundation_stereo_node \
    s:/foundation_stereo/get_depth \
    a:/foundation_stereo/infer_depth \
    --

summary
