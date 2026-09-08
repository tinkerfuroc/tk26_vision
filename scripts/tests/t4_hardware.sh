#!/usr/bin/env bash
# T4 — hardware-in-the-loop cases. Invoke subcommands individually:
#   t4_hardware.sh servo_motion
#   t4_hardware.sh shelf_scene
#   t4_hardware.sh person
#   t4_hardware.sh person_phase2      (Phase-2 recovery/geometry; staged operator scenes)
#   t4_hardware.sh follow_regression   (replay-scored person-tracker bags; SKIPs if none)
#   t4_hardware.sh all        (default)
# Requires: servo at /dev/ttyUSB1, live cameras. Shelf/person cases need staged scenes.
# follow_regression needs no hardware — it replays recorded bags through the offline
# scorer; set PTBENCH_BAGS_DIR to point at a dir of labeled clips (default
# $WS_ROOT/benchmarks/person_tracker/bags).

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$HERE/lib.sh"
source_envs

SERVO_DEVICE="${SERVO_DEVICE:-/dev/ttyUSB1}"

t4_servo_motion() {
    section "T4.1 — servo commanded motion"
    if [ ! -c "$SERVO_DEVICE" ]; then skip "T4.1" "$SERVO_DEVICE not present"; return; fi
    start_launch ctrl_t4.1 pan_tilt pan_tilt.launch.py "device:=$SERVO_DEVICE"
    sleep 3
    printf '  -> commanding pan=0.3, tilt=0.0. Watch the servo move.\n'
    ros2 topic pub --once /pan_tilt_controller/cmd tinker_vision_msgs_26/msg/PanTiltCommand \
        '{mode: 1, pan_rad: 0.3, tilt_rad: 0.0, speed_raw: 0, accel_raw: 0}' >/dev/null 2>&1 || true
    sleep 2
    timeout 3 ros2 run tf2_ros tf2_echo base_link head_camera_link >"$LOG_DIR/T4.1_after.tf" 2>&1 || true
    printf '  -> returning to home (pan=0.0, tilt=0.0)\n'
    ros2 topic pub --once /pan_tilt_controller/cmd tinker_vision_msgs_26/msg/PanTiltCommand \
        '{mode: 0, pan_rad: 0.0, tilt_rad: 0.0, speed_raw: 0, accel_raw: 0}' >/dev/null 2>&1 || true
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
    start_launch ctrl_t4.2 pan_tilt pan_tilt.launch.py "device:=$SERVO_DEVICE"
    sleep 2
    start_node follow_head_t4.2 pan_tilt follow_head
    wait_for_action /follow_head_action 15 || { fail "T4.2" "follow_head_action missing"; stop_all_nodes; return; }
    printf '  -> wave a hand in front of the orbbec camera for ~15 s now.\n'
    ros2 action send_goal /follow_head_action tinker_vision_msgs_26/action/FollowHeadAction '{start_following: true}' --feedback >"$LOG_DIR/T4.2.actout" 2>&1 &
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

# T4.6 — Phase-2 recovery policy + geometry robustness (operator-in-the-loop).
# Mirrors Task 4 (Steps 2-5) of
# docs/superpowers/plans/2026-06-03-person-tracker-phase2-recovery-geometry.md.
# This is an INTERACTIVE harness: it boots the node with the installed default.yaml,
# streams TrackPerson feedback + /target_points to log files, and walks the operator
# through three staged scenes (occlusion re-entry, crosser, hard-lost). The pass/fail
# lines flag the few invariants we can assert from the captured streams; the rest is
# explicitly visual and called out as "visual inspection required". Nothing here is a
# substitute for a live run — it just makes the live run repeatable and self-documenting.
t4_person_phase2() {
    section "T4.6 — person tracker Phase 2 (recovery hysteresis + crosser reject + geometry)"
    local CFG
    CFG="$(ros2 pkg prefix vision_track 2>/dev/null)/share/vision_track/config/default.yaml"
    if [ ! -f "$CFG" ]; then
        skip "T4.6" "vision_track not built/installed (no $CFG) — run build.sh --packages-select vision_track first"
        return
    fi

    # -- T4.6.0  startup: node boots with Phase-2 params, no indefinite-recovery line (plan Step 2)
    start_node person_track_p2 vision_track person_track_server --ros-args --params-file "$CFG"
    local nlog="$LAST_LOG"
    if ! wait_for_action /track_person 30; then fail "T4.6.0" "track_person missing"; stop_all_nodes; return; fi
    sleep 2
    if assert_log_grep "$nlog" 'Max recovery frames: 45' \
        && assert_log_grep "$nlog" 'Person Track Node initialized successfully' \
        && assert_log_nogrep "$nlog" 'indefinite recovery|allow_indefinite'; then
        pass "T4.6.0 node boots with max_recovery_frames=45, no indefinite-recovery coast"
    else
        fail "T4.6.0" "missing 'Max recovery frames: 45' / init line, or stale indefinite-recovery line present (see $nlog)"
    fi

    # Common goal: debug overlay on so the operator can watch the green TARGET box.
    local goal="{return_rgb_img: false, return_depth_img: false, return_segment: false, debug: true, target_frame: ''}"

    # -- T4.6.1  provisional-vs-committed re-lock across a full occlusion (plan Step 3)
    printf '\n  T4.6.1 OCCLUSION RE-ENTRY (plan Step 3)\n'
    printf '    Operator: stand in orbbec view until the green TARGET box locks, then step\n'
    printf '    FULLY behind a pillar/wall for ~3 s, then re-emerge in the same spot.\n'
    printf '    Press enter when ready, then perform the move.\n'
    read -r _ || true
    local fb1="$LOG_DIR/T4.6.1.feedback" tp1="$LOG_DIR/T4.6.1.target_points"
    : >"$fb1"; : >"$tp1"
    setsid ros2 topic echo /target_points geometry_msgs/msg/PointStamped >"$tp1" 2>&1 &
    local tppid=$!
    timeout 35 ros2 action send_goal -f /track_person tinker_vision_msgs_26/action/TrackPerson "$goal" >"$fb1" 2>&1 &
    local ap=$!
    sleep 30
    kill "$ap" "$tppid" 2>/dev/null || true; wait "$ap" 2>/dev/null || true; wait "$tppid" 2>/dev/null || true
    # Invariants we CAN assert from the streams:
    #   - the action surfaced target_lost:true at least once (the coast happened), AND
    #   - it recovered to target_lost:false afterwards (re-lock fired).
    # Note: /target_points is NOT silent during the coast — the node publishes a
    # NaN-coordinate sentinel each lost frame (person_track_node.py ~L967). The real
    # invariant is "no FINITE point during the coast", so we count finite-x samples.
    local saw_lost saw_relock finite_pts
    saw_lost=$(grep -cE 'target_lost:[[:space:]]*true' "$fb1" || true)
    saw_relock=$(grep -cE 'target_lost:[[:space:]]*false' "$fb1" || true)
    finite_pts=$(grep -E 'x:' "$tp1" | grep -ivcE 'nan' || true)
    printf '    captured: target_lost:true x%s, target_lost:false x%s, finite /target_points samples x%s\n' \
        "$saw_lost" "$saw_relock" "$finite_pts"
    if [ "${saw_lost:-0}" -ge 1 ] && [ "${saw_relock:-0}" -ge 1 ]; then
        pass "T4.6.1 coast (target_lost:true) then re-lock (target_lost:false) observed — VISUAL: re-lock felt <= ~1 s?"
    else
        fail "T4.6.1" "did not see both a coast and a re-lock in feedback (see $fb1)"
    fi
    printf '    VISUAL CHECK: /target_points (%s) should carry NaN sentinels through the coast,\n' "$tp1"
    printf '    then resume FINITE x/y after the brief provisional window. Re-lock <= ~1 s at 12-15 Hz.\n'

    # -- T4.6.2  crosser rejection (plan Step 3, depth gate)
    printf '\n  T4.6.2 CROSSER REJECTION (plan Step 3, depth gate)\n'
    printf '    Operator stands locked at ~2.5 m. A BYSTANDER walks between the robot and the\n'
    printf '    operator (nearer to the camera) and out the other side. The green TARGET box\n'
    printf '    must STAY on the far operator; the lock must NOT jump to the nearer crosser.\n'
    printf '    Press enter when ready, then perform the cross.\n'
    read -r _ || true
    local fb2="$LOG_DIR/T4.6.2.feedback"
    : >"$fb2"
    timeout 25 ros2 action send_goal -f /track_person tinker_vision_msgs_26/action/TrackPerson "$goal" >"$fb2" 2>&1 &
    ap=$!
    sleep 20
    kill "$ap" 2>/dev/null || true; wait "$ap" 2>/dev/null || true
    # Best automated signal: the committed track id stayed stable across the cross.
    local nids
    nids=$(grep -oE 'target_track_id:[[:space:]]*[0-9-]+' "$fb2" | awk -F: '{print $2}' | tr -d ' ' | sort -u | grep -vc '^-1$' || true)
    printf '    distinct non-(-1) target_track_id values during the cross: %s\n' "$nids"
    if [ "${nids:-0}" -le 1 ]; then
        pass "T4.6.2 committed target_track_id stable across the cross (<=1 id) — VISUAL: box stayed on operator?"
    else
        fail "T4.6.2" "target_track_id changed during cross ($nids distinct ids) — possible crosser capture (see $fb2)"
    fi
    printf '    VISUAL CHECK (authoritative): in the debug overlay the green TARGET box never\n'
    printf '    latched onto the nearer crosser.\n'

    # -- T4.6.3  hard-lost bound (plan Step 4)
    printf '\n  T4.6.3 HARD-LOST BOUND (plan Step 4)\n'
    printf '    Operator: get locked, then LEAVE the scene entirely and do NOT return.\n'
    printf '    Expected: after ~max_recovery_frames coast frames the action ABORTS with\n'
    printf '    message containing "hard-lost (recovery cap)".\n'
    printf '    Press enter when ready, then walk out and stay out.\n'
    read -r _ || true
    local fb3="$LOG_DIR/T4.6.3.feedback"
    : >"$fb3"
    # No timeout kill here — we WANT to see the node abort on its own.
    timeout 40 ros2 action send_goal -f /track_person tinker_vision_msgs_26/action/TrackPerson "$goal" >"$fb3" 2>&1 || true
    if grep -qE 'hard-lost \(recovery cap\)' "$fb3" "$nlog"; then
        pass "T4.6.3 action aborted with 'hard-lost (recovery cap)' after the recovery bound"
    else
        fail "T4.6.3" "no 'hard-lost (recovery cap)' abort within 40 s — check it didn't coast forever (see $fb3 / $nlog)"
    fi

    # -- T4.6.4  lateral-accuracy + jitter smoke (plan Step 5)
    printf '\n  T4.6.4 LATERAL ACCURACY + JITTER (plan Step 5)\n'
    printf '    Operator stands at a TAPE-MEASURED lateral offset (e.g. 0.5 m left of the\n'
    printf '    optical axis at 2.5 m range) and holds still for ~15 s.\n'
    printf '    Press enter when in position.\n'
    read -r _ || true
    local tp4="$LOG_DIR/T4.6.4.target_points"
    : >"$tp4"
    setsid ros2 topic echo /target_points geometry_msgs/msg/PointStamped >"$tp4" 2>&1 &
    tppid=$!
    timeout 18 ros2 action send_goal -f /track_person tinker_vision_msgs_26/action/TrackPerson "$goal" >/dev/null 2>&1 &
    ap=$!
    sleep 15
    kill "$ap" "$tppid" 2>/dev/null || true; wait "$ap" 2>/dev/null || true; wait "$tppid" 2>/dev/null || true
    local n_samples
    n_samples=$(grep -cE '^[[:space:]]*x:' "$tp4" || true)
    printf '    captured %s /target_points samples (%s)\n' "$n_samples" "$tp4"
    if [ "${n_samples:-0}" -ge 5 ]; then
        pass "T4.6.4 captured a /target_points stream — MANUAL: compare x/y to tape, check jitter < pre-Phase-2"
    else
        fail "T4.6.4" "too few /target_points samples ($n_samples) — was the operator in view? (see $tp4)"
    fi
    printf '    MANUAL: record observed x/y vs measured offset (torso band should keep the centroid\n'
    printf '    off legs/feet; EMA should visibly reduce frame-to-frame jitter) in src/tk26_vision/DEV_NOTES.md.\n'

    stop_all_nodes
}

t4_follow_regression() {
    section "T4.5 — follow regression (replay scored bags)"
    local BAGS_DIR="${PTBENCH_BAGS_DIR:-$WS_ROOT/benchmarks/person_tracker/bags}"
    local PT_DIR="$WS_ROOT/benchmarks/person_tracker"

    # Collect every <clip> subdir that carries a gt.json.
    local clips=()
    if [ -d "$BAGS_DIR" ]; then
        local gt bag
        for gt in "$BAGS_DIR"/*/gt.json; do
            [ -f "$gt" ] || continue
            bag="$(dirname "$gt")"
            clips+=("$bag")
        done
    fi

    if [ "${#clips[@]}" -eq 0 ]; then
        skip "T4.5" "no labeled bags in $BAGS_DIR (record + label first)"
        return
    fi

    local bag clip out
    for bag in "${clips[@]}"; do
        clip="$(basename "$bag")"
        out="$LOG_DIR/T4.5_$clip.out"
        ( cd "$PT_DIR" && "$VENV/bin/python" -m ptbench.replay.score_cli \
            --bag "$bag" --gt "$bag/gt.json" --backend offline ) >"$out" 2>&1
        if grep -qE '^OVERALL[[:space:]].*PASS' "$out"; then
            pass "T4.5 $clip OVERALL PASS"
        else
            local snippet
            snippet="$(grep -E '^OVERALL' "$out" | head -1)"
            [ -n "$snippet" ] || snippet="$(tail -3 "$out" | tr '\n' ' ')"
            fail "T4.5" "$clip not PASS: ${snippet:-no scoreboard (see $out)}"
        fi
    done
}

case "${1:-all}" in
    servo_motion) t4_servo_motion ;;
    servo_tracking) t4_servo_tracking ;;
    shelf_scene) t4_shelf_scene ;;
    person) t4_person ;;
    person_phase2) t4_person_phase2 ;;
    follow_regression) t4_follow_regression ;;
    all) t4_servo_motion; t4_servo_tracking; t4_shelf_scene; t4_person; t4_person_phase2; t4_follow_regression ;;
    *) printf 'usage: %s {servo_motion|servo_tracking|shelf_scene|person|person_phase2|follow_regression|all}\n' "$0"; exit 2 ;;
esac

summary
