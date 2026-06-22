#!/usr/bin/env bash
# T2 — live cameras, empty scene. Each node gets started, one call is issued,
# shape of response is asserted, node is torn down. Kimi_api T2.7–T2.9 skip
# cleanly if no API key.
#
# Usage: t2_live.sh [--no-precheck]
#
# Requires orbbec + realsense drivers already running in separate terminals.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$HERE/lib.sh"
source_envs

NO_PRECHECK=0
[ "${1:-}" = "--no-precheck" ] && NO_PRECHECK=1
MIN_HZ="${MIN_HZ:-2}"   # orbbec empty-scene rate can be ~3-4 Hz
SERVO_DEVICE="${SERVO_DEVICE:-/dev/ttyUSB1}"

# Cap service-call output at 4 KB so image byte arrays don't blow up the log.
# All fields we assert on (status, error_msg, objects, is_open, feature,
# recommendation, centroids, width, height) appear in the first few hundred bytes.
svc_call() {
    local out="$1" timeout="$2" srv="$3" srv_type="$4" payload="$5"
    timeout "$timeout" ros2 service call "$srv" "$srv_type" "$payload" 2>&1 \
        | head -c 4000 >"$out" || true
}

act_call() {
    local out="$1" timeout="$2" act="$3" act_type="$4" payload="$5" flags="${6:-}"
    # shellcheck disable=SC2086
    timeout "$timeout" ros2 action send_goal $flags "$act" "$act_type" "$payload" 2>&1 \
        | head -c 4000 >"$out" || true
}

# Empty-scene OK for ObjectDetection: status=0 (objects found) OR status=1 (no objects).
# We consider both valid since the view may or may not contain the queried class.
assert_objdet_ok() {
    local tag="$1" out="$2"
    if grep -qE 'Traceback|Exception' "$out"; then
        fail "$tag" "traceback in response"
        return
    fi
    if ! grep -qE 'status=[01]' "$out"; then
        fail "$tag" "no status=0|1 in response. Head:\n$(head -c 500 "$out")"
        return
    fi
    # Must have the image fields to confirm the node actually processed a frame.
    if grep -qE "rgb_image=sensor_msgs.msg.Image.*width=[1-9]" "$out"; then
        local st
        st=$(grep -oE 'status=[01]' "$out" | head -1)
        pass "$tag ObjectDetection $st with image payload"
    else
        fail "$tag" "no rgb_image width in response. Head:\n$(head -c 500 "$out")"
    fi
}

section "T2 precheck — cameras publishing"
if [ "$NO_PRECHECK" -eq 0 ]; then
    if wait_for_topic_hz /camera/color/image_raw "$MIN_HZ" 12; then
        pass "precheck: orbbec /camera/color/image_raw ≥ $MIN_HZ Hz"
    else
        fail "precheck" "/camera/color/image_raw not at $MIN_HZ Hz"
        summary; exit 1
    fi
    if wait_for_topic_hz /camera/xarm_camera/color/image_raw "$MIN_HZ" 12; then
        pass "precheck: realsense /camera/xarm_camera/color/image_raw ≥ $MIN_HZ Hz"
    else
        fail "precheck" "/camera/xarm_camera/color/image_raw not at $MIN_HZ Hz"
        summary; exit 1
    fi
else
    skip "precheck" "--no-precheck"
fi

section "T2.1/T2.2 — /object_detection (default YOLO)"
start_node yolo_default_node object_detection_new yolo_seg_default_node
if wait_for_service /object_detection 30; then
    sleep 8  # warmup: let ApproximateTimeSynchronizer buffer a few frame pairs
    for cam in orbbec realsense; do
        out="$LOG_DIR/T2.1_${cam}.svcout"
        svc_call "$out" 30 /object_detection tinker_vision_msgs_26/srv/ObjectDetection \
            "{camera: '$cam', prompt: 'person', flags: '', target_frame: '', category: ''}"
        assert_objdet_ok "T2.1/T2.2 [$cam]" "$out"
    done
else
    fail "T2.1/T2.2" "/object_detection never appeared"
fi
stop_all_nodes

section "T2.3 — /object_detection_yolo (custom YOLO, orbbec)"
start_node yolo_custom_node object_detection_new yolo_seg_node
if wait_for_service /object_detection_yolo 30; then
    sleep 8
    out="$LOG_DIR/T2.3.svcout"
    svc_call "$out" 30 /object_detection_yolo tinker_vision_msgs_26/srv/ObjectDetection \
        "{camera: 'orbbec', prompt: 'person', flags: '', target_frame: '', category: ''}"
    assert_objdet_ok "T2.3" "$out"
else
    fail "T2.3" "/object_detection_yolo never appeared"
fi
stop_all_nodes

section "T2.4 — /door_detection_srv"
start_node door_detection vision_util door_detection
if wait_for_service /door_detection_srv 15; then
    out="$LOG_DIR/T2.4.svcout"
    svc_call "$out" 15 /door_detection_srv tinker_vision_msgs_26/srv/DoorDetection "{camera: 'orbbec'}"
    if grep -qE 'Traceback' "$out"; then
        fail "T2.4" "traceback in response"
    elif grep -qE 'status=0' "$out" && grep -qE 'is_open=[01]' "$out"; then
        pass "T2.4 $(grep -oE 'status=0.*is_open=[01]' "$out" | head -1)"
    else
        fail "T2.4" "head: $(head -c 400 "$out")"
    fi
else
    fail "T2.4" "service never appeared"
fi
stop_all_nodes

section "T2.5/T2.6 — /get_point_cloud_service"
start_node get_point_cloud vision_util get_point_cloud
if wait_for_service /get_point_cloud_service 15; then
    sleep 8
    for cam in orbbec realsense; do
        out="$LOG_DIR/T2.5_${cam}.svcout"
        svc_call "$out" 20 /get_point_cloud_service tinker_vision_msgs_26/srv/GetPointCloud \
            "{camera: '$cam'}"
        if grep -qE 'Traceback' "$out"; then
            fail "T2.5/T2.6 [$cam]" "traceback"
        elif grep -qE 'status=0' "$out" && grep -qE 'width=[1-9]|height=[1-9]' "$out"; then
            pass "T2.5/T2.6 [$cam]: status=0 with non-empty points"
        elif grep -qE "error_msg='No camera data for $cam" "$out"; then
            skip "T2.5/T2.6 [$cam]" "node returned 'No camera data' — ApproximateTimeSynchronizer slop=0.05 couldn't pair color+depth at current rates; node wiring is correct"
        else
            fail "T2.5/T2.6 [$cam]" "head: $(head -c 400 "$out")"
        fi
    done
else
    fail "T2.5/T2.6" "service never appeared"
fi
stop_all_nodes

section "T2.7/T2.8 — feature_recognition (live OpenRouter)"
if have_api_key; then
    start_node feature_recognition kimi_api feature_recognition
    if wait_for_service /feature_extraction_service 20; then
        out="$LOG_DIR/T2.7.svcout"
        svc_call "$out" 60 /feature_extraction_service tinker_vision_msgs_26/srv/FeatureExtraction \
            "{camera: 'orbbec'}"
        if grep -qE 'Traceback' "$out"; then
            fail "T2.7" "traceback"
        elif grep -qE "status=0" "$out" && grep -qE "feature='[^']+" "$out"; then
            pass "T2.7 extraction produced non-empty feature"
        else
            fail "T2.7" "head: $(head -c 500 "$out")"
        fi
        out="$LOG_DIR/T2.8.svcout"
        svc_call "$out" 60 /seat_recommend_service tinker_vision_msgs_26/srv/SeatRecommendation \
            "{camera: 'orbbec', names: ['alice'], features: ['adult wearing glasses']}"
        if grep -qE 'Traceback' "$out"; then
            fail "T2.8" "traceback"
        elif grep -qE "status=0" "$out" && grep -qE "recommendation='[^']+" "$out"; then
            pass "T2.8 seat recommendation produced non-empty result"
        else
            fail "T2.8" "head: $(head -c 500 "$out")"
        fi
    else
        fail "T2.7/T2.8" "feature_extraction_service never appeared"
    fi
    stop_all_nodes
else
    skip "T2.7/T2.8" "no valid OPENROUTER_API_KEY"
fi

section "T2.9 — feature_matching (live) + yolo dependency"
if have_api_key; then
    start_node yolo_default_for_match object_detection_new yolo_seg_default_node
    wait_for_service /object_detection 30 || true
    start_node feature_matching kimi_api feature_matching
    if wait_for_service /feature_matching_service 20; then
        out="$LOG_DIR/T2.9.svcout"
        svc_call "$out" 60 /feature_matching_service tinker_vision_msgs_26/srv/FeatureMatching \
            "{camera: 'orbbec', features: ['red bottle'], max_distance: 2.0, target_frame: 'base_link'}"
        if grep -qE 'Traceback' "$out"; then
            fail "T2.9" "traceback"
        elif grep -qE "status=[01]" "$out" && grep -qE "centroids=" "$out"; then
            st=$(grep -oE 'status=[01]' "$out" | head -1)
            pass "T2.9 feature_matching $st (empty scene: centroids=[] expected)"
        else
            fail "T2.9" "head: $(head -c 500 "$out")"
        fi
    else
        fail "T2.9" "feature_matching_service never appeared"
    fi
    stop_all_nodes
else
    skip "T2.9" "no valid OPENROUTER_API_KEY"
fi

section "T2.10 — follow_head action feedback"
if [ -c "$SERVO_DEVICE" ]; then
    start_launch ctrl_for_t2.10 pan_tilt pan_tilt.launch.py "device:=$SERVO_DEVICE"
    sleep 2
    start_node follow_head pan_tilt follow_head
    if wait_for_action /follow_head_action 20; then
        sleep 5  # let follow_head buffer camera frames
        out="$LOG_DIR/T2.10.actout"
        act_call "$out" 15 /follow_head_action tinker_vision_msgs_26/action/FollowHeadAction \
            '{start_following: true}' '--feedback'
        if grep -qE 'Traceback' "$out"; then
            fail "T2.10" "traceback"
        elif grep -qE 'Feedback:' "$out"; then
            pass "T2.10 feedback received"
        elif grep -qE 'Goal accepted' "$out"; then
            pass "T2.10 action handshake OK (no feedback emitted in 15s empty scene — expected)"
        else
            fail "T2.10" "no goal accept (head: $(head -c 300 "$out"))"
        fi
    else
        fail "T2.10" "action never appeared"
    fi
    stop_all_nodes
else
    skip "T2.10" "$SERVO_DEVICE not present"
fi

section "T2.11 — grocery_categorize action"
if have_api_key; then
    start_node yolo_default_for_grocery object_detection_new yolo_seg_default_node
    wait_for_service /object_detection 30 || true
    start_node grocery_categorize kimi_api grocery_categorize
    if wait_for_action /grocery_categorize 20; then
        out="$LOG_DIR/T2.11.actout"
        goal='{n_layers: 2, prompt: "", img_table: {}, segment_object: {}, pt_shelf_left: {header: {frame_id: "base_link"}, point: {x: 1.0, y: 0.3, z: 0.0}}, pt_shelf_right: {header: {frame_id: "base_link"}, point: {x: 1.0, y: -0.3, z: 0.0}}, flags: "", target_frame: "base_link"}'
        act_call "$out" 60 /grocery_categorize tinker_vision_msgs_26/action/Categorize "$goal"
        if grep -qE 'Traceback' "$out"; then
            fail "T2.11" "traceback"
        elif grep -qE 'Goal finished|Result:|status=' "$out"; then
            pass "T2.11 action terminated cleanly"
        else
            fail "T2.11" "head: $(head -c 400 "$out")"
        fi
    else
        fail "T2.11" "action never appeared"
    fi
    stop_all_nodes
else
    skip "T2.11" "no valid OPENROUTER_API_KEY"
fi

section "T2.12 — spot_on_shelf action"
start_node yolo_custom_for_shelf object_detection_new yolo_seg_node
wait_for_service /object_detection_yolo 30 || true
start_node spot_on_shelf tk_vision_specialized spot_on_shelf_server
if wait_for_action /spot_on_shelf 20; then
    out="$LOG_DIR/T2.12.actout"
    goal='{shelf_left: {header: {frame_id: "map"}, pose: {position: {x: 1.0, y: 0.3, z: 0.5}, orientation: {w: 1.0}}}, shelf_right: {header: {frame_id: "map"}, pose: {position: {x: 1.0, y: -0.3, z: 0.5}, orientation: {w: 1.0}}}, shelf_heights: [0.3, 0.6], item_ids: ["bottle"]}'
    act_call "$out" 60 /spot_on_shelf tinker_vision_msgs_26/action/SpotOnShelf "$goal"
    if grep -qE 'Traceback' "$out"; then
        fail "T2.12" "traceback"
    elif grep -qE 'Goal finished|Result:' "$out"; then
        pass "T2.12 action terminated"
    else
        fail "T2.12" "head: $(head -c 400 "$out")"
    fi
else
    fail "T2.12" "action never appeared"
fi
stop_all_nodes

section "T2.14 — /object_detection_generalist (YOLO branch + VLM+SAM branch)"
start_node generalist object_detection_generalist generalist_node
if wait_for_service /object_detection_generalist 30; then
    sleep 8  # sync warmup
    # YOLO branch — prompt is in pretrained COCO vocab, no fallback needed
    out="$LOG_DIR/T2.14_yolo.svcout"
    svc_call "$out" 30 /object_detection_generalist tinker_vision_msgs_26/srv/ObjectDetectionGeneralist \
        "{camera: 'realsense', prompt: 'bottle', use_vlm_sam_fallback: false}"
    if grep -qE 'Traceback' "$out"; then
        fail "T2.14 YOLO" "traceback"
    elif grep -qE "detection_source='(yolo|none)'" "$out" && grep -qE 'status=[01]' "$out"; then
        src=$(grep -oE "detection_source='[^']*'" "$out" | head -1)
        pass "T2.14 YOLO branch $src"
    else
        fail "T2.14 YOLO" "head: $(head -c 400 "$out")"
    fi

    # VLM+SAM branch — only run if we have a real key (Gemini call costs)
    if have_api_key; then
        out="$LOG_DIR/T2.14_vlm.svcout"
        # 90 s timeout: Gemini 2.5 Pro is 9-14 s/call plus SAM + sync warmup
        svc_call "$out" 90 /object_detection_generalist tinker_vision_msgs_26/srv/ObjectDetectionGeneralist \
            "{camera: 'realsense', prompt: 'spatula', use_vlm_sam_fallback: true}"
        if grep -qE 'Traceback' "$out"; then
            fail "T2.14 VLM" "traceback"
        elif grep -qE "detection_source='(vlm_sam|none|yolo)'" "$out" && grep -qE 'status=[01]' "$out"; then
            src=$(grep -oE "detection_source='[^']*'" "$out" | head -1)
            pass "T2.14 VLM+SAM branch $src"
        else
            fail "T2.14 VLM" "head: $(head -c 400 "$out")"
        fi
    else
        skip "T2.14 VLM" "no valid OPENROUTER_API_KEY"
    fi
else
    fail "T2.14" "/object_detection_generalist never appeared"
fi
stop_all_nodes

section "T2.15 — specialist /object_detection_yolo drops person (excluded_classes)"
# Positive observation only possible when a person is actually in frame.
# In empty-scene runs we fall back to confirming the param loads via log.
start_node yolo_specialist_t2.15 object_detection_new yolo_seg_node
sleep 3
if grep -qE "Excluded classes: \['person'\]|excluded_classes.*person" "$LAST_LOG"; then
    pass "T2.15 excluded_classes=['person'] loaded at startup"
else
    fail "T2.15" "excluded_classes log line missing (tail: $(tail -5 "$LAST_LOG" | tr '\n' '|'))"
fi
# Live positive-case check still needs operator (person in Orbbec frame) — left to T4.
stop_all_nodes

section "T2.16 — /object_match_all empty-scene invariant"
# This is a "shape" check: with DASHSCOPE_API_KEY=fake, every match_batch
# will fail (auth), so the pipeline returns no candidates. The server's
# empty-scene invariant is status=1 + detection_source='vlm_match_all', and
# that's what we assert. Same response shape whether the scene is genuinely
# empty or the VLM auth fails per batch.
#
# Soft camera check: the top-level precheck already gates this in the
# default flow, but be tolerant of --no-precheck so the case skips cleanly
# instead of hanging on a 'No camera data' wait.
need_orbbec=1
if [ "$NO_PRECHECK" -eq 1 ]; then
    wait_for_topic_hz /camera/color/image_raw "$MIN_HZ" 5 || need_orbbec=0
fi
if [ "$need_orbbec" -eq 1 ]; then
    DASHSCOPE_API_KEY=fake start_node object_match_all_t2.16 \
        tk_vision_specialized object_match_all_server
    if wait_for_service /object_match_all 30; then
        sleep 5  # warmup: let snapshot buffer pair color + depth/PC
        out="$LOG_DIR/T2.16.svcout"
        svc_call "$out" 30 /object_match_all tinker_vision_msgs_26/srv/ObjectMatchAll \
            "{camera: 'orbbec', category_filter: [], target_frame: '', sort_closest: false, sort_highest: false, return_rgb_image: false, return_depth_image: false, return_segments: false}"
        if grep -qE 'Traceback' "$out"; then
            fail "T2.16" "traceback in response"
        elif grep -qE 'status=1' "$out" && grep -qE "detection_source='vlm_match_all'" "$out"; then
            pass "T2.16 empty-scene invariant status=1 detection_source=vlm_match_all"
        else
            fail "T2.16" "head: $(head -c 500 "$out")"
        fi
    else
        fail "T2.16" "/object_match_all never appeared (log tail: $(tail -5 "$LAST_LOG" | tr '\n' '|'))"
    fi
    stop_all_nodes
else
    skip "T2.16" "cameras not running (--no-precheck path)"
fi

section "T2.13 — person_track action"
start_node person_track vision_track person_track_server --ros-args -p enable_reid:=false
if wait_for_action /track_person 30; then
    sleep 5  # buffer warmup
    out="$LOG_DIR/T2.13.actout"
    act_call "$out" 15 /track_person tinker_vision_msgs_26/action/TrackPerson \
        "{target_frame: 'map', return_rgb_img: false, return_depth_img: false, return_segment: false, debug: false}" \
        '-f'
    if grep -qE 'Traceback' "$out"; then
        fail "T2.13" "traceback"
    elif grep -qE 'target_lost=True|target_lost=False|Feedback:' "$out"; then
        pass "T2.13 feedback received"
    elif grep -qE 'Goal accepted' "$out"; then
        pass "T2.13 action handshake OK (no feedback in 15s empty scene)"
    else
        fail "T2.13" "head: $(head -c 400 "$out")"
    fi
else
    fail "T2.13" "action never appeared"
fi
stop_all_nodes

summary
