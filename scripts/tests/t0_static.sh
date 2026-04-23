#!/usr/bin/env bash
# T0 — static/smoke checks for tk26_vision migrated packages.
# No ROS traffic, no cameras, no background nodes. Target wall-clock < 30 s.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./lib.sh
source "$HERE/lib.sh"
source_envs

VENV_PY="$VENV/bin/python"
EXPECTED_SHEBANG="#!$VENV_PY"

section "T0.1 — Shebangs point to venv python"
bad=0; seen=0
while IFS= read -r f; do
    seen=$((seen+1))
    first=$(head -1 "$f" 2>/dev/null || true)
    if [ "$first" != "$EXPECTED_SHEBANG" ]; then
        fail "T0.1" "bad shebang in $f: $first"; bad=1
    fi
done < <(find "$WS_ROOT/install"/{object_detection_new,object_detection_generalist,vision_util,pan_tilt,kimi_api,vision_track,tk_vision_specialized}/lib -maxdepth 2 -type f 2>/dev/null)
[ "$seen" -eq 0 ] && fail "T0.1" "no install-tree entry scripts found — has the workspace been built?"
[ "$bad" -eq 0 ] && [ "$seen" -gt 0 ] && pass "T0.1 ($seen scripts, all correct)"

section "T0.2 — fix_venv_shebangs.sh is idempotent"
out=$("$WS_ROOT/scripts/fix_venv_shebangs.sh" 2>&1)
echo "$out" | tail -1
out2=$("$WS_ROOT/scripts/fix_venv_shebangs.sh" 2>&1)
if echo "$out2" | tail -1 | grep -qE 'done — 0 script\(s\) updated'; then
    pass "T0.2"
else
    fail "T0.2" "second run did not report 0 updates: $(echo "$out2" | tail -1)"
fi

section "T0.3 — Venv deps importable"
if "$VENV_PY" -c "import ultralytics, openai, dotenv, serial, scipy, torch, cv2; from ultralytics import FastSAM" 2>"$LOG_DIR/t0.3.err"; then
    pass "T0.3"
else
    fail "T0.3" "$(cat "$LOG_DIR/t0.3.err")"
fi

section "T0.3b — Generalist module importable"
if "$VENV_PY" -c "from object_detection_generalist.generalist_node import GeneralistDetectionNode; from object_detection_generalist.vlm_bbox import request_bboxes; from object_detection_generalist.sam_mask import FastSAMPredictor" 2>"$LOG_DIR/t0.3b.err"; then
    pass "T0.3b"
else
    fail "T0.3b" "$(cat "$LOG_DIR/t0.3b.err")"
fi

section "T0.4 — ROS interfaces built"
ifaces=(
    tinker_vision_msgs_26/action/TrackPerson
    tinker_vision_msgs_26/action/SpotOnShelf
    tinker_vision_msgs_26/srv/ObjectDetection
    tinker_vision_msgs_26/srv/ObjectDetectionGeneralist
    tinker_vision_msgs_26/srv/DoorDetection
    tinker_vision_msgs_26/srv/GetPointCloud
    tinker_vision_msgs_26/srv/FeatureExtraction
    tinker_vision_msgs_26/srv/FeatureMatching
    tinker_vision_msgs_26/srv/SeatRecommendation
    tinker_vision_msgs_26/srv/FollowHead
    tinker_vision_msgs_26/action/FollowHeadAction
    tinker_vision_msgs_26/action/Categorize
    tinker_vision_msgs_26/msg/PanTiltCommand
    tinker_vision_msgs_26/msg/PanTiltState
    tinker_vision_msgs_26/srv/SetTorque
    tinker_vision_msgs_26/srv/SetZero
)
iface_bad=0
for i in "${ifaces[@]}"; do
    if ! ros2 interface show "$i" >/dev/null 2>&1; then
        fail "T0.4" "missing: $i"; iface_bad=1
    fi
done
[ "$iface_bad" -eq 0 ] && pass "T0.4 (${#ifaces[@]} interfaces)"

section "T0.5 — Entry-point imports (--help exit clean)"
# (pkg, entry) pairs
entries=(
    object_detection_new:yolo_seg_node
    object_detection_new:yolo_seg_default_node
    object_detection_generalist:generalist_node
    vision_util:door_detection
    vision_util:get_point_cloud
    pan_tilt:controller
    pan_tilt:state_publisher
    pan_tilt:follow_head
    kimi_api:feature_recognition
    kimi_api:feature_matching
    kimi_api:grocery_categorize
    tk_vision_specialized:spot_on_shelf_server
    vision_track:person_track_server
)
# For kimi_api nodes, give a smoke key so _env.require_api_key() doesn't kill them during node boot.
for pair in "${entries[@]}"; do
    pkg="${pair%%:*}"; entry="${pair##*:}"
    log="$LOG_DIR/t0.5_${pkg}_${entry}.log"
    extra_args=()
    env_prefix=()
    case "$pkg" in
        # The generalist checks the key lazily on the VLM branch, so it starts
        # without one; pass a smoke key to keep `env` symmetry with kimi_api.
        kimi_api|object_detection_generalist) env_prefix=(env OPENROUTER_API_KEY=smoke) ;;
    esac
    case "$entry" in
        controller) extra_args=(--ros-args -p device:=/dev/ttyUSB_nonexistent) ;;
    esac
    # --ros-args --help exits immediately after printing; we just want to confirm
    # no ImportError/ModuleNotFoundError in the first 3 seconds of node boot.
    "${env_prefix[@]}" timeout 3 ros2 run "$pkg" "$entry" "${extra_args[@]}" \
        >"$log" 2>&1 || true
    if grep -qE 'ModuleNotFoundError|ImportError' "$log"; then
        fail "T0.5" "$pkg/$entry import error (see $log)"
    else
        pass "T0.5 $pkg/$entry"
    fi
done

section "T0.6 — weights_cache import + cache dir"
if python3 -c "from vision_util.weights_cache import resolve_weights, _writable_cache; _writable_cache()" 2>"$LOG_DIR/t0.6_weights_cache.log"; then
    pass "T0.6 (vision_util.weights_cache importable, cache dir writable)"
else
    fail "T0.6" "import/cache-dir failure (see $LOG_DIR/t0.6_weights_cache.log)"
fi

section "T0.7 — .env sanity"
if have_api_key; then
    pass "T0.7 (API key populated)"
elif [ -f "$ENV_FILE" ]; then
    skip "T0.7" "$ENV_FILE exists but key is placeholder; live-LLM tests will skip"
else
    skip "T0.7" "$ENV_FILE absent; live-LLM tests will skip"
fi

summary
