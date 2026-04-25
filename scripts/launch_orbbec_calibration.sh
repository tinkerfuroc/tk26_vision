#!/usr/bin/env bash
# Launch the Orbbec Femto Bolt with exposure / white-balance / gain locked for
# pan_tilt extrinsic calibration. Same FastDDS SHM profile and tk26 overrides
# as launch_orbbec_shm.sh (depth_registration on, IR off, SDK frame-sync off),
# but with all auto-tuning disabled so the ChArUco detection RMS doesn't ride
# on the office's ambient lighting.
#
# Why this matters for calibration:
#   - Auto-exposure responds to scene content (e.g. a person walking past the
#     camera): brightness shifts frame-to-frame and ArUco's adaptive-threshold
#     sweep picks a different winning window size each frame, flipping
#     detection on/off even with a static EE.
#   - Auto-gain inflates sensor noise when ambient light drops, defeating
#     sub-pixel corner refinement (CORNER_REFINE_SUBPIX). Noisy corners push
#     solvePnP reprojection rms past the `max_reproj_px=1.5` gate.
#   - Auto-WB changes the grayscale mapping; thresholds that separated black
#     from white marker bits stop separating them.
# Locking all three makes detection a function of the board pose alone, which
# is what the whole calibration pipeline assumes.
#
# Overrides (export before invoking):
#   COLOR_EXPOSURE   microseconds. 20000 (20 ms) is a forgiving starting point
#                    for typical office lighting on a matte ChArUco print on
#                    aluminium composite; the matte surface absorbs ~60% of
#                    incident light, which is why the first-pass default of
#                    10 ms came out too dark. If the image looks clipped in
#                    highlights, cut to 12000–15000. If still dark after also
#                    raising COLOR_GAIN, go up to 25000–30000 (keep < 33000
#                    or 30 fps will stall). Auto typically lands at ~18–25 ms
#                    in office light, so this default tracks that.
#   COLOR_GAIN       0–255 typ. 100 = moderate, roughly ISO-200-ish. Low
#                    values preserve sub-pixel corner refinement (which is
#                    what makes PnP reprojection rms stay under the 1.5 px
#                    gate). Raise if the board reads dark even at 25 ms
#                    exposure; lower if you see visible luminance noise in
#                    shadowed parts of the board.
#   COLOR_WB_K       white-balance colour temperature in Kelvin. **Note: the
#                    Femto Bolt's WB parameter is inverted from photo-camera
#                    convention.** It sets the sensor's white-point reference,
#                    so a higher Kelvin value makes the output image WARMER
#                    (camera boosts red to match the cooler "target"), not
#                    cooler. Default 3500 K was tuned empirically against the
#                    tk25 arena's LED lighting: at 5500 the image runs warm
#                    (R/G ≈ 1.23, visible yellow tint); at 3500 it balances
#                    to R/G ≈ 1.00, B/G ≈ 1.01. Shift lower (3000) if still
#                    yellow; raise (4500–5000) if it turns bluish.
#   COLOR_WIDTH/HEIGHT/FPS   Default 1920x1080@30 — Femto Bolt's factory
#                    intrinsics auto-switch with resolution (the driver
#                    publishes the matching fx/fy/cx/cy on
#                    /camera/color/camera_info), so this resolution bump is
#                    free from the pipeline's perspective. The ~50% linear
#                    resolution gain over 720p translates to ~50% more pixels
#                    per marker-bit, which matters because ArUco needs about
#                    3 px per bit to clear its border-quiet-zone check
#                    (with a 10 cm board at ~0.75 m, 720p gives ~2 px/bit,
#                    1080p gives ~3 px/bit — decisive margin). Supported
#                    higher modes on Femto Bolt MJPG: 2560x1440@30,
#                    3840x2160@25 (drop COLOR_FPS to 25 if you go 4K).
#
# Quick tuning loop: launch this, point the EE at the camera, watch the
# detection badge in calibrate_web. The badge-shown `corners=N` will trend
# upward as exposure/gain land in the right zone. Aim for rms ≤ 1.0 px and
# stable `corners = 16` (all interior corners of the 5x5 compact board) — the
# pipeline accepts down to 6, but a comfortable margin keeps detection stable
# when the arm moves through the grid.
#
# Extra args forward to `ros2 launch`, e.g.
#   ./launch_orbbec_calibration.sh enable_colored_point_cloud:=false
#
# The workspace install/setup must already be sourced in the calling shell.

set -euo pipefail

: "${COLOR_EXPOSURE:=20000}"
: "${COLOR_GAIN:=100}"
: "${COLOR_WB_K:=3500}"
: "${COLOR_WIDTH:=1920}"
: "${COLOR_HEIGHT:=1080}"
: "${COLOR_FPS:=30}"

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
PROFILES_FILE="$REPO_ROOT/config/fastdds_shm.xml"

if [ ! -f "$PROFILES_FILE" ]; then
    echo "error: FastDDS SHM profile not found: $PROFILES_FILE" >&2
    exit 1
fi

export FASTRTPS_DEFAULT_PROFILES_FILE="$PROFILES_FILE"

echo "[launch_orbbec_calibration] locked: exposure=${COLOR_EXPOSURE}us gain=${COLOR_GAIN} wb=${COLOR_WB_K}K  stream=${COLOR_WIDTH}x${COLOR_HEIGHT}@${COLOR_FPS}" >&2

exec ros2 launch orbbec_camera femto_bolt.launch.py \
    depth_registration:=true \
    enable_ir:=false \
    enable_frame_sync:=false \
    enable_color_auto_exposure:=false \
    enable_color_auto_white_balance:=false \
    color_exposure:="${COLOR_EXPOSURE}" \
    color_gain:="${COLOR_GAIN}" \
    color_white_balance:="${COLOR_WB_K}" \
    color_width:="${COLOR_WIDTH}" \
    color_height:="${COLOR_HEIGHT}" \
    color_fps:="${COLOR_FPS}" \
    "$@"
