#!/usr/bin/env bash
# Launch the Orbbec Femto Bolt via orbbec_camera with the FastDDS SHM profile
# and the tk26_vision tuned overrides (depth registration on, IR off, SDK-side
# frame sync off). Without these, the vendored launch drops to ~3 Hz; see
# src/tk26_vision/CAMERA_BRINGUP.md for the full root-cause writeup.
#
# Extra args are forwarded to ros2 launch, e.g.
#   ./launch_orbbec_shm.sh enable_point_cloud:=true
#
# The workspace install/setup must already be sourced in the calling shell
# (this script does not source it — consistent with tk26_vision/scripts/build.sh
# only sourcing what it owns).

set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
PROFILES_FILE="$REPO_ROOT/config/fastdds_shm.xml"

if [ ! -f "$PROFILES_FILE" ]; then
    echo "error: FastDDS SHM profile not found: $PROFILES_FILE" >&2
    exit 1
fi

export FASTRTPS_DEFAULT_PROFILES_FILE="$PROFILES_FILE"

exec ros2 launch orbbec_camera femto_bolt.launch.py \
    depth_registration:=true \
    enable_colored_point_cloud:=true \
    enable_ir:=false \
    enable_frame_sync:=false \
    "$@"
