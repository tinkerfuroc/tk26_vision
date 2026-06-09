#!/usr/bin/env bash
# kill_stale_bench.sh — manual cleanup mirror of the launch's kill_stale guard.
#
# Mirrors the STALE_PATTERNS in track_web_bench.launch.py exactly. Use this
# when you need to clean up orphaned bench processes by hand (e.g. after an
# ungraceful terminal close that reparented nodes to PID 1).
#
# Patterns are scoped to the installed lib/<pkg>/ exec paths so that an
# editor buffer, a grep, or a running ros2 launch are never matched.
# SIGTERM (default, NOT -9) lets each stale node release cameras/GPU cleanly.
set -u

echo "[kill_stale] SIGTERM stale: lib/vision_track/person_track_server"
pkill -f 'lib/vision_track/person_track_server' || true

echo "[kill_stale] SIGTERM stale: lib/vision_track/track_web"
pkill -f 'lib/vision_track/track_web' || true

echo "[kill_stale] SIGTERM stale: lib/tk_vision_specialized/waving_person_server"
pkill -f 'lib/tk_vision_specialized/waving_person_server' || true
