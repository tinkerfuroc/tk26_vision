#!/usr/bin/env bash
# Shared helpers for tk26_vision integration tests. Source, don't execute.
#
# Exports: WS_ROOT, VENV, ROS_SETUP, LOG_DIR, ENV_FILE
# Functions: source_envs, pass/fail/skip/summary, have_api_key,
#            start_node, stop_all_nodes, wait_for_service, wait_for_action,
#            wait_for_topic_hz, assert_log_grep, assert_log_nogrep

WS_ROOT="${WS_ROOT:-$HOME/tk25_ws}"
VENV="$WS_ROOT/src/tk26_vision/.venv-vision-main"
ROS_SETUP="${ROS_SETUP:-/opt/ros/humble/setup.bash}"
LOG_DIR="${LOG_DIR:-$WS_ROOT/src/tk26_vision/scripts/tests/logs}"
ENV_FILE="${ENV_FILE:-$WS_ROOT/.env}"

mkdir -p "$LOG_DIR"

PASS_N=0; FAIL_N=0; SKIP_N=0
FAILURES=()

pass()    { PASS_N=$((PASS_N+1)); printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail()    { FAIL_N=$((FAIL_N+1)); FAILURES+=("$1${2:+ — $2}"); printf '  \033[31mFAIL\033[0m  %s%s\n' "$1" "${2:+ — $2}"; }
skip()    { SKIP_N=$((SKIP_N+1)); printf '  \033[33mSKIP\033[0m  %s%s\n' "$1" "${2:+ — $2}"; }
section() { printf '\n=== %s ===\n' "$1"; }

summary() {
    printf '\n==== SUMMARY: %d pass / %d fail / %d skip ====\n' "$PASS_N" "$FAIL_N" "$SKIP_N"
    if [ "$FAIL_N" -gt 0 ]; then
        printf 'Failures:\n'
        for f in "${FAILURES[@]}"; do printf '  - %s\n' "$f"; done
        return 1
    fi
    return 0
}

source_envs() {
    set +u
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    # shellcheck disable=SC1091
    source "$ROS_SETUP"
    [ -f "$WS_ROOT/install/setup.bash" ] && source "$WS_ROOT/install/setup.bash"
    set -u
    export ROS2_PTH_WARNED=1
}

have_api_key() {
    [ -f "$ENV_FILE" ] || return 1
    grep -qE '^OPENROUTER_API_KEY=sk-or-[^R]' "$ENV_FILE" 2>/dev/null
}

# -- Background node management --
NODE_PIDS=(); NODE_PGIDS=(); LAST_LOG=""

# start_node <tag> <pkg> <entry> [ros2 run args...]
start_node() {
    local tag="$1" pkg="$2" entry="$3"
    shift 3
    local log="$LOG_DIR/$tag.log"
    : >"$log"
    setsid ros2 run "$pkg" "$entry" "$@" >>"$log" 2>&1 &
    local pid=$!
    sleep 0.1
    local pgid
    pgid=$(ps -o pgid= "$pid" 2>/dev/null | tr -d ' ' || echo "$pid")
    NODE_PIDS+=("$pid")
    NODE_PGIDS+=("$pgid")
    LAST_LOG="$log"
}

stop_all_nodes() {
    local pgid pid
    for pgid in "${NODE_PGIDS[@]:-}"; do
        [ -n "$pgid" ] && kill -TERM -"$pgid" 2>/dev/null || true
    done
    sleep 0.5
    for pgid in "${NODE_PGIDS[@]:-}"; do
        [ -n "$pgid" ] && kill -KILL -"$pgid" 2>/dev/null || true
    done
    for pid in "${NODE_PIDS[@]:-}"; do
        [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
    done
    NODE_PIDS=(); NODE_PGIDS=()
}

wait_for_service() {
    local srv="$1" timeout="${2:-10}" elapsed=0
    while [ "$elapsed" -lt "$timeout" ]; do
        ros2 service list 2>/dev/null | grep -qx "$srv" && return 0
        sleep 1; elapsed=$((elapsed+1))
    done
    return 1
}

wait_for_action() {
    local act="$1" timeout="${2:-10}" elapsed=0
    while [ "$elapsed" -lt "$timeout" ]; do
        ros2 action list 2>/dev/null | grep -qx "$act" && return 0
        sleep 1; elapsed=$((elapsed+1))
    done
    return 1
}

wait_for_topic_hz() {
    local topic="$1" min_hz="${2:-5}" timeout="${3:-15}"
    local hz
    hz=$(timeout "$timeout" ros2 topic hz "$topic" 2>&1 | grep -oP 'average rate: \K[0-9.]+' | head -1 || true)
    [ -n "$hz" ] && awk "BEGIN{exit !($hz >= $min_hz)}"
}

# assert_log_grep <log> <pattern> — return 0 if pattern matches (silent)
assert_log_grep() { grep -qE "$2" "$1" 2>/dev/null; }
assert_log_nogrep() { ! grep -qE "$2" "$1" 2>/dev/null; }

# On any exit, kill everything we spawned
cleanup_all() { stop_all_nodes; }
trap cleanup_all EXIT INT TERM
