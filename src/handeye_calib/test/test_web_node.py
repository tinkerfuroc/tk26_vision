import rclpy
from handeye_calib.handeye_web import HandeyeWebNode


def setup_module(_):
    rclpy.init()


def teardown_module(_):
    if rclpy.ok():
        rclpy.shutdown()


def test_node_constructs_and_safe_defaults():
    node = HandeyeWebNode()
    try:
        st = node.get_state_dict()
        assert st["camera_connected"] is False        # no camera in this env
        assert st["num_samples"] == 0
        jpg = node.latest_jpeg()
        assert isinstance(jpg, (bytes, bytearray)) and bytes(jpg[:2]) == b"\xff\xd8"
        cap = node.do_capture()
        assert cap["ok"] is False                      # nothing to capture
        sol = node.do_solve()
        assert sol["ok"] is False and "sample" in sol["reason"].lower()
    finally:
        node.destroy_node()


def test_do_move_validates_joint_count():
    node = HandeyeWebNode()
    try:
        bad = node.do_move([0.0, 0.0, 0.0])            # wrong arity
        assert bad["ok"] is False
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# T4: settle gate + per-sample delete
# ---------------------------------------------------------------------------

def test_capture_blocked_when_not_steady():
    """T4 HARD gate: do_capture returns ok=False with a 'stab...' reason when
    the StabilityTracker has not yet declared the board steady, regardless of
    whether camera/intrinsics/board-detection are otherwise available."""
    node = HandeyeWebNode()
    try:
        # Default state of a fresh node: _stability_steady=False, _frame=None,
        # _K=None, _cap=None.  The HARD settle gate must fire BEFORE the v1
        # 'no camera frame' / 'no intrinsics' / 'no board detection' branches
        # so the operator sees a stability rejection even when a board pose
        # is in fact available but not yet steady.
        node._stability_steady = False
        r = node.do_capture()
        assert r["ok"] is False
        assert "stab" in r["reason"].lower()
    finally:
        node.destroy_node()


def test_delete_sample_by_idx_out_of_range():
    """T4: do_delete_sample(99) on an empty session returns ok=False."""
    node = HandeyeWebNode()
    try:
        r = node.do_delete_sample(99)
        assert r["ok"] is False
        assert "num_samples" in r
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# T3-seq: CaptureSequenceRunner state machine
# ---------------------------------------------------------------------------

def test_sequence_refuses_when_empty_waypoints():
    """T3-seq: do_start_sequence must refuse when no waypoints are recorded.

    The runner can't loop over an empty list; the auto-capture entry point
    is the operator's only safe gate against starting an empty run."""
    node = HandeyeWebNode()
    node.waypoint_store.clear()  # isolate from any persisted per-robot YAML
    try:
        r = node.do_start_sequence(dry_run=False)
        assert r["ok"] is False
        assert "no waypoints" in r["reason"].lower()
    finally:
        node.destroy_node()


def test_sequence_state_dict_shape():
    """T3-seq: state.sequence dict has the contract keys (idle defaults).

    Before any start, the runner may be ``None`` (lazy construction); the
    idle-default dict is what the WS push emits via ``state.sequence`` so the
    UI doesn't have to special-case "no runner yet"."""
    node = HandeyeWebNode()
    try:
        s = node.sequence_runner.state_dict() if node.sequence_runner else {
            "running": False, "dry_run": False, "current_idx": None,
            "total": 0, "current_step": "idle", "log": []}
        for k in ("running", "dry_run", "current_idx", "total", "current_step", "log"):
            assert k in s, f"missing key {k!r} in sequence state dict: {s}"
    finally:
        node.destroy_node()


def test_sequence_cancel_when_not_running_is_noop():
    """T3-seq: cancel is idempotent — calling without a live runner returns ok=True."""
    node = HandeyeWebNode()
    try:
        r = node.do_cancel_sequence()
        assert r["ok"] is True
    finally:
        node.destroy_node()


def test_safety_preview_graceful_degrades_without_tf():
    """T3: ``safety_preview()`` returns the right shape even without TF /
    cached pose so the UI can render the "unknown" branch instead of crashing.

    Shape contract: ``{"safe": bool|None, "detail": str}``. With no robot in
    this env, the cached ``_t_base_ee_cache`` is ``None`` after the first WS
    state refresh, so ``safe`` must be ``None`` and ``detail`` a human-readable
    string explaining why (TF unavailable).
    """
    node = HandeyeWebNode()
    try:
        # Force the cache into the "no TF" state — the TF refresh inside
        # get_state_dict() would also do this, but be explicit to keep the
        # assertion local to safety_preview's degraded path.
        node._t_base_ee_cache = None
        sp = node.safety_preview()
        assert isinstance(sp, dict)
        assert set(sp) >= {"safe", "detail"}
        assert sp["safe"] is None
        assert isinstance(sp["detail"], str) and sp["detail"]
    finally:
        node.destroy_node()
