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
