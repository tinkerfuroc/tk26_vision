import numpy as np
import rclpy
from fastapi.testclient import TestClient
from handeye_calib.handeye_web import HandeyeWebNode, make_app


def setup_module(_):
    rclpy.init()


def teardown_module(_):
    if rclpy.ok():
        rclpy.shutdown()


def _client():
    node = HandeyeWebNode()
    return node, TestClient(make_app(node))


def test_index_served():
    node, c = _client()
    try:
        r = c.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert "<html" in r.text.lower()
    finally:
        node.destroy_node()


def test_state_endpoint_no_hardware():
    node, c = _client()
    try:
        r = c.get("/api/state")
        assert r.status_code == 200
        assert r.json()["camera_connected"] is False
    finally:
        node.destroy_node()


def test_frame_endpoint_returns_jpeg_placeholder():
    node, c = _client()
    try:
        r = c.get("/api/frame.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"
    finally:
        node.destroy_node()


def test_frame_raw_query_returns_jpeg():
    """T2: ?raw=1 returns a JPEG (placeholder is fine with no camera)."""
    node, c = _client()
    try:
        r = c.get("/api/frame.jpg?raw=1")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"
    finally:
        node.destroy_node()


def test_action_endpoints_degrade_gracefully():
    node, c = _client()
    try:
        assert c.post("/api/capture", json={}).json()["ok"] is False
        assert c.post("/api/solve", json={}).json()["ok"] is False
        assert c.post("/api/move", json={"joints": [0, 0, 0]}).json()["ok"] is False
    finally:
        node.destroy_node()


def test_static_index_served_from_disk():
    node, c = _client()
    try:
        r = c.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        # the new static index must reference the static stylesheet path
        assert "/static/style.css" in r.text
    finally:
        node.destroy_node()


def test_settings_controls_served_on_info_tab():
    """The calib_frame + depth + emitter settings controls ship in index.html
    and app.js wires /api/config."""
    node, c = _client()
    try:
        html = c.get("/").text
        assert 'name="calib-frame"' in html and 'value="ir"' in html
        assert 'id="apply-config-btn"' in html and 'id="depth-weight-input"' in html
        assert 'id="ir-emitter-input"' in html
        js = c.get("/static/app.js").text
        assert "/api/config" in js and "renderSettings" in js
    finally:
        node.destroy_node()


def test_static_assets_served():
    node, c = _client()
    try:
        for asset, ct in (("style.css", "text/css"), ("app.js", "javascript")):
            r = c.get(f"/static/{asset}")
            assert r.status_code == 200, f"{asset} -> {r.status_code}"
            assert ct in r.headers["content-type"]
    finally:
        node.destroy_node()


def test_websocket_pushes_state():
    node, c = _client()
    try:
        with c.websocket_connect("/ws") as ws_conn:
            msg = ws_conn.receive_json()
            for key in ("camera_connected", "frame_hz", "samples", "stability", "diversity"):
                assert key in msg, f"missing {key} in WS message"
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# T4: capture gallery thumbnails + per-sample delete
# ---------------------------------------------------------------------------

def test_sample_thumb_404_for_missing_idx():
    """T4: GET /api/samples/{idx}/thumb.jpg returns 404 when no such sample."""
    node, c = _client()
    try:
        r = c.get("/api/samples/0/thumb.jpg")
        assert r.status_code == 404
    finally:
        node.destroy_node()


def test_delete_sample_endpoint_returns_json():
    """T4: DELETE /api/samples/{idx} returns JSON with ok+num_samples even
    when idx is out of range (graceful degradation, mirrors do_delete_sample)."""
    node, c = _client()
    try:
        r = c.delete("/api/samples/99")
        assert r.status_code == 200
        body = r.json()
        assert "ok" in body and "num_samples" in body
        assert body["ok"] is False
    finally:
        node.destroy_node()


def test_capture_rejected_without_settle():
    """T4: /api/capture must reject when the StabilityTracker is not steady.
    With no camera in this env the tracker is never steady, so we expect the
    settle-gate rejection reason ('not stable yet ...') rather than the v1
    'no camera frame' / 'no board detection' message."""
    node, c = _client()
    try:
        r = c.post("/api/capture", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert "stab" in body["reason"].lower()
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# T5: solve method picker
# ---------------------------------------------------------------------------

def test_api_solve_accepts_method_param():
    """T5: ``POST /api/solve`` accepts a JSON body ``{method: "auto"|"TSAI"|...}``.

    Without any captured samples the call still has to degrade gracefully —
    ``{"ok": False, "reason": ...}``. The contract is the route accepting the
    ``method`` param and forwarding it to ``do_solve``, not a successful solve.
    """
    node, c = _client()
    try:
        for method in ("auto", "TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"):
            r = c.post("/api/solve", json={"method": method})
            assert r.status_code == 200, f"{method} -> HTTP {r.status_code}"
            body = r.json()
            assert body["ok"] is False
            assert "reason" in body
        # And the default (no body) still works
        r = c.post("/api/solve", json={})
        assert r.status_code == 200 and r.json()["ok"] is False
    finally:
        node.destroy_node()


def test_do_solve_accepts_method_kwarg():
    """T5: ``HandeyeWebNode.do_solve(method=...)`` is the contract /api/solve forwards to.

    Calling with the canonical method strings must not raise even with no
    samples — degraded shape only.
    """
    node, _ = _client()
    try:
        for method in ("auto", "TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"):
            out = node.do_solve(method=method)
            assert out["ok"] is False
            assert "reason" in out
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# T6: Promote tab — diff endpoint returns BOTH yaml + xacro halves; apply with
# no ROBOT_NAME refuses the xacro half; vendor-path refusal
# ---------------------------------------------------------------------------

def _forge_last_solve(node):
    """Stamp ``node.last_solve`` with an identity-X SolveResult so the diff
    path is reachable in a no-hardware test environment."""
    from handeye_calib.handeye_solve import SolveResult
    node.last_solve = SolveResult(
        X=np.eye(4), Tbb=np.eye(4),
        train_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.001, "reproj_px": 0.5},
        heldout_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.001, "reproj_px": 0.5},
        status="PASS", per_method=[])


def test_promote_diff_no_solve_returns_ok_false():
    node, c = _client()
    try:
        body = c.get("/api/promote/diff").json()
        assert body["ok"] is False  # no solve run
    finally:
        node.destroy_node()


def test_promote_diff_yaml_only_when_robot_name_unset(monkeypatch):
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    node, c = _client()
    try:
        # forge a last_solve so the diff path is exercised
        _forge_last_solve(node)
        body = c.get("/api/promote/diff").json()
        assert body["ok"] is True
        assert body["xacro"] is None  # no ROBOT_NAME → no xacro half
        assert body["yaml"] is not None
        assert "target_path" in body["yaml"] and "diff" in body["yaml"]
    finally:
        node.destroy_node()


def test_promote_apply_xacro_refuses_when_robot_unset(monkeypatch):
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    node, c = _client()
    try:
        _forge_last_solve(node)
        r = c.post("/api/promote/apply", json={"which": "xacro"})
        body = r.json()
        assert body["ok"] is False and "ROBOT_NAME" in body["reason"]
    finally:
        node.destroy_node()


def test_promote_diff_endpoint_exists():
    """Smoke: the new GET /api/promote/diff route is registered."""
    node, c = _client()
    try:
        r = c.get("/api/promote/diff")
        # whether or not a solve has run, the route should be reachable (200).
        assert r.status_code == 200
        body = r.json()
        assert "ok" in body
    finally:
        node.destroy_node()


def test_promote_reload_endpoint_exists():
    """Smoke: POST /api/promote/reload returns JSON with ``ok``."""
    node, c = _client()
    try:
        r = c.post("/api/promote/reload")
        assert r.status_code == 200
        body = r.json()
        assert "ok" in body
    finally:
        node.destroy_node()


def test_state_payload_includes_safety_preview():
    """T3: ``state.safety_preview`` is wired into the WS/state payload (likely
    ``None`` here since there's no TF), so the Move tab's safety-status line can
    read it without duplicating SafetyEnvelope math in JS."""
    node, c = _client()
    try:
        r = c.get("/api/state")
        assert r.status_code == 200
        body = r.json()
        assert "safety_preview" in body
        # No TF in this env => the degraded shape: {"safe": None, "detail": str}
        sp = body["safety_preview"]
        if sp is not None:
            assert "safe" in sp and "detail" in sp
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# T1-wp: waypoint CRUD + per-robot YAML persistence
# ---------------------------------------------------------------------------

def test_waypoint_add_returns_no_current_joints_in_test_env():
    """Test env has no JointState → add must degrade gracefully."""
    node, c = _client()
    try:
        r = c.post("/api/waypoints", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False and "current joints" in body["reason"].lower()
    finally:
        node.destroy_node()


def test_waypoint_delete_out_of_range():
    node, c = _client()
    node.waypoint_store.clear()  # isolate from any persisted per-robot YAML
    try:
        r = c.delete("/api/waypoints/99")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert body["count"] == 0
    finally:
        node.destroy_node()


def test_waypoint_save_refuses_without_robot_name(monkeypatch):
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    node, c = _client()
    try:
        r = c.post("/api/waypoints/save", json={})
        body = r.json()
        assert body["ok"] is False and "ROBOT_NAME" in body["reason"]
    finally:
        node.destroy_node()


def test_state_payload_includes_waypoints():
    node, c = _client()
    node.waypoint_store.clear()  # isolate from any persisted per-robot YAML
    try:
        r = c.get("/api/state")
        assert r.status_code == 200
        body = r.json()
        assert "waypoints" in body
        assert body["waypoints"] == []  # empty after isolation
    finally:
        node.destroy_node()


# ---------------------------------------------------------------------------
# T3-seq: CaptureSequenceRunner endpoints
# ---------------------------------------------------------------------------

def test_sequence_start_refuses_empty():
    """T3-seq: POST /api/sequence/start refuses when no waypoints are recorded."""
    node, c = _client()
    node.waypoint_store.clear()  # isolate from any persisted per-robot YAML
    try:
        r = c.post("/api/sequence/start", json={"dry_run": False})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert "reason" in body
    finally:
        node.destroy_node()


def test_sequence_cancel_when_not_running():
    """T3-seq: POST /api/sequence/cancel is idempotent when no runner is live."""
    node, c = _client()
    try:
        r = c.post("/api/sequence/cancel", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
    finally:
        node.destroy_node()


def test_state_payload_includes_sequence():
    """T3-seq: state.sequence is present in WS/state payload (idle defaults)."""
    node, c = _client()
    try:
        r = c.get("/api/state")
        assert r.status_code == 200
        body = r.json()
        assert "sequence" in body
        seq = body["sequence"]
        assert seq["running"] is False
        assert seq["current_step"] == "idle"
        assert seq["current_idx"] is None
        assert seq["total"] == 0
        assert seq["dry_run"] is False
        assert seq["log"] == []
    finally:
        node.destroy_node()


def test_try_ffs_depth_returns_none_when_service_absent():
    """FFS graceful fallback: with no foundation_stereo service running, the
    depth client must time out on wait_for_service and return None (so capture
    degrades to monocular) rather than block forever or raise."""
    node, _ = _client()
    try:
        node._ffs_wait_for_service_s = 0.05   # don't make the test slow
        assert node._try_ffs_depth() is None
    finally:
        node.destroy_node()


def test_try_ffs_depth_disabled_returns_none():
    """use_ffs_depth=False short-circuits the client entirely."""
    node, _ = _client()
    try:
        node._use_ffs_depth = False
        assert node._try_ffs_depth() is None
    finally:
        node.destroy_node()


def test_try_ffs_depth_swallows_rclpy_errors():
    """Any rclpy-side failure (create_client / wait_for_service / call_async)
    must degrade to None, never raise out of do_capture (which would 500 the
    /api/capture request or abort the auto-capture sequence). Review finding #2."""
    node, _ = _client()
    try:
        def _boom(*a, **k):
            raise RuntimeError("simulated rclpy handle error")
        node.create_client = _boom            # force the create path to raise
        node._ffs_cli = None
        assert node._try_ffs_depth() is None   # swallowed, not propagated
    finally:
        node.destroy_node()


def _prime_for_capture(node, depth_hw=(48, 64)):
    """Drive a no-hardware node into a capturable state: steady, a fake color
    frame + K + ChArUco cap, and a monkeypatched identity base->eef TF so
    do_capture reaches the FFS depth block + try_add."""
    from types import SimpleNamespace
    M = 12
    obs_px = np.array([[10 + i * 3, 10 + i * 2] for i in range(M)], float)
    node._cap = {"T_cam_board": np.eye(4), "obs_px": obs_px,
                 "corner_idx": np.arange(M), "reproj_px": 0.5, "area_frac": 0.3}
    node._K = np.array([[60., 0, 32.], [0, 60., 24.], [0, 0, 1.]])
    node._frame = np.zeros((depth_hw[0], depth_hw[1], 3), np.uint8)
    node._frame_stamp = None
    node._stability_steady = True
    node._stability_since_frames = 9
    tfm = SimpleNamespace(transform=SimpleNamespace(
        translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
        rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0)))
    node.tf_buffer.lookup_transform = lambda *a, **k: tfm


def test_do_capture_ffs_branch_stores_depth():
    """Valid FFS depth at the corners -> sample carries obs_xyz_cam (the only
    on-hardware behavior change; review finding #3 coverage gap)."""
    node, _ = _client()
    try:
        _prime_for_capture(node)
        node._try_ffs_depth = lambda: np.full((48, 64), 0.5, np.float32)
        out = node.do_capture()
        assert out["ok"] is True
        assert node._last_depth_source == "ffs"
        s = node.session.samples[-1]
        assert s.obs_xyz_cam is not None and s.obs_xyz_valid.all()
    finally:
        node.destroy_node()


def test_do_capture_shape_mismatch_falls_back_monocular():
    node, _ = _client()
    try:
        _prime_for_capture(node)
        node._try_ffs_depth = lambda: np.full((10, 10), 0.5, np.float32)  # wrong shape
        out = node.do_capture()
        assert out["ok"] is True and node._last_depth_source == "shape-mismatch"
        assert node.session.samples[-1].obs_xyz_cam is None
    finally:
        node.destroy_node()


def test_do_capture_too_sparse_depth_falls_back_monocular():
    node, _ = _client()
    try:
        _prime_for_capture(node)
        node._try_ffs_depth = lambda: np.zeros((48, 64), np.float32)  # all invalid
        out = node.do_capture()
        assert out["ok"] is True and node._last_depth_source == "ffs-too-sparse"
        assert node.session.samples[-1].obs_xyz_cam is None
    finally:
        node.destroy_node()


def test_do_capture_drops_depth_if_pose_moved_during_ffs():
    """Review finding #4: the FFS call can block ~seconds; if the pose moved
    during it (steady flips False), the fresh stereo no longer matches the
    cached corners, so depth must be dropped (kept monocular)."""
    node, _ = _client()
    try:
        _prime_for_capture(node)

        def _moved():
            node._stability_steady = False     # simulate motion during the call
            return np.full((48, 64), 0.5, np.float32)
        node._try_ffs_depth = _moved
        out = node.do_capture()
        assert out["ok"] is True and node._last_depth_source == "moved-during-ffs"
        assert node.session.samples[-1].obs_xyz_cam is None
    finally:
        node.destroy_node()


def test_promote_yaml_invariant_across_calib_frame(monkeypatch):
    """The stored artifact must be frame-agnostic: observing the SAME physical
    camera body in color vs IR yields the identical camera_link mount
    (arm_to_camera) AND the identical color_optical reference. The solver's X is
    T_eef->observed_optical; the promote path composes back to camera_link via
    the right internal transform and always derives the color_optical reference.
    """
    import yaml as _yaml
    from handeye_calib.handeye_solve import SolveResult
    from scipy.spatial.transform import Rotation as _Rot
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    node, _ = _client()
    try:
        np = node._np
        # A plausible ground-truth camera-body pose T_eef->camera_link.
        M = np.eye(4)
        M[:3, :3] = _Rot.from_euler("xyz", [0.1, -0.2, 0.05]).as_matrix()
        M[:3, 3] = [0.067, -0.018, 0.024]
        T_cl_color = node._mount_to_color_matrix()
        T_cl_ir = node._mount_to_ir_optical_matrix()
        X_color = M @ T_cl_color   # T_eef->color_optical
        X_ir = M @ T_cl_ir         # T_eef->ir_optical

        def hand_eye_for(X, frame):
            node._calib_frame = frame
            node.last_solve = SolveResult(
                X=X, Tbb=np.eye(4),
                train_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.001, "reproj_px": 0.5},
                heldout_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.001, "reproj_px": 0.5},
                status="PASS", per_method=[])
            diff = node.compute_promote_diff()
            return _yaml.safe_load(diff["yaml"]["proposed_text"])["hand_eye"]

        yc = hand_eye_for(X_color, "color")
        yi = hand_eye_for(X_ir, "ir")
        for key in ("arm_to_camera_xyz", "arm_to_camera_rpy",
                    "color_optical_xyz", "color_optical_rpy"):
            np.testing.assert_allclose(
                [float(v) for v in yc[key].split()],
                [float(v) for v in yi[key].split()], atol=1e-6,
                err_msg=f"{key} differs between color and IR calibration")
        # And arm_to_camera must equal the ground-truth camera_link pose M.
        np.testing.assert_allclose(
            [float(v) for v in yc["arm_to_camera_xyz"].split()], M[:3, 3], atol=1e-6)
    finally:
        node.destroy_node()


def test_do_capture_ir_mode_uses_cached_ir_depth():
    """In calib_frame='ir', depth comes from the cached native-IR FFS stream
    (_get_ir_depth), not the color-aligned get_depth service."""
    node, _ = _client()
    try:
        _prime_for_capture(node)
        node._calib_frame = "ir"
        node._get_ir_depth = lambda: np.full((48, 64), 0.5, np.float32)
        # If the code wrongly used the color service path it'd return None here:
        node._try_ffs_depth = lambda: None
        out = node.do_capture()
        assert out["ok"] is True and node._last_depth_source == "ffs"
        assert node.session.samples[-1].obs_xyz_cam is not None
    finally:
        node.destroy_node()


def test_set_calib_frame_clears_frame_specific_samples():
    """Switching observation frame must discard the accumulated samples — they
    are tied to a specific frame's intrinsics; mixing color-K and IR-K samples
    would corrupt the solve."""
    node, _ = _client()
    try:
        _prime_for_capture(node)
        node._try_ffs_depth = lambda: None      # color, monocular
        node.do_capture()
        assert len(node.session.samples) == 1
        node._set_calib_frame("ir")
        assert node._calib_frame == "ir"
        assert len(node.session.samples) == 0
        assert node._sample_depth_source == {} and node._thumbs == {}
        # Same-frame "switch" is a no-op (doesn't clear).
        _prime_for_capture(node)
        node._get_ir_depth = lambda: None
        node.do_capture()
        assert len(node.session.samples) == 1
        node._set_calib_frame("ir")
        assert len(node.session.samples) == 1
    finally:
        node.destroy_node()


def test_api_config_sets_knobs_and_switches_frame():
    """POST /api/config live-updates the depth knobs + calib_frame and surfaces
    them in state.config."""
    node, c = _client()
    try:
        r = c.post("/api/config", json={
            "calib_frame": "ir", "depth_weight": 0.8,
            "use_ffs_depth": False, "depth_win": 3, "depth_sigma_m": 0.004})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert node._calib_frame == "ir"
        assert abs(node._depth_weight - 0.8) < 1e-9
        assert node._use_ffs_depth is False
        assert node._depth_win == 3
        assert abs(node._depth_sigma_m - 0.004) < 1e-12
        cfg = c.get("/api/state").json()["config"]
        assert cfg["calib_frame"] == "ir"
        assert abs(cfg["depth_weight"] - 0.8) < 1e-9
        assert cfg["use_ffs_depth"] is False
    finally:
        node.destroy_node()


def test_api_config_emitter_degrades_without_camera_node():
    """Setting the IR emitter must degrade gracefully (no camera node in the test
    env) — the other knobs still apply, the emitter sub-result reports the miss."""
    node, c = _client()
    try:
        node._emitter_wait_s = 0.05  # don't make the test slow
        r = c.post("/api/config", json={"depth_weight": 0.6, "ir_emitter_enabled": False})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True                 # knobs applied regardless
        assert abs(node._depth_weight - 0.6) < 1e-9
        assert "emitter" in body and body["emitter"]["ok"] is False
    finally:
        node.destroy_node()


def test_api_config_ignores_bad_calib_frame():
    node, c = _client()
    try:
        before = node._calib_frame
        r = c.post("/api/config", json={"calib_frame": "lidar"})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert node._calib_frame == before       # unchanged, no crash
    finally:
        node.destroy_node()


def test_do_capture_drops_sample_if_frame_switched_mid_capture():
    """Review HIGH: a concurrent calib_frame switch (which clears the session)
    during the depth call must not let the stale-frame sample land in the new
    session — do_capture re-checks the frame before committing."""
    node, _ = _client()
    try:
        _prime_for_capture(node)

        def _switch_then_none():
            node._set_calib_frame("ir")   # concurrent frame switch during depth call
            return None
        node._try_ffs_depth = _switch_then_none
        out = node.do_capture()
        assert out["ok"] is False and "switched" in out["reason"]
        assert len(node.session.samples) == 0
    finally:
        node.destroy_node()


def test_do_capture_ir_drops_stale_depth():
    """Review MED: a frozen/old native-IR depth stream must not feed stale
    geometry — depth older than the frame is dropped (monocular)."""
    from builtin_interfaces.msg import Time as TimeMsg
    node, _ = _client()
    try:
        _prime_for_capture(node)
        node._calib_frame = "ir"
        now = node.get_clock().now().to_msg()
        node._frame_stamp = now
        old = TimeMsg(); old.sec = int(now.sec) - 5; old.nanosec = int(now.nanosec)
        node._ir_depth = np.full((48, 64), 0.5, np.float32)
        node._ir_depth_stamp = old        # 5 s stale vs the 1 s max age
        out = node.do_capture()           # real _get_ir_depth + staleness check
        assert out["ok"] is True and node._last_depth_source == "ir-depth-stale"
        assert node.session.samples[-1].obs_xyz_cam is None
    finally:
        node.destroy_node()


def test_api_config_ignores_unknown_and_reserved_keys():
    """Review low: a stray body key (e.g. 'self') must not 500 the request."""
    node, c = _client()
    try:
        r = c.post("/api/config", json={"self": 1, "garbage": "x", "depth_weight": 0.7})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert abs(node._depth_weight - 0.7) < 1e-9
    finally:
        node.destroy_node()


def test_api_config_without_emitter_key_does_not_command_emitter():
    """Review MED: a config POST that omits ir_emitter_enabled must not issue a
    SetParameters call (the UI now only sends it when the operator toggled it)."""
    node, c = _client()
    try:
        body = c.post("/api/config", json={"depth_weight": 0.5}).json()
        assert body["ok"] is True and "emitter" not in body
    finally:
        node.destroy_node()


def test_json_response_scrubs_nan_inf():
    """A NaN/Inf in any do_* return must not break the wire.

    Starlette's ``JSONResponse.render`` calls ``json.dumps(allow_nan=False)``;
    pre-fix a non-finite float triggered a plain-text 500 the browser surfaced
    as ``JSON.parse: unexpected character at line 1 column 1``. The boundary
    scrub in ``make_app`` must turn NaN/Inf into ``null`` so the call stays
    200 + parseable JSON.
    """
    node, c = _client()
    try:
        node.do_solve = lambda method="auto", reject_sigma=None: {
            "ok": True,
            "X_xyz_mm": [1.0, float("nan"), float("inf")],
            "train_metrics": {"trans_rmse_m": float("nan"), "reproj_px": 0.5},
            "per_sample_reproj_px": [0.1, float("nan"), 0.2],
        }
        r = c.post("/api/solve", json={"method": "auto"})
        assert r.status_code == 200
        body = r.json()  # would raise pre-fix
        assert body["X_xyz_mm"] == [1.0, None, None]
        assert body["train_metrics"]["trans_rmse_m"] is None
        assert body["per_sample_reproj_px"][1] is None
    finally:
        node.destroy_node()


def test_anchor_endpoint_graceful_without_head_camera():
    node, c = _client()
    try:
        r = c.post("/api/anchor")
        assert r.status_code == 200
        body = r.json()
        # No head frames have arrived -> ok:False with a clear reason, never 500.
        assert body["ok"] is False
        assert "head" in body["reason"].lower()
        assert body["n_anchor_obs"] == 0
    finally:
        node.destroy_node()


def test_anchor_clear_is_idempotent():
    node, c = _client()
    try:
        r = c.post("/api/anchor/clear")
        assert r.status_code == 200 and r.json()["ok"] is True
        assert r.json()["n_anchor_obs"] == 0
    finally:
        node.destroy_node()
