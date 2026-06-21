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
        node.do_solve = lambda method="auto": {
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
