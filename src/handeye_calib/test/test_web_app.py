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
