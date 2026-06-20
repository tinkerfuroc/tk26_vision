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
