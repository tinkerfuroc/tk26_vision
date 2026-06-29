"""Endpoint tests for the capture-session history browser.

Drives the real FastAPI app (no hardware) with HANDEYE_DUMP_DIR pointed at a tmp
dir, so persistence + the /api/sessions browse/load/delete routes are exercised
end-to-end without touching the real calibration_data tree.
"""
import numpy as np
import pytest
import rclpy
from fastapi.testclient import TestClient
from handeye_calib.handeye_web import HandeyeWebNode, make_app
from handeye_calib import handeye_model as hm


def setup_module(_):
    rclpy.init()


def teardown_module(_):
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDEYE_DUMP_DIR", str(tmp_path))
    return tmp_path


def _client():
    node = HandeyeWebNode()
    return node, TestClient(make_app(node))


def _inject_capture(node, n=3):
    """Mimic n successful captures (no hardware): samples + sidecars + thumbs."""
    with node.lock:
        node.session.samples = [
            hm.Sample(np.eye(4), np.eye(4), np.zeros((4, 2)), np.arange(4))
            for _ in range(n)
        ]
        for i in range(n):
            node._thumbs[i] = b"\xff\xd8" + f"thumb{i}".encode()
            node._sample_reproj_px[i] = 0.2 + 0.01 * i
            node._sample_depth_source[i] = "ffs"
            node._sample_joints[i] = [0.0] * 7


def test_sessions_list_empty(env):
    node, c = _client()
    try:
        r = c.get("/api/sessions")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True and body["sessions"] == []
    finally:
        node.destroy_node()


def test_capture_persists_and_is_browsable(env):
    node, c = _client()
    try:
        _inject_capture(node, 3)
        name = node._persist_session()           # what do_capture calls
        assert name and (env / "wrist_handeye_sessions" / name / "session.json").is_file()

        # listed
        sessions = c.get("/api/sessions").json()["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["name"] == name and sessions[0]["n_samples"] == 3
        assert sessions[0]["has_solve"] is False

        # detail carries the samples
        detail = c.get(f"/api/sessions/{name}").json()
        assert detail["ok"] is True and len(detail["samples"]) == 3

        # thumbnail served from disk
        t = c.get(f"/api/sessions/{name}/samples/1/thumb.jpg")
        assert t.status_code == 200 and t.content[:2] == b"\xff\xd8"
    finally:
        node.destroy_node()


def test_detail_404_for_unknown_session(env):
    node, c = _client()
    try:
        r = c.get("/api/sessions/wrist_handeye_nope")
        assert r.status_code == 404 and r.json()["ok"] is False
    finally:
        node.destroy_node()


def test_thumb_placeholder_when_missing(env):
    node, c = _client()
    try:
        _inject_capture(node, 1)
        name = node._persist_session()
        # index 5 was never captured -> placeholder JPEG, never a 500
        r = c.get(f"/api/sessions/{name}/samples/5/thumb.jpg")
        assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"
    finally:
        node.destroy_node()


def test_load_session_rehydrates_live_state(env):
    node, c = _client()
    try:
        _inject_capture(node, 4)
        name = node._persist_session()
        # Simulate a restart: live state wiped, but the session is on disk.
        with node.lock:
            node.session.samples.clear()
            node._thumbs.clear()
            node._session_name = None
        r = c.post(f"/api/sessions/{name}/load")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True and body["num_samples"] == 4
        assert len(node.session.samples) == 4
        assert node._session_name == name          # re-solve folds back into it
        # thumbnails came back from disk too
        assert node.sample_thumb(0)[:2] == b"\xff\xd8"
    finally:
        node.destroy_node()


def test_delete_session(env):
    node, c = _client()
    try:
        _inject_capture(node, 2)
        name = node._persist_session()
        r = c.delete(f"/api/sessions/{name}")
        assert r.status_code == 200 and r.json()["ok"] is True
        assert c.get("/api/sessions").json()["sessions"] == []
    finally:
        node.destroy_node()


def test_delete_then_capture_starts_new_session(env):
    """After a calib-frame clear the next capture must start a NEW session dir,
    leaving the prior one intact as history."""
    node, c = _client()
    try:
        _inject_capture(node, 2)
        first = node._persist_session()
        # _set_calib_frame clear resets _session_name; emulate it directly.
        with node.lock:
            node.session.samples.clear()
            node._thumbs.clear()
            node._session_name = None
        _inject_capture(node, 3)
        second = node._persist_session()
        assert second != first
        names = {s["name"] for s in c.get("/api/sessions").json()["sessions"]}
        assert {first, second} <= names
    finally:
        node.destroy_node()
