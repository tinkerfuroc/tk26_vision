# Copyright 2026 Tinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""track_web_app endpoints against a fake bridge (no ROS)."""
import asyncio
import json

from fastapi.testclient import TestClient

from vision_track.track_web_app import create_app


class FakeBridge:
    def __init__(self):
        self.calls = []
        self._state = {"ts": 1.0, "reacquisition_state": 2, "candidates": []}
        self._jpeg = b"\xff\xd8fakejpeg\xff\xd9"

    def snapshot(self):
        return {"state": self._state, "state_age_s": 0.1,
                "goal": {"held": False, "observer": False}, "gallery_version": 0}

    def latest_state(self):
        return 7, self._state

    def latest_gallery(self):
        return {"version": 0, "thumbs": []}

    def latest_jpeg(self):
        return 3, self._jpeg

    def start_goal(self):
        self.calls.append("start")
        return {"ok": True, "message": "goal sent"}

    def stop_goal(self):
        self.calls.append("stop")
        return {"ok": True, "message": "cancelled"}

    def reseed(self, bbox):
        self.calls.append(("reseed", tuple(bbox)))
        return {"success": True, "target_track_id": 9, "message": "reseeded"}

    def wave(self):
        self.calls.append("wave")
        return {"status": 0, "boxes": [[1, 2, 3, 4]], "points": [[0.5, 0.1, 2.0]]}


def _client():
    b = FakeBridge()
    return b, TestClient(create_app(b, webui_dir=None))


def test_status():
    b, c = _client()
    r = c.get("/api/status")
    assert r.status_code == 200 and r.json()["goal"] == {"held": False, "observer": False}


def test_goal_and_wave_roundtrip():
    b, c = _client()
    assert c.post("/api/goal/start").json()["ok"] is True
    assert c.post("/api/goal/stop").json()["ok"] is True
    assert c.post("/api/wave").json()["boxes"] == [[1, 2, 3, 4]]
    assert b.calls[:3] == ["start", "stop", "wave"]


def test_reseed_validates_bbox():
    b, c = _client()
    assert c.post("/api/reseed", json={"bbox": [1, 2, 30, 40]}).json()["success"] is True
    assert ("reseed", (1, 2, 30, 40)) in b.calls
    assert c.post("/api/reseed", json={"bbox": [1, 2]}).status_code == 422
    assert c.post("/api/reseed", json={"bbox": [30, 40, 1, 2]}).status_code == 422


def test_ws_pushes_state():
    b, c = _client()
    with c.websocket_connect("/ws/state") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "state" and msg["data"]["reacquisition_state"] == 2


def test_mjpeg_stream_first_frame_and_media_type():
    # Starlette's TestClient buffers responses to completion, so it cannot
    # consume an (intentionally) infinite MJPEG stream — drive the endpoint's
    # generator directly instead: call the route function with a stub Request,
    # check the response shape, and pull exactly one frame.
    b = FakeBridge()
    app = create_app(b, webui_dir=None)
    route = next(r for r in app.routes if getattr(r, "path", None) == "/stream.mjpg")

    class _ConnectedReq:
        async def is_disconnected(self):
            return False

    async def _first_chunk():
        resp = route.endpoint(_ConnectedReq())
        assert "multipart/x-mixed-replace" in resp.media_type
        chunk = await resp.body_iterator.__anext__()
        await resp.body_iterator.aclose()
        return chunk

    chunk = asyncio.run(_first_chunk())
    # Pin the full part format: boundary, headers, header/body separator,
    # hand-computed Content-Length, and trailing CRLF.
    assert chunk.startswith(b"--frame\r\n")
    assert b"Content-Type: image/jpeg\r\n" in chunk
    assert b"Content-Length: " + str(len(b._jpeg)).encode() + b"\r\n\r\n" in chunk
    assert b"fakejpeg" in chunk and chunk.endswith(b"\r\n")


def test_unexpected_bridge_error_returns_json_500():
    class BoomBridge(FakeBridge):
        def wave(self):
            raise RuntimeError("boom")

    c = TestClient(create_app(BoomBridge(), webui_dir=None),
                   raise_server_exceptions=False)
    r = c.post("/api/wave")
    assert r.status_code == 500
    assert "boom" in r.json()["error"]


def test_mjpeg_stream_ends_on_client_disconnect():
    b = FakeBridge()
    app = create_app(b, webui_dir=None)
    route = next(r for r in app.routes if getattr(r, "path", None) == "/stream.mjpg")

    class _GoneReq:
        async def is_disconnected(self):
            return True

    async def _consume():
        resp = route.endpoint(_GoneReq())
        return [chunk async for chunk in resp.body_iterator]

    assert asyncio.run(_consume()) == []   # generator exits without yielding


def test_webui_served_from_dir(tmp_path):
    (tmp_path / "index.html").write_text("<html>ok</html>")
    (tmp_path / "style.css").write_text("body{}")
    (tmp_path / "app.js").write_text("'use strict';")
    b = FakeBridge()
    c = TestClient(create_app(b, webui_dir=tmp_path))
    assert c.get("/").status_code == 200
    assert "javascript" in c.get("/app.js").headers["content-type"]
