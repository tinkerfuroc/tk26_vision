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
"""ROS-free FastAPI app for the track_web dashboard.

``create_app(bridge, webui_dir)`` wires HTTP/WS/MJPEG endpoints to a bridge
object (the ROS node in production, a fake in tests). The bridge contract:

    snapshot() -> dict                      # /api/status payload
    latest_state() -> (seq:int, dict|None)  # newest ~/debug_state
    latest_gallery() -> dict|None           # newest ~/debug_gallery payload
    latest_jpeg() -> (seq:int, bytes|None)  # newest annotated frame as JPEG
    start_goal() / stop_goal() -> dict      # {ok: bool, message: str}
    reseed(bbox:[x1,y1,x2,y2]) -> dict      # ReseedTarget response fields
    wave() -> dict                          # DetectWaving boxes/points or error
    record_start() / record_stop() -> dict  # ros2 bag record control (offline replay)

All bridge methods must be thread-safe; handlers poll (no cross-thread asyncio).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, field_validator

_STATE_POLL_S = 0.03    # ~33 Hz cap on WS pushes
_MJPEG_POLL_S = 1 / 15  # 15 fps cap on the stream


class ReseedRequest(BaseModel):
    bbox: List[int]

    @field_validator("bbox")
    @classmethod
    def _valid_box(cls, v):
        if len(v) != 4 or v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError("bbox must be [x1,y1,x2,y2] with x2>x1, y2>y1")
        return v


def create_app(bridge, webui_dir: Optional[Path] = None) -> FastAPI:
    """Build the FastAPI app around a (real or fake) tracker bridge."""
    app = FastAPI(title="track_web")

    # Spec §4: the server must never crash and always answers JSON. Expected
    # failures (service timeout/unavailable) come back as error dicts from the
    # bridge; this catches the UNEXPECTED ones (a real bug) so clients doing
    # response.json() still get a parseable body instead of a text 500.
    @app.exception_handler(Exception)
    async def _unhandled(request, exc):  # noqa: ARG001 - FastAPI handler shape
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"},
                            status_code=500)

    if webui_dir is not None and Path(webui_dir).exists():
        webui = Path(webui_dir)

        @app.get("/")
        def index():
            return FileResponse(webui / "index.html", media_type="text/html")

        @app.get("/style.css")
        def style():
            return FileResponse(webui / "style.css", media_type="text/css")

        @app.get("/app.js")
        def appjs():
            return FileResponse(webui / "app.js",
                                media_type="application/javascript")
    else:
        @app.get("/")
        def index_missing():
            return JSONResponse({"error": "webui dir not found"}, status_code=500)

    @app.get("/api/status")
    def status():
        return bridge.snapshot()

    @app.post("/api/goal/start")
    def goal_start():
        return bridge.start_goal()

    @app.post("/api/goal/stop")
    def goal_stop():
        return bridge.stop_goal()

    @app.post("/api/reseed")
    def reseed(req: ReseedRequest):
        return bridge.reseed(req.bbox)

    @app.post("/api/wave")
    def wave():
        return bridge.wave()

    @app.post("/api/record/start")
    def record_start():
        return bridge.record_start()

    @app.post("/api/record/stop")
    def record_stop():
        return bridge.record_stop()

    # Group routes are declared BEFORE the single-name routes: FastAPI matches
    # in declaration order, so the literal "group" segment in /api/proc/group/...
    # would otherwise be captured by the {name} param of /api/proc/{name}/...
    @app.post("/api/proc/group/{group}/start")
    def proc_group_start(group: str):
        return bridge.proc_group_start(group)

    @app.post("/api/proc/group/{group}/stop")
    def proc_group_stop(group: str):
        return bridge.proc_group_stop(group)

    @app.get("/api/follow/status")
    def follow_status():
        return bridge.follow_status()

    @app.post("/api/proc/{name}/start")
    def proc_start(name: str):
        return bridge.proc_start(name)

    @app.post("/api/proc/{name}/stop")
    def proc_stop(name: str):
        return bridge.proc_stop(name)

    @app.get("/api/proc/status")
    def proc_status():
        return bridge.proc_status()

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        await ws.accept()
        last_state_seq = -1
        last_gallery_version = -1
        last_proc = None
        last_follow = None
        try:
            while True:
                seq, state = bridge.latest_state()
                if state is not None and seq != last_state_seq:
                    last_state_seq = seq
                    await ws.send_text(json.dumps({"type": "state", "data": state}))
                    gal = bridge.latest_gallery()
                    if gal is not None and gal.get("version", -1) != last_gallery_version:
                        last_gallery_version = gal["version"]
                        await ws.send_text(json.dumps({"type": "gallery", "data": gal}))
                # Push the bringup-process map whenever it changes (and on the
                # first iteration, since last_proc starts as a sentinel None that
                # never equals a real status dict).
                proc = bridge.proc_status()
                if proc != last_proc:
                    last_proc = proc
                    await ws.send_text(json.dumps({"type": "proc", "data": proc}))
                # Push live follow state alongside proc so the panel updates as
                # /follow_server/status arrives. Guarded for older bridges/fakes.
                follow = (bridge.follow_status()
                          if hasattr(bridge, "follow_status") else {})
                if follow != last_follow:
                    last_follow = follow
                    await ws.send_text(json.dumps({"type": "follow", "data": follow}))
                await asyncio.sleep(_STATE_POLL_S)
        except WebSocketDisconnect:
            return

    async def _mjpeg_gen(request: Request):
        # Infinite by design: the stream outlives source stalls (goal stopped,
        # tracker idle) so the <img> resumes the moment frames return. The only
        # exit is the HTTP client going away. NOTE: Starlette's TestClient
        # buffers responses to completion and so cannot consume this endpoint —
        # tests drive this generator directly instead (see test_track_web_app).
        # Re-emit the last frame on a heartbeat (~every 0.5s) even when the source
        # seq is unchanged. A multipart <img> stream that goes quiet (target lost,
        # goal idle) otherwise latches the browser on the last frame and won't
        # resume when frames return; the heartbeat keeps the stream flowing.
        heartbeat_polls = max(1, int(0.5 / _MJPEG_POLL_S))
        last_seq = -1
        idle_polls = 0
        while True:
            if await request.is_disconnected():
                return
            seq, jpeg = bridge.latest_jpeg()
            idle_polls += 1
            fresh = jpeg is not None and seq != last_seq
            if jpeg is not None and (fresh or idle_polls >= heartbeat_polls):
                last_seq = seq
                idle_polls = 0
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
            await asyncio.sleep(_MJPEG_POLL_S)

    @app.get("/stream.mjpg")
    def stream(request: Request):
        return StreamingResponse(
            _mjpeg_gen(request),
            media_type="multipart/x-mixed-replace; boundary=frame")

    return app
