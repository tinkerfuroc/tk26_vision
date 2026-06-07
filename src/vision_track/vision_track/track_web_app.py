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

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        await ws.accept()
        last_state_seq = -1
        last_gallery_version = -1
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
                await asyncio.sleep(_STATE_POLL_S)
        except WebSocketDisconnect:
            return

    async def _mjpeg_gen(request: Request):
        # Infinite by design: the stream outlives source stalls (goal stopped,
        # tracker idle) so the <img> resumes the moment frames return. The only
        # exit is the HTTP client going away. NOTE: Starlette's TestClient
        # buffers responses to completion and so cannot consume this endpoint —
        # tests drive this generator directly instead (see test_track_web_app).
        last_seq = -1
        while True:
            if await request.is_disconnected():
                return
            seq, jpeg = bridge.latest_jpeg()
            if jpeg is not None and seq != last_seq:
                last_seq = seq
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
