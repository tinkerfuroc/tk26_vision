"""ROS-free FastAPI factory for the restaurant nav-test dashboard.

Bridge contract (the node implements these):
  snapshot() -> dict                  # {state, readiness, proc}
  latest_state() -> (seq:int, dict|None)
  latest_jpeg() -> (seq:int, bytes|None)
  start_test(mock: bool=False) -> dict
  stop_test() -> dict
  proc_status() -> dict
  proc_start(name) -> dict ; proc_stop(name) -> dict
  proc_group_start(group) -> list ; proc_group_stop(group) -> list
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

_NO_CACHE = {"Cache-Control": "no-cache"}
_STATE_POLL_S = 0.05
_MJPEG_POLL_S = 1 / 15


def create_app(bridge, webui_dir: Optional[Path] = None) -> FastAPI:
    app = FastAPI(title="restaurant_nav_test_web")
    webui = Path(webui_dir) if webui_dir else Path(__file__).resolve().parents[1] / "webui"

    @app.exception_handler(Exception)
    async def _unhandled(request, exc):  # noqa: ANN001
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)

    @app.get("/")
    def index():
        return FileResponse(webui / "index.html", media_type="text/html", headers=_NO_CACHE)

    @app.get("/style.css")
    def style():
        return FileResponse(webui / "style.css", media_type="text/css", headers=_NO_CACHE)

    @app.get("/app.js")
    def appjs():
        return FileResponse(webui / "app.js", media_type="application/javascript", headers=_NO_CACHE)

    @app.get("/api/status")
    def status():
        return bridge.snapshot()

    @app.post("/api/test/start")
    def test_start(mock: bool = False):
        return bridge.start_test(mock=mock)

    @app.post("/api/test/stop")
    def test_stop():
        return bridge.stop_test()

    @app.get("/api/proc/status")
    def proc_status():
        return bridge.proc_status()

    # NOTE: group routes BEFORE /{name} routes — FastAPI matches in declaration order.
    @app.post("/api/proc/group/{group}/start")
    def proc_group_start(group: str):
        return bridge.proc_group_start(group)

    @app.post("/api/proc/group/{group}/stop")
    def proc_group_stop(group: str):
        return bridge.proc_group_stop(group)

    @app.post("/api/proc/{name}/start")
    def proc_start(name: str):
        return bridge.proc_start(name)

    @app.post("/api/proc/{name}/stop")
    def proc_stop(name: str):
        return bridge.proc_stop(name)

    @app.websocket("/ws/state")
    async def ws_state(ws: WebSocket):
        await ws.accept()
        last_state_seq = -1
        last_proc = None
        try:
            while True:
                seq, state = bridge.latest_state()
                if state is not None and seq != last_state_seq:
                    last_state_seq = seq
                    await ws.send_text(json.dumps({"type": "state", "data": state}))
                snap = bridge.snapshot()
                proc = {"proc": snap.get("proc"), "readiness": snap.get("readiness")}
                if proc != last_proc:
                    last_proc = proc
                    await ws.send_text(json.dumps({"type": "proc", "data": proc}))
                await asyncio.sleep(_STATE_POLL_S)
        except WebSocketDisconnect:
            return

    async def _mjpeg_gen(request: Request):
        heartbeat_polls = max(1, int(0.5 / _MJPEG_POLL_S))
        last_seq = -1
        idle = 0
        while True:
            if await request.is_disconnected():
                return
            seq, jpeg = bridge.latest_jpeg()
            idle += 1
            fresh = jpeg is not None and seq != last_seq
            if jpeg is not None and (fresh or idle >= heartbeat_polls):
                last_seq = seq
                idle = 0
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
            await asyncio.sleep(_MJPEG_POLL_S)

    @app.get("/stream.mjpg")
    def stream(request: Request):
        return StreamingResponse(
            _mjpeg_gen(request),
            media_type="multipart/x-mixed-replace; boundary=frame")

    return app
