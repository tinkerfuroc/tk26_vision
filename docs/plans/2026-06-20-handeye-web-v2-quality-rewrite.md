# handeye_web v2 — Quality Rewrite Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task (fresh implementer per task + two-stage spec + code-quality review). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `handeye_calib`'s web UI to the visual + interactive quality of `pan_tilt/calib_web`. Replace the 30-line inline `INDEX_HTML` with a proper static `webui/` (index.html + style.css + app.js), WebSocket state push, tabbed layout (Info / Move / Capture / Solve / Promote), live frame with annotated/raw toggle + detection badge + resize handle, joint editor with safety preview, sample gallery with diversity meter and settle gate, solve panel with per-method comparison + residual visualizations, and a unified-diff promote flow with backup.

**Architecture:** Same three-layer split as v1 — `web_support.py` (ROS-free helpers), `HandeyeWebNode` (rclpy data plane), `make_app(node)` (FastAPI HTTP plane + WS). New: a `webui/` directory served by `StaticFiles` + `FileResponse`, mirroring `pan_tilt`'s pattern. The v1 in-process solve/promote path is preserved (no subprocesses / sessions — handeye solve is fast and single-shot, unlike pan_tilt's multi-phase CLI runner).

**Tech Stack:** Python 3.10, rclpy, tf2_ros, cv_bridge, FastAPI/uvicorn + `WebSocket`, OpenCV (`cv2.aruco`), numpy/scipy, vanilla JS (no framework — matches pan_tilt's `app.js`).

**Reference UI for visual/UX patterns:** `src/tk26_vision/src/pan_tilt/webui/{index.html,style.css,app.js}` and `src/tk26_vision/src/pan_tilt/pan_tilt/calib_web.py`. Re-use CSS variable names, status-line classes, gate-pill colors, and confirmation-dialog conventions verbatim where they apply.

**Supersedes:** `docs/plans/2026-06-15-handeye-web-server.md`. That plan's `INDEX_HTML` constant and its `test_index_html_is_nonempty_html` test are removed by Task 1.

## Global Constraints

- Package: `src/tk26_vision/src/handeye_calib/`. Git repo is `src/tk26_vision` (branch `dev`). Workspace root is NOT a git repo — run git inside `src/tk26_vision`.
- **Concurrent committer present** (foundation_stereo, kimi_api, object_detection_generalist, vision_track may have uncommitted changes). Commit ONLY the files each task names. Never `git add -A`/`.`, never `--amend`, never rebase.
- `import handeye_calib.handeye_web` and `import handeye_calib.web_support` MUST stay ROS-free: rclpy/fastapi imports live INSIDE functions (`main`, `make_app`) or methods, never at module top. The Task 2 `__getattr__` lazy hook for `HandeyeWebNode` is preserved.
- Do NOT modify `validate_pose_set` / `diff_payload` in `handeye_web.py` — their 3 tests in `test/test_web_helpers.py` must stay green.
- Venv python: `src/tk26_vision/.venv-vision-main/bin/python`. Pure tests need no sourcing. Node/app tests need `source src/tk26_vision/install/setup.bash` first (for rclpy/pan_tilt/tinker_arm_msgs) and `PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH"`.
- **Build wrapper:** rebuild ONLY via `tkbuild tk26_vision --packages-select handeye_calib` (NEVER plain `colcon build`). After each task that adds an entry-point file or `data_files`, rebuild to refresh `install/`.
- **Identity:** all commits authored as `Ccindy0171 <cindy.w0135@gmail.com>` (repo-local git config already enforces this).
- Units: backend stores SI (meters, radians). Frontend renders **mm and degrees** for all human-facing numbers (match pan_tilt). Convert at the JS layer, not the server.
- Frames contract (unchanged from v1): `T_base_eef = A_i` = TF `link_base`→`link_eef`. `T_cam_board = B_i` = ChArUco board in color **optical** frame. Solver returns `X = T_eef→color_optical`.

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `handeye_calib/webui/index.html` | NEW (T1) | Static HTML shell — header, tab bar, panels. Loads style.css + app.js. |
| `handeye_calib/webui/style.css` | NEW (T1) | Dark-theme stylesheet — re-uses pan_tilt's CSS variables and status-line/gate-pill classes. |
| `handeye_calib/webui/app.js` | NEW (T1) | WebSocket client, render loop, all UI event handlers. Grows across T1→T6. |
| `handeye_calib/handeye_calib/web_support.py` | MODIFY (T1) | DROP `INDEX_HTML`. Keep all pure helpers. Add `enriched_state_payload`, `mm`, `deg`, `solve_payload_v2` (per-sample residuals, mm/deg units). |
| `handeye_calib/handeye_calib/handeye_web.py` | MODIFY (T1, T3, T4, T5, T6) | Add `/ws` endpoint, static-file mounts, enriched state fields, new endpoints per task. |
| `handeye_calib/setup.py` | MODIFY (T1) | Add `webui/*` to `data_files` (mirror pan_tilt). |
| `handeye_calib/test/test_web_support.py` | MODIFY (T1) | Drop `test_index_html_is_nonempty_html`. Add tests for new helpers. |
| `handeye_calib/test/test_web_app.py` | MODIFY (T1, T4, T5, T6) | Add WS handshake test, static-asset tests, gallery/diff endpoint tests. |
| `handeye_calib/test/test_web_node.py` | MODIFY (T4, T5) | Add tests for stability tracker integration + per-method solve payload. |
| `handeye_calib/README.md` | MODIFY (T6) | New `## UI` section, `0.4.0` changelog entry. |

---

## Task 1 — Static webui/ scaffold + WebSocket state stream + connection pill

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/webui/index.html`
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/webui/style.css`
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/webui/app.js`
- Modify: `src/tk26_vision/src/handeye_calib/handeye_calib/web_support.py` (drop `INDEX_HTML`; add `enriched_state_payload`, `mm`, `deg`)
- Modify: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_web.py` (mount static files; replace `GET /` with `FileResponse`; add `GET /ws` WebSocket; expand `get_state_dict`)
- Modify: `src/tk26_vision/src/handeye_calib/setup.py` (install `webui/*`)
- Modify: `src/tk26_vision/src/handeye_calib/test/test_web_support.py` (drop INDEX_HTML test; add helpers tests)
- Modify: `src/tk26_vision/src/handeye_calib/test/test_web_app.py` (add WS + static-asset tests)

**Interfaces:**
- Consumes: existing `HandeyeWebNode` (T2/T3 from v1), `web_support` helpers.
- Produces (downstream tasks rely on these):
  - `web_support.enriched_state_payload(...)` → dict with keys: `camera_connected`, `intrinsics_ok`, `num_samples`, `last_detection`, `status_msg`, **`frame_count`, `frame_hz`, `frame_age_sec`, `image_topic`, `ros_domain_id`, `t_base_ee` (4×4 list or None), `xarm_joint_positions` (list[7] or None), `board` (spec dict), `safety_envelope` (dict), `stability` (dict with `steady: bool, since_frames: int, target_frames: int`), `samples` (list of metadata dicts — empty in T1, populated in T4), `diversity` (dict with `coverage_deg: float, target_deg: float`), `last_solve` (None in T1, set in T5)**.
  - `web_support.mm(x_m: float) -> float` (round to 4 dp), `web_support.deg(x_rad: float) -> float` (round to 4 dp).
  - `HandeyeWebNode.get_state_dict()` returns the enriched payload (extended in this task; further extended in T4/T5).
  - HTTP `GET /` → `webui/index.html` via `FileResponse`.
  - HTTP `GET /static/{path}` → mounted at `webui/`.
  - WebSocket `GET /ws` → server pushes `enriched_state_payload()` JSON every 200 ms (5 Hz).
  - JS global `state` (the last WS message) — every later task reads from it.
  - JS function `setStatus(elementId, text, kind)` where `kind ∈ {"", "ok", "warn", "err"}` — used for all status lines in later tasks.

- [ ] **Step 1: Write the failing tests** (append to `test/test_web_support.py`):
```python
def test_mm_and_deg_round_to_4dp():
    assert ws.mm(0.0012345) == 1.2345
    assert ws.deg(0.0174533) == 1.0  # 1 deg in rad → ≈ 1.0
    assert ws.mm(-0.001) == -1.0


def test_enriched_state_payload_has_all_keys():
    d = ws.enriched_state_payload(
        camera_connected=False, intrinsics_ok=False, num_samples=0,
        last_detection=None, status_msg="idle",
        frame_count=0, frame_hz=0.0, frame_age_sec=None,
        image_topic="/foo", ros_domain_id=0,
        t_base_ee=None, xarm_joint_positions=None,
        board={"squares_x": 5}, safety_envelope={"z_floor_m": 0.0},
        stability={"steady": False, "since_frames": 0, "target_frames": 3},
        samples=[], diversity={"coverage_deg": 0.0, "target_deg": 30.0},
        last_solve=None,
    )
    required = {
        "camera_connected", "intrinsics_ok", "num_samples", "last_detection",
        "status_msg", "frame_count", "frame_hz", "frame_age_sec", "image_topic",
        "ros_domain_id", "t_base_ee", "xarm_joint_positions", "board",
        "safety_envelope", "stability", "samples", "diversity", "last_solve",
    }
    assert set(d) >= required
```

And REPLACE the old `test_index_html_is_nonempty_html` block with — nothing. (INDEX_HTML is being removed.)

Append to `test/test_web_app.py`:
```python
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
```

- [ ] **Step 2: Run the tests to verify they fail**
```bash
cd /home/tinker/tk25_ws
source src/tk26_vision/install/setup.bash
PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH" src/tk26_vision/.venv-vision-main/bin/python \
  -m pytest src/tk26_vision/src/handeye_calib/test/test_web_support.py src/tk26_vision/src/handeye_calib/test/test_web_app.py -v
```
Expected: the four new tests FAIL.

- [ ] **Step 3: Implement `web_support.py` changes**
- DELETE the `INDEX_HTML = """..."""` block at the bottom of `web_support.py`.
- Add helpers:
```python
def mm(x_m):
    return round(float(x_m) * 1000.0, 4)


def deg(x_rad):
    import math
    return round(float(x_rad) * 180.0 / math.pi, 4)


def enriched_state_payload(*, camera_connected, intrinsics_ok, num_samples,
                            last_detection, status_msg,
                            frame_count, frame_hz, frame_age_sec,
                            image_topic, ros_domain_id,
                            t_base_ee, xarm_joint_positions,
                            board, safety_envelope,
                            stability, samples, diversity, last_solve):
    base = state_payload(camera_connected, intrinsics_ok, num_samples,
                          last_detection, status_msg)
    base.update({
        "frame_count": int(frame_count),
        "frame_hz": float(frame_hz),
        "frame_age_sec": (None if frame_age_sec is None else float(frame_age_sec)),
        "image_topic": str(image_topic),
        "ros_domain_id": int(ros_domain_id),
        "t_base_ee": (None if t_base_ee is None else
                      [list(map(float, row)) for row in t_base_ee]),
        "xarm_joint_positions": (None if xarm_joint_positions is None else
                                  [float(j) for j in xarm_joint_positions]),
        "board": dict(board),
        "safety_envelope": dict(safety_envelope),
        "stability": dict(stability),
        "samples": list(samples),
        "diversity": dict(diversity),
        "last_solve": last_solve,
    })
    return base
```

- [ ] **Step 4: Implement static-asset + WebSocket wiring in `handeye_web.py`**

Replace the existing `@app.get("/")` route with `FileResponse`. Add a `StaticFiles` mount at `/static`. Add a `/ws` endpoint. Resolve the `webui/` path via `importlib.resources` (works from both source-tree and install-tree). Wire `HandeyeWebNode.get_state_dict()` to call `ws.enriched_state_payload(...)` with sensible defaults for fields T4/T5 will populate (empty `samples=[]`, `last_solve=None`, etc.). Track `frame_count` and a rolling `frame_hz` (windowed over last 30 frames).

Sketch of the new route block in `make_app`:
```python
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import WebSocket, WebSocketDisconnect

webui_dir = Path(__file__).parent / "webui"
app.mount("/static", StaticFiles(directory=str(webui_dir)), name="static")

@app.get("/")
def index():
    return FileResponse(str(webui_dir / "index.html"), media_type="text/html")

@app.websocket("/ws")
async def ws_state(ws: WebSocket):
    await ws.accept()
    import asyncio
    try:
        while True:
            await ws.send_json(node.get_state_dict())
            await asyncio.sleep(0.2)  # 5 Hz
    except WebSocketDisconnect:
        pass
```

`HandeyeWebNode` additions:
- In `_on_image`: bump `self._frame_count`; push `time.monotonic()` onto a `collections.deque(maxlen=30)`; compute `_frame_hz` from deltas.
- `_frame_age_sec`: `time.monotonic() - last_frame_ts` or `None` if no frames.
- Expose `ros_domain_id`: `int(os.environ.get("ROS_DOMAIN_ID", "0"))`.
- Expose `t_base_ee`: cache the most recent successful TF lookup (try in `get_state_dict` with a 50ms timeout; cache result + invalidate on failure).
- Expose `xarm_joint_positions`: subscribe to `/joint_states` (best-effort; populate from JointState whose name list contains the 7 xArm joints).
- Stability tracker: instantiate `gates.StabilityTracker(...)` (READ `handeye_calib/gates.py` first for actual constructor args + tick API) and feed it from `_on_image`. Expose `stability = {"steady": bool, "since_frames": int, "target_frames": int}`. (Task 4 turns this into a hard pre-capture gate; in T1 it's just observable.)
- Diversity: `coverage_deg = max angular spread across all currently-accepted samples` (use `pan_tilt.calibration` rotation helpers OR compute directly: max pairwise `arccos((tr(R_i R_j^T) - 1) / 2)` in degrees across accepted samples; cap at `target_deg`). T1 returns `0.0` because no samples yet — wire the field, not the math (math comes in T4).

- [ ] **Step 5: Implement the static UI (skeleton — header + tab bar + connection pill, no panel bodies yet)**

`webui/style.css` — copy the CSS variables and status-line + button classes verbatim from `src/tk26_vision/src/pan_tilt/webui/style.css` (variables block, `body`, `.bg-panel`, `.status-line`, `.status-line.ok|.warn|.err`, `button`, `button.primary`, `button:disabled`, `.conn-indicator{.connected,.dropped}`, `.gate-pill{.pass,.warn,.fail}`, `.side-tabs`, `.side-tab`, `.side-tab.active`, `.side-panel-content`, `.side-panel-content.active{display:block}`). DO NOT copy pan_tilt-specific selectors like `.xarm-panels` or `.calib-resid-grid` — those come in T4/T5 with handeye-specific names.

`webui/index.html` — structure:
```html
<!doctype html><html lang="en"><head>
  <meta charset="utf-8"><title>handeye_web · calibrate</title>
  <link rel="stylesheet" href="/static/style.css">
</head><body>
  <header class="bg-panel">
    <span class="title">handeye_calib · calibrate_web</span>
    <span id="conn-indicator" class="conn-indicator">WS: connecting…</span>
  </header>
  <main>
    <nav class="side-tabs">
      <button class="side-tab active" data-tab="info">Info</button>
      <button class="side-tab" data-tab="move">Move</button>
      <button class="side-tab" data-tab="capture">Capture</button>
      <button class="side-tab" data-tab="solve">Solve</button>
      <button class="side-tab" data-tab="promote">Promote</button>
    </nav>
    <section class="side-panel-content active" data-panel="info">
      <div class="status-line" id="info-camera">camera: …</div>
      <div class="status-line" id="info-tf">tf: …</div>
    </section>
    <section class="side-panel-content" data-panel="move"></section>
    <section class="side-panel-content" data-panel="capture"></section>
    <section class="side-panel-content" data-panel="solve"></section>
    <section class="side-panel-content" data-panel="promote"></section>
  </main>
  <script src="/static/app.js"></script>
</body></html>
```

`webui/app.js` — exports nothing; runs on load:
- Global `let state = null;` and `function setStatus(id, text, kind="") { ... }` (mirrors pan_tilt's pattern; sets `textContent` + class `.status-line ok|warn|err`).
- Tab switching: click handler on `.side-tab` toggles `.active` on the button and on the matching `.side-panel-content`.
- WebSocket: open `/ws`, with auto-reconnect every 1.5 s on close/error. On message: `state = JSON.parse(ev.data); render();`.
- `function render()`: updates `#conn-indicator` to `WS: live` (`.connected`); updates info-tab status lines (camera connected/disconnected, frame Hz, frame age).
- On WS close/error: set `#conn-indicator` to `WS: dropped — retrying` (`.dropped`).

This skeleton must successfully render and tab-switch; the panel bodies stay empty for now. T2–T6 fill them in.

- [ ] **Step 6: Wire setup.py** — add to `data_files`:
```python
import os
from glob import glob
# in setup() data_files=[...]:
(os.path.join('share', package_name, 'webui'), glob('handeye_calib/webui/*')),
```
And ensure the source-tree `handeye_calib/webui/` is also reachable by the runtime `FileResponse` path (the `Path(__file__).parent / "webui"` lookup works for an ament_python install because `handeye_web.py` is imported from `install/.../site-packages/handeye_calib/handeye_web.py` and webui ships alongside it). Verify with:
```bash
ls /home/tinker/tk25_ws/install/handeye_calib/lib/python3.10/site-packages/handeye_calib/webui/ 2>&1 | head
```

If webui doesn't land next to `handeye_web.py`, add it to `packages` data via `package_data` in `setup.py`:
```python
package_data={'handeye_calib': ['webui/*']},
include_package_data=True,
```

- [ ] **Step 7: Run the tests + live smoke**
```bash
tkbuild tk26_vision --packages-select handeye_calib
PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH" src/tk26_vision/.venv-vision-main/bin/python \
  -m pytest src/tk26_vision/src/handeye_calib/test/ -v
# live smoke (server starts + WS reachable + static assets):
ros2 launch handeye_calib handeye_web.launch.py port:=8792 &
sleep 5
curl -fsS -o /dev/null -w "GET / -> %{http_code}\n" http://127.0.0.1:8792/
curl -fsS -o /dev/null -w "GET /static/style.css -> %{http_code}\n" http://127.0.0.1:8792/static/style.css
curl -fsS -o /dev/null -w "GET /static/app.js -> %{http_code}\n" http://127.0.0.1:8792/static/app.js
kill %1 2>/dev/null; wait 2>/dev/null
```
Expected: all-green test suite (note: the `test_index_html_is_nonempty_html` count is one fewer, the 4 new tests add to web_app + web_support); all curl probes return 200.

- [ ] **Step 8: Commit**
```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/webui/ \
        src/handeye_calib/handeye_calib/web_support.py \
        src/handeye_calib/handeye_calib/handeye_web.py \
        src/handeye_calib/setup.py \
        src/handeye_calib/test/test_web_support.py \
        src/handeye_calib/test/test_web_app.py
git commit -m "feat(handeye_calib): static webui/ scaffold + WebSocket state stream

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2 — Live frame panel (resizable, annotated/raw toggle, detection badge, FPS, placeholder)

**Files:**
- Modify: `webui/index.html` (add the `<div class="cam-panel">` with `<img>`, resize handle, frame-toolbar overlay, detection badge, placeholder)
- Modify: `webui/style.css` (add `.cam-panel`, `.cam-img`, `.cam-resize`, `.frame-toolbar`, `.detection-badge`, `.frame-placeholder` rules — copy from `pan_tilt/webui/style.css` and rename selectors).
- Modify: `webui/app.js` (frame polling at 3 Hz with cache-bust, raw/annotated radio handler, resize-handle pointer events with localStorage `handeye-cam-w`, detection badge driven by `state.last_detection`, placeholder visibility toggle on img.load/error).
- Modify: `handeye_calib/handeye_web.py` (extend `latest_jpeg(raw: bool = False)`; add `raw` query param to `/api/frame.jpg`).
- Modify: `handeye_calib/web_support.py` (extend `draw_charuco_overlay` to optionally render corner indices + RMS text overlay + 4-point board outline).
- Modify: `test/test_web_support.py` (test overlay with indices argument).
- Modify: `test/test_web_app.py` (test `?raw=1` returns JPEG with content differing from default).

**Interfaces:**
- Consumes: T1 WS state (`frame_count`, `frame_hz`, `last_detection`).
- Produces:
  - `HandeyeWebNode.latest_jpeg(raw: bool = False) -> bytes`
  - `web_support.draw_charuco_overlay(bgr, corners_xy, ids=None, rms_px=None, image_topic=None) -> np.ndarray`
  - DOM IDs available to T3–T6: `#cam-img`, `#detection-badge`, `#frame-placeholder`, `#cam-resize`, `input[name="frame-mode"]` radios.

- [ ] **Step 1: Write failing tests**
```python
# in test_web_support.py
def test_overlay_with_indices_preserves_shape():
    img = np.zeros((100, 200, 3), np.uint8)
    out = ws.draw_charuco_overlay(img,
        corners_xy=np.array([[50, 50], [150, 80]]),
        ids=np.array([3, 7]), rms_px=0.42, image_topic="/foo")
    assert out.shape == img.shape


# in test_web_app.py
def test_frame_raw_query_returns_jpeg():
    node, c = _client()
    try:
        r = c.get("/api/frame.jpg?raw=1")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert r.content[:2] == b"\xff\xd8"
    finally:
        node.destroy_node()
```
Run — expected FAIL.

- [ ] **Step 2: Implement**

`web_support.draw_charuco_overlay` — when `ids is not None`, render each ID as `cv2.putText` next to the corresponding green corner circle. When `rms_px is not None`, render `f"rms={rms_px:.2f}px"` in the top-left header bar. Bandwidth: downscale to max 960 px wide before encoding (matches pan_tilt — see its `_downscale`).

`HandeyeWebNode.latest_jpeg(raw=False)` — same lock pattern; if `raw=False` and a frame + corners exist, render overlay; if `raw=True`, return the raw BGR encoded directly; placeholder when no frame.

Endpoint:
```python
@app.get("/api/frame.jpg")
def frame(raw: int = 0):
    return Response(content=node.latest_jpeg(raw=bool(raw)), media_type="image/jpeg")
```

Frontend `index.html` — insert at top of `<main>` (or as a floating panel — match pan_tilt's resizable left float):
```html
<div class="cam-panel">
  <img id="cam-img" class="cam-img" alt="">
  <div class="cam-resize" id="cam-resize" title="drag to resize"></div>
  <div class="frame-toolbar">
    <label><input type="radio" name="frame-mode" value="annotated" checked> annotated</label>
    <label><input type="radio" name="frame-mode" value="raw"> raw</label>
  </div>
  <div class="detection-badge" id="detection-badge">corners=0  NO DETECTION</div>
  <div class="frame-placeholder" id="frame-placeholder">waiting for frames on <span id="placeholder-topic">/xarm_camera/color/image_raw</span></div>
</div>
```

`app.js` additions:
- `function refreshFrame()`: `document.getElementById('cam-img').src = '/api/frame.jpg?t=' + Date.now() + (currentMode === 'raw' ? '&raw=1' : '');` Called every 333 ms (3 Hz).
- Resize: pointer-down on `#cam-resize` captures pointer; on pointer-move, set CSS variable `--cam-w` (`document.documentElement.style.setProperty('--cam-w', newPx + 'px')`); clamp 240–800 px; on pointer-up persist to `localStorage.setItem('handeye-cam-w', newPx)`. On load: restore from localStorage.
- Detection badge — driven by `state.last_detection`: `corners=N rms=X.XXpx OK` (green, `.ok`) or `corners=0 NO DETECTION` (red, `.err`).
- Placeholder visibility: hide on `<img>.onload`, show on `.onerror`.

CSS:
```css
:root { --cam-w: 480px; }
.cam-panel { position: relative; width: var(--cam-w); ... }
.cam-img { width: 100%; aspect-ratio: 16/9; background: #000; ... }
.cam-resize { position: absolute; bottom: 0; right: 0; width: 14px; height: 14px;
              cursor: nwse-resize; background: var(--accent); opacity: 0.5; }
.frame-toolbar { position: absolute; top: 4px; left: 4px;
                 background: rgba(0,0,0,0.6); padding: 4px 6px; font-size: 11px; }
.detection-badge { position: absolute; bottom: 4px; left: 4px;
                   background: rgba(0,0,0,0.7); padding: 4px 8px; font-size: 11px; }
.detection-badge.ok { color: var(--ok); }
.detection-badge.err { color: var(--err); }
.frame-placeholder { position: absolute; inset: 0; display: flex;
                     align-items: center; justify-content: center; color: var(--fg-muted); }
.cam-img.has-frame + .cam-resize ~ .frame-placeholder { display: none; }
```

- [ ] **Step 3: Run tests + browser smoke**
```bash
tkbuild tk26_vision --packages-select handeye_calib
PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH" src/tk26_vision/.venv-vision-main/bin/python \
  -m pytest src/tk26_vision/src/handeye_calib/test/ -v
ros2 launch handeye_calib handeye_web.launch.py port:=8793 &
sleep 4
curl -fsS -o /tmp/a.jpg http://127.0.0.1:8793/api/frame.jpg
curl -fsS -o /tmp/b.jpg "http://127.0.0.1:8793/api/frame.jpg?raw=1"
file /tmp/a.jpg /tmp/b.jpg
kill %1; wait 2>/dev/null
```
Expected: both files are JPEGs.

- [ ] **Step 4: Commit**
```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/webui/ \
        src/handeye_calib/handeye_calib/web_support.py \
        src/handeye_calib/handeye_calib/handeye_web.py \
        src/handeye_calib/test/test_web_support.py \
        src/handeye_calib/test/test_web_app.py
git commit -m "feat(handeye_calib): live frame panel — resize, raw/annotated toggle, detection badge

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3 — Info tab content + Move tab (joint editor, unit toggle, safety preview, presets)

**Files:**
- Modify: `webui/index.html` (fill `[data-panel="info"]` and `[data-panel="move"]` bodies).
- Modify: `webui/style.css` (add `.kv-table`, `.matrix-block`, `.joint-grid`, `.unit-toggle`, `.preset-bar` rules).
- Modify: `webui/app.js` (info-panel renderer reading from `state`; move-panel form handlers).
- Modify: `handeye_calib/handeye_web.py` (`HandeyeWebNode.do_move` already returns `{ok, reason}`; add a `safety_preview()` method that runs the SafetyEnvelope check against the cached `t_base_ee` for the UI to display live).

**Interfaces:**
- Consumes: T1 WS state (camera/TF/joints/board/safety/ros_domain_id).
- Produces:
  - JS function `formatMatrix(rows, dp=4)` → monospace string.
  - JS function `applyMoveConfirm(angles_rad: number[7])` → handler that calls `POST /api/move` after `confirm()`.
  - DOM IDs for downstream layout: `#info-kv-camera`, `#info-kv-robot`, `#info-matrix-tbe`, `#info-board`, `#info-safety`, `#move-joint-inputs`, `#move-unit-toggle`, `#move-status`, `#move-safety-status`, `#move-load-current`, `#move-zero`, `#move-send`, `#move-presets`.

- [ ] **Step 1: Info tab renderer (in `render()`)**

Show as `.kv-table` rows:
- **Camera:** subscribed topic, `ROS_DOMAIN_ID`, status (streaming/—), frames received, last frame age (s), frame Hz.
- **Robot state:** xArm joints (7 floats, mm/deg formatted), TF status.
- **T_base_eef:** 4×4 matrix in monospace pre block (4 dp).
- **ChArUco board:** squares, square_len_m (mm), marker_len_m (mm), aruco dict.
- **Safety envelope:** JSON dump (z_floor_m, mast_*).

These are all read from `state.*` — no new endpoints. Each row uses `.status-line` with `.ok`/`.warn`/`.err` driven by current values.

- [ ] **Step 2: Move tab form**

```html
<div class="move-panel">
  <div class="unit-toggle" id="move-unit-toggle">
    <label><input type="radio" name="move-unit" value="rad" checked> rad</label>
    <label><input type="radio" name="move-unit" value="deg"> deg</label>
  </div>
  <div class="joint-grid" id="move-joint-inputs">
    <!-- 7 numbered inputs labeled J0..J6, populated by JS -->
  </div>
  <div class="row">
    <button id="move-load-current">Load current</button>
    <button id="move-zero">Zero all</button>
    <button class="primary" id="move-send">Move (joints)</button>
  </div>
  <div class="status-line" id="move-status"></div>
  <div class="status-line" id="move-safety-status"></div>
  <div class="preset-bar" id="move-presets">
    <button data-preset="home">Home [0,0,0,0,0,0,0]</button>
    <button data-preset="look-forward">Look forward at board</button>
  </div>
</div>
```

JS handlers:
- Unit toggle: when switching rad↔deg, convert all input values (track `prevUnit`).
- "Load current": read `state.xarm_joint_positions`, populate inputs.
- "Zero all": fill with 0.
- "Move (joints)": parse 7 floats, convert to rad if deg-mode, `confirm('Send xArm to these joints now?\n' + angles.join(', '))`; on accept, `setStatus('move-status', 'moving…', 'warn')`; POST `/api/move` with `{joints: [7 rads]}`; on response set `ok`/`err`.
- Safety status (live, driven by `state.t_base_ee` + `state.safety_envelope`): render `safe (z=…, r_mast=…)` (green) or `VIOLATION: …` (red). The math may run client-side OR server-side via a new `GET /api/safety` route — pick whichever is simpler given that `SafetyEnvelope` already lives in `pan_tilt.calibration.safety`. **Recommendation:** add server-side `node.safety_preview()` returning `{safe: bool, detail: str}` and include it in the WS state under `state.safety_preview` so the UI just renders it.
- Preset buttons: a `home` preset fills `[0,0,0,0,0,0,0]`. A `look-forward` preset fills a hardcoded 7-joint pose where the wrist camera faces forward at ~50 cm range (pick a sensible default from `src/tk26_vision/CAMERA_BRINGUP.md` or use a recorded pose from a tinker2 calibration session; if neither exists, leave the preset wired but commented in the JS with a TODO note — do not invent unsafe values).

- [ ] **Step 3: Test + smoke + commit**
```bash
tkbuild tk26_vision --packages-select handeye_calib
PYTHONPATH=... pytest src/tk26_vision/src/handeye_calib/test/ -v
ros2 launch handeye_calib handeye_web.launch.py port:=8794 &
# manual: open http://127.0.0.1:8794, switch tabs, verify Info populates from WS,
# verify Move form responds to unit toggle.
```

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/webui/ \
        src/handeye_calib/handeye_calib/handeye_web.py
git commit -m "feat(handeye_calib): Info + Move tabs — kv tables, joint editor, safety preview, presets

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4 — Capture tab (settle gate, sample gallery with thumbnails, diversity meter, per-sample delete)

**Files:**
- Modify: `handeye_calib/handeye_web.py` (settle gate enforcement in `do_capture`; store per-sample thumbnail JPEG + metadata; add `do_delete_sample(idx)`; expose enriched `samples` + `diversity` + `stability` in WS state)
- Modify: `handeye_calib/web_support.py` (helper `sample_metadata(idx, sample, prev_sample) -> dict`; helper `compute_diversity_deg(samples) -> float`)
- Modify: `handeye_calib/handeye_web.py` (make_app): add `GET /api/samples/{idx}/thumb.jpg`, `DELETE /api/samples/{idx}`.
- Modify: `webui/index.html` (capture panel: pre-capture badge, capture button, gallery grid, diversity meter)
- Modify: `webui/style.css` (`.capture-panel`, `.stability-badge`, `.gallery`, `.gallery-item`, `.diversity-meter`)
- Modify: `webui/app.js` (capture handler, gallery renderer, delete handler, diversity bar update)
- Modify: `test/test_web_support.py` (test `sample_metadata`, `compute_diversity_deg`)
- Modify: `test/test_web_node.py` (test settle gate blocks capture when not steady; test delete-by-index)
- Modify: `test/test_web_app.py` (test gallery thumb endpoint; test DELETE; test that `/api/capture` is rejected without settle)

**Interfaces:**
- Consumes: T1 enriched state + thumbnails.
- Produces:
  - `web_support.sample_metadata(idx: int, sample: hm.Sample, prev_sample: hm.Sample | None) -> dict` with keys `idx, n_corners, reproj_px, area_frac, angular_delta_deg (vs prev), joint_positions (or null), ts`.
  - `web_support.compute_diversity_deg(samples: list[hm.Sample]) -> float` — max pairwise rotation angle (deg) between any two `T_base_eef` rotation parts.
  - `HandeyeWebNode.do_capture()` — same return shape; now BLOCKED by `stability.steady == False` (returns `{ok: False, reason: "not stable yet (1/3 steady frames)"}`).
  - `HandeyeWebNode.do_delete_sample(idx: int) -> {ok, num_samples}`.
  - State extension: `state.samples` populated with `sample_metadata` per accepted sample; `state.diversity.coverage_deg` from `compute_diversity_deg`; `state.stability` reflects current StabilityTracker.
  - HTTP: `GET /api/samples/{idx}/thumb.jpg` (320 px wide JPEG of the captured frame + overlay); `DELETE /api/samples/{idx}` → `{ok, num_samples}`.

- [ ] **Step 1: Tests**

Add to `test_web_support.py`:
```python
def test_compute_diversity_zero_for_zero_or_one_sample():
    from handeye_calib import handeye_model as hm
    assert ws.compute_diversity_deg([]) == 0.0
    s = hm.Sample(np.eye(4), np.eye(4), np.zeros((0,2)), np.zeros((0,), int))
    assert ws.compute_diversity_deg([s]) == 0.0


def test_compute_diversity_max_pairwise_deg():
    from handeye_calib import handeye_model as hm
    from scipy.spatial.transform import Rotation as R
    def mk(rpy_deg):
        T = np.eye(4); T[:3,:3] = R.from_euler('xyz', rpy_deg, degrees=True).as_matrix()
        return hm.Sample(T, np.eye(4), np.zeros((0,2)), np.zeros((0,), int))
    s = [mk([0,0,0]), mk([45,0,0]), mk([0,30,0])]
    cov = ws.compute_diversity_deg(s)
    assert cov >= 45.0  # at least the 0→45 about X
```

Add to `test_web_node.py`:
```python
def test_capture_blocked_when_not_steady(monkeypatch):
    node = HandeyeWebNode()
    try:
        # force stability tracker to NOT steady (override its is_steady method)
        node.stability_steady = False  # whatever the implementation uses
        r = node.do_capture()
        assert r["ok"] is False and "stab" in r["reason"].lower()
    finally:
        node.destroy_node()


def test_delete_sample_by_idx_out_of_range():
    node = HandeyeWebNode()
    try:
        r = node.do_delete_sample(99)
        assert r["ok"] is False
    finally:
        node.destroy_node()
```

Add to `test_web_app.py`:
```python
def test_sample_thumb_404_for_missing_idx():
    node, c = _client()
    try:
        r = c.get("/api/samples/0/thumb.jpg")
        assert r.status_code == 404
    finally:
        node.destroy_node()
```

- [ ] **Step 2: Implement**

`web_support.compute_diversity_deg`: extract `R_i = sample.T_base_eef[:3, :3]` for each accepted sample. For all pairs `(i, j)`, compute `angle_deg = degrees(arccos(clip((trace(R_i.T @ R_j) - 1) / 2, -1, 1)))`. Return the max (or 0.0 for ≤1 sample).

`web_support.sample_metadata`: build a dict per accepted sample. `angular_delta_deg` = the rotation angle between this sample's `T_base_eef[:3,:3]` and `prev_sample.T_base_eef[:3,:3]` (or `None` for the first).

`HandeyeWebNode`:
- Add `self._stability = gates.StabilityTracker(...)` — READ `gates.py` for actual API; pass appropriate window/tolerance for ~3 consecutive steady frames at 2 px corner motion (this is the v1 deferral being closed).
- In `_on_image`: call `self._stability.update(corners_xy_or_pose)` and cache the result; expose `{steady, since_frames, target_frames}` in state.
- `do_capture`: if `not self._stability.is_steady()`, return `{ok: False, reason: "not stable yet (since_frames/target_frames)"}`. Otherwise proceed as v1. After a successful `session.try_add`, also store `self._thumbs[idx] = ws.encode_jpeg(downscale(frame_with_overlay, max_w=320))` and `self._sample_joints[idx] = last_xarm_joint_positions or None`.
- `do_delete_sample(idx)`: under lock, validate idx, pop `self.session.samples[idx]`, pop `self._thumbs[idx]`, pop `self._sample_joints[idx]`. Recompute diversity. Return `{ok: True, num_samples: ...}`.
- `get_state_dict`: build `state.samples = [ws.sample_metadata(i, s, samples[i-1] if i else None) for i, s in enumerate(samples)]` + `state.diversity = {"coverage_deg": ws.compute_diversity_deg(samples), "target_deg": self._min_diversity_deg}`.

HTTP routes:
```python
@app.get("/api/samples/{idx}/thumb.jpg")
def thumb(idx: int):
    jpg = node.sample_thumb(idx)  # returns bytes or None
    if jpg is None:
        return Response(status_code=404)
    return Response(content=jpg, media_type="image/jpeg")

@app.delete("/api/samples/{idx}")
def delete_sample(idx: int):
    return JSONResponse(node.do_delete_sample(idx))
```

`webui/index.html` capture panel:
```html
<div class="capture-panel">
  <div class="stability-badge" id="capture-stability">stability: …</div>
  <button class="primary" id="capture-btn" disabled>Capture pose</button>
  <div class="status-line" id="capture-status"></div>
  <div class="diversity-meter">
    <div class="bar"><div id="diversity-fill" class="fill"></div></div>
    <div class="label" id="diversity-label">0° / 30°</div>
  </div>
  <div class="gallery" id="gallery"></div>
</div>
```

`app.js`:
- Render `#capture-stability` from `state.stability`: `steady ✓` (green) or `stabilizing… i/n` (warn).
- Enable `#capture-btn` only when `state.camera_connected && state.intrinsics_ok && state.last_detection && state.stability.steady`.
- Click handler: POST `/api/capture`; setStatus to `accepted` / reject reason; on success the gallery refreshes automatically via WS push.
- Gallery: for each `state.samples[i]`, render a `.gallery-item` containing `<img src="/api/samples/i/thumb.jpg">`, idx, n_corners, reproj_px (px), area_frac (%), angular_delta_deg (°), and a small "✕ delete" button → `DELETE /api/samples/{i}`.
- Diversity meter: width % = `min(100, coverage_deg / target_deg * 100)`; color = green when ≥100%, warn when ≥50%, err otherwise; label `"NN° / 30°"`.

- [ ] **Step 3: Test + smoke + commit**

```bash
tkbuild tk26_vision --packages-select handeye_calib
PYTHONPATH=... pytest src/tk26_vision/src/handeye_calib/test/ -v
```
Then commit:
```bash
git add src/handeye_calib/handeye_calib/ src/handeye_calib/test/
git commit -m "feat(handeye_calib): Capture tab — settle gate, gallery thumbnails, diversity meter, delete

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5 — Solve tab (method picker, per-method comparison, residual canvases, coverage canvas, mm/deg)

**Files:**
- Modify: `handeye_calib/handeye_web.py` (`do_solve(method: str = "auto") -> dict` with per-sample residuals + per-method summary)
- Modify: `handeye_calib/web_support.py` (helper `solve_payload_v2(res, samples, K, dist, board_pts) -> dict` — adds `per_sample_reproj_px`, `per_method_summary`, `X_xyz_mm`, `X_rpy_deg`, units-rendered metrics)
- Modify: `webui/index.html` (solve panel with method dropdown, comparison table, canvases)
- Modify: `webui/style.css` (`.solve-panel`, `.method-table`, `.canvas-row`)
- Modify: `webui/app.js` (solve handler, comparison table renderer, two canvas drawing routines, coverage canvas)
- Modify: `test/test_web_support.py` (test `solve_payload_v2` schema)
- Modify: `test/test_web_app.py` (test `/api/solve` with method param)

**Interfaces:**
- Produces:
  - `web_support.solve_payload_v2(res, samples, K, dist, board_pts) -> dict` with keys: `status, X_xyz_mm (list[3]), X_rpy_deg (list[3]), train_metrics_mm_deg, heldout_metrics_mm_deg, per_method_summary (list of {name, reproj_px}), per_sample_reproj_px (list[float], len == len(samples))`.
  - `HandeyeWebNode.do_solve(method: str = "auto") -> dict` (POST body `{method: "auto"|"TSAI"|"PARK"|"HORAUD"|"ANDREFF"|"DANIILIDIS"}`).
  - JS canvas helpers `drawHistogram(canvasId, values, opts)`, `drawScatter(canvasId, values, opts)`, `drawCoverage(canvasId, samples, K)`.

- [ ] **Step 1: Tests**
```python
def test_solve_payload_v2_units_and_keys():
    from handeye_calib.handeye_solve import SolveResult
    res = SolveResult(X=np.eye(4), Tbb=np.eye(4),
                      train_metrics={"trans_rmse_m": 0.001, "rot_rmse_rad": 0.00174, "reproj_px": 0.3},
                      heldout_metrics={"trans_rmse_m": 0.002, "rot_rmse_rad": 0.00349, "reproj_px": 0.5},
                      status="PASS",
                      per_method=[{"name": "TSAI", "X": np.eye(4), "Tbb": np.eye(4), "reproj_px": 0.31}])
    p = ws.solve_payload_v2(res, samples=[], K=np.eye(3), dist=None, board_pts=np.zeros((0,3)))
    assert p["X_xyz_mm"] == [0.0, 0.0, 0.0]
    assert p["train_metrics_mm_deg"]["trans_rmse_mm"] == 1.0
    assert abs(p["train_metrics_mm_deg"]["rot_rmse_deg"] - 0.1) < 0.01
    assert p["per_method_summary"][0]["name"] == "TSAI"
    assert isinstance(p["per_sample_reproj_px"], list)
```

- [ ] **Step 2: Implement**

`solve_payload_v2` — call `handeye_solve._reproj_rms` per-sample (or use the existing `_residuals` and split into per-sample chunks) to populate `per_sample_reproj_px`. Convert all metric distances mm and rotations to deg using `ws.mm` / `ws.deg`.

`HandeyeWebNode.do_solve(method)` — if `method != "auto"`, override `handeye_solve._METHODS` to only that key when invoking `seed_handeye` (cleanest: factor `seed_handeye` to accept an optional `methods_subset` arg; or temporarily monkey-patch in node code; **preferred**: add an optional `methods: dict | None = None` kwarg to `seed_handeye` and thread it through `solve(...)`).

Endpoint:
```python
@app.post("/api/solve")
async def solve(request: Request):
    body = await request.json()
    return JSONResponse(node.do_solve(method=body.get("method", "auto")))
```

Frontend solve panel:
```html
<div class="solve-panel">
  <div class="row">
    <label>Method:
      <select id="solve-method">
        <option value="auto" selected>auto (best of all)</option>
        <option>TSAI</option><option>PARK</option><option>HORAUD</option>
        <option>ANDREFF</option><option>DANIILIDIS</option>
      </select>
    </label>
    <button class="primary" id="solve-btn">Solve</button>
  </div>
  <div class="status-line" id="solve-status"></div>
  <span class="gate-pill" id="solve-verdict" hidden></span>
  <table class="method-table" id="method-table"></table>
  <div class="kv-table" id="solve-metrics"></div>
  <div class="kv-table" id="solve-X"></div>
  <div class="canvas-row">
    <canvas id="resid-hist" width="360" height="140"></canvas>
    <canvas id="resid-scatter" width="360" height="140"></canvas>
  </div>
  <canvas id="coverage" width="600" height="400"></canvas>
</div>
```

`app.js`:
- Solve click: setStatus 'solving…' (warn); POST `/api/solve` with `{method: select.value}`; on response render verdict pill (`.gate-pill.pass|.warn|.fail` driven by status); fill `method-table` rows `{name, reproj_px}`; fill `solve-metrics` with train + heldout in mm/deg/px; fill `solve-X` with xyz (mm) + rpy (deg); draw all three canvases.
- `drawHistogram(canvasId, values, {bins=20})`: simple bar chart.
- `drawScatter(canvasId, values)`: dots at `(i, v[i])`.
- `drawCoverage(canvasId, samples, K)`: for each sample, project `T_cam_board * [0,0,0]` to image with `K`; plot as a labeled dot, color-coded by depth (warm = closer).

- [ ] **Step 3: Test + smoke + commit**

`tkbuild` then `pytest`. Manual: capture ≥6 synthetic poses via the existing `test_synthetic` data path (or by running `handeye_collect` against a recorded bag if available) and verify the solve panel renders.

```bash
git add src/handeye_calib/handeye_calib/ src/handeye_calib/test/
git commit -m "feat(handeye_calib): Solve tab — method picker, per-method table, residual + coverage canvases

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6 — Promote tab (unified diff modal: **yaml + xacro**, ROBOT_NAME-scoped, confirm, write-with-backup, reload) + docs/changelog

**Background — why the xacro side is non-trivial:** the wrist-camera mount is
defined in a **shared vendor xacro** at
`src/tk25_manipulation/src/xarm_ros2/xarm_description/urdf/camera/realsense_d435i.urdf.xacro:40`
(joint name `camera_link_joint`, current origin
`xyz="0.06746 -0.0175 0.0237" rpy="${M_PI} ${-M_PI/2} 0"`). Patching it in place
would overwrite tinker1's calibration when tinker2 calibrates and vice versa.
pan_tilt sidesteps this by having per-robot xacro files
(`pan_tilt.urdf_tinker1.xacro`, `pan_tilt.urdf.xacro` for tinker2) — handeye
doesn't have that infrastructure yet.

This task introduces a **per-robot xacro override** convention:

```
src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/wrist_camera.xacro
```

- If the file exists, T6 patches it via `apply_handeye.patch_urdf_origin`.
- If it does NOT exist AND `ROBOT_NAME` is set, T6 creates a minimal seed file
  in the diff preview (a one-joint `<robot>` block with `camera_link_joint`'s
  `<origin>` set to the calibrated values) so the operator can review and
  apply. The seed includes a comment-line `<!-- include this from the robot's
  main xacro with <xacro:include filename="..."/> -->` — wiring the include
  into the main xacro is a separate one-time setup step (out of scope for
  this task; called out in the README).
- If `ROBOT_NAME` is unset, the UI offers **yaml-only promote** and the xacro
  diff section shows a red banner "set ROBOT_NAME=tinker1|tinker2 to enable
  per-robot xacro write".
- If the resolved target path points at the **shared vendor xacro** (i.e. the
  per-robot override resolution falls through to the vendor file as a last
  resort), the apply endpoint REFUSES with `{ok: False, reason: "refusing to
  write shared vendor xacro — set up per-robot override at <path>"}`. No
  surprise cross-robot writes.

**Files:**
- Modify: `handeye_calib/handeye_web.py` (split `do_promote` into `compute_promote_diff` + `apply_promote`; new routes `GET /api/promote/diff`, `POST /api/promote/apply`, `POST /api/promote/reload`; both endpoints return BOTH yaml and xacro halves)
- Modify: `handeye_calib/apply_handeye.py` (add `resolve_robot_xacro_path(robot_name, basic_repo_root) -> Path | None`; add `seed_handeye_override_xacro(joint_name, xyz_str, rpy_str) -> str`; verify `patch_urdf_origin` raises a typed error when the joint isn't found so the UI can suggest the seed-file fallback)
- Modify: `webui/index.html` (promote panel: TWO diff blocks — yaml and xacro — each with its own 'Show', 'Apply' button, status, backup line; one 'Reload from disk' button)
- Modify: `webui/style.css` (`.diff-block`, `.diff-add`, `.diff-del`, `.diff-hunk`, `.diff-warn` for the vendor-fallback banner)
- Modify: `webui/app.js` (two parallel diff renderers; per-half confirm + apply; show backup paths; collapsible blocks when long)
- Modify: `handeye_calib/README.md` (new `## UI` section + `## Per-robot xacro override (one-time setup)` section + `0.4.0` changelog entry)
- Modify: `test/test_apply.py` (test `resolve_robot_xacro_path` for tinker1/tinker2/missing; test `seed_handeye_override_xacro` returns valid xacro; test `patch_urdf_origin` round-trip against a realsense_d435i.urdf.xacro-shaped fixture)
- Modify: `test/test_web_app.py` (diff endpoint returns BOTH halves; apply with no ROBOT_NAME refuses xacro half; apply against vendor-path target refuses)

**Interfaces:**
- Produces:
  - `apply_handeye.resolve_robot_xacro_path(robot_name: str | None, basic_repo_root: Path) -> Path | None`
    - Returns `basic_repo_root / "src" / "tinker_robot_config" / "robots" / robot_name / "wrist_camera.xacro"` when `robot_name` truthy, else `None`. Does NOT check existence — that's the caller's job.
  - `apply_handeye.seed_handeye_override_xacro(joint_name: str, xyz_str: str, rpy_str: str) -> str`
    - Returns a complete xacro file body:
      ```xml
      <?xml version="1.0"?>
      <robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="handeye_override">
        <!-- Generated by handeye_web. Include this from the robot's main xacro:
             <xacro:include filename="$(find tinker_robot_config)/robots/<ROBOT>/wrist_camera.xacro"/>
             and remove the corresponding <joint> block from the vendor d435i xacro
             (or guard the vendor block with a xacro:unless). -->
        <joint name="{joint_name}" type="fixed">
          <parent link="link_eef"/>
          <child link="camera_link"/>
          <origin xyz="{xyz_str}" rpy="{rpy_str}"/>
        </joint>
      </robot>
      ```
  - `HandeyeWebNode.compute_promote_diff() -> {ok, yaml: {...} | None, xacro: {...} | None, reason?: str}`
    where each sub-dict has `{target_path: str, current_text: str, proposed_text: str, diff: str, mode: "patch"|"seed"|"refuse-vendor", warning?: str}`. The `xacro` sub-dict is `None` when `ROBOT_NAME` is unset (UI shows "ROBOT_NAME required" banner).
  - `HandeyeWebNode.apply_promote(*, which: str = "both") -> {ok, yaml?: {written_path, backup_path}, xacro?: {written_path, backup_path}, reason?: str}`
    where `which ∈ {"yaml", "xacro", "both"}` lets the UI offer per-half apply. Both writes go through `apply_handeye.write_with_backup`; if `which == "both"` and the xacro write fails after the yaml write succeeded, the response surfaces the partial-success state explicitly (`{ok: True, yaml: {...}, xacro: {ok: False, reason: ...}}`).
  - HTTP `GET /api/promote/diff`, `POST /api/promote/apply` (body `{which: "yaml"|"xacro"|"both"}`), `POST /api/promote/reload`.

- [ ] **Step 1: Tests**

Add to `test/test_apply.py`:
```python
from pathlib import Path
import pytest
from handeye_calib import apply_handeye as ah


def test_resolve_robot_xacro_path_for_tinker2(tmp_path):
    # synthesize a basic-repo-shaped fixture
    (tmp_path / "src/tinker_robot_config/robots/tinker2").mkdir(parents=True)
    p = ah.resolve_robot_xacro_path("tinker2", tmp_path)
    assert p == tmp_path / "src/tinker_robot_config/robots/tinker2/wrist_camera.xacro"


def test_resolve_robot_xacro_path_none_when_robot_unset(tmp_path):
    assert ah.resolve_robot_xacro_path(None, tmp_path) is None
    assert ah.resolve_robot_xacro_path("", tmp_path) is None


def test_seed_handeye_override_xacro_well_formed():
    body = ah.seed_handeye_override_xacro("camera_link_joint",
                                          "0.07 -0.02 0.024", "3.14 -1.57 0")
    assert "<?xml" in body
    assert 'name="handeye_override"' in body
    assert 'name="camera_link_joint"' in body
    assert 'xyz="0.07 -0.02 0.024"' in body
    assert 'rpy="3.14 -1.57 0"' in body
    assert '<parent link="link_eef"' in body and '<child link="camera_link"' in body


def test_patch_urdf_origin_against_realsense_d435i_shape():
    sample = ('<robot><joint name="camera_link_joint" type="fixed">\n'
              '  <parent link="link_eef"/><child link="camera_link"/>\n'
              '  <origin xyz="0.06746 -0.0175 0.0237" rpy="3.14 -1.57 0"/>\n'
              '</joint></robot>\n')
    patched = ah.patch_urdf_origin(sample, "camera_link_joint",
                                    [0.08, -0.01, 0.02], [3.1, -1.6, 0.0])
    assert 'xyz="0.08 -0.01 0.02"' in patched
    assert 'rpy="3.1 -1.6 0.0"' in patched
    assert 'xyz="0.06746' not in patched
```

Add to `test/test_web_app.py`:
```python
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
        from handeye_calib.handeye_solve import SolveResult
        node.last_solve = SolveResult(X=np.eye(4), Tbb=np.eye(4),
            train_metrics={"trans_rmse_m":0.001,"rot_rmse_rad":0.001,"reproj_px":0.5},
            heldout_metrics={"trans_rmse_m":0.001,"rot_rmse_rad":0.001,"reproj_px":0.5},
            status="PASS", per_method=[])
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
        # (forge last_solve as above)
        ...
        r = c.post("/api/promote/apply", json={"which": "xacro"})
        body = r.json()
        assert body["ok"] is False and "ROBOT_NAME" in body["reason"]
    finally:
        node.destroy_node()
```

Run — expected FAIL.

- [ ] **Step 2: Implement `apply_handeye` additions**

In `handeye_calib/apply_handeye.py`:
```python
from pathlib import Path


def resolve_robot_xacro_path(robot_name, basic_repo_root):
    if not robot_name:
        return None
    return Path(basic_repo_root) / "src" / "tinker_robot_config" / "robots" \
           / robot_name / "wrist_camera.xacro"


_OVERRIDE_TEMPLATE = """<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="handeye_override">
  <!-- Generated by handeye_web. Include this from the robot's main xacro:
       <xacro:include filename="$(find tinker_robot_config)/robots/<ROBOT>/wrist_camera.xacro"/>
       and remove (or guard with <xacro:unless>) the corresponding <joint> in
       the vendor d435i xacro at
       xarm_description/urdf/camera/realsense_d435i.urdf.xacro -->
  <joint name="{joint_name}" type="fixed">
    <parent link="link_eef"/>
    <child link="camera_link"/>
    <origin xyz="{xyz_str}" rpy="{rpy_str}"/>
  </joint>
</robot>
"""


def seed_handeye_override_xacro(joint_name, xyz_str, rpy_str):
    return _OVERRIDE_TEMPLATE.format(joint_name=joint_name, xyz_str=xyz_str, rpy_str=rpy_str)
```

- [ ] **Step 3: Implement `compute_promote_diff` + `apply_promote` in `HandeyeWebNode`**

Pseudocode (substitute real names/paths):
```python
def compute_promote_diff(self):
    import os, difflib, yaml
    if self.last_solve is None:
        return {"ok": False, "reason": "run solve first"}

    # ---- yaml half (always computed) --------------------------------------
    T_mount_color = self._mount_to_color_matrix()  # from launch params
    T_eef_mount   = ah.compose_eef_to_mount(self.last_solve.X, T_mount_color)
    proposed_dict = ah.handeye_yaml_dict(T_eef_mount, self.last_solve.X,
                                         len(self.session.samples),
                                         self.last_solve.heldout_metrics,
                                         date=self._calibration_date_or_unset(),
                                         square_len_m=self._square_len_m)
    proposed_yaml = yaml.safe_dump(proposed_dict, sort_keys=False)
    yaml_target   = self._robot_handeye_yaml_path()  # existing v1 logic
    current_yaml  = yaml_target.read_text() if yaml_target and yaml_target.exists() else ""
    yaml_diff     = "".join(difflib.unified_diff(
        current_yaml.splitlines(keepends=True),
        proposed_yaml.splitlines(keepends=True),
        fromfile=str(yaml_target) if yaml_target else "(no path)",
        tofile="proposed", lineterm=""))
    yaml_half = ({"target_path": str(yaml_target),
                  "current_text": current_yaml,
                  "proposed_text": proposed_yaml,
                  "diff": yaml_diff,
                  "mode": "patch"} if yaml_target else None)

    # ---- xacro half (gated on ROBOT_NAME) ---------------------------------
    robot_name = os.environ.get("ROBOT_NAME") or self._robot_name_param()
    basic_root = self._tk25_basic_repo_root()  # parameter or env-resolved
    xacro_target = ah.resolve_robot_xacro_path(robot_name, basic_root) if basic_root else None

    xacro_half = None
    if xacro_target:
        # build the proposed xacro text
        # T_eef_mount above is what gets written into camera_link_joint origin
        xyz_str = " ".join(f"{v:.9g}" for v in T_eef_mount[:3, 3])
        from scipy.spatial.transform import Rotation as R
        rpy = R.from_matrix(T_eef_mount[:3, :3]).as_euler('xyz')
        rpy_str = " ".join(f"{v:.9g}" for v in rpy)
        joint_name = self._mount_joint_name_param()  # default "camera_link_joint"

        if xacro_target.exists():
            current_xacro = xacro_target.read_text()
            try:
                proposed_xacro = ah.patch_urdf_origin(current_xacro, joint_name, xyz_str.split(), rpy_str.split())
                mode = "patch"
            except ValueError:
                # joint not in existing override file → re-seed
                proposed_xacro = ah.seed_handeye_override_xacro(joint_name, xyz_str, rpy_str)
                mode = "seed"
        else:
            current_xacro  = ""
            proposed_xacro = ah.seed_handeye_override_xacro(joint_name, xyz_str, rpy_str)
            mode = "seed"

        xacro_diff = "".join(difflib.unified_diff(
            current_xacro.splitlines(keepends=True),
            proposed_xacro.splitlines(keepends=True),
            fromfile=str(xacro_target), tofile="proposed", lineterm=""))
        xacro_half = {"target_path": str(xacro_target),
                      "current_text": current_xacro,
                      "proposed_text": proposed_xacro,
                      "diff": xacro_diff,
                      "mode": mode}

    return {"ok": True, "yaml": yaml_half, "xacro": xacro_half,
            "robot_name": robot_name or None}


def apply_promote(self, which="both"):
    if self.last_solve is None:
        return {"ok": False, "reason": "run solve first"}
    diff = self.compute_promote_diff()
    if not diff["ok"]:
        return diff
    out = {"ok": True}

    if which in ("yaml", "both") and diff["yaml"]:
        y = diff["yaml"]
        # refuse if target_path is None or escapes the per-robot config dir
        ah.write_with_backup(y["target_path"], y["proposed_text"])
        out["yaml"] = {"written_path": y["target_path"],
                       "backup_path": _latest_backup_for(y["target_path"])}

    if which in ("xacro", "both"):
        if diff["xacro"] is None:
            out["xacro"] = {"ok": False, "reason": "ROBOT_NAME unset — cannot write per-robot xacro"}
        else:
            x = diff["xacro"]
            # safety: refuse to write the shared vendor xacro by path-prefix check
            if "xarm_description/urdf/camera/realsense_d435i.urdf.xacro" in x["target_path"]:
                out["xacro"] = {"ok": False,
                                "reason": "refusing to write shared vendor xacro — set up per-robot override"}
            else:
                ah.write_with_backup(x["target_path"], x["proposed_text"])
                out["xacro"] = {"written_path": x["target_path"],
                                "backup_path": _latest_backup_for(x["target_path"])}

    return out
```

Notes: `_latest_backup_for(path)` should glob `<path>.old-*` and return the most recent. `_tk25_basic_repo_root()` resolves via `ament_index_python.get_package_share_directory('tinker_robot_config')` and walks up to the repo root, or accepts an explicit launch param `tk25_basic_repo_root` for testing; if unresolvable, `xacro_half` stays `None` with a `reason: "tk25_basic repo root not resolvable"`.

- [ ] **Step 4: Frontend (two diff blocks, per-half apply)**

```html
<div class="promote-panel">
  <div class="diff-half">
    <h3>hand_eye.yaml <span class="badge" id="yaml-target"></span></h3>
    <button id="promote-yaml-diff-btn">Show diff</button>
    <pre class="diff-block" id="promote-yaml-diff"></pre>
    <button class="primary" id="promote-yaml-apply-btn" disabled>Apply yaml</button>
    <div class="status-line" id="promote-yaml-status"></div>
    <div class="status-line" id="promote-yaml-backup"></div>
  </div>
  <div class="diff-half">
    <h3>wrist_camera.xacro <span class="badge" id="xacro-target"></span></h3>
    <div class="status-line" id="xacro-mode"></div>     <!-- "patch"/"seed"/"ROBOT_NAME unset" -->
    <div class="status-line diff-warn" id="xacro-warn" hidden></div>
    <button id="promote-xacro-diff-btn">Show diff</button>
    <pre class="diff-block" id="promote-xacro-diff"></pre>
    <button class="primary" id="promote-xacro-apply-btn" disabled>Apply xacro</button>
    <div class="status-line" id="promote-xacro-status"></div>
    <div class="status-line" id="promote-xacro-backup"></div>
  </div>
  <div class="row"><button id="promote-reload-btn">Reload from disk</button></div>
</div>
```

JS:
- `promote-yaml-diff-btn` / `promote-xacro-diff-btn` click: GET `/api/promote/diff` once, render `body.yaml.diff` into `#promote-yaml-diff` and `body.xacro.diff` into `#promote-xacro-diff`. Apply line coloring (`.diff-add`/`.diff-del`/`.diff-hunk`). Set `#yaml-target` / `#xacro-target` text to the resolved paths. Set `#xacro-mode` to `mode: patch` (existing file) / `mode: seed (new per-robot xacro)` / `ROBOT_NAME unset — yaml-only promote` when `body.xacro === null`. Enable each Apply button only if its diff is non-empty AND its `mode != "refuse-vendor"`.
- `promote-yaml-apply-btn` click: `confirm('Overwrite ' + yamlPath + '?\nA timestamped backup will be made.')`; POST `/api/promote/apply` with `{which: "yaml"}`; on `body.yaml.written_path` show backup line (green).
- `promote-xacro-apply-btn` click: same with `{which: "xacro"}`.

CSS:
```css
.promote-panel { display: grid; grid-template-columns: 1fr; gap: 16px; }
@media (min-width: 1200px) { .promote-panel { grid-template-columns: 1fr 1fr; } }
.diff-half { background: var(--bg-panel); padding: 12px; border-radius: 4px; }
.diff-block { background: #0a0a0a; padding: 8px; font-size: 11px;
              white-space: pre; max-height: 50vh; overflow: auto; }
.diff-add { color: var(--ok); }
.diff-del { color: var(--err); }
.diff-hunk { color: var(--accent); }
.diff-warn { color: var(--err); }
.badge { font-size: 10px; color: var(--fg-muted); background: #0a0a0a;
         padding: 2px 6px; border-radius: 3px; margin-left: 6px; }
```

- [ ] **Step 5: README additions**

```markdown
## UI

The handeye_web tool is a single-page calibration UI with five tabs:

- **Info** — camera/TF/robot/board/safety status, T_base_eef matrix.
- **Move** — joint editor (rad/deg toggle), Load-current, Zero, presets, with a live SafetyEnvelope preview before sending.
- **Capture** — stability-gated capture (3 steady frames), sample gallery with thumbnails, per-sample delete, diversity meter (max pairwise rotation° / 30° target).
- **Solve** — method picker (auto/TSAI/PARK/HORAUD/ANDREFF/DANIILIDIS), per-method reprojection comparison, residual histogram + scatter, sample-coverage canvas, PASS/WARN/FAIL pill with mm/deg metrics.
- **Promote** — side-by-side unified diff for both `hand_eye.yaml` AND a per-robot
  `wrist_camera.xacro` override, ROBOT_NAME-scoped, confirm-before-apply, backup
  paths surfaced for each write.

Live state pushed via WebSocket at 5 Hz (no polling). Live camera feed polls at 3 Hz with annotated/raw toggle and a resizable panel (persisted in `localStorage`).

## Per-robot xacro override (one-time setup)

The wrist camera mount is defined in a shared vendor xacro
(`xarm_description/urdf/camera/realsense_d435i.urdf.xacro`, joint
`camera_link_joint`). Patching it in place would write tinker1's
calibration over tinker2's. Instead, handeye_web writes a per-robot
override at:

```
src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/wrist_camera.xacro
```

For the override to take effect, **the operator must one-time include
this file from the main robot xacro** (e.g.
`src/tk25_basic/src/tinker_urdf/src/mobile_manipulator.urdf.xacro`):

```xml
<xacro:include filename="$(find tinker_robot_config)/robots/$(arg robot_name)/wrist_camera.xacro"/>
```

and either remove the corresponding `<joint name="camera_link_joint">`
block from the vendor d435i xacro, or guard it with
`<xacro:unless value="$(arg use_handeye_override)">…</xacro:unless>`.
This setup happens once per workspace; afterwards every
`handeye_web` calibration just overwrites the per-robot override file.

If `ROBOT_NAME` is unset when promoting, the UI offers **yaml-only
promote** (the `hand_eye.yaml` is still written; the xacro half is
disabled with a banner explaining why). The promote endpoint refuses
to write the shared vendor xacro under any circumstance.
```

Changelog (top of existing list):
```markdown
### 0.4.0 (2026-06-20)
- **handeye_web quality rewrite to pan_tilt parity.** Inline 30-line UI replaced
  by a static `webui/` (index.html + style.css + app.js, ~3 kLoC frontend).
  New: WebSocket state stream, tabbed layout, resizable live frame with
  annotated/raw toggle and detection badge, joint editor with rad/deg toggle
  and live safety preview, settle-gated capture with sample gallery + per-sample
  delete + diversity meter, solve panel with method picker + per-method
  comparison + residual histogram/scatter + coverage canvas (mm/deg
  units), promote panel with side-by-side unified-diff preview for BOTH
  `hand_eye.yaml` AND a per-robot `wrist_camera.xacro` override
  (ROBOT_NAME-scoped, refuses to overwrite the shared vendor xacro),
  confirm-before-apply, backup paths surfaced.
- Closes v1 deferral: StabilityTracker is now a hard pre-capture gate.
- New `apply_handeye` helpers: `resolve_robot_xacro_path`,
  `seed_handeye_override_xacro`.
- New endpoints: `/ws`, `/api/samples/{idx}/thumb.jpg`, `DELETE /api/samples/{idx}`,
  `/api/promote/diff` (returns both yaml + xacro halves), `/api/promote/apply`
  (accepts `which ∈ {yaml,xacro,both}`), `/api/promote/reload`.
- `/api/solve` accepts `{method}` body; `/api/frame.jpg` accepts `?raw=1`.
```

- [ ] **Step 6: Final verify + commit**

```bash
tkbuild tk26_vision --packages-select handeye_calib
PYTHONPATH=... pytest src/tk26_vision/src/handeye_calib/test/ -v
# launch + click through all 5 tabs in a real browser:
ROBOT_NAME=tinker2 ros2 launch handeye_calib handeye_web.launch.py port:=8800
# verify the Promote tab shows BOTH yaml AND xacro halves;
# verify with ROBOT_NAME unset the xacro half shows "ROBOT_NAME unset" banner.
```

```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/ src/handeye_calib/test/ src/handeye_calib/README.md
git commit -m "feat(handeye_calib): Promote tab — yaml+xacro unified diff, ROBOT_NAME-scoped, backup + UI docs

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage vs the user's complaint ("nowhere near pan_tilt quality"):**

| pan_tilt feature | Closed by |
|---|---|
| Separate `webui/` (index.html + style.css + app.js) | T1 |
| Dark theme + CSS variables + status-line classes | T1 |
| Connection pill (live / dropped) with auto-reconnect | T1 |
| WebSocket state push (no polling for state) | T1 |
| Resizable live camera with localStorage width | T2 |
| Annotated / raw frame toggle | T2 |
| Detection badge (corners + RMS, color-coded) | T2 |
| Frame placeholder when no camera | T2 |
| Tab layout (5 panels) | T3 |
| Info kv tables (camera/topic/ROS_DOMAIN/robot/board/safety) | T3 |
| T_base_eef matrix display | T3 |
| Joint editor with rad/deg unit toggle + Load-current/Zero/presets | T3 |
| Live SafetyEnvelope preview before move | T3 |
| Confirmation dialog before robot motion | T3 |
| Settle gate (StabilityTracker) hard-gating capture | T4 |
| Sample gallery with per-capture thumbnails | T4 |
| Per-sample delete | T4 |
| Diversity meter (coverage° / target°) | T4 |
| Solve method picker | T5 |
| Per-method comparison table | T5 |
| Residual histogram + scatter canvas | T5 |
| Coverage canvas | T5 |
| mm + deg units in UI | T5 |
| PASS / WARN / FAIL gate pill | T5 |
| Unified diff preview before write (yaml + xacro) | T6 |
| ROBOT_NAME-scoped per-robot xacro override write | T6 |
| Refuse to overwrite shared vendor xacro | T6 |
| Confirm-before-apply + backup path display (per half) | T6 |
| Reload-from-disk button | T6 |

**Intentionally NOT included (called out so the implementer doesn't backfill them):**

- pan_tilt's session picker + subprocess runner + `/ws/calib-log` log stream. Handeye solves in-process; sessions buy nothing.
- pan_tilt's Cartesian move editor (`/api/xarm/move_cartesian`). Adds a planning dependency. Defer to v3 if the operator misses it.
- pan_tilt's per-phase waypoint authoring (Phase 1/2/4 buttons). Handeye is a single capture loop, not phased.

**Placeholder scan:** the only structured-skeleton blocks are in T3 ("look-forward" preset, intentionally TODO with a `// FIXME` instruction NOT to invent unsafe joint values) and T4's settle gate (instruction is to READ `gates.py` first for the actual `StabilityTracker` constructor — the API is real, not inventable from the plan). All other steps give complete tests + code.

**Type consistency:** `state.samples` schema (from `sample_metadata`) is consumed by the gallery JS in T4 with the same key names (`idx, n_corners, reproj_px, area_frac, angular_delta_deg, joint_positions`). `solve_payload_v2` keys (`X_xyz_mm, X_rpy_deg, per_method_summary, per_sample_reproj_px`) are used identically in the Solve panel JS. `enriched_state_payload` field names are reused across all five tabs without renames.

**Commit discipline:** 6 commits, one per task. Each task names its exact files. No `--amend` or rebase. All identity = `Ccindy0171 <cindy.w0135@gmail.com>` via the repo-local git config.
