# track_web bringup control (launch BT + audio + dummy nav from the webui) — design

**Date:** 2026-06-10
**Package:** `vision_track` (`src/tk26_vision`)
**Status:** approved (user, 2026-06-10)

Upgrade the `track_web` dashboard so it can start/stop the follow-person demo
components on demand, and add a launch file that brings up the control surface.
The webui gains a **process-manager** (fixed allowlist) with **per-component**
start/stop toggles for: complete audio, dummy navigation, and the follow-person
behaviour tree.

## Decisions (user, 2026-06-10)
- **On-demand process-manager in the webui** (not an all-in-one launch): a webui
  control spawns/stops the components as `ros2 launch`/`ros2 run` subprocesses.
- **Per-component toggles** — audio, dummy_nav, bt each independently
  start/stop/status.
- **Homed in `vision_track`** (with the webui); it spawns the `behavior_tree` /
  `audio_pakage` bringups at runtime (no build dep — just `ros2 run`/`ros2 launch`).

## Architecture
```
ros2 launch vision_track track_web_control.launch.py
  ├─ cleanup (SIGTERM stale person_track_server / track_web)
  ├─ person_track_server         (the tracker)
  └─ track_web  (FastAPI + ROS bridge + ProcessManager)
                       │ webui Bringup panel → /api/proc/{name}/start|stop
                       ▼
        ProcessManager (fixed allowlist) spawns:
          audio     → ros2 launch audio_pakage audio.launch.py
          dummy_nav → ros2 run behavior_tree dummy-nav
          bt        → ros2 run behavior_tree follow-person
```

## Components

### 1. `process_manager.py` (new, `vision_track/`)
A standalone, ROS-free, unit-testable supervisor. **Fixed allowlist** — the API
takes a *name*, never a command, so the browser cannot run arbitrary processes.

```python
REGISTRY = {
    "audio":     ["ros2", "launch", "audio_pakage", "audio.launch.py"],
    "dummy_nav": ["ros2", "run", "behavior_tree", "dummy-nav"],
    "bt":        ["ros2", "run", "behavior_tree", "follow-person"],
}
```
`ProcessManager` (constructor takes `registry=REGISTRY` so tests inject a fake
command, e.g. `["sleep","30"]`):
- `start(name) -> dict` — unknown name → `{"error": ...}` (never raises). If
  already running → return current status (idempotent). Else
  `subprocess.Popen(cmd, start_new_session=True, env=os.environ.copy())` (own
  process group so the whole `ros2 launch` tree can be signalled); record the
  Popen. Return `status(name)`.
- `stop(name) -> dict` — if running: `os.killpg(os.getpgid(pid), SIGTERM)`, wait
  up to `term_timeout_s` (default 5), then `SIGKILL` the group if still alive.
  Return `status(name)`.
- `status(name) -> dict` — `{"name", "running": bool, "pid": int|None,
  "returncode": int|None}` (poll the Popen to reap; cache last returncode).
- `status_all() -> dict[name -> status]`.
- `shutdown_all()` — stop every running child; called on track_web exit.
- Env note: children inherit the track_web process env (ROS + `tk25_ws/install`
  sourced), so `ros2 run/launch` resolve the spawned packages; the entry-point
  shebangs (tkbuild-set) route each to its own venv.

### 2. Bridge wiring (`track_web.py` `TrackWebNode`)
Model on the existing `record_start`/`record_stop` subprocess handling. The node
owns one `ProcessManager`; expose the bridge contract methods `proc_start(name)`,
`proc_stop(name)`, `proc_status()` (delegate to the manager). Call
`manager.shutdown_all()` in `main()`'s shutdown path (the `finally`/SIGINT
handler that already stops recording) so children die with the dashboard.

### 3. Endpoints (`track_web_app.py`)
- `POST /api/proc/{name}/start` → `bridge.proc_start(name)`.
- `POST /api/proc/{name}/stop` → `bridge.proc_stop(name)`.
- `GET /api/proc/status` → `bridge.proc_status()`.
- `/ws/state`: each loop, also push `{"type": "proc", "data": bridge.proc_status()}`
  when it changes (track a `last_proc` snapshot like `last_state_seq`).
The existing global exception handler already guarantees JSON on error.

### 4. Frontend (`webui/index.html` + `app.js` + `style.css`)
A new **"Bringup"** panel (in the controls column) with three rows — **Audio**,
**Dummy Nav**, **Follow BT** — each: a start/stop toggle button + a status dot
(grey=stopped, green=running, red=exited-nonzero). Toggles POST to
`/api/proc/{name}/start|stop`; status comes from the `proc` ws message (fallback
poll `/api/proc/status`). **Disable the existing manual `#btn-start`
(goal start) while `bt` is running** — the BT owns the `/track_person` goal, so a
second manual goal would conflict; show a hint ("Follow BT owns the goal"). Style
to match the existing dashboard (reuse the controls/badge CSS).

### 5. Launch (`launch/track_web_control.launch.py`)
Model on `track_web_bench.launch.py` but minimal: a cleanup `ExecuteProcess`
(SIGTERM stale `person_track_server` / `track_web`), then `person_track_server`
(camera-topic + tracker args) and `track_web` (`output='screen'`). Args:
`launch_tracker` (default true), camera topics, web `host`/`port`. The follow-demo
components are NOT here — the webui spawns them.

## Interaction / prerequisites
- **BT owns the goal** when `bt` is running → manual goal-start disabled in the UI
  (designed-in, §4).
- Recommended order surfaced in the UI: Audio + Dummy Nav, then Follow BT (the BT
  retries the `announce` service / `/track_person` action, so strict order isn't
  required).
- **Prerequisite:** `behavior_tree` + `audio_pakage` must be in `tk25_ws/install`
  (built via `tkbuild tk25_decision` + `tkbuild tk_24_audio`) so the webui's
  `ros2 run`/`ros2 launch` resolve them. If a package is missing, `start` returns
  an error dict and the UI shows it (no crash).
- **Safety:** fixed name→command registry; same trust boundary as the existing
  goal/record control (local-LAN robot dashboard).

## Testing
- `test/test_process_manager.py`: with a harmless command (`["sleep","30"]` /
  `["bash","-c","sleep 30"]`): start → `running:true`+pid; double-start is
  idempotent (same pid); stop → not running + the process is actually gone
  (`os.kill(pid,0)` raises); unknown name → error dict (no raise);
  `shutdown_all()` kills everything; a fast-exiting command
  (`["true"]`) reports `running:false` + returncode 0 after a poll.
- `test/test_track_web_app.py` (extend): FastAPI `TestClient` with a fake bridge
  exposing `proc_start/stop/status` — `POST /api/proc/audio/start` returns the
  status dict; `/api/proc/bogus/start` returns an error dict (validated by the
  manager, not a 500); `GET /api/proc/status` returns the map.
- Frontend: manual (toggles + status dots + manual-start disabled while bt runs).
- Build via `tkbuild tk26_vision --packages-select vision_track`; `ros2 launch
  vision_track track_web_control.launch.py --show-args`; no new flake8 on touched
  lines.

## Files
- Create: `vision_track/process_manager.py`,
  `launch/track_web_control.launch.py`, `test/test_process_manager.py`.
- Modify: `vision_track/track_web.py` (manager + bridge methods + shutdown),
  `vision_track/track_web_app.py` (endpoints + ws proc push),
  `webui/index.html`, `webui/app.js`, `webui/style.css`,
  `test/test_track_web_app.py`, `readme.md` (changelog). (`setup.py` already
  globs `launch/` + `webui/`.)

## Invariants / risks
- The process-manager never raises into the request handler (always returns a
  dict); unknown names are rejected against the registry.
- Children are spawned in their own process group and reliably killed on stop +
  on dashboard shutdown (no orphans).
- Cross-package coupling is runtime-only (`ros2 run`/`ros2 launch`); no build
  dependency added to `vision_track`.
