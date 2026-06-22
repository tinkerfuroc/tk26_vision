# track_web bringup control — implementation plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD. Phase = one commit.

**Spec:** `docs/superpowers/specs/2026-06-10-track-web-bringup-control-design.md`
**Package:** `vision_track` (`src/tk26_vision`).
**Build:** `bash ./tkbuild tk26_vision --packages-select vision_track` (from `/home/tinker/tk25_ws`).
**Test:** `cd /home/tinker/tk25_ws/src/tk26_vision && source .venv-vision-main/bin/activate && cd src/vision_track && python -m pytest test/<file> -q -p no:cacheprovider`.
**Patterns to model:** `vision_track/track_web.py` (`TrackWebNode`, esp. `record_start`/`record_stop` subprocess handling + `main()` shutdown), `vision_track/track_web_app.py` (`create_app`, the `/api/*` + `/ws/state` shapes, the global exception handler), `test/test_track_web_app.py` (fake-bridge TestClient style), `launch/track_web_bench.launch.py` (cleanup ExecuteProcess + Node actions + args).
**Invariants:** process-manager never raises into a handler (returns dicts); fixed name→command registry (no arbitrary commands); children in own process groups, killed on stop + dashboard shutdown; runtime-only coupling (no build dep added).

---

## Task 1 — `ProcessManager` (one commit)
**Files:** create `vision_track/process_manager.py`; test `test/test_process_manager.py`.

- [ ] **Failing test** `test/test_process_manager.py` (Apache header from any `test/test_*.py`): build `ProcessManager(registry={"sleeper": ["sleep","30"], "quick": ["true"]})`. Assert: `start("sleeper")` → `running True` + int pid; second `start("sleeper")` → same pid (idempotent); `os.kill(pid,0)` ok while running; `stop("sleeper")` → `running False` and `os.kill(pid,0)` raises `ProcessLookupError`; `start("bogus")` → dict with `"error"` (no raise); `quick` after a short poll → `running False`, `returncode 0`; `shutdown_all()` after starting `sleeper` → not running. Use real short-lived commands (no ROS).
- [ ] **Run → FAIL.**
- [ ] **Implement** `process_manager.py` per spec §1: module-level `REGISTRY` (audio/dummy_nav/bt commands), `ProcessManager(registry=REGISTRY)` with `start/stop/status/status_all/shutdown_all`. `Popen(..., start_new_session=True, env=os.environ.copy())`; `stop` uses `os.killpg(os.getpgid(pid), SIGTERM)` then `SIGKILL` after `term_timeout_s` (5.0); `status` polls+reaps and caches `returncode`; all methods validate name against the registry and return dicts (never raise). Thread-safe with a `threading.Lock` (uvicorn worker threads call it).
- [ ] **Run → PASS.**
- [ ] **Commit:** `feat(vision_track): ProcessManager — fixed-allowlist subprocess supervisor for track_web`

---

## Task 2 — bridge wiring + endpoints + ws push (one commit)
**Files:** modify `vision_track/track_web.py`, `vision_track/track_web_app.py`; extend `test/test_track_web_app.py`.

- [ ] **Failing test** in `test/test_track_web_app.py`: extend the fake bridge with `proc_start(name)`, `proc_stop(name)`, `proc_status()` (return canned dicts). Add tests (FastAPI `TestClient`): `POST /api/proc/audio/start` → the status dict; `POST /api/proc/audio/stop` → status; `GET /api/proc/status` → the map; a bogus name still returns JSON (delegated to the bridge/manager, not a 500).
- [ ] **Run → FAIL** (endpoints missing).
- [ ] **Implement endpoints** in `track_web_app.py` `create_app`: `@app.post("/api/proc/{name}/start")` → `bridge.proc_start(name)`; `/stop` → `bridge.proc_stop(name)`; `@app.get("/api/proc/status")` → `bridge.proc_status()`. In `ws_state`, after the state/gallery push, track `last_proc` and push `{"type":"proc","data": bridge.proc_status()}` when changed.
- [ ] **Implement bridge** in `track_web.py` `TrackWebNode`: import + construct `ProcessManager`; add `proc_start/proc_stop/proc_status` (delegate). In `main()`'s shutdown path (where recording is stopped / on SIGINT/finally), call `node.proc_manager.shutdown_all()` so spawned children die with the dashboard. Model env/subprocess handling on the existing `record_start`.
- [ ] **Run → PASS** the new endpoint tests + the existing `test_track_web_app.py` stay green.
- [ ] **Commit:** `feat(vision_track): /api/proc endpoints + ProcessManager wired into track_web bridge`

---

## Task 3 — frontend Bringup panel (one commit)
**Files:** modify `webui/index.html`, `webui/app.js`, `webui/style.css`.

- [ ] **Implement** a "Bringup" panel in `index.html` (in the `#controls` column): three rows (`Audio` / `Dummy Nav` / `Follow BT`) each with a toggle button (`id="proc-audio"`, `proc-dummy_nav`, `proc-bt`) + a status dot (`id="dot-audio"`, …). In `app.js`: clicking a toggle POSTs `/api/proc/{name}/start` or `/stop` based on current state; handle the `{"type":"proc"}` ws message (and a startup `GET /api/proc/status`) to set dot colour (grey stopped / green running / red exited-nonzero) + button label (Start/Stop). When `proc.bt.running` is true, **disable `#btn-start`** (manual goal) + show a hint; re-enable when bt stops. In `style.css`: `.proc-dot` states + the panel layout, matching the existing controls/badge styling.
- [ ] **Verify** (no unit framework for the static UI): `python -c "import json"`-style is N/A; instead confirm the JS references match the new endpoints + ws `proc` shape from Task 2, and the element ids are consistent between html/js. (Manual browser check is operator-side.)
- [ ] **Commit:** `feat(vision_track): track_web Bringup panel — per-component start/stop toggles`

---

## Task 4 — control launch + README (one commit)
**Files:** create `launch/track_web_control.launch.py`; modify `readme.md`.

- [ ] **Implement** `track_web_control.launch.py` (model on `track_web_bench.launch.py`): a cleanup `ExecuteProcess` (`bash -lc` SIGTERM stale `person_track_server` + `track_web` via narrow `lib/<pkg>/` pkill patterns — copy the bench's safe patterns), then `Node(package='vision_track', executable='person_track_server', ...)` gated on `launch_tracker` (default true) with camera-topic args, and `Node(package='vision_track', executable='track_web', output='screen')` with `host`/`port` args. Declare args: `launch_tracker`, `image_topic`/`depth_topic`/`camera_info_topic`, `host`, `port`.
- [ ] **README changelog** (`readme.md`): document `track_web_control.launch.py` + the Bringup panel (per-component audio/dummy_nav/bt toggles), the prerequisite (`tkbuild tk25_decision` + `tkbuild tk_24_audio` so the spawns resolve), and the BT-owns-the-goal note.
- [ ] **Verify:** `ros2 launch vision_track track_web_control.launch.py --show-args` lists the args (after build).
- [ ] **Commit:** `feat(vision_track): track_web_control.launch.py (tracker + upgraded webui)`

---

## Final
- [ ] `bash ./tkbuild tk26_vision --packages-select vision_track`; run `test/test_process_manager.py` + `test/test_track_web_app.py` green; `ros2 launch vision_track track_web_control.launch.py --show-args`; confirm `webui/` + the new launch install to `share/`; no new flake8 on touched files. DEV_NOTES entry (2026-06-10) summarizing the feature + the operator checks (build behavior_tree/audio via tkbuild into tk25_ws/install; click each toggle; manual-start disabled while bt runs; children die on dashboard exit). Commit `docs(vision_track): DEV_NOTES — track_web bringup control`.
