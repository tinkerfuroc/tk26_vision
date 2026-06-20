# handeye_calib — Waypoint authoring + auto-capture sequence

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement task-by-task (fresh implementer per task; per-task reviews skipped only on explicit "go faster" signal — otherwise dispatched on Sonnet for mechanical fixes, Opus for the state-machine task).

**Goal:** Add the pan_tilt-style "author a list of arm waypoints, then run an automatic move→settle→capture loop" workflow to the handeye_calib web UI. Today the operator drives each capture manually (UI button + a teach-mode pose). With this feature they record a sequence once, save it per-robot, and re-run the full capture sweep with one click.

**Architecture:** Backend extends `HandeyeWebNode` with (1) a waypoint list with CRUD endpoints + per-robot YAML persistence, and (2) a `CaptureSequenceRunner` daemon-thread state machine that drives `JointMove` → `StabilityTracker` settle → `do_capture()` → next. Frontend extends the **Capture tab** with a waypoints sub-panel (list, add-current, delete, save/reload, run/cancel/dry-run) above the existing settle-gated manual capture button + gallery. Progress is surfaced via the existing 5 Hz WS state push (no subprocess + log stream; the sequence loop fires the same `do_capture()` the manual button uses, so the gallery updates organically).

**Reference UX patterns** to copy verbatim from pan_tilt: the per-row "load / remove" button pair on a waypoint list (`src/pan_tilt/webui/app.js` xArm Waypoints tab), the `confirm()` dialog text before any motion ("Send xArm to these joints now?\n…"), the cancel button + log fan-out idiom (we'll simplify — no separate subprocess, just a state line), and the "Save to draft" / "Reload from draft" persistence buttons.

**Supersedes:** nothing. Adds a new feature on top of `handeye_web` v2 (commit `efda3d7` and prior). Compatible with the per-robot config convention introduced in T6 of the v2 plan ([[2026-06-20-handeye-web-v2-quality-rewrite.md]]).

## Global Constraints

- Package: `src/tk26_vision/src/handeye_calib/`. Git repo is `src/tk26_vision` (branch `dev`).
- **Concurrent committer present** (foundation_stereo, kimi_api, etc.). Commit ONLY the files each task names. Never `git add -A`/`.`, never `--amend`, never rebase.
- `import handeye_calib.handeye_web` and `import handeye_calib.web_support` MUST stay ROS-free (rclpy/fastapi nested inside functions/methods).
- Do NOT modify `validate_pose_set` / `diff_payload` / `MIN_POSES` in `handeye_web.py`.
- **Build wrapper:** rebuild ONLY via `tkbuild tk26_vision --packages-select handeye_calib`.
- **Identity:** all commits authored as `Ccindy0171 <cindy.w0135@gmail.com>` (repo-local git config enforces).
- **Safety invariant:** every move sent during the sequence loop goes through the same `do_move()` path the manual UI uses — same `SafetyEnvelope.validate(t_base_ee)` check, same `JointMove` action client. No fast path that bypasses the envelope.
- **Cancel is cooperative + immediate:** the runner thread checks a `threading.Event` between EVERY state transition (post-move, post-settle, post-capture). On cancel, the in-flight `JointMove` goal is also cancelled via `goal_handle.cancel_goal_async()` so the arm doesn't keep moving toward a now-stale target.
- **Settle is reused, NOT re-invented:** the same `gates.StabilityTracker` that hard-gates manual capture (T4) gates each sequence step. If the arm doesn't settle within `settle_timeout_s` (default 5.0), the step is skipped and logged; the sequence does NOT abort on a single settle failure — it moves on and the operator decides whether to re-run.
- **Persistence is per-robot.** Waypoint YAML lives at `src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/handeye_waypoints.yaml`. When `ROBOT_NAME` is unset, save endpoints refuse (mirrors T6's xacro convention); load on startup is best-effort (silently skip if file absent).
- **Dry-run mode** drives moves + settles WITHOUT calling `do_capture()` — so the operator can verify a recorded sequence is safe + reachable before committing to a capture run that pollutes the sample gallery.
- Units: backend stores SI (rad). Frontend renders rad or deg per the existing Move tab's unit toggle (waypoint list shows whichever the user has selected; persistence is always rad).

---

## File structure

| File | Status | Responsibility |
|---|---|---|
| `handeye_calib/handeye_calib/waypoints.py` | NEW (T1) | Pure ROS-free helpers: `WaypointStore` (in-memory list with add/delete/load/save), YAML schema, `resolve_waypoints_path(robot_name, basic_repo_root)`. |
| `handeye_calib/handeye_calib/handeye_web.py` | MODIFY (T1, T3) | T1: WaypointStore wired into `HandeyeWebNode`, CRUD + save/reload endpoints, `state.waypoints` in WS push. T3: `CaptureSequenceRunner` + sequence endpoints + `state.sequence` in WS push. |
| `handeye_calib/handeye_calib/web_support.py` | MODIFY (T1) | New `waypoint_metadata(idx, joints_rad) -> dict` helper (idx + 7 floats + abbreviation string). |
| `handeye_calib/handeye_calib/webui/index.html` | MODIFY (T2, T4) | T2: waypoints sub-panel inside Capture tab (above gallery). T4: sequence controls (run / dry / cancel) + progress line + per-step log. |
| `handeye_calib/handeye_calib/webui/style.css` | MODIFY (T2, T4) | `.waypoints-panel`, `.waypoint-row`, `.waypoint-joints`, `.sequence-controls`, `.sequence-progress`, `.sequence-log` classes. |
| `handeye_calib/handeye_calib/webui/app.js` | MODIFY (T2, T4) | T2: waypoint list renderer + add/delete/load/save/reload handlers. T4: sequence controls + progress renderer + cancel handler. |
| `handeye_calib/README.md` | MODIFY (T5) | New `## Waypoints + auto-capture` section, `0.5.0 (2026-06-20)` changelog entry. |
| `handeye_calib/test/test_waypoints.py` | NEW (T1) | Unit tests for `WaypointStore`, YAML round-trip, `resolve_waypoints_path`. |
| `handeye_calib/test/test_web_app.py` | MODIFY (T1, T3) | T1: CRUD endpoint tests. T3: sequence start/cancel + state shape tests. |
| `handeye_calib/test/test_web_node.py` | MODIFY (T3) | Tests for `CaptureSequenceRunner` state machine (mock JointMove + stability). |

---

## Task 1 — Waypoint CRUD backend + per-robot YAML persistence

**Files:**
- Create: `src/tk26_vision/src/handeye_calib/handeye_calib/waypoints.py`
- Modify: `src/tk26_vision/src/handeye_calib/handeye_calib/web_support.py` (add `waypoint_metadata` helper)
- Modify: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_web.py` (wire WaypointStore + 5 new endpoints + `state.waypoints` field)
- Create: `src/tk26_vision/src/handeye_calib/test/test_waypoints.py`
- Modify: `src/tk26_vision/src/handeye_calib/test/test_web_app.py` (CRUD endpoint coverage)

**Interfaces produced:**
- `waypoints.WaypointStore`:
  - `__init__(self)` — empty list.
  - `.list() -> list[list[float]]` — copy of the 7-float lists.
  - `.add(joints_rad: Sequence[float]) -> int` — validates length 7, returns new idx.
  - `.delete(idx: int) -> bool` — returns True on success, False on out-of-range.
  - `.clear() -> None`
  - `.load_yaml(path: Path) -> int` — replaces in-memory list; returns count loaded. Raises on parse error.
  - `.save_yaml(path: Path) -> None` — atomic write via `apply_handeye.write_with_backup` (so existing waypoints aren't lost on a typo).
- `waypoints.resolve_waypoints_path(robot_name, basic_repo_root) -> Path | None` — mirrors T6's `resolve_robot_xacro_path` pattern. Returns `basic_repo_root / "src/tinker_robot_config/robots/<robot>/handeye_waypoints.yaml"` when robot truthy; `None` otherwise.
- `waypoints.YAML_SCHEMA_VERSION = 1` — top-level YAML keys: `schema_version, recorded_for_robot, waypoints: [[j0..j6], ...]`.
- `web_support.waypoint_metadata(idx: int, joints_rad: Sequence[float]) -> dict` with keys: `idx, joints_rad (list[7] float), abbrev (string, e.g. "0.42, -0.30, 1.57, …" — first 3 rounded, ellipsis)`.
- `HandeyeWebNode.do_add_waypoint() -> {ok, count, reason?}` — uses live `self._xarm_joints` cache, returns `{ok:false, reason: "no current joints"}` when unavailable.
- `HandeyeWebNode.do_delete_waypoint(idx: int) -> {ok, count, reason?}`.
- `HandeyeWebNode.do_save_waypoints() -> {ok, path, count, reason?}` — refuses when `ROBOT_NAME` unset (mirrors T6 promote pattern); returns the resolved YAML path on success.
- `HandeyeWebNode.do_reload_waypoints() -> {ok, count, path, reason?}` — re-reads the per-robot YAML; replaces in-memory list.
- WS state extension: `state.waypoints = [waypoint_metadata(i, w) for i, w in enumerate(store.list())]` (empty list when none).
- New HTTP routes:
  - `GET /api/waypoints` → `{count, items: [waypoint_metadata(...)]}` (mirror of `state.waypoints`).
  - `POST /api/waypoints` body `{}` → `do_add_waypoint()`.
  - `DELETE /api/waypoints/{idx}` → `do_delete_waypoint(idx)`.
  - `POST /api/waypoints/save` → `do_save_waypoints()`.
  - `POST /api/waypoints/reload` → `do_reload_waypoints()`.

**Startup behavior:** when `ROBOT_NAME` is set AND the per-robot YAML exists, `HandeyeWebNode.__init__` calls `do_reload_waypoints()` once (silently — failures logged but don't block startup). Operator's last-saved sequence persists across restarts.

**Implementation notes:**
- WaypointStore is ROS-free (pure Python + pyyaml). Lives in its own module so the unit tests don't need rclpy at all.
- `do_add_waypoint` reads under the same `self.lock` the rest of the node uses — `self._xarm_joints` is the cache populated by `_on_joint_state` (T1 of v2 plan).
- `do_save_waypoints` uses `apply_handeye.write_with_backup(path, yaml.safe_dump(payload))` so a previous waypoints file is backed up to `handeye_waypoints.yaml.old-<timestamp>` (consistent with promote's backup discipline).

- [ ] **Step 1: Write failing tests**

In `test/test_waypoints.py` (new):
```python
from pathlib import Path
import pytest
import yaml
from handeye_calib import waypoints as wp


def test_store_starts_empty():
    s = wp.WaypointStore()
    assert s.list() == []


def test_store_add_validates_length():
    s = wp.WaypointStore()
    s.add([0, 0, 0, 0, 0, 0, 0])
    assert len(s.list()) == 1
    with pytest.raises(ValueError):
        s.add([0, 0, 0])  # too short


def test_store_delete_returns_true_on_hit_false_on_miss():
    s = wp.WaypointStore()
    s.add([0.1] * 7)
    s.add([0.2] * 7)
    assert s.delete(0) is True
    assert s.list() == [[0.2] * 7]
    assert s.delete(99) is False


def test_store_yaml_roundtrip(tmp_path):
    s = wp.WaypointStore()
    s.add([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    s.add([-0.1] * 7)
    p = tmp_path / "handeye_waypoints.yaml"
    s.save_yaml(p, recorded_for_robot="tinker2")
    s2 = wp.WaypointStore()
    n = s2.load_yaml(p)
    assert n == 2
    assert s2.list() == s.list()
    # schema version + robot name persisted in the file
    on_disk = yaml.safe_load(p.read_text())
    assert on_disk["schema_version"] == wp.YAML_SCHEMA_VERSION
    assert on_disk["recorded_for_robot"] == "tinker2"


def test_resolve_waypoints_path(tmp_path):
    p = wp.resolve_waypoints_path("tinker2", tmp_path)
    assert p == tmp_path / "src/tinker_robot_config/robots/tinker2/handeye_waypoints.yaml"
    assert wp.resolve_waypoints_path(None, tmp_path) is None
    assert wp.resolve_waypoints_path("", tmp_path) is None
```

In `test/test_web_app.py`, append:
```python
def test_waypoint_add_returns_no_current_joints_in_test_env():
    """Test env has no JointState → add must degrade gracefully."""
    node, c = _client()
    try:
        r = c.post("/api/waypoints", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False and "current joints" in body["reason"].lower()
    finally:
        node.destroy_node()


def test_waypoint_delete_out_of_range():
    node, c = _client()
    try:
        r = c.delete("/api/waypoints/99")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is False
        assert body["count"] == 0
    finally:
        node.destroy_node()


def test_waypoint_save_refuses_without_robot_name(monkeypatch):
    monkeypatch.delenv("ROBOT_NAME", raising=False)
    node, c = _client()
    try:
        r = c.post("/api/waypoints/save", json={})
        body = r.json()
        assert body["ok"] is False and "ROBOT_NAME" in body["reason"]
    finally:
        node.destroy_node()


def test_state_payload_includes_waypoints():
    node, c = _client()
    try:
        r = c.get("/api/state")
        assert r.status_code == 200
        body = r.json()
        assert "waypoints" in body
        assert body["waypoints"] == []  # empty in test env
    finally:
        node.destroy_node()
```

Run — expected FAIL.

- [ ] **Step 2: Implement `waypoints.py`**
- `WaypointStore` as above. `load_yaml` accepts the schema_version 1 layout; raises `ValueError` on schema mismatch. `save_yaml` writes via `apply_handeye.write_with_backup` (existing import).
- `resolve_waypoints_path` is a 3-line function mirroring `resolve_robot_xacro_path`.

- [ ] **Step 3: Implement node-side wiring in `handeye_web.py`**
- Inside `_make_node_class`, `import` `from handeye_calib import waypoints as hwp` and `from handeye_calib.waypoints import WaypointStore`.
- In `__init__`: `self.waypoint_store = WaypointStore()`. After all params are declared, attempt `self.do_reload_waypoints()` (silently log + skip on failure).
- Add `do_add_waypoint`, `do_delete_waypoint`, `do_save_waypoints`, `do_reload_waypoints` methods. Each holds `self.lock` while touching `self.waypoint_store`.
- Extend `get_state_dict` to compute `waypoints = [ws.waypoint_metadata(i, w) for i, w in enumerate(self.waypoint_store.list())]`, passed through `enriched_state_payload(waypoints=...)`. **Don't forget to add the `waypoints` kwarg to `enriched_state_payload` in `web_support.py`** with default `[]`.
- Wire 5 routes in `make_app` per the interfaces section.

- [ ] **Step 4: Run tests + smoke + commit**
```bash
cd /home/tinker/tk25_ws
tkbuild tk26_vision --packages-select handeye_calib
PYTHONPATH="src/tk26_vision/src/handeye_calib:$PYTHONPATH" src/tk26_vision/.venv-vision-main/bin/python \
  -m pytest src/tk26_vision/src/handeye_calib/test/ -q
ros2 launch handeye_calib handeye_web.launch.py port:=8801 &
sleep 4
curl -fsS http://127.0.0.1:8801/api/state | python3 -c "import sys,json; print('waypoints' in json.load(sys.stdin))"
curl -fsS http://127.0.0.1:8801/api/waypoints
curl -fsS -X POST http://127.0.0.1:8801/api/waypoints/save -d '{}' -H 'Content-Type: application/json'
kill %1; wait 2>/dev/null
```
Commit:
```bash
cd src/tk26_vision
git add src/handeye_calib/handeye_calib/waypoints.py \
        src/handeye_calib/handeye_calib/web_support.py \
        src/handeye_calib/handeye_calib/handeye_web.py \
        src/handeye_calib/test/test_waypoints.py \
        src/handeye_calib/test/test_web_app.py
git commit -m "feat(handeye_calib): waypoint CRUD + per-robot YAML persistence

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2 — Waypoint UI in the Capture tab (list + add-current + delete + save/reload)

**Files:**
- Modify: `webui/index.html` (waypoints sub-panel inside `[data-panel="capture"]`, ABOVE the existing stability badge + capture button + gallery).
- Modify: `webui/style.css` (`.waypoints-panel`, `.waypoints-list`, `.waypoint-row`, `.waypoint-joints`, `.waypoint-actions`).
- Modify: `webui/app.js` (renderer reads `state.waypoints`; add-current / delete / save / reload click handlers).

**No backend changes** — T1 already shipped them.

**Interfaces produced:**
- DOM IDs (the only ones T4 will reach for):
  - `#waypoints-list` — the `<ul>` rendered from `state.waypoints`.
  - `#waypoint-add-current-btn`, `#waypoint-save-btn`, `#waypoint-reload-btn`.
  - `#waypoints-status` (status line for save / reload feedback).
- JS function `renderWaypointsList()` — driven by every WS state push.

- [ ] **Step 1: HTML structure**

Insert at the top of `[data-panel="capture"]`:
```html
<div class="waypoints-panel">
  <h3>Capture waypoints</h3>
  <p class="muted">
    Record a sequence of arm poses, save them to disk (per-robot), and the
    auto-capture run-button below will cycle through them. Each move goes
    through the same SafetyEnvelope + StabilityTracker the manual Capture
    button does — settle failures skip a step but don't abort the run.
  </p>
  <div class="row">
    <button id="waypoint-add-current-btn" type="button">+ Add current joints</button>
    <button id="waypoint-save-btn" type="button">Save to disk</button>
    <button id="waypoint-reload-btn" type="button">Reload from disk</button>
  </div>
  <div class="status-line" id="waypoints-status"></div>
  <ul class="waypoints-list" id="waypoints-list">
    <li class="waypoints-empty">no waypoints recorded yet</li>
  </ul>
</div>
```

- [ ] **Step 2: JS handlers**

```js
const WP_LIST     = $("#waypoints-list");
const WP_ADD_BTN  = $("#waypoint-add-current-btn");
const WP_SAVE_BTN = $("#waypoint-save-btn");
const WP_REL_BTN  = $("#waypoint-reload-btn");

if (WP_ADD_BTN) WP_ADD_BTN.addEventListener("click", async () => {
  setStatus("waypoints-status", "adding…", "warn");
  const r = await fetch("/api/waypoints", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
  const body = await r.json();
  setStatus("waypoints-status",
    body.ok ? `added — ${body.count} waypoint(s)` : `add failed: ${body.reason}`,
    body.ok ? "ok" : "err");
});

if (WP_SAVE_BTN) WP_SAVE_BTN.addEventListener("click", async () => {
  if (!confirm("Save the current waypoint sequence to disk?\n(Existing per-robot waypoints YAML will be backed up first.)")) return;
  setStatus("waypoints-status", "saving…", "warn");
  const r = await fetch("/api/waypoints/save", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
  const body = await r.json();
  setStatus("waypoints-status",
    body.ok ? `saved to ${body.path}` : `save failed: ${body.reason}`,
    body.ok ? "ok" : "err");
});

if (WP_REL_BTN) WP_REL_BTN.addEventListener("click", async () => {
  setStatus("waypoints-status", "reloading…", "warn");
  const r = await fetch("/api/waypoints/reload", {method: "POST", headers: {"Content-Type": "application/json"}, body: "{}"});
  const body = await r.json();
  setStatus("waypoints-status",
    body.ok ? `loaded ${body.count} waypoint(s) from ${body.path}` : `reload failed: ${body.reason}`,
    body.ok ? "ok" : "err");
});

function renderWaypointsList() {
  if (!WP_LIST) return;
  const wps = (state && Array.isArray(state.waypoints)) ? state.waypoints : [];
  if (wps.length === 0) {
    WP_LIST.innerHTML = '<li class="waypoints-empty">no waypoints recorded yet</li>';
    return;
  }
  WP_LIST.innerHTML = wps.map(w =>
    `<li class="waypoint-row" data-idx="${w.idx}">
       <span class="waypoint-idx">#${w.idx}</span>
       <span class="waypoint-joints" title="${w.joints_rad.map(j => j.toFixed(4)).join(', ')} rad">${w.abbrev}</span>
       <span class="waypoint-actions">
         <button data-act="load" type="button">Load</button>
         <button data-act="delete" type="button">Delete</button>
       </span>
     </li>`
  ).join("");
}

// Delegate clicks for per-row buttons (Load fills the move-tab joint inputs;
// Delete fires DELETE /api/waypoints/{idx}).
if (WP_LIST) {
  WP_LIST.addEventListener("click", async (ev) => {
    const btn = ev.target.closest("button[data-act]");
    if (!btn) return;
    const row = btn.closest("li[data-idx]");
    const idx = parseInt(row.dataset.idx, 10);
    if (btn.dataset.act === "load") {
      const wp = state.waypoints.find(w => w.idx === idx);
      if (wp) {
        writeMoveJoints(wp.joints_rad);
        setStatus("waypoints-status", `loaded #${idx} into Move tab`, "");
      }
    } else if (btn.dataset.act === "delete") {
      if (!confirm(`Delete waypoint #${idx}?`)) return;
      const r = await fetch(`/api/waypoints/${idx}`, {method: "DELETE"});
      const body = await r.json();
      setStatus("waypoints-status",
        body.ok ? `deleted #${idx} — ${body.count} remaining` : `delete failed: ${body.reason}`,
        body.ok ? "ok" : "err");
    }
  });
}
```

Wire `renderWaypointsList()` into the main `render()` so it updates on every WS push.

- [ ] **Step 3: CSS**

```css
.waypoints-panel { margin-bottom: 20px; }
.waypoints-list {
  list-style: none; padding: 0; margin: 6px 0 0;
  border: 1px solid var(--border); border-radius: 4px;
  max-height: 220px; overflow-y: auto;
}
.waypoint-row {
  display: grid;
  grid-template-columns: 40px 1fr auto;
  gap: 8px; align-items: center;
  padding: 4px 8px;
  border-bottom: 1px solid var(--border);
  font-family: var(--mono); font-size: 12px;
}
.waypoint-row:last-child { border-bottom: none; }
.waypoint-idx { color: var(--accent); font-weight: 600; }
.waypoint-joints { color: var(--fg-muted); }
.waypoint-actions button { padding: 2px 8px; font-size: 11px; }
.waypoints-empty {
  padding: 12px;
  color: var(--fg-muted);
  font-style: italic; text-align: center;
}
```

- [ ] **Step 4: Smoke + commit**
```bash
tkbuild tk26_vision --packages-select handeye_calib
ros2 launch handeye_calib handeye_web.launch.py port:=8802 &
sleep 4
curl -fsS http://127.0.0.1:8802/ | grep -E 'waypoints-panel|waypoints-list' | head
kill %1; wait 2>/dev/null
```
Commit:
```bash
git add src/handeye_calib/handeye_calib/webui/
git commit -m "feat(handeye_calib): waypoint UI in Capture tab (list + add/delete/save/reload)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3 — Auto-capture state machine backend (CaptureSequenceRunner)

**Files:**
- Modify: `src/tk26_vision/src/handeye_calib/handeye_calib/handeye_web.py` (new `CaptureSequenceRunner` class inside `_make_node_class`; sequence start/cancel endpoints; `state.sequence` field).
- Modify: `src/tk26_vision/src/handeye_calib/test/test_web_node.py` (state-machine tests with mocked JointMove + stability).
- Modify: `src/tk26_vision/src/handeye_calib/test/test_web_app.py` (sequence endpoint shape tests).

**This is the heart of the feature — dispatch on Opus.**

**Interfaces produced:**
- `CaptureSequenceRunner` (inner class, instantiated lazily on first `start`):
  - `__init__(self, node)` — owns the daemon thread + stop event + state dict + log deque.
  - `.start(dry_run: bool = False, settle_timeout_s: float = 5.0) -> {ok, reason?}` — returns immediately; thread runs the loop. Refuses if `state.sequence.running` is True.
  - `.cancel() -> {ok}` — sets stop event; in-flight goal handle is cancelled. Idempotent.
  - `.state_dict() -> dict` with keys: `running, dry_run, current_idx (int|None), total, current_step ("idle"|"moving"|"settling"|"capturing"|"done"|"cancelled"|"error"), log (list[str], last 20)`.
- `HandeyeWebNode.sequence_runner: CaptureSequenceRunner | None` (set in `__init__`).
- `HandeyeWebNode.do_start_sequence(dry_run: bool) -> {ok, reason?}` — refuses on empty waypoints; otherwise delegates to `runner.start(...)`.
- `HandeyeWebNode.do_cancel_sequence() -> {ok}`.
- WS state extension: `state.sequence = sequence_runner.state_dict()`.
- New HTTP routes:
  - `POST /api/sequence/start` body `{dry_run: bool = false}` → `do_start_sequence(...)`.
  - `POST /api/sequence/cancel` → `do_cancel_sequence()`.

**State machine flow (one iteration of the loop):**

```
for idx, waypoint in enumerate(self.node.waypoint_store.list()):
    if self._stop.is_set(): break
    self._set_state(current_idx=idx, current_step="moving")
    self._log(f"#{idx}: moving to {waypoint[:3]} …")
    goal_handle = self.node._jm.send_goal_async(JointMove.Goal(joint0=..., add_octomap=True))
    # await goal acceptance (poll with stop-event check)
    if self._stop.is_set(): self._cancel_inflight(goal_handle); break
    # await result (poll with stop-event check) — or short timeout (10 s)
    result = self._await_result(goal_handle, deadline=10.0)
    if self._stop.is_set(): break
    if not result.ok:
        self._log(f"#{idx}: move failed ({result.reason}); skipping")
        continue
    self._set_state(current_step="settling")
    settled = self._wait_for_settle(settle_timeout_s)
    if self._stop.is_set(): break
    if not settled:
        self._log(f"#{idx}: settle timeout after {settle_timeout_s}s; skipping")
        continue
    if self.dry_run:
        self._log(f"#{idx}: dry-run — settled but skipping capture")
        continue
    self._set_state(current_step="capturing")
    cap = self.node.do_capture()
    self._log(f"#{idx}: " + ("captured" if cap["ok"] else f"capture skipped ({cap['reason']})"))
self._set_state(running=False, current_step="cancelled" if self._stop.is_set() else "done")
```

**Implementation notes:**
- Daemon thread holds a reference to the node; uses `node.lock` only when reading shared state (joints, samples). The state dict + log are owned by the runner itself (its own threading.Lock).
- `_await_result` is a polling loop on `rclpy.spin_until_future_complete` with a short timeout (50 ms) inside an outer `while time < deadline and not self._stop.is_set()` loop. Avoids a blocking wait that misses the cancel signal.
- `_wait_for_settle` polls `state.stability.steady` at 10 Hz until `True` for 3 consecutive ticks OR timeout fires. (Reuses the existing T4 stability machinery — don't re-tune thresholds.)
- The log is a `collections.deque(maxlen=20)` of strings.
- On `cancel`, the runner attempts to call `goal_handle.cancel_goal_async()` on the in-flight JointMove (best-effort — the arm may already be at the waypoint).
- The runner is recreated per `start()` call (`self.node.sequence_runner = CaptureSequenceRunner(self.node); .start(...)`) so state from a prior run doesn't leak.

- [ ] **Step 1: Failing tests**

In `test/test_web_node.py` append:
```python
def test_sequence_refuses_when_empty_waypoints():
    node = HandeyeWebNode()
    try:
        r = node.do_start_sequence(dry_run=False)
        assert r["ok"] is False and "no waypoints" in r["reason"].lower()
    finally:
        node.destroy_node()


def test_sequence_state_dict_shape():
    node = HandeyeWebNode()
    try:
        s = node.sequence_runner.state_dict() if node.sequence_runner else {
            "running": False, "dry_run": False, "current_idx": None,
            "total": 0, "current_step": "idle", "log": []}
        for k in ("running", "dry_run", "current_idx", "total", "current_step", "log"):
            assert k in s
    finally:
        node.destroy_node()


def test_sequence_cancel_when_not_running_is_noop():
    node = HandeyeWebNode()
    try:
        r = node.do_cancel_sequence()
        assert r["ok"] is True
    finally:
        node.destroy_node()
```

In `test/test_web_app.py` append:
```python
def test_sequence_start_refuses_empty():
    node, c = _client()
    try:
        r = c.post("/api/sequence/start", json={"dry_run": False})
        body = r.json()
        assert body["ok"] is False
    finally:
        node.destroy_node()


def test_state_payload_includes_sequence():
    node, c = _client()
    try:
        r = c.get("/api/state")
        body = r.json()
        assert "sequence" in body
        assert body["sequence"]["running"] is False
        assert body["sequence"]["current_step"] == "idle"
    finally:
        node.destroy_node()
```

- [ ] **Step 2: Implement `CaptureSequenceRunner`**

Inside `_make_node_class`, declare the runner class with the state machine flow above. Key snippets:

```python
import threading, collections, time

class CaptureSequenceRunner:
    def __init__(self, node):
        self.node = node
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._state = {
            "running": False, "dry_run": False,
            "current_idx": None, "total": 0,
            "current_step": "idle",
        }
        self._log = collections.deque(maxlen=20)

    def state_dict(self):
        with self._lock:
            return {**self._state, "log": list(self._log)}

    def start(self, dry_run=False, settle_timeout_s=5.0):
        with self._lock:
            if self._state["running"]:
                return {"ok": False, "reason": "sequence already running"}
            wps = self.node.waypoint_store.list()
            if not wps:
                return {"ok": False, "reason": "no waypoints recorded"}
            self._stop.clear()
            self._state.update({
                "running": True, "dry_run": dry_run,
                "current_idx": None, "total": len(wps),
                "current_step": "starting",
            })
            self._log.clear()
            self._append_log_locked(f"starting sequence ({len(wps)} waypoints, dry_run={dry_run})")
        self._thread = threading.Thread(target=self._run, args=(wps, dry_run, settle_timeout_s),
                                          daemon=True, name="capture-sequence")
        self._thread.start()
        return {"ok": True}

    def cancel(self):
        self._stop.set()
        self._append_log("cancel requested")
        return {"ok": True}

    # ... internal: _run, _await_result, _wait_for_settle, _set_state,
    # _append_log, _append_log_locked.
```

The full implementation should be ~150 lines including the move/settle/capture inner methods. Write it carefully — this is the safety-critical surface. Reference `pan_tilt/pan_tilt/calib_web.py`'s subprocess-runner threading patterns for inspiration (kill-signal handling, deadline polling).

- [ ] **Step 3: Wire endpoints**
```python
@app.post("/api/sequence/start")
async def sequence_start(request: Request):
    body = await request.json()
    return JSONResponse(node.do_start_sequence(dry_run=bool(body.get("dry_run", False))))

@app.post("/api/sequence/cancel")
def sequence_cancel():
    return JSONResponse(node.do_cancel_sequence())
```

And `get_state_dict` calls `sequence = self.sequence_runner.state_dict() if self.sequence_runner else <idle default>` and passes via `enriched_state_payload(sequence=...)` (add the kwarg to `web_support.enriched_state_payload`).

- [ ] **Step 4: Smoke + commit**
```bash
tkbuild tk26_vision --packages-select handeye_calib
PYTHONPATH=... pytest src/tk26_vision/src/handeye_calib/test/ -q
ros2 launch handeye_calib handeye_web.launch.py port:=8803 &
sleep 4
curl -fsS http://127.0.0.1:8803/api/state | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['sequence'])"
curl -fsS -X POST http://127.0.0.1:8803/api/sequence/start -d '{"dry_run":false}' -H 'Content-Type: application/json'
kill %1; wait 2>/dev/null
```
Expected: state.sequence shows `current_step: "idle", running: false`; start returns `{ok:false, reason:"no waypoints..."}`.

Commit:
```bash
git add src/handeye_calib/handeye_calib/handeye_web.py \
        src/handeye_calib/handeye_calib/web_support.py \
        src/handeye_calib/test/test_web_node.py \
        src/handeye_calib/test/test_web_app.py
git commit -m "feat(handeye_calib): CaptureSequenceRunner state machine + sequence endpoints

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4 — Auto-capture UI (run / dry-run / cancel + live progress)

**Files:**
- Modify: `webui/index.html` (sequence controls below the waypoints list, above the manual capture button).
- Modify: `webui/style.css` (`.sequence-controls`, `.sequence-progress`, `.sequence-log`).
- Modify: `webui/app.js` (run/cancel/dry handlers; `renderSequenceUI()` reads `state.sequence`).

**Interfaces produced:**
- DOM IDs: `#sequence-run-btn`, `#sequence-dry-btn`, `#sequence-cancel-btn`, `#sequence-status`, `#sequence-progress` (text line), `#sequence-log` (scrollable `<ul>`).

- [ ] **Step 1: HTML**

Below the waypoints-panel (still inside `[data-panel="capture"]`):
```html
<div class="sequence-controls">
  <h3>Auto-capture sequence</h3>
  <div class="row">
    <button class="primary" id="sequence-run-btn" type="button" disabled>Run sequence</button>
    <button id="sequence-dry-btn" type="button" disabled>Run dry (move + settle only)</button>
    <button id="sequence-cancel-btn" type="button" disabled>Cancel</button>
  </div>
  <div class="status-line" id="sequence-progress">idle</div>
  <div class="status-line" id="sequence-status"></div>
  <ul class="sequence-log" id="sequence-log"></ul>
</div>
```

- [ ] **Step 2: JS handlers + renderer**

```js
const SEQ_RUN    = $("#sequence-run-btn");
const SEQ_DRY    = $("#sequence-dry-btn");
const SEQ_CANCEL = $("#sequence-cancel-btn");
const SEQ_PROG   = $("#sequence-progress");
const SEQ_LOG    = $("#sequence-log");

async function startSequence(dryRun) {
  const total = (state.waypoints || []).length;
  const verb = dryRun ? "dry-run (move + settle only)" : "RUN CAPTURE";
  if (!confirm(`${verb} the ${total}-waypoint sequence?\nThe arm will move to each pose in order. Click Cancel to stop.`)) return;
  setStatus("sequence-status", "starting…", "warn");
  const r = await fetch("/api/sequence/start", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({dry_run: dryRun})});
  const body = await r.json();
  if (!body.ok) setStatus("sequence-status", `start failed: ${body.reason}`, "err");
}

if (SEQ_RUN)    SEQ_RUN.addEventListener("click", () => startSequence(false));
if (SEQ_DRY)    SEQ_DRY.addEventListener("click", () => startSequence(true));
if (SEQ_CANCEL) SEQ_CANCEL.addEventListener("click", async () => {
  setStatus("sequence-status", "cancelling…", "warn");
  await fetch("/api/sequence/cancel", {method: "POST"});
});

function renderSequenceUI() {
  if (!SEQ_RUN) return;
  const seq = (state && state.sequence) || {running: false, current_step: "idle", total: 0, log: []};
  const wps = (state && state.waypoints) || [];
  const canRun = wps.length > 0 && !seq.running;
  SEQ_RUN.disabled    = !canRun;
  SEQ_DRY.disabled    = !canRun;
  SEQ_CANCEL.disabled = !seq.running;
  if (seq.running) {
    SEQ_PROG.textContent = `${seq.current_step} — #${seq.current_idx ?? "?"} / ${seq.total}`;
    SEQ_PROG.className = "status-line warn";
  } else {
    SEQ_PROG.textContent = seq.current_step === "done" ? "done" :
                           seq.current_step === "cancelled" ? "cancelled" : "idle";
    SEQ_PROG.className = "status-line " + (seq.current_step === "done" ? "ok" :
                                             seq.current_step === "cancelled" ? "warn" : "");
  }
  SEQ_LOG.innerHTML = seq.log.map(line => `<li>${line.replace(/</g, "&lt;")}</li>`).join("");
}
```

Wire `renderSequenceUI()` into the main `render()`.

- [ ] **Step 3: CSS**

```css
.sequence-controls { margin-top: 16px; }
.sequence-log {
  list-style: none; padding: 6px 10px; margin: 6px 0 0;
  background: var(--bg);
  border: 1px solid var(--border); border-radius: 3px;
  font-family: var(--mono); font-size: 11px;
  color: var(--fg-muted);
  max-height: 180px; overflow-y: auto;
}
.sequence-log li { padding: 2px 0; border-bottom: 1px solid var(--border); }
.sequence-log li:last-child { border-bottom: none; }
```

- [ ] **Step 4: Smoke + commit**
```bash
tkbuild tk26_vision --packages-select handeye_calib
ros2 launch handeye_calib handeye_web.launch.py port:=8804 &
sleep 4
curl -fsS http://127.0.0.1:8804/ | grep -E 'sequence-(run|dry|cancel)-btn' | head
kill %1; wait 2>/dev/null
```
Commit:
```bash
git add src/handeye_calib/handeye_calib/webui/
git commit -m "feat(handeye_calib): auto-capture sequence UI (run / dry / cancel + progress + log)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5 — README + changelog + final HW-ready smoke

**Files:**
- Modify: `src/handeye_calib/README.md` (new `## Waypoints + auto-capture` section, `0.5.0 (2026-06-20)` changelog entry).
- This task adds no code beyond the README; the live smoke covers the whole feature.

**README additions:**

```markdown
## Waypoints + auto-capture

The Capture tab supports an ordered list of arm waypoints and a one-click
auto-capture sweep:

1. Move the arm to a pose you want (xArm teach mode, or the Move tab's
   joint editor). Click **+ Add current joints** to append the live
   `xArm joints` to the waypoint list.
2. Repeat until you have 12-20 diverse poses (the diversity meter helps —
   you want > 30° of rotation spread).
3. Click **Save to disk** — the list persists to
   `src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/handeye_waypoints.yaml`
   (atomic write with timestamped backup). Re-runs of the web server
   reload the file on startup; you only record the sequence once per
   robot.
4. Click **Run dry** first to verify every waypoint is reachable
   (move + settle, NO capture). The arm sweeps through the list; any
   waypoint that times out on settle is logged and skipped. Cancel
   stops the run after the current step.
5. When the dry-run looks clean, click **Run sequence** for the real
   thing. Each settled pose feeds the manual capture path — same
   StabilityTracker gate, same SafetyEnvelope check, same sample
   gallery population. The Solve tab is reachable as soon as ≥ 6
   diverse samples are in the gallery.

The sequence runner is single-instance — `Run sequence` is disabled
while another run is in flight. **Cancel** is cooperative + immediate:
the in-flight JointMove goal is cancelled and the runner exits at the
next state transition (typically within 50 ms). Operator safety is on
the arm driver + SafetyEnvelope, not on this UI.
```

**Changelog (top of existing list):**

```markdown
### 0.5.0 (2026-06-20)
- **Waypoint authoring + auto-capture sequence.** New backend module
  `waypoints.py` with `WaypointStore` + per-robot YAML persistence
  (`src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/handeye_waypoints.yaml`).
  Auto-loaded on startup; refuses save when `ROBOT_NAME` unset.
- Capture tab gained a waypoints sub-panel: add-current / per-row load+delete /
  save+reload buttons, and a sequence controls strip below it:
  **Run sequence**, **Run dry**, **Cancel**, with a live progress line and
  bounded log.
- New `CaptureSequenceRunner` daemon-thread state machine drives
  move → settle → capture per waypoint. Reuses the existing JointMove
  client + StabilityTracker + `do_capture` path — no fast path bypassing
  the SafetyEnvelope. Cancel is cooperative + immediate (in-flight goal
  cancelled, runner exits at next transition).
- New endpoints: `GET/POST /api/waypoints`, `DELETE /api/waypoints/{idx}`,
  `POST /api/waypoints/save`, `POST /api/waypoints/reload`,
  `POST /api/sequence/start` (body `{dry_run: bool}`),
  `POST /api/sequence/cancel`. WS state extended with `state.waypoints`
  + `state.sequence`.
```

- [ ] **Step 1: Hardware-ready smoke (operator does this)**

The smoke for this task is hardware-in-the-loop, so the implementer's job is just to verify the no-hardware path:

```bash
tkbuild tk26_vision --packages-select handeye_calib
PYTHONPATH=... pytest src/tk26_vision/src/handeye_calib/test/ -q
ROBOT_NAME=tinker2 ros2 launch handeye_calib handeye_web.launch.py port:=8805 &
sleep 4
curl -fsS http://127.0.0.1:8805/api/state | python3 -c "import sys,json; d=json.load(sys.stdin); print('waypoints' in d, 'sequence' in d, d['sequence'])"
curl -fsS -X POST http://127.0.0.1:8805/api/sequence/start -d '{"dry_run":true}' -H 'Content-Type: application/json'
kill %1; wait 2>/dev/null
```
Expected:
- `True True {'running': False, 'dry_run': False, 'current_idx': None, 'total': 0, 'current_step': 'idle', 'log': []}`
- start returns `{ok:false, reason:"no waypoints recorded"}`

Operator-in-the-loop smoke (not part of this PR; record in `DEV_NOTES.md` after):
- Manual: record 3 waypoints via the UI → Save → Reload (verify count survives) → Run dry (arm sweeps to all 3, no capture) → Run sequence (each successful pose appends to gallery; Cancel after the second pose verifies clean stop).

- [ ] **Step 2: Commit**
```bash
git add src/handeye_calib/README.md
git commit -m "docs(handeye_calib): README + 0.5.0 changelog — waypoints + auto-capture

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-review

**Spec coverage:**

| User requirement | Closed by |
|---|---|
| Record a *sequence* of joint sets | T1 (WaypointStore) + T2 (UI) |
| Persist across restarts | T1 (per-robot YAML, auto-load on startup) |
| Arm cycles through them automatically | T3 (CaptureSequenceRunner state machine) |
| Settle between moves, capture each pose | T3 (reuses T4 StabilityTracker + do_capture) |
| Dry-run mode | T3 + T4 (boolean flag + button) |
| Cancel mid-run | T3 (cooperative stop event + in-flight goal cancel) + T4 (button) |
| Live progress | T4 (state.sequence in WS push, progress line + log) |
| Per-robot config (matches T6 convention) | T1 (`resolve_waypoints_path`) |
| Same safety gates as manual capture | T3 (no fast path; reuses do_move → SafetyEnvelope, do_capture → settle gate) |

**Intentionally NOT included (call out so the implementer doesn't backfill):**
- Reorder via drag-drop. v2 follow-up if requested.
- Multiple named sequences (e.g. "near board" vs "far board"). v2 follow-up.
- WebSocket log fan-out at higher granularity than 5 Hz state push. The bounded log + per-waypoint progress is enough for operator feedback at this cadence; a separate WS stream is over-engineering.
- Pan_tilt-style "phases" (Phase 1 / Phase 2 sweep / sanity). Handeye's solver doesn't care about phase semantics; one ordered list is enough.
- Subprocess + log capture (pan_tilt uses this because its solver is a CLI). Handeye solves in-process; the sequence runner is in-process too.
- A "Test reach" button per waypoint (move to that single pose without running the sequence). The existing per-row "Load" button feeds the Move tab's joint editor; the operator can then click Move (joints) to test reach — same safety path, no new code.

**Risks the implementer should be careful about:**
1. **Race on `state.sequence` between WS push thread and runner thread.** The runner mutates `_state` under its own `_lock`; the WS push reads via `state_dict()` which copies under the same lock. Don't share the dict reference.
2. **Stop event vs in-flight goal.** Both must fire on cancel. If only the stop event fires, the arm may still complete the current goal and end up at a stale waypoint. If only the goal cancels, the runner loops on the next waypoint immediately. Both, in order: cancel goal first, then set stop event.
3. **rclpy executor context.** `send_goal_async` returns a future that resolves on the rclpy spin thread. The runner thread polling on the future without spinning will hang forever. Use `rclpy.spin_until_future_complete(node, future, timeout_sec=0.05)` in a loop, OR rely on the existing MultiThreadedExecutor in `main()` — confirm which by reading the current `main()`. If the executor is single-threaded, this is a known gotcha; switch to MultiThreadedExecutor for the sequence path.
4. **JointMove field names.** `tinker_arm_msgs/action/JointMove` has `joint0..joint6` (NOT `joints: [...]`) plus `add_octomap: bool`. Construct the goal accordingly — the existing `do_move()` code in T3 of the v2 plan already does this; copy that pattern.
