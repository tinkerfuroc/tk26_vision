# track_web follow-pipeline integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the track_web dashboard a working bring-up surface for the current follow pipeline (nav executive + rewired BT) with a preserved vision+audio-only mode.

**Architecture:** A BT `--no-nav` flag builds the tree with/without the follow child; `follow_server` publishes a `std_msgs/String` JSON status; the dashboard's `ProcessManager` gains a fixed `_GROUPS` layer and the webui gets a mode selector + Start/Stop + a live Follow-state panel.

**Tech Stack:** ROS2 Humble (rclpy), py_trees, FastAPI + TestClient, vanilla JS webui.

**Spec:** `src/tk26_vision/docs/superpowers/specs/2026-06-13-web-ui-follow-integration-design.md`

**Cross-repo commit discipline:** one git commit per Task, staged with explicit pathspecs only (NEVER `git add -A` / `git add .` — unrelated parallel work is committed in these repos constantly). Build with `tkbuild <sub-ws>` per memory, not bare colcon. Each Task lives in exactly one repo.

---

### Task 1: BT `--no-nav` mode (tk25_decision)

**Repo:** `src/tk25_decision` · **Files:**
- Modify: `src/behavior_tree/behavior_tree/FollowPerson/follow_person.py`
- Modify: `src/behavior_tree/behavior_tree/FollowPerson/cli.py`
- Test: `src/behavior_tree/test/test_follow_tree_build.py`
- Modify: `src/behavior_tree/README.md` (changelog)

- [ ] **Step 1: Read the existing build test** to match its construction/mock setup.

Run: `sed -n '1,80p' src/behavior_tree/test/test_follow_tree_build.py`
Note how it imports `create_follow_person_tree` and whether it needs any mock-mode env. Mirror that setup in the new tests below.

- [ ] **Step 2: Write the failing tests** — append to `test/test_follow_tree_build.py`:

```python
def test_tree_with_nav_includes_follow_child():
    root = create_follow_person_tree(enable_navigation=True)
    names = [c.name for c in root.children]
    assert "Follow Navigation" in names
    assert len(root.children) == 3   # Track Person, Follow Navigation, Follow Reactions


def test_tree_no_nav_omits_follow_child():
    root = create_follow_person_tree(enable_navigation=False)
    names = [c.name for c in root.children]
    assert "Follow Navigation" not in names
    assert "Track Person" in names           # tracking still present
    assert len(root.children) == 2           # Track Person, Follow Reactions
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cd src/tk25_decision && python3 -m pytest src/behavior_tree/test/test_follow_tree_build.py -k "no_nav or with_nav" -v`
Expected: FAIL — `create_follow_person_tree() got an unexpected keyword argument 'enable_navigation'`.

- [ ] **Step 4: Add the `enable_navigation` branch** in `follow_person.py`. Replace the `create_follow_person_tree` signature + body construction:

```python
def create_follow_person_tree(
    target_frame: str = "",
    enable_navigation: bool = True,
) -> py_trees.behaviour.Behaviour:
    """Build and return the follow-person tree root.

    Args:
        target_frame: TF frame for the tracked position output. Defaults to ``""``
            (camera frame, no per-frame TF lookup) — see the long note below.
        enable_navigation: When True (default) the tree includes the
            ``BtNode_FollowAction`` child that drives ``follow_server`` (full
            pipeline). When False the tree is built WITHOUT that child
            (``Parallel[track, reactions]``) — the vision+audio-only mode: the
            tracker stays alive and the reacq announcer fires, but the robot base
            never moves (no Follow goal is dispatched). Nothing reads ``follow/*``,
            so the no-nav tree is self-consistent.
    """
    track = BtNode_TrackPersonAction(
        name="Track Person",
        target_frame=target_frame,
    )
    announce = BtNode_ReacqAnnounce(name="Reacq Announce")
    reactions = py_trees.composites.Sequence(
        name="Follow Reactions",
        memory=False,
        children=[announce],
    )

    children = [track]
    if enable_navigation:
        follow = BtNode_FollowAction(
            name="Follow Navigation",
            use_breadcrumbs=True,
            timeout=0.0,
        )
        children.append(follow)
    children.append(reactions)

    root = py_trees.composites.Parallel(
        name="Follow Person",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False),
        children=children,
    )
    return root
```

(Keep the existing long `target_frame` docstring paragraph; only the signature, the `Args` block, and the children-list construction change. The `BtNode_FollowAction` import stays at module top — it is only *instantiated* conditionally.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd src/tk25_decision && python3 -m pytest src/behavior_tree/test/test_follow_tree_build.py -v`
Expected: PASS (all, including pre-existing cases).

- [ ] **Step 6: Add the `--no-nav` CLI flag** — replace `cli.py:main()`:

```python
def main():
    """Run the follow-person behaviour tree until interrupted.

    ``--no-nav`` builds the vision+audio-only tree (no follow-navigation child,
    no base motion). ``parse_known_args`` ignores ``--ros-args`` so the script
    still works under ``ros2 run behavior_tree follow-person [--no-nav]``.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="follow-person")
    parser.add_argument(
        "--no-nav", action="store_true",
        help="vision+audio only: omit the follow-navigation child (no base motion)",
    )
    args, _ = parser.parse_known_args()
    enable_navigation = not args.no_nav

    from behavior_tree.runtime import run_tree
    from behavior_tree.FollowPerson.follow_person import create_follow_person_tree

    run_tree(
        lambda: create_follow_person_tree(enable_navigation=enable_navigation),
        period_ms=200.0,
        title="Follow Person" if enable_navigation else "Follow Person (vision+audio)",
    )
```

(Move the `from behavior_tree.runtime import run_tree` import inside `main` as shown, or leave the existing module-level import and drop the duplicate — either works; do not import it twice.)

- [ ] **Step 7: Build + smoke the CLI parses the flag**

Run: `tkbuild tk25_decision --packages-select behavior_tree && source install/setup.zsh`
Run: `ros2 run behavior_tree follow-person --help`
Expected: argparse usage shows `--no-nav`. (Ctrl-C any tree that starts; we only check the flag is accepted.)

- [ ] **Step 8: Changelog + commit**

Add a `README.md` changelog line: `- follow-person BT: --no-nav flag builds the vision+audio-only tree (no follow-navigation child).`

```bash
cd src/tk25_decision
git add src/behavior_tree/behavior_tree/FollowPerson/follow_person.py \
        src/behavior_tree/behavior_tree/FollowPerson/cli.py \
        src/behavior_tree/test/test_follow_tree_build.py \
        src/behavior_tree/README.md
git commit -m "feat(behavior_tree): follow-person --no-nav vision+audio-only mode"
```

---

### Task 2: `follow_server` status publisher (tk26_navigation)

**Repo:** `src/tk26_navigation` · **Files:**
- Modify: `src/following/following/follow_server.py`
- Test: `src/following/test/test_f2_follow_server.py`
- Modify: `src/following/README.md` (changelog)

- [ ] **Step 1: Add a failing assertion to the F2 suite.** Read how a goal is driven in `test_f2_follow_server.py` (find a test that sends a Follow goal and spins). Add a subscriber to `/follow_server/status` in (or alongside) one active-goal test and assert a JSON status arrives:

```python
def test_status_topic_emits_json(f2):   # reuse the existing fixture name/shape
    import json
    from std_msgs.msg import String
    received = []
    sub = f2.client_node.create_subscription(
        String, "/follow_server/status",
        lambda m: received.append(m.data), 10)
    # drive a normal follow goal the way the other tests do (send goal, spin ~3 s)
    f2.send_follow_goal()                 # use the suite's existing helper/sequence
    f2.spin_for(3.0)
    assert received, "no /follow_server/status message during an active goal"
    doc = json.loads(received[-1])
    assert set(doc) == {"state", "distance_to_person", "reacq_state",
                        "breadcrumbs_pending", "goal_held"}
    f2.client_node.destroy_subscription(sub)
```

(Adapt `f2`, `client_node`, `send_follow_goal`, `spin_for` to the actual fixture/helpers in the file — mirror an existing test exactly; do not invent new harness.)

- [ ] **Step 2: Run it to verify it fails**

Run: `cd src/tk26_navigation && python3 -m pytest src/following/test/test_f2_follow_server.py -k status_topic -v`
Expected: FAIL — assertion "no /follow_server/status message" (topic not advertised yet).

- [ ] **Step 3: Add the imports** at the top of `follow_server.py` (after the existing `import time` / near `from std_msgs.msg import UInt8`):

```python
import json
from std_msgs.msg import String
```

- [ ] **Step 4: Create the publisher** where the other publishers/subscriptions are created (near line 142, the `reacq_state_topic` subscription). Add:

```python
# Compact JSON status mirror of the Follow feedback, for the track_web
# dashboard's Follow-state panel. std_msgs/String keeps vision_track free of a
# tinker_nav_msgs dependency (it does not subscribe the action feedback topic).
self._status_pub = self.create_publisher(String, '~/status', 10)
```

- [ ] **Step 5: Publish the status** immediately after `goal_handle.publish_feedback(fb)` (currently `follow_server.py:521`):

```python
self._status_pub.publish(String(data=json.dumps({
    "state": int(fb.state),
    "distance_to_person": float(fb.distance_to_person),
    "reacq_state": int(fb.reacq_state),
    "breadcrumbs_pending": int(fb.breadcrumbs_pending),
    "goal_held": bool(fb.goal_held),
})))
```

- [ ] **Step 6: Run the F2 test to verify it passes**

Run: `cd src/tk26_navigation && python3 -m pytest src/following/test/test_f2_follow_server.py -k status_topic -v`
Expected: PASS. Then run the full F2 file to confirm no regression: `python3 -m pytest src/following/test/test_f2_follow_server.py -v`.

- [ ] **Step 7: Changelog + commit**

`README.md` changelog: `- follow_server: publishes ~/status (std_msgs/String JSON: state/distance/reacq/breadcrumbs_pending/goal_held) for the track_web Follow-state panel.`

```bash
cd src/tk26_navigation
git add src/following/following/follow_server.py \
        src/following/test/test_f2_follow_server.py \
        src/following/README.md
git commit -m "feat(following): follow_server ~/status JSON publisher for dashboards"
```

---

### Task 3: `ProcessManager` allowlist + group layer (tk26_vision)

**Repo:** `src/tk26_vision` · **Files:**
- Modify: `src/vision_track/vision_track/process_manager.py`
- Test: `src/vision_track/test/test_process_manager.py`

- [ ] **Step 1: Write failing group tests** — append to `test_process_manager.py` (use a registry/group of harmless `true`/`sleep` commands like the existing tests do; read the file first to match its fixture style):

```python
def test_start_group_starts_all_members_in_order():
    reg = {"a": ["sleep", "5"], "b": ["sleep", "5"], "c": ["sleep", "5"]}
    groups = {"g": ["a", "b", "c"]}
    pm = ProcessManager(registry=reg, groups=groups, stagger_sec=0.0)
    res = pm.start_group("g")
    try:
        assert [r["name"] for r in res] == ["a", "b", "c"]
        assert all(r["running"] for r in res)
    finally:
        pm.shutdown_all()


def test_start_group_unknown_is_error_not_raise():
    pm = ProcessManager(registry={}, groups={}, stagger_sec=0.0)
    res = pm.start_group("nope")
    assert isinstance(res, dict) and "error" in res


def test_stop_group_stops_reverse_order():
    reg = {"a": ["sleep", "5"], "b": ["sleep", "5"]}
    groups = {"g": ["a", "b"]}
    pm = ProcessManager(registry=reg, groups=groups, stagger_sec=0.0)
    pm.start_group("g")
    res = pm.stop_group("g")
    assert [r["name"] for r in res] == ["b", "a"]
    assert all(not r["running"] for r in res)
```

- [ ] **Step 2: Run to verify failure**

Run: `cd src/tk26_vision && python3 -m pytest src/vision_track/test/test_process_manager.py -k group -v`
Expected: FAIL — `ProcessManager.__init__() got an unexpected keyword argument 'groups'`.

- [ ] **Step 3: Update the REGISTRY + add GROUPS** in `process_manager.py` (replace the `REGISTRY` block at lines 45-49):

```python
# Fixed allowlist. Keys are the only values the API will accept; the argv lists
# are never built from caller input.
REGISTRY = {
    "audio":         ["ros2", "launch", "audio_pakage", "audio.launch.py"],
    "follow_server": ["ros2", "run", "following", "follow_server",
                      "--ros-args", "-p", "working_frame:=map"],
    "bt_vision":     ["ros2", "run", "behavior_tree", "follow-person", "--no-nav"],
    "bt_nav":        ["ros2", "run", "behavior_tree", "follow-person"],
}

# Fixed group allowlist for the two follow demo modes. Members are REGISTRY keys
# started in listed order (audio + TTS up, then follow_server, then the BT).
GROUPS = {
    "follow_vision": ["audio", "bt_vision"],
    "follow_nav":    ["audio", "follow_server", "bt_nav"],
}
```

- [ ] **Step 4: Extend `__init__` + add group methods.** Change the constructor signature and add the two methods:

```python
def __init__(self, registry=REGISTRY, groups=GROUPS, stagger_sec=1.5):
    self._registry = dict(registry)
    self._groups = dict(groups)
    self._stagger_s = float(stagger_sec)
    self._procs: dict[str, subprocess.Popen] = {}
    self._last_rc: dict[str, int | None] = {}
    self._lock = threading.Lock()
    self.term_timeout_s = 5.0
```

```python
def start_group(self, group) -> list | dict:
    """Start every member of ``group`` in listed order, staggered.

    Returns the list of per-member status dicts, or an error dict for an
    unknown group. Never raises. Each member goes through ``start`` (its own
    lock acquisition), so the stagger sleep happens BETWEEN members, outside
    the lock — the dashboard stays responsive.
    """
    if group not in self._groups:
        return {"group": group, "error": f"unknown group '{group}'"}
    out = []
    members = self._groups[group]
    for i, name in enumerate(members):
        out.append(self.start(name))
        if self._stagger_s and i < len(members) - 1:
            time.sleep(self._stagger_s)
    return out

def stop_group(self, group) -> list | dict:
    """Stop every member of ``group`` in REVERSE order. Never raises."""
    if group not in self._groups:
        return {"group": group, "error": f"unknown group '{group}'"}
    return [self.stop(name) for name in reversed(self._groups[group])]
```

Update the module docstring line 17-18 ("audio, dummy nav, behavior tree") to "audio, follow_server, and the follow-person BT (vision-only or with-nav)".

- [ ] **Step 5: Run to verify pass**

Run: `cd src/tk26_vision && python3 -m pytest src/vision_track/test/test_process_manager.py -v`
Expected: PASS (all, including pre-existing).

- [ ] **Step 6: Commit** (the README/changelog for vision_track lands with Task 5, the user-facing UI change):

```bash
cd src/tk26_vision
git add src/vision_track/vision_track/process_manager.py \
        src/vision_track/test/test_process_manager.py
git commit -m "feat(vision_track): ProcessManager follow allowlist + group layer"
```

---

### Task 4: Dashboard bridge + endpoints (tk26_vision)

**Repo:** `src/tk26_vision` · **Files:**
- Modify: `src/vision_track/vision_track/track_web.py` (bridge: group methods + follow-status sub)
- Modify: `src/vision_track/vision_track/track_web_app.py` (group endpoints + payload)
- Test: `src/vision_track/test/test_track_web_app.py`

- [ ] **Step 1: Write failing app tests** — append to `test_track_web_app.py`. Extend `FakeBridge` (around line 23) with the new methods, then add endpoint tests mirroring `test_proc_start_stop_known` (line 184):

```python
    # --- add to FakeBridge ---
    def proc_group_start(self, group):
        self.calls.append(("proc_group_start", group))
        return [{"name": "audio", "running": True}]

    def proc_group_stop(self, group):
        self.calls.append(("proc_group_stop", group))
        return [{"name": "audio", "running": False}]

    def follow_status(self):
        self.calls.append("follow_status")
        return {"state": 1, "distance_to_person": 1.2, "reacq_state": 0,
                "breadcrumbs_pending": 0, "goal_held": False, "stale": False}
```

```python
def test_proc_group_start_stop():
    b, c = client()                      # reuse the file's existing client() helper
    r = c.post("/api/proc/group/follow_nav/start")
    assert r.status_code == 200
    r = c.post("/api/proc/group/follow_nav/stop")
    assert r.status_code == 200
    assert b.calls[-2:] == [("proc_group_start", "follow_nav"),
                            ("proc_group_stop", "follow_nav")]


def test_follow_status_endpoint():
    b, c = client()
    r = c.get("/api/follow/status")
    assert r.status_code == 200
    assert r.json()["state"] == 1
    assert "follow_status" in b.calls
```

(If the file uses a `client()` helper, reuse it; otherwise build `TestClient(create_app(b, webui_dir=None))` as the other tests do.)

- [ ] **Step 2: Run to verify failure**

Run: `cd src/tk26_vision && python3 -m pytest src/vision_track/test/test_track_web_app.py -k "group or follow_status" -v`
Expected: FAIL — 404 (routes absent).

- [ ] **Step 3: Add the routes** in `track_web_app.py`, beside the existing `/api/proc/...` handlers (the `proc_start`/`proc_stop`/`proc_status` block):

```python
    @app.post("/api/proc/group/{group}/start")
    def proc_group_start(group: str):
        return bridge.proc_group_start(group)

    @app.post("/api/proc/group/{group}/stop")
    def proc_group_stop(group: str):
        return bridge.proc_group_stop(group)

    @app.get("/api/follow/status")
    def follow_status():
        return bridge.follow_status()
```

- [ ] **Step 4: Add the bridge methods + follow-status subscription** in `track_web.py`. In `__init__` near the ProcessManager line (`self.proc_manager = ProcessManager()`, ~line 96) add the subscriber + cache:

```python
from std_msgs.msg import String   # add to imports if not present
# ...
self._follow_status = None       # latest parsed /follow_server/status dict
self._follow_status_t = 0.0
self.create_subscription(
    String, "/follow_server/status", self._on_follow_status, 10)
```

Add the callback + the three delegating bridge methods (near `proc_start`, ~line 324):

```python
def _on_follow_status(self, msg):
    import json
    try:
        with self._lock:
            self._follow_status = json.loads(msg.data)
            self._follow_status_t = time.time()
    except (ValueError, TypeError):
        pass

def follow_status(self):
    with self._lock:
        data = dict(self._follow_status) if self._follow_status else {}
        age = time.time() - self._follow_status_t if self._follow_status else 1e9
    data["stale"] = age > 2.0      # no fresh status in 2 s -> panel shows "—"
    return data

def proc_group_start(self, group):
    return self.proc_manager.start_group(group)

def proc_group_stop(self, group):
    return self.proc_manager.stop_group(group)
```

- [ ] **Step 5: Fold follow_status into the WS payload** so the panel updates live. Find the periodic payload that already carries `proc` (the WS loop calling `bridge.proc_status()`, `track_web_app.py` ~line 148) and add a sibling key, guarded so a bridge without the method (older fakes) doesn't break:

```python
                "follow": bridge.follow_status() if hasattr(bridge, "follow_status") else {},
```

- [ ] **Step 6: Run to verify pass**

Run: `cd src/tk26_vision && python3 -m pytest src/vision_track/test/test_track_web_app.py -v`
Expected: PASS (all).

- [ ] **Step 7: Commit**

```bash
cd src/tk26_vision
git add src/vision_track/vision_track/track_web.py \
        src/vision_track/vision_track/track_web_app.py \
        src/vision_track/test/test_track_web_app.py
git commit -m "feat(vision_track): dashboard group endpoints + follow-status bridge"
```

---

### Task 5: Webui mode selector + Follow-state panel + docs (tk26_vision)

**Repo:** `src/tk26_vision` · **Files:**
- Modify: `src/vision_track/webui/index.html`
- Modify: `src/vision_track/webui/app.js`
- Modify: `src/vision_track/webui/style.css`
- Modify: `src/vision_track/launch/track_web_control.launch.py` (docstring)
- Modify: `src/vision_track/README.md` (changelog)

- [ ] **Step 1: Read the current Bringup panel markup + JS** so the additions match existing element IDs, fetch helpers, and the WS payload handler.

Run: `sed -n '1,200p' src/vision_track/webui/index.html` and `grep -n "proc\|fetch\|ws\.\|onmessage\|bringup" src/vision_track/webui/app.js`

- [ ] **Step 2: Add the Follow panel markup** to `index.html` (inside or next to the existing Bringup panel). Use these exact IDs (the JS in Step 3 binds them):

```html
<div class="card" id="follow-card">
  <h3>Follow demo</h3>
  <label><input type="radio" name="follow-mode" value="follow_vision" checked> vision + audio</label>
  <label><input type="radio" name="follow-mode" value="follow_nav"> with navigation</label>
  <div class="row">
    <button id="follow-start">Start follow</button>
    <button id="follow-stop">Stop follow</button>
  </div>
  <div id="follow-state" class="follow-state">Follow state: —</div>
</div>
```

- [ ] **Step 3: Wire the buttons + state panel** in `app.js`. Add (reuse the file's existing POST/GET helper if present rather than raw fetch):

```javascript
function selectedFollowMode() {
  const r = document.querySelector('input[name="follow-mode"]:checked');
  return r ? r.value : 'follow_vision';
}
function setFollowControlsRunning(running) {
  document.querySelectorAll('input[name="follow-mode"]').forEach(el => el.disabled = running);
  document.getElementById('follow-start').disabled = running;
  document.getElementById('follow-stop').disabled = !running;
}
document.getElementById('follow-start').addEventListener('click', async () => {
  await fetch(`/api/proc/group/${selectedFollowMode()}/start`, {method: 'POST'});
  setFollowControlsRunning(true);
});
document.getElementById('follow-stop').addEventListener('click', async () => {
  await fetch(`/api/proc/group/${selectedFollowMode()}/stop`, {method: 'POST'});
  setFollowControlsRunning(false);
});

const FOLLOW_STATES = {0: 'IDLE', 1: 'TRACKING', 2: 'PURSUIT_LAST_SEEN',
                       3: 'APPROACHING_FINAL', 4: 'SUCCEEDED', 5: 'FAILED'};
function renderFollow(f) {
  const el = document.getElementById('follow-state');
  if (!f || f.stale || f.state === undefined) { el.textContent = 'Follow state: —'; return; }
  const name = FOLLOW_STATES[f.state] ?? `state ${f.state}`;
  const d = (f.distance_to_person >= 0) ? `, ${f.distance_to_person.toFixed(2)} m` : '';
  el.textContent = `Follow state: ${name}${d}${f.goal_held ? ' (HOLDING)' : ''}`;
}
```

In the existing WS `onmessage` handler, where the payload's `proc` field is consumed, add: `if (msg.follow) renderFollow(msg.follow);` (match the actual parsed-message variable name in that handler).

- [ ] **Step 4: Minimal styling** — append to `style.css`:

```css
.follow-state { margin-top: 6px; font-family: monospace; }
#follow-card .row { display: flex; gap: 8px; margin-top: 6px; }
```

- [ ] **Step 5: Update the launch docstring** in `track_web_control.launch.py` (lines 11-19): drop the `dummy_nav` mention; describe the two follow modes and the with-nav prerequisite. Replace that paragraph with:

```python
# The upgraded dashboard carries a **Bringup panel** with a follow-mode selector
# (vision+audio | with navigation) + Start/Stop. "vision+audio" spawns
# audio + ``follow-person --no-nav`` (tracking + voice, no base motion);
# "with navigation" spawns audio + ``following follow_server`` + ``follow-person``
# and drives the base — Nav2 must ALREADY be running (this launch does not start
# it). A live Follow-state panel reads ``/follow_server/status``. Components are
# spawned on demand AT RUNTIME by the dashboard's fixed-allowlist ProcessManager
# (see process_manager.py REGISTRY/GROUPS); for the spawns to resolve, build
# ``behavior_tree`` + ``following`` + ``audio_pakage`` into the overlay first.
```

- [ ] **Step 6: Build + manual UI smoke**

Run: `./src/tk26_vision/scripts/build.sh --packages-select vision_track && source src/tk26_vision/install/setup.zsh`
Run: `python3 -m pytest src/tk26_vision/src/vision_track/test/test_track_web_app.py src/tk26_vision/src/vision_track/test/test_process_manager.py -v`
Expected: PASS. (Full live dashboard verification is operator-in-the-loop per DEV_NOTES — note it, don't block on it.)

- [ ] **Step 7: Changelog + commit**

`README.md` changelog: `- track_web: follow-mode selector (vision+audio | with-nav) + Start/Stop + live Follow-state panel; dropped dead dummy_nav; allowlist now audio/follow_server/bt_vision/bt_nav.`

```bash
cd src/tk26_vision
git add src/vision_track/webui/index.html src/vision_track/webui/app.js \
        src/vision_track/webui/style.css \
        src/vision_track/launch/track_web_control.launch.py \
        src/vision_track/README.md
git commit -m "feat(vision_track): track_web follow-mode selector + Follow-state panel"
```

---

## Self-review notes (author)

- **Spec coverage:** D1 → Task 1; D4 status publisher → Task 2; D2 allowlist+groups → Task 3; bridge/endpoints + D4 panel data → Task 4; D3 mode selector + panel + docstring cleanup → Task 5. Goal-ownership decision needs no code (verified). ✓
- **No new deps:** Task 2 uses `std_msgs/String`; Task 4 dashboard subscribes `std_msgs/String` only. ✓
- **Type consistency:** group names `follow_vision`/`follow_nav` and registry keys `audio`/`follow_server`/`bt_vision`/`bt_nav` are identical in Tasks 3/4/5; `follow_status()` dict keys match the JSON `follow_server` emits in Task 2 (`state/distance_to_person/reacq_state/breadcrumbs_pending/goal_held`) plus the `stale` flag added by the bridge. ✓
- **Order:** Tasks 1+2 (BT flag, status topic) precede 3-5 (allowlist references `bt_vision`/`follow_server`; panel reads the status). ✓
