# track_web follow-pipeline integration (nav executive + vision-only mode) — design

**Date:** 2026-06-13
**Packages:** `vision_track` (tk26_vision, primary), `following` (tk26_navigation),
`behavior_tree` (tk25_decision)
**Status:** approved (user, 2026-06-13)
**Supersedes the bringup half of:** `docs/superpowers/specs/2026-06-10-track-web-bringup-control-design.md`
**Related:** tk26_navigation `docs/specs/2026-06-12-f4-bt-in-the-loop-design.md`
(the scripted/automated follow surface; this is the operator-driven one).

## Goal

Make the track_web "Follow Demo" dashboard a working bring-up surface for the
**current** follow pipeline — start the navigation executive + the rewired
follow-person BT from the UI and watch live follow state — while keeping a
**vision+audio-only** follow mode (perception + voice reactions, no base motion).

## Motivation — the dashboard's follow demo is stale

The dashboard's `ProcessManager` allowlist (`process_manager.py:46-48`) predates
the 2026-06-10 follow rewire:

```python
"audio":     ros2 launch audio_pakage audio.launch.py
"dummy_nav": ros2 run behavior_tree dummy-nav      # DEAD
"bt":        ros2 run behavior_tree follow-person   # now requires follow_server
```

After the rewire, `follow_person.py` drives navigation through the `Follow` action
on `follow_server` and **dropped** the `/follow_target` publisher. Consequences:

- **`dummy_nav` is dead** — it subscribes `/follow_target`, which has had no
  publisher since the rewire.
- **`bt` is broken in the UI** — the tree's `BtNode_FollowAction` waits on
  `follow_server`, which is **not** in the allowlist and nothing launches it.

So starting the follow demo from the dashboard today yields a BT whose follow child
never connects and a nav stub that does nothing.

## Track-goal ownership (settled by the code, no BT change)

`person_track_server` is **single-goal**: the dashboard's `start_goal()` already
handles *"goal REJECTED (another client is tracking?)"* (`track_web.py:202`). The
follow-person BT's child A (`BtNode_TrackPersonAction`) sends its own
`/track_person` goal. They cannot both hold it. **`reseed` is a separate service**
(`ReseedTarget`, `track_web.py:238`) — target selection by drawing a box works
regardless of which client owns the action goal.

Therefore the ownership rule is:
- **The BT owns the `/track_person` goal** in the follow demo (its child A).
- **The dashboard is viewer + target-selector + bring-up controller:** state /
  gallery / video flow off the tracker's debug topics regardless of goal owner;
  `reseed` selects the target; the Bringup panel starts/stops the follow stack.
- The dashboard's own `start_goal`/`stop_goal` remain for the **standalone
  pure-tracking demo**; pressed during a follow they cleanly report "REJECTED —
  another client is tracking" (the BT). Stopping the BT frees the goal again.

No change to `BtNode_TrackPersonAction` or `person_track_server`.

## Design decisions

### D1 — Two follow modes via a BT `--no-nav` flag

`create_follow_person_tree(target_frame="", enable_navigation=True)`:
- `enable_navigation=True` → `Parallel[track, follow, reactions]` (today's full
  tree).
- `enable_navigation=False` → `Parallel[track, reactions]` — **no** follow child.
  This is the vision+audio-only path; it is self-consistent because the announcer
  (`BtNode_ReacqAnnounce`) only reads `track/reacquisition_state`, written by child
  A, and nothing reads `follow/*`.

`cli.py:main()` parses `--no-nav` (argparse `parse_known_args`, ignoring
`--ros-args`) and passes `enable_navigation = not args.no_nav` into the tree
factory handed to `run_tree`.

### D2 — Allowlist + a thin group layer in `ProcessManager`

New fixed allowlist (drop `dummy_nav`; split `bt`):
```python
_REGISTRY = {
    "audio":         ["ros2","launch","audio_pakage","audio.launch.py"],
    "follow_server": ["ros2","run","following","follow_server",
                      "--ros-args","-p","working_frame:=map"],
    "bt_vision":     ["ros2","run","behavior_tree","follow-person","--no-nav"],
    "bt_nav":        ["ros2","run","behavior_tree","follow-person"],
}
_GROUPS = {
    "follow_vision": ["audio", "bt_vision"],
    "follow_nav":    ["audio", "follow_server", "bt_nav"],
}
```
`start_group(group)` validates `group` against `_GROUPS` (fixed-allowlist model
preserved), then starts each member **in listed order** with a small
`stagger_sec` (default `1.5`) between starts so the `TextToSpeech` service and
`follow_server` are up before the BT's first tick (belt-and-suspenders —
`BtNode_FollowAction` also waits for its server). `stop_group(group)` stops members
in **reverse** order. Both never raise on unknown/already-running/already-stopped
(same contract as `start`/`stop`). Per-member `start`/`stop` stay public for
debugging.

**`working_frame:=map`** is baked into the `follow_server` entry (arena case). The
out-of-arena `odom` variant is a documented manual run, not a UI control.

### D3 — Mode selector + single Start/Stop in the UI

`webui/index.html` + `app.js`: a **Follow mode** radio (`vision+audio` |
`with navigation`) and one **Start follow** / **Stop follow** pair. Start posts the
selected group (`follow_vision` / `follow_nav`); Stop posts the same group's stop.
The mode radio disables while a group is running. The existing per-component status
chips remain (driven by `proc_status`).

### D4 — Surface live follow state in the dashboard

`follow_server` publishes a compact **`std_msgs/String` JSON** status on `~/status`
(resolves to `/follow_server/status`) at its feedback cadence, mirroring
`{state, distance_to_person, reacq_state, breadcrumbs_pending, goal_held}`. This is
additive and keeps the coupling minimal — **no** `tinker_nav_msgs` dependency leaks
into `vision_track` (the dashboard subscribes `std_msgs/String`, which it already
depends on; it does not subscribe the hidden action-feedback topic). The dashboard
parses the JSON and renders a small **Follow state** panel
(`TRACKING / PURSUIT_LAST_SEEN / APPROACHING_FINAL`, distance, `HOLDING` when
`goal_held`). Stale (no message in ~2 s) → shown as "—".

## Components / changes

| Change | File | Role |
|---|---|---|
| `enable_navigation` arg + tree branch | `behavior_tree/FollowPerson/follow_person.py` | build with/without the follow child |
| `--no-nav` flag | `behavior_tree/FollowPerson/cli.py` | select the mode at launch |
| `~/status` JSON publisher (additive) | `following/following/follow_server.py` | feed the dashboard follow-state panel |
| allowlist + `start_group`/`stop_group` | `vision_track/vision_track/process_manager.py` | spawn the right component set per mode |
| `/follow/status` in payload; group endpoints; status sub | `vision_track/vision_track/track_web.py`, `track_web_app.py` | bridge + HTTP/WS surface |
| mode selector, Start/Stop, follow-state panel | `vision_track/webui/{index.html,app.js,style.css}` | the operator UI |
| docstring: drop `dummy_nav`, document modes + prereqs | `vision_track/launch/track_web_control.launch.py` | bring-up doc |

## Data flow (with-navigation mode)

```
Prereq (NOT started by the UI): Nav2 up + cameras up (CAMERA_BRINGUP.md)
track_web_control.launch.py: person_track_server + track_web dashboard

operator: pick "with navigation" → Start follow
  dashboard → ProcessManager.start_group("follow_nav")
     → audio.launch.py → follow_server (working_frame:=map) → follow-person BT
  BT child A → /track_person goal (owns tracking)   [reseed selects target]
  BT child B → Follow goal → follow_server → Nav2 → /cmd_vel → robot
  follow_server → /follow_server/status (JSON) ─┐
  tracker debug topics ─────────────────────────┼─▶ dashboard: video+gallery+state
                                                 └─▶ dashboard: Follow-state panel
```

Vision+audio-only mode is identical minus `follow_server`/`bt_nav` (it runs
`bt_vision`); the Follow-state panel reads "—" (no `follow_server`), which is the
correct signal that navigation is intentionally off.

## Prerequisites (documented, not started by the UI)

- **with-navigation:** Nav2 already running (`bringup_launch.py`); cameras up per
  `CAMERA_BRINGUP.md`. The dashboard does not spawn Nav2 (heavy). If Nav2 is down,
  the Follow-state panel shows no progress / nav failure — visible, not silent.
- **vision+audio-only:** cameras up. No Nav2 needed.

## Testing

- `vision_track` `test/test_process_manager*.py` (extend): `start_group` starts all
  members of a known group in order and returns their statuses; unknown group →
  error dict (never raises); `stop_group` stops in reverse; a member spawn failure
  is reported without aborting the rest.
- `vision_track` `test/test_track_web_app.py` (extend): `/proc/group/start` +
  `/proc/group/stop` call the bridge with the posted group name; the periodic
  payload carries `follow_status`; a fake bridge drives the assertions (no ROS
  graph), matching the existing fake-bridge pattern.
- `behavior_tree` `test/test_follow_tree_build.py` (extend): `enable_navigation=
  False` builds a 2-child Parallel (no `BtNode_FollowAction`); `True` builds 3.
- `following`: assert `follow_server` advertises `/follow_server/status`
  (`std_msgs/String`) and that one feedback cycle emits parseable JSON with the
  five keys (extend an existing follow_server test).
- Live: the dashboard follow demo is exercised on the robot per the vision DEV_NOTES
  follow-demo round; this spec adds a checklist entry (both modes start/stop; reseed
  selects target; follow-state panel tracks the run).

## Files

**Modify**
- tk25_decision: `behavior_tree/FollowPerson/follow_person.py`,
  `behavior_tree/FollowPerson/cli.py`, `behavior_tree/test/test_follow_tree_build.py`,
  `README.md` (changelog).
- tk26_navigation: `src/following/following/follow_server.py`, its test,
  `src/following/README.md` (changelog).
- tk26_vision: `vision_track/vision_track/process_manager.py`,
  `vision_track/vision_track/track_web.py`,
  `vision_track/vision_track/track_web_app.py`,
  `vision_track/webui/{index.html,app.js,style.css}`,
  `vision_track/launch/track_web_control.launch.py` (docstring),
  `vision_track/test/test_process_manager*.py`, `vision_track/test/test_track_web_app.py`,
  `vision_track/README.md` (changelog).

**No new package dependencies** — `follow_server` status is `std_msgs/String`
(both sides already depend on `std_msgs`); `bt_vision` is the same executable with a
flag; the group layer is ROS-free.

## Dependencies / direction

No new cross-package message deps. The dashboard does **not** import
`tinker_nav_msgs` (it reads the JSON status topic, not the action feedback). The
ProcessManager argv references `following` / `behavior_tree` / `audio_pakage`
executables resolved at runtime from the overlay (the same way the current
allowlist already references `behavior_tree`), so no build/exec-dep change to
`vision_track` is required.

## Invariants / risks

- **Goal ownership is unambiguous:** the BT owns `/track_person`; the dashboard's
  start_goal is rejected during a follow and that rejection is a clear, handled
  message — not a crash. `reseed` stays functional for target selection.
- **`--no-nav` tree is self-consistent:** nothing reads `follow/*`; the announcer
  only needs `track/reacquisition_state`.
- **Ordered group start** + per-member `wait_for_server` means the BT never races
  ahead of `follow_server`/audio.
- **Fixed-allowlist security preserved:** groups are a fixed `_GROUPS` map of
  fixed `_REGISTRY` names; no operator-supplied argv ever reaches a shell.
- **Additive follow_server publisher:** `~/status` is a new topic; existing
  consumers and the F4 observer (which uses TF + cmd_vel) are unaffected.
- **Nav2 is a prerequisite, not auto-spawned:** a missing Nav2 surfaces as a
  stalled Follow-state panel rather than the UI trying to boot a heavy stack.

## Out of scope

- Spawning Nav2 / cameras from the dashboard (heavy; remain documented prereqs).
- Production task rewire (HelpMeCarry / GPSR) — separate effort.
- The F4 automated sim harness — its own spec; this is the operator surface.
- `odom`/out-of-arena `working_frame` as a UI control (manual run only).
