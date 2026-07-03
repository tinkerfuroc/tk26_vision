# Waving detection: VLM-only default + live-person prompt

- **Date:** 2026-07-04
- **Status:** Approved (brainstorm), pending implementation plan
- **Package:** `src/tk26_vision/src/tk_vision_specialized`
- **Touches:** `waving_person_server.py`, `_waving_vlm.py`, `launch/detect_waving.launch.py`, tests, README, `src/tk26_vision/CLAUDE.md`
- **Supersedes/extends:** `2026-06-02-waving-vlm-fallback-design.md`, `2026-07-03-restaurant-resolution-and-waving-vlm-concurrency-design.md`

## Goal

Two user-requested changes to the `detect_waving_persons` service:

1. **Force waving detection to use the VLM as the sole waver source** (not the MediaPipe pose heuristic), by default.
2. **Add a "must be a LIVE person" clause to the VLM prompt** — a detected waver must be a real, physically-present human, not a figure printed/displayed on a wall mural, advertisement, poster, screen, photo, etc.

Constraint: **no regressions**. Existing callers, the offline/no-API-key smoke tests, and the current fallback semantics must keep working; the change must be revertible via a parameter.

## Background: current architecture

`waving_person_server.py` (`DetectWavingPersonsNode`) serves `tinker_vision_msgs_26/srv/DetectWaving`. Per request it runs **two** detectors:

1. **MediaPipe CV path — always runs.** YOLO11-seg finds `person` boxes; MediaPipe Pose runs per-ROI; `is_waving()` applies a geometric heuristic (wrist above nose, or wrist above elbow with elbow near shoulder). Each waver's 3D centroid comes from the YOLO seg mask (or bbox fallback) over the back-projected depth grid.
2. **VLM path — opt-in.** Launched only when the request sets `min_waving_persons > 0` (`_start_vlm_call` returns `None` otherwise). Runs `request_waving_persons_chain` on a background thread in parallel with the CV pass. After the CV loop, `should_wait_for_vlm(cv_count, min_waving_persons)` decides whether to block on the VLM result and **merge** its boxes into the CV waver list (`_merge_vlm_result`, deduped by IoU), or abandon the call.

So today the VLM **augments** MediaPipe, gated by `min_waving_persons`. The prompt (`_SYSTEM_PROMPT` in `_waving_vlm.py`) asks for whole-person boxes of everyone waving; `_SCHEMA` is `{persons:[{box_2d,waving}]}`; `select_boxes()` keeps entries with `waving == true` and a decodable box.

**The regression trap.** The two runtime callers both send `min_waving_persons = 0`:
- `BtNode_ScanForWavingPersonNew` (GPSR/EGPSR/Restaurant BT) — never sets the field.
- `track_web.py wave()` (person-tracker "wave to resume") — deliberately default.

So today they get **MediaPipe-only**. If we bypass MediaPipe without also neutralizing the `min_waving_persons` gate, these callers would get **zero detection**. The design must flip both together.

## Ground-truth test images

Three JPEGs at the workspace root (`/home/tinker/tk25_ws/`), all 4096×3072, all from the same venue (a "Premium Lounge" wall mural of two mountain climbers, one with a raised arm that reads like a wave):

| Image | Scene | Expected LIVE wavers |
|---|---|---|
| `waving_background` | Mural only; one live seated woman (lower-left) not waving | **0** |
| `waving_real_and_background` | Mural + 3 live people; bearded man **and** center woman both have a raised open palm (overlapping arms), third man holds a phone | **2** |
| `waving_two_hands` | Entrance; 3 live people, left (curly hair) + center (glasses) raise open palms, right holds a controller; a fourth person walks in the background | **2** |

These exercise both failure modes: `waving_background` = false-positive on a printed figure; `waving_real_and_background` = reject the printed figure while keeping two real overlapping wavers; `waving_two_hands` = don't under-count real wavers (positive control, no printed trap).

## Design

### 1. New parameter: `waving_detector`

Add a string ROS param `waving_detector` to `DetectWavingPersonsNode`, default `'vlm'`. Values:

- `'vlm'` *(new default)* — VLM is the **sole** waver source. MediaPipe `is_waving` is skipped. YOLO still runs (person masks feed the 3D centroid; it is **not** the wave decision).
- `'hybrid'` — today's behavior: MediaPipe wavers + VLM augmentation gated by `min_waving_persons`.
- `'mediapipe'` — legacy CV-only; VLM never called.

`enable_vlm_fallback` (existing, default `true`) is retained as a hard kill-switch: when `false`, the VLM is never called and the node behaves as `mediapipe` regardless of `waving_detector`. This preserves its current meaning and gives a single "no VLM ever" flag.

### 2. `effective_mode` resolution (graceful degrade)

At the top of `detect_waving_callback`, resolve the mode actually used this call:

```
if not enable_vlm_fallback or not self._vlm_chain:
    effective_mode = 'mediapipe'          # kill-switch OR no provider key
    if waving_detector in ('vlm', 'hybrid'):
        log.warn("waving_detector=<x> but VLM unavailable (no key / disabled); "
                 "falling back to MediaPipe for this call.")
elif waving_detector == 'vlm':
    effective_mode = 'vlm'
elif waving_detector == 'hybrid':
    effective_mode = 'hybrid'
else:
    effective_mode = 'mediapipe'
```

This is the no-regression guarantee: an offline / no-`OPENROUTER_API_KEY` / no-`DASHSCOPE_API_KEY` box (T1/T2 smoke tests, dev machines) transparently keeps the old MediaPipe behavior. Only boxes with a working provider key are truly VLM-only.

### 3. Control-flow changes in `detect_waving_callback`

Minimal edits; the merge/dedup/centroid code is reused unchanged.

- **Launch VLM:** call `_start_vlm_call(rgb_image, request.min_waving_persons, force=(effective_mode == 'vlm'))`. Add a `force` param to `_start_vlm_call`: when `True`, skip the `min_waving_persons <= 0` early-return (still returns `None` if `not enable_vlm_fallback or not self._vlm_chain`). Launch when `effective_mode in ('vlm', 'hybrid')`; in `hybrid` the existing `min_waving_persons` gate still applies.
- **MediaPipe pass:** compute `run_mediapipe = effective_mode in ('mediapipe', 'hybrid')`. In the per-person loop, only call `self.pose.process()` / `is_waving()` when `run_mediapipe`; otherwise set `landmarks=None, is_wave=False`. Still build `person_records` (seg masks → centroid depth) and `all_person_annotations` (for the debug overlay) in every mode. The `if is_wave:` centroid block therefore only runs in mediapipe/hybrid — VLM supplies centroids in `vlm` mode.
- **Wait/merge:** compute `wait = (effective_mode == 'vlm') or should_wait_for_vlm(len(cv_wavers), request.min_waving_persons)`. When waiting, block on `vlm_future.result(timeout=self.vlm_timeout_s)` and `_merge_vlm_result(...)`. In `vlm` mode the CV waver lists are empty, so the merge simply appends every VLM box (dedup against an empty list is a no-op) — a clean VLM-only result with no new merge code.

Everything downstream (sort-by-depth, TF transform, debug overlay, vision logging, response population) is unchanged. In `vlm` mode the overlay draws all YOLO persons green and VLM wavers orange (`waving (vlm)`), which the existing code already does.

### 4. Prompt change (`_waving_vlm.py`, prompt-text-only)

Extend `_SYSTEM_PROMPT` with a live-person clause. No `_SCHEMA` / `select_boxes` change (user chose prompt-only for now; a structured `live_person` field is a possible future hardening). Proposed addition (wording finalized in the plan):

> Only count REAL, LIVE people who are physically present in the scene. Do NOT count any person who is printed, drawn, or displayed on a poster, advertisement, banner, wall mural, painting, photograph, magazine, product packaging, television, monitor, phone, tablet, or any other screen or flat surface — even if their pose looks exactly like a wave. A climber, model, or figure that is part of a picture on the wall is NOT a waving person.

Rationale: both adversarial test images share a wall mural whose printed climber has a raised arm; the clause must name "wall mural / advertisement / printed figure" explicitly because that is the concrete distractor.

### 5. Launch wiring

`launch/detect_waving.launch.py`: add a `waving_detector` `DeclareLaunchArgument` (default `vlm`) and pass it through to the node `parameters`. No change to defaults elsewhere.

## Non-goals

- No change to `DetectWaving.srv` (no new request/response fields). `min_waving_persons` keeps its type; in `vlm` mode it no longer gates whether the VLM runs, but it is not removed.
- No structured `live_person` schema field (deferred; prompt-only for now).
- No change to `_merge_vlm_result`, `_waving_geometry.centroid_from_box`, dedup, depth, TF, or vision-logging logic.
- No per-caller mode overrides in the BT / tracker in this change (they inherit the `vlm` default; a node can set `waving_detector` if it needs a fast path).

## Backward-compatibility / regression analysis

- **No-key / offline boxes:** `effective_mode` degrades to `mediapipe` → identical to today. T0/T1/T2 smoke tests unaffected.
- **`hybrid` / `mediapipe` modes:** old code paths preserved verbatim; `should_wait_for_vlm` and the `min_waving_persons` gate untouched in those modes. Selecting `waving_detector:=hybrid` reproduces the exact current behavior.
- **Callers:** `BtNode_ScanForWavingPersonNew` is async (polls `done()`), no client timeout — tolerates VLM latency. `track_web.wave()` uses a 30 s client timeout — covers the ≤20 s VLM timeout. `test_scan.py` sets `min_waving_persons=2`; harmless in `vlm` mode (VLM runs regardless).
- **Latency (accepted tradeoff):** in `vlm` mode every call waits ~5–20 s (VLM) vs ~100–300 ms (MediaPipe). This is the point of the change. Documented; a fast path remains one param flip away (`waving_detector:=mediapipe`/`hybrid`).

## Testing

1. **Unit (no network), extend `test/test_waving_vlm.py`:**
   - `_SYSTEM_PROMPT` contains the live-person clause (assert substrings like "wall mural"/"printed"/"advertisement" and "LIVE").
   - `select_boxes` still filters on `waving == true` (existing tests stay green — proves prompt-only didn't change the decoder).
2. **Unit (no network) for mode logic:** a small helper `resolve_effective_mode(waving_detector, enable_vlm_fallback, chain_nonempty)` extracted as a pure function so it is testable without ROS; assert the truth table (vlm+key→vlm, vlm+nokey→mediapipe, hybrid→hybrid, mediapipe→mediapipe, kill-switch→mediapipe). And `_start_vlm_call(force=True)` bypasses the `min_waving_persons<=0` gate but still returns `None` with an empty chain (test via a light stub or by asserting the branch).
3. **Offline live-VLM image test (new file, e.g. `test/test_waving_vlm_live_images.py`):** `pytest.mark.skipif` when no `OPENROUTER_API_KEY`/`DASHSCOPE_API_KEY`. `cv2.imread` each workspace image, **downscale to ~1280 px wide** (production camera scale; `encode_data_url` does not resize, so full 4096-px frames waste tokens/latency), call `request_waving_persons_chain`, assert box counts: `waving_background → 0`, `waving_real_and_background → 2`, `waving_two_hands → 2`; for the first two, assert no returned box centre lands in the top-right mural region. Image paths resolved from an env var or workspace-root default, skip if absent. Mark as non-deterministic in a comment (counts are the contract; allow a documented retry).
4. **Build + node smoke:** `./src/tk26_vision/scripts/build.sh --packages-select tk_vision_specialized`, then `ros2 run tk_vision_specialized waving_person_server` — confirm it advertises `detect_waving_persons`, logs the resolved `waving_detector`/chain, and (no key) logs the degrade warning.

## Files changed

- `tk_vision_specialized/tk_vision_specialized/waving_person_server.py` — param, `effective_mode`, `_start_vlm_call(force=)`, guarded MediaPipe pass, wait/merge gate.
- `tk_vision_specialized/tk_vision_specialized/_waving_vlm.py` — `_SYSTEM_PROMPT` live-person clause.
- `tk_vision_specialized/launch/detect_waving.launch.py` — `waving_detector` arg.
- `tk_vision_specialized/test/test_waving_vlm.py` — prompt-clause + mode-logic unit tests.
- `tk_vision_specialized/test/test_waving_vlm_live_images.py` — new key-gated image test.
- `tk_vision_specialized/README.md` — Changelog entry + behavior note.
- `src/tk26_vision/CLAUDE.md` — update the `waving_person_server` param blurb (`waving_detector` default `vlm`, degrade behavior).

## Rollout / build / verify

- Build with the wrapper (`build.sh`), not raw colcon (venv shebangs).
- Verify order: unit tests (green offline) → node startup smoke → offline image test where a key exists.
- Commit per phase (one commit per plan phase); README/Changelog updated in the same commit as the code it documents.

## Risks & tradeoffs

- **VLM non-determinism:** the image test counts are the contract but the VLM may occasionally miscount; the test documents this and the prompt is tuned against the three known images. Prompt-only (no enforced `live_person` field) is a deliberate, weaker-but-simpler choice the user selected; if false positives on printed figures persist, the follow-up is the structured `live_person` schema field + `select_boxes` filter.
- **Latency** on every waving call in `vlm` mode (see above) — accepted, documented, param-revertible.
