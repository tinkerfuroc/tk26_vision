# Passive recovery (post-NEEDS_HELP) + upright+mask-fill gallery gate + larger seg model — design

**Date:** 2026-06-09 (revised 2026-06-10)
**Package:** `src/vision_track`
**Branch:** `feat/track-web-idle-video`
**Status:** approved (user, 2026-06-09; revised 2026-06-10 — Issue 2 pseudo-mask dropped, passive window → 5 s)

Changes to `person_track_server`. **Issue 1** extends the passive-recovery window
to ~5 s and adds a precision-bounded auto-recovery escape hatch once latched in
NEEDS_HELP. **Issue 2** (OSNet background parity) was investigated and resolved
with NO code change — the query already uses the real seg mask. **Issue 3**
gates gallery admission on BOTH bbox w/h ratio (upright) AND mask-fill (clean).
**Issue 4** is the model-quality upgrade (larger YOLO-seg + TensorRT) that reduces
maskless frames and lifts the raw ReID operating point Issue 1's bar depends on.
They compose: a bigger model (#4) raises similarity and mask coverage; the
upright+mask-fill gate (#3) keeps the gallery to clean standing operator views;
the post-NEEDS_HELP relaxation (#1) catches a returning operator once #4 has
lifted their similarity above the relaxed bar.

---

## Issue 1 — passive reacquisition does not recover after NEEDS_HELP

### Root cause
The lone-candidate **commit bar is held at `single_person_commit_bar` (0.72)**
(`core/tracking_pipeline.py:_confirm_reid_candidate`, ~line 629), and `pending`
arms only on commit-bar hits. A person returning after a real absence
(lighting/pose/background changed) scores in the **[0.55, 0.72) dead band**:
Phase 3 *pursues* them every frame but `is_hit` is always False → never arms,
never commits → stuck in `REACQ_NEEDS_HELP` until a manual wave/reseed. The FSM
re-arm-on-`committed_swap` path works and nothing aborts (the help-latch holds
the goal) — it is purely "pursued, never commits." There is no post-NEEDS_HELP
escape hatch.

### Fix — relaxed lone-candidate recovery while latched in NEEDS_HELP
Add a precision-bounded escape hatch, but a **conservative** one — stricter than
the wave/reseed reclaim (which commits at 0.55 over 5 frames). The relaxed bar
relies on Issue 2 + Issue 4 having lifted the returning operator's similarity, so
it can stay high and still recover:

- **Gate:** apply ONLY when `tracker.in_needs_help` is True AND
  `num_candidates == 1` (exactly one person visible → no ambiguity).
  `tracker.in_needs_help` is set by the node each iteration from the **latched**
  help state `self._help_latched` (NOT the instantaneous `reacq_state`, which
  would oscillate because the relaxed bar resets `frames_lost`). `_help_latched`
  is set at `frames_lost >= active_help_after_frames` and cleared only on a TRUE
  re-lock, so it is stable across the recovery window.
- **Relaxed bar + sustained N-of-M:** new params
  `single_person_commit_bar_help` (default **0.62**) and a dedicated N-of-M
  window: commit when **12 of the last 16** frames cleared the relaxed bar
  (`needs_help_confirm_frames` = **12**, `needs_help_commit_window` = **16**).
  In `_confirm_reid_candidate`, when the gate holds, override exactly three
  locals before the window machinery: `commit_bar = single_person_commit_bar_help`,
  `required_confirmation = needs_help_confirm_frames` (no `post_shake_extra`
  addition — keep it exactly 12), and `window_m = needs_help_commit_window` (16).
  Arming preconfirm stays `reid_preconfirm_frames` (3) commit-bar hits. Outside
  the gate, behavior is UNCHANGED (lone 0.72 / multi `reid_threshold`,
  N = `reid_confirmation_frames` + post_shake, window = `provisional_commit_window`
  18).
- **Why 0.62, not the wave's 0.55:** 0.55 over 5 frames was chosen to be
  *generous* for a cooperative wave. Passive auto-recovery must not auto-lock a
  marginal [0.55, 0.62) match, so the bar is held at 0.62 and the sustained
  requirement is much stronger (12/16 vs 5). The returning operator clears 0.62
  because Issue 2 (query/gallery background parity) and Issue 4 (larger seg model)
  raise their deep cosine ~0.10–0.15 off the maskless-background floor. The two
  fixes are designed to compose: #2/#4 lift the score, #1 catches it.
- **Re-lock must clear the latch:** the relaxed commit must produce a genuine
  `target_track_id` swap → `committed_swap` → `fsm.start(new_id)` + present=True
  step → `feedback.target_lost=False` → `_help_latched` cleared. (This is the
  existing commit mechanism; only the bar/count/window change.) Do NOT relax in a
  way that only publishes a provisional (target_lost stays True) — that would not
  clear the latch and would leave the same stall.

### Precision invariants
- Multi-person scenes (`num_candidates > 1`) are UNCHANGED — the relaxation never
  applies when more than one person is visible, so a bystander in a crowd cannot
  be locked more easily.
- The relaxation activates only AFTER the human has already been asked for help
  (latched NEEDS_HELP). Normal tracking and the in-window passive reacq
  (`frames_lost < active_help_after_frames`) keep the strict 0.72 lone bar.
- The N-of-M sustained-streak gate is preserved and made *stronger* (12/16) — no
  single-frame false lock; a brief look-alike cannot accumulate 12 of 16.
- `OTHER_PERSON_MAX_TARGET_SIM` (0.72) is unchanged (lone case doesn't run
  `register_other_persons` anyway).

### Passive-recovery window — WALL-CLOCK, ~5 s (not frames)
Escalation to NEEDS_HELP was frame-based (`reacq_state(...)` / `_is_awaiting_help`
flip once `frames_lost >= active_help_after_frames`). **Operator constraint
(2026-06-10): do not base need-help on frames — frame rate is unreliable during a
tournament** (GPU contention, other nodes), so a frame count gives an
unpredictable wall-clock window. Convert escalation to **wall-clock time**:

- New param `active_help_after_sec` (default **5.0**); **remove**
  `active_help_after_frames` (param, reads, telemetry, debug-state field).
- `reacq_state(tracked, time_since_lost, help_after_sec)` — NEEDS_HELP once
  `time_since_lost >= help_after_sec`, else PASSIVE (TRACKING while held). Pure,
  time-based; `help_after_sec <= 0` escalates immediately when lost (disable
  semantics preserved).
- Time anchor: node tracks `self._last_confirmed_time` = wall-clock of the last
  TRUE lock (`feedback.target_lost == False`), set at tracking start, reset per
  goal, refreshed only on a confirmed lock (NOT on a provisional/pre-commit coast
  — so a coast doesn't reset the clock). `time_since_lost = time.time() -
  self._last_confirmed_time` feeds both the telemetry and the
  `_is_awaiting_help` latch. This is robust to fps AND to the `frames_lost`-reset
  problem the latch was created to paper over.
- `_is_awaiting_help` latches at `time_since_lost >= active_help_after_sec`
  (replacing the frame check); the `active_help_timeout_sec` post-escalation hold
  bound is already time-based and unchanged. The latch still clears only on a true
  re-lock, so the abort-mid-reacquire protection is intact.

The latch (and thus Issue-1's relaxation) now engages at a *deterministic 5 s*
regardless of fps: 5 s of strict passive recovery, then escalate + relax.

### New params (Issue 1)
| Param | Default | Meaning |
|---|---|---|
| `single_person_commit_bar_help` | `0.62` | lone-candidate commit bar while latched in NEEDS_HELP |
| `needs_help_confirm_frames` | `12` | confirm-hits required (N) to commit while latched in NEEDS_HELP |
| `needs_help_commit_window` | `16` | N-of-M window length (M) for the help commit |
| `active_help_after_sec` | `5.0` | wall-clock seconds lost before escalating to NEEDS_HELP (replaces `active_help_after_frames` — frame counts are unreliable when fps varies in a tournament) |

---

## Issue 2 — OSNet query/gallery background parity — RESOLVED, NO CODE CHANGE

### Decision (operator, 2026-06-09)
**Rejected the pseudo-mask fallback.** "For the OSNet query you should have the
seg mask — do not synthesize a pseudo mask." Verified the mask plumbing is
correct and symmetric: `core/result_parser._extract_mask` populates
`DetectionResult.mask`, and **every** OSNet query call site already passes the
real `result.mask` to `extract_features` → `_segment_crop_for_reid` (init/present
`tracking_pipeline.py:144`, reacq `:266`, verify `:319`, reid_match `:608`,
periodic `:762`, register `:945`). There is no query-specific mask drop: the
query uses the same real seg mask the gallery does whenever YOLO-seg emits one.

The only asymmetry is inherent and acceptable: the gallery is *quality-gated*
(rejects low-coverage frames), the query runs every frame. On the rare frame
where YOLO-seg genuinely emits a box without a mask, `_segment_crop_for_reid`
keeps its **existing** raw-crop behavior (unchanged). The right lever for fewer
maskless frames is a better seg model (Issue 4), not a synthesized mask. Phase A
was reverted; `reid/reid.py` is unchanged from `main`. No tests, no params.

### Superseded fix (NOT taken — kept for the record)
The first draft synthesized a bbox-inscribed ellipse pseudo-mask in
`_segment_crop_for_reid` when the mask was None/empty/all-zero, so a maskless
query crop would still get the gallery's background neutralization. The operator
rejected this: the OSNet query already has the real seg mask (plumbing verified
above), so fabricating one papers over a non-problem and a coarse ellipse can't
match a real tight-crop. Implementation reverted.

---

## Issue 3 — gallery admission gate on BOTH bbox w/h ratio AND mask-fill

### Rationale (operator decision, revised 2026-06-10)
The operator is **always standing**, so only clean, UPRIGHT, well-segmented
operator views should enrich the ReID gallery. The first draft (mask-fill only,
aspect relaxed to 2.0) was reconsidered: the operator opted to **keep an upright
bbox w/h gate AND the mask-fill gate** — admit a view only when BOTH hold. The
existing gate (`reid/quality.py:crop_quality_ok`, fed by
`reid/appearance_manager.py:update_appearance`) already checks both
`min_mask_coverage` and `max_aspect_ratio` and ANDs all checks, so this is a
threshold + wiring change, not new gate logic.

### Fix
- **Aspect (uprightness) gate:** launch param `gallery_max_aspect_ratio`
  (default **0.5**). `crop_quality_ok` rejects when `aspect_ratio` (= w/h) `>`
  this. An upright standing operator is taller-than-wide (w/h ~0.4), so 0.5
  admits clearly-upright boxes and rejects square/wide (occluded/merged/
  non-standing) ones. Adjustable at launch.
- **Mask-fill gate:** launch param `gallery_min_mask_fill` (default **0.35** =
  mask_pixels/bbox_area). `crop_quality_ok` rejects when `mask_coverage <=` this.
  A merged/garbage/occlusion-inflated box has the operator mask as a thin slice
  (low fill → reject). `mask_coverage` is `None` when no seg mask is present that
  frame → not rejected on fill, but still subject to the aspect gate.
- Both are wired into the per-call gate dict in `update_appearance` (read from
  tracker attrs, falling back to `DEFAULT_GATE` for `min_crop_h` / `min_blur_var`).
  `crop_quality_ok` ANDs all checks, so admission requires upright AND clean.

### New params (Issue 3)
| Param | Default | Meaning |
|---|---|---|
| `gallery_max_aspect_ratio` | `0.5` | max bbox w/h to admit (upright gate — operator is taller-than-wide; reject square/wide); adjustable at launch |
| `gallery_min_mask_fill` | `0.35` | min mask_pixels/bbox_area to admit (clean-detection gate, strict `<=`); adjustable at launch |

### Invariants / risks
- Admission-only gate — never affects the query/matching path or the live lock.
  A rejected frame still refreshes motion/last-seen (existing behaviour kept).
- Both gates AND'd: a view must be upright (w/h ≤ 0.5) AND clean (fill > 0.35).
  Trade-off the operator accepted: a genuinely-upright operator whose bbox is
  clipped square in a dense crowd will be skipped for gallery enrichment that
  frame (tracking continues via ByteTrack regardless; the gallery just waits for a
  cleaner upright view). Loosen `gallery_max_aspect_ratio` at launch if a venue
  proves too crowded.
- `mask_coverage` is `None` when no mask is available → not rejected on fill that
  frame, but the aspect gate still applies.

---

## Issue 4 — larger YOLO-seg model + TensorRT engine (operational; better masks)

### Rationale
Issues 2 and 3 both lean on mask quality: #2's pseudo-mask only fires when the
mask is *missing* (a bigger model misses fewer), and #3 gates on mask coverage (a
bigger model segments tighter and more completely). A larger seg model also lifts
detection recall on small/occluded persons — exactly the hard reacquisition
frames — raising the raw ReID operating point that Issue 1's 0.62 bar depends on.

### What already exists (no new code)
- `YOLOTracker` `model_path` param accepts `yolo11{m,l,x}-seg.pt` (SUPPORTED list)
  and loads `.engine` (TensorRT) transparently.
- `inference_size` (imgsz) param exists; the engine is resolution/batch-locked to
  its export imgsz, and the node must run the same imgsz.
- `scripts/export_yolo_trt.py` exports a FP16 TensorRT engine for a given model +
  imgsz on the deployment GPU.

### Fix (operational)
1. Provision `tensorrt` in the export venv (it is not in `.venv-vision-main`).
2. Export `yolo11m-seg.pt` → `yolo11m-seg.engine` at `inference_size` 736 on the
   robot GPU (RTX 5070 Ti, idle).
3. Benchmark end-to-end tracker latency with the engine vs the current `s` `.pt`;
   confirm it sustains the camera rate (≥ ~25–30 Hz) and stays within the 10 s/call
   vision compute budget. If `m` is too slow even with TRT, fall back to `s`-seg
   engine (still a TRT speedup); if `m` has ample headroom, optionally evaluate
   `l`-seg.
4. **Do NOT change the default** `model_path` (stays `yolo11s-seg.pt` as the safe,
   portable `.pt` fallback). Document the recommended production launch override
   (`model_path:=<abs>/yolo11m-seg.engine inference_size:=736`) and, if a launch
   file is the deploy entrypoint, add a commented/opt-in arg for it.

### Invariants / risks
- Engines are hardware/TensorRT-version specific and non-portable — must be
  re-exported per deployment box; the `.pt` fallback always works.
- Larger model raises GPU memory + per-frame latency; the benchmark gates the
  choice. No tracker logic changes — purely which weights are loaded.
- This phase is benchmark-and-document; it does not block Issues 1–3.

---

## Testing (TDD)
- **Issue 1:** unit-test the gated commit-bar/count/window selection in
  `_confirm_reid_candidate` (SimpleNamespace stub, monkeypatch
  `compute_similarity`): with `in_needs_help=True, num_candidates==1`, a lone
  candidate at sim 0.65 (≥ 0.62) ARMS and COMMITS after 12 hits within a 16-frame
  window (and a sim 0.60 lone candidate does NOT, being below 0.62); with
  `in_needs_help=False` it does NOT commit at 0.65 (stays strict 0.72); with
  `num_candidates>1` it does NOT use the relaxed path. Confirm the commit produces
  a `target_track_id` swap (latch-clearing path).
- **Issue 2:** no test — no code change (pseudo-mask dropped; the real seg mask is
  already used for the OSNet query). `reid/reid.py` stays at its `main` contract,
  so its existing `test_deep_crop_segmentation.py` tests remain valid.
- **Issue 3:** unit-test the admission gate via `crop_quality_ok` with the new
  thresholds (`gallery_max_aspect_ratio=0.5, gallery_min_mask_fill=0.35`): an
  upright clean box (w/h = 0.4, mask_coverage = 0.45) ADMITTED; a square box
  (w/h = 1.0) REJECTED even with good fill (aspect gate); an upright low-fill box
  (mask_coverage = 0.20) REJECTED (mask-fill gate); a None-coverage upright box
  ADMITTED but a None-coverage square box REJECTED (aspect still applies); aspect
  boundary w/h == 0.5 admits / 0.51 rejects (gate is `>`); mask boundary
  mask_coverage == 0.35 rejects / 0.36 admits (gate is `<=`); plus an
  `update_appearance` wiring test that both tracker attrs reach `crop_quality_ok`.
- **Issue 4:** no unit test (operational); manual latency benchmark recorded in
  DEV_NOTES.
- Full suite green; no NEW flake8 in touched lines (pre-existing baseline only).

## Files
- `core/tracking_pipeline.py` — Issue 1 gated commit bar/count/window in
  `_confirm_reid_candidate`.
- `person_track_node.py` — Issue 1 `tracker.in_needs_help = self._help_latched`
  wiring + new param declares/reads (Issues 1 & 3) + `active_help_after_frames`
  default 45→150 (Issue 1 passive window).
- `reid/reid.py` — UNCHANGED (Issue 2 pseudo-mask reverted).
- `reid/appearance_manager.py` — Issue 3 gate dict from tracker-configured
  thresholds.
- `reid/quality.py` — Issue 3 (only if a default needs touching; signature already
  takes the thresholds as kwargs).
- `yolo_tracker.py` — tracker defaults for the new attrs (Issues 1 & 3).
- `scripts/export_yolo_trt.py` / launch docs — Issue 4 (operational).
- `test/…` — new tests for Issues 1 & 3.
- `readme.md` Changelog + `DEV_NOTES.md`.
