# Passive recovery (post-NEEDS_HELP) + OSNet background parity + mask-fill gallery gate — design

**Date:** 2026-06-09
**Package:** `src/vision_track`
**Branch:** `feat/track-web-idle-video`
**Status:** approved (user, 2026-06-09 — numbers revised after first draft)

Four changes to `person_track_server`. The first three are root-caused tracker
fixes; the fourth is the model-quality upgrade they all benefit from. They
compose: background parity (#2) and a larger seg model (#4) raise the raw ReID
operating point and produce better/more-frequent masks; the mask-fill gallery
gate (#3) keeps the gallery clean without rejecting square-but-upright crowd
views; the post-NEEDS_HELP relaxation (#1) is the gated escape hatch that catches
a returning operator once #2/#4 have lifted their similarity above the relaxed
bar.

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

### New params (Issue 1)
| Param | Default | Meaning |
|---|---|---|
| `single_person_commit_bar_help` | `0.62` | lone-candidate commit bar while latched in NEEDS_HELP |
| `needs_help_confirm_frames` | `12` | confirm-hits required (N) to commit while latched in NEEDS_HELP |
| `needs_help_commit_window` | `16` | N-of-M window length (M) for the help commit |

---

## Issue 2 — OSNet query/gallery background parity (mask-None fallback)

### Root cause
The query and gallery deep pipelines are structurally identical (both call
`reid/reid.py:_segment_crop_for_reid` with the detection mask, same
resize/normalize/channel order) EXCEPT: `_segment_crop_for_reid` **early-returns
the RAW, full-background crop when the mask is None / empty / all-zero**
(`reid/reid.py`, the `if mask_crop is None or mask_crop.size == 0: return crop`
guard plus the all-zero guard). The gallery is admitted only from quality-gated,
mask-present frames (always background-neutralized); the query runs every frame,
including hard ones (small / occluded / edge-clipped persons) where the seg mask
is missing → a **background-laden query embedding** is cosine-compared against
neutralized gallery views → background leaks into the deep cosine (intermittent,
biased toward exactly the hard reacquisition frames). This also drags the deep
cosine down ~0.10–0.15, feeding Issue 1's dead-band.

### Fix — pseudo-mask fallback (Policy A)
In `_segment_crop_for_reid`, when the mask is None/empty/all-zero, do NOT return
the raw crop. Instead synthesize a **bbox-inscribed ellipse pseudo-mask** (an
upright-person prior: ellipse centered in the crop, semi-axes ≈ the crop
half-extents) and run the SAME pipeline (dilate → tight-crop → resize 128×256 →
GaussianBlur → `0.15*fg + 0.85*blur` background attenuation). This keeps the
OSNet input in the same neutralized-background distribution as the gallery even
with no real mask. One edit point; both the gallery and query paths call this
function, so parity is enforced everywhere (init, reseed, register, verify,
periodic) with no other call-site change.

### Invariants / risks
- Existing gallery views were built mask-present (neutralized) — a forward fix
  does not invalidate them; no gallery clear needed.
- Keep ALL neutralization at the 128×256 OSNet size (the perf-fix invariant —
  never reintroduce a full-resolution blur). The pseudo-mask synth is
  constant-time; per-candidate CPU budget unaffected.
- The ellipse prior is coarser than a real seg mask: it removes corner/background
  but cannot evict a co-bbox bystander the way a real tight-crop does. It is
  strictly better than raw background for parity, not a substitute for a mask.
- This is independent of the `mask_coverage` feature used by Issue 3 (that is
  computed from the bbox mask in `extract_features`, not from
  `_segment_crop_for_reid`); the pseudo-mask only affects the OSNet deep crop.

---

## Issue 3 — mask-fill gallery admission gate (not bbox aspect ratio)

### Root cause / rationale
The operator stands throughout, so we want only clean, well-segmented operator
views in the ReID gallery. The **existing** gate (`reid/quality.py:crop_quality_ok`,
fed by `reid/appearance_manager.py:update_appearance` via `DEFAULT_GATE`) already
rejects on `min_mask_coverage` (0.4) AND `max_aspect_ratio` (w/h ≤ 0.9). The
**aspect-ratio gate is wrong for crowds**: occlusion / box-clipping makes a
genuinely upright operator's bbox square-ish (w/h ≈ 1.0 > 0.9) → such clean views
are *rejected*, starving the gallery exactly when reacquisition is hardest. Pose
uprightness is guaranteed by operator behaviour, not by bbox shape, so it should
not be inferred from bbox shape.

The robust signal is **how much of the bbox the mask fills**
(`mask_coverage = mask_pixels / bbox_area`, already computed as `area_ratio` in
`extract_features`): a merged/garbage/occlusion-inflated box has the operator's
mask as a thin slice (low fill → reject); a clean view — including a
square-but-clean crowd view — fills its box well (admit).

### Fix
- **Parameterize the mask-fill floor:** new launch param `gallery_min_mask_fill`
  (default **0.35**, configurable at launch). It overrides `min_mask_coverage` in
  the gate. (0.35 is slightly below the old 0.4 — a clean upright silhouette fills
  ~0.35–0.55 of its tight bbox, so 0.35 keeps margin while still rejecting
  merged/garbage boxes whose target mask is a thin slice.)
- **Neutralize the aspect-ratio rejection:** make `max_aspect_ratio` a launch
  param `gallery_max_aspect_ratio` with a **permissive default (2.0)** so
  square-but-upright crowd boxes are admitted. Truly degenerate wide boxes
  (merged neighbours) are still caught by the mask-fill floor (a wide merged box
  has low target-mask coverage), so mask-fill subsumes the old aspect rejection.
- Wire both into the gate dict built in `update_appearance` (read from tracker
  attrs, falling back to `DEFAULT_GATE` for the untouched `min_crop_h` /
  `min_blur_var`). `crop_quality_ok` already accepts these as kwargs — no change
  to its signature. `min_mask_coverage` is still skipped when `mask_coverage` is
  None (no mask that frame), so a maskless frame is never rejected on fill alone.

### New params (Issue 3)
| Param | Default | Meaning |
|---|---|---|
| `gallery_min_mask_fill` | `0.35` | min mask_pixels/bbox_area to admit a view into the ReID gallery (clean-detection gate; configurable at launch) |
| `gallery_max_aspect_ratio` | `2.0` | max bbox w/h to admit (permissive — degenerate-box backstop; mask-fill is the primary signal) |

### Invariants / risks
- Admission-only gate — never affects the query/matching path or the live lock.
  A rejected frame still refreshes motion/last-seen (existing behaviour kept).
- `mask_coverage` is `None` when no mask is available → not rejected on fill that
  frame (unchanged). The gate only enriches the gallery; tracking continues via
  ByteTrack regardless.
- Loosening the aspect gate cannot admit garbage: the mask-fill floor (now the
  controlling signal) rejects the wide/merged boxes the aspect gate used to.

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
- **Issue 2:** unit-test `_segment_crop_for_reid` with `mask=None` and an all-zero
  mask: output is 128×256 (=(256,128,3)) and background-neutralized (NOT byte-equal
  to a plain resize of the raw crop), with the center closer to the original than
  the corners; a real nonzero mask is unchanged (still segments to the mask).
- **Issue 3:** unit-test the admission gate via `crop_quality_ok` with the new
  thresholds: a square-but-clean box (w/h = 1.0, mask_coverage = 0.45) is ADMITTED
  with `gallery_max_aspect_ratio=2.0, gallery_min_mask_fill=0.35` (regression vs
  the old 0.9 aspect reject); a low-fill box (mask_coverage = 0.20) is REJECTED; a
  None mask_coverage is not rejected on fill; the boundary mask_coverage == 0.35
  is rejected (gate is strict `<=`), 0.36 admitted.
- **Issue 4:** no unit test (operational); manual latency benchmark recorded in
  DEV_NOTES.
- Full suite green; no NEW flake8 in touched lines (pre-existing baseline only).

## Files
- `core/tracking_pipeline.py` — Issue 1 gated commit bar/count/window in
  `_confirm_reid_candidate`.
- `person_track_node.py` — Issue 1 `tracker.in_needs_help = self._help_latched`
  wiring + the new param declares/reads (Issues 1 & 3).
- `reid/reid.py` — Issue 2 pseudo-mask fallback in `_segment_crop_for_reid`.
- `reid/appearance_manager.py` — Issue 3 gate dict from tracker-configured
  thresholds.
- `reid/quality.py` — Issue 3 (only if a default needs touching; signature already
  takes the thresholds as kwargs).
- `yolo_tracker.py` — tracker defaults for the new attrs (Issues 1 & 3).
- `scripts/export_yolo_trt.py` / launch docs — Issue 4 (operational).
- `test/…` — new tests per issue.
- `readme.md` Changelog + `DEV_NOTES.md`.
