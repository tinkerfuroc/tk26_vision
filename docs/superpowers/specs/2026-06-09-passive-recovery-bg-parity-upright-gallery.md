# Passive recovery (post-NEEDS_HELP) + OSNet background parity + upright-only gallery — design

**Date:** 2026-06-09
**Package:** `src/vision_track`
**Branch:** `feat/track-web-idle-video`
**Status:** approved (user, 2026-06-09)

Three changes to `person_track_server`, each root-caused by a read-only
investigation. They compose: background parity (#2) raises the raw ReID operating
point; upright-only gallery (#3) keeps the gallery clean; the post-NEEDS_HELP
relaxation (#1) is the gated escape hatch for the residual dead-band so a
returning operator auto-recovers without a wave.

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

### Fix — relaxed lone-candidate recovery while latched in NEEDS_HELP, matching the wave bar
The wave/reseed reclaim (`_step_reseed_probation`) commits at `sim >=
reid_threshold` (0.55) sustained for `reseed_confirmation_frames` (5) frames and
has been reliable in practice. Mirror that for the **passive** path, but only in
the precision-safe envelope the operator authorized:

- **Gate:** apply ONLY when `tracker.in_needs_help` is True AND
  `num_candidates == 1` (exactly one person visible → no ambiguity).
  `tracker.in_needs_help` is set by the node each iteration from the **latched**
  help state `self._help_latched` (NOT the instantaneous `reacq_state`, which
  would oscillate because the relaxed bar resets `frames_lost`). `_help_latched`
  is set at `frames_lost >= active_help_after_frames` and cleared only on a TRUE
  re-lock, so it is stable across the recovery window.
- **Relaxed bar:** new param `single_person_commit_bar_help` (default **0.55**,
  == `reid_threshold` == the wave bar; "same, if not higher" per operator). In
  `_confirm_reid_candidate`, when the gate holds:
  `commit_bar = single_person_commit_bar_help` and the required confirm count
  `required = needs_help_confirm_frames` (new param, default **5**, mirroring the
  wave's `reseed_confirmation_frames`). Keep the existing N-of-M window mechanism
  (dip tolerance) — i.e. commit when `sum(reid_confirm_window) >= 5` of the last
  M frames cleared `commit_bar`; arming preconfirm stays `reid_preconfirm_frames`
  (3) commit-bar hits. Outside the gate, behavior is UNCHANGED (lone 0.72 / multi
  `reid_threshold`, N = `reid_confirmation_frames`+post_shake).
- **Re-lock must clear the latch:** the relaxed commit must produce a genuine
  `target_track_id` swap → `committed_swap` → `fsm.start(new_id)` + present=True
  step → `feedback.target_lost=False` → `_help_latched` cleared. (This is the
  existing commit mechanism; only the bar/count change.) Do NOT relax in a way
  that only publishes a provisional (target_lost stays True) — that would not
  clear the latch and would leave the same stall.

### Precision invariants
- Multi-person scenes (`num_candidates > 1`) are UNCHANGED — the relaxation never
  applies when more than one person is visible, so a bystander in a crowd cannot
  be locked more easily.
- The relaxation activates only AFTER the human has already been asked for help
  (latched NEEDS_HELP). Normal tracking and the in-window passive reacq
  (`frames_lost < active_help_after_frames`) keep the strict 0.72 lone bar.
- The N-of-M sustained-streak gate is preserved (no single-frame false lock).
- `OTHER_PERSON_MAX_TARGET_SIM` (0.72) is unchanged (lone case doesn't run
  `register_other_persons` anyway).

### New params (Issue 1)
| Param | Default | Meaning |
|---|---|---|
| `single_person_commit_bar_help` | `0.55` | lone-candidate commit bar while latched in NEEDS_HELP (== wave bar) |
| `needs_help_confirm_frames` | `5` | confirm-hits required to commit while latched in NEEDS_HELP (mirrors `reseed_confirmation_frames`) |

---

## Issue 2 — OSNet query/gallery background parity (mask-None fallback)

### Root cause
The query and gallery deep pipelines are structurally identical (both call
`reid/reid.py:_segment_crop_for_reid` with the detection mask, same
resize/normalize/channel order) EXCEPT: `_segment_crop_for_reid` **early-returns
the RAW, full-background crop when the mask is None / empty / all-zero**
(`reid/reid.py:235-239`). The gallery is admitted only from quality-gated,
mask-present frames (always background-neutralized); the query runs every frame,
including hard ones (small / occluded / edge-clipped persons) where the seg mask
is missing → a **background-laden query embedding** is cosine-compared against
neutralized gallery views → background leaks into the deep cosine (intermittent,
biased toward exactly the hard reacquisition frames). This also drags the deep
cosine down ~0.10-0.15, feeding Issue 1's dead-band.

### Fix — pseudo-mask fallback (Policy A)
In `_segment_crop_for_reid`, when the mask is None/empty/all-zero, do NOT return
the raw crop. Instead synthesize a **bbox-inscribed ellipse pseudo-mask** (an
upright person prior: ellipse centered in the crop, axes ~ the crop half-extents)
and run the SAME pipeline (dilate → tight-crop → resize 128×256 → GaussianBlur →
`0.15*fg + 0.85*blur` background attenuation). This keeps the OSNet input in the
same neutralized-background distribution as the gallery even with no real mask.
One edit point; both the gallery (`reid.py:318`) and query (`reid.py:387`) paths
call this function, so parity is enforced everywhere (init, reseed, register,
verify, periodic) with no other call-site change.

### Invariants / risks
- Existing gallery views were built mask-present (neutralized) — a forward fix
  does not invalidate them; no gallery clear needed.
- Keep ALL neutralization at the 128×256 OSNet size (the perf-fix invariant —
  never reintroduce a full-resolution blur). The pseudo-mask synth is
  constant-time; per-candidate CPU budget unaffected.
- The ellipse prior is coarser than a real seg mask: it removes corner/background
  but cannot evict a co-bbox bystander the way a real tight-crop does. It is
  strictly better than raw background for parity, not a substitute for a mask.

---

## Issue 3 — upright-only gallery admission

### Rationale
The operator stands throughout tracking, so every legitimate gallery view should
be an **upright** person. Admitting non-upright crops (sitting/bent fragments,
wide merged boxes, partial detections) pollutes the gallery and degrades
matching. Gate gallery admission on bbox uprightness.

### Fix
In the gallery-admission quality path (`reid/appearance_manager.py`
`update_appearance` / `crop_quality_ok`, ~lines 104-145), add an aspect-ratio
gate: admit a view only if `bbox_height / bbox_width >= gallery_min_aspect_ratio`
(new param, default **1.5** — an upright standing person is typically 2-3; a
sitting/bent/merged box is < 1.5). Reject non-upright crops from gallery
admission (do NOT block tracking — the live target id still tracks via ByteTrack;
only gallery *enrichment* is gated). Apply to all admissions including the anchor
(the standing operator's first clean view is upright).

### New param (Issue 3)
| Param | Default | Meaning |
|---|---|---|
| `gallery_min_aspect_ratio` | `1.5` | min bbox height/width to admit a view into the ReID gallery (upright-person gate) |

### Invariants / risks
- Admission-only gate — never affects the query/matching path or the live lock.
- If the operator's first views are all non-upright (unlikely, they stand), the
  gallery simply admits fewer views until an upright one appears; tracking is
  unaffected. Default 1.5 is permissive enough to admit normal standing views
  with margin while rejecting clearly non-upright boxes.

---

## Testing (TDD)
- **Issue 1:** unit-test the gated commit-bar/count selection in
  `_confirm_reid_candidate`: with `in_needs_help=True, num_candidates==1`, a lone
  candidate at sim 0.60 (dead-band) now ARMS and COMMITS after
  `needs_help_confirm_frames` hits at `single_person_commit_bar_help`; with
  `in_needs_help=False` it does NOT (stays strict 0.72); with `num_candidates>1`
  it does NOT (strict). Confirm the commit produces a `target_track_id` swap
  (latch-clearing path). Reuse the `SimpleNamespace`-stub style.
- **Issue 2:** unit-test `_segment_crop_for_reid` with `mask=None` and empty mask:
  output is 128×256 and background-neutralized (NOT the raw crop), and a maskless
  crop yields the SAME neutralization on the (would-be) gallery and query paths.
- **Issue 3:** unit-test the admission gate: an upright bbox (h/w >= 1.5) is
  admitted; a wide bbox (h/w < 1.5) is rejected; the param is honored.
- Full suite green; no NEW flake8 in touched lines (pre-existing baseline only).

## Files
- `core/tracking_pipeline.py` — Issue 1 gated commit bar/count in
  `_confirm_reid_candidate`.
- `person_track_node.py` — Issue 1 `tracker.in_needs_help = self._help_latched`
  wiring + the three/four new param declares/reads.
- `reid/reid.py` — Issue 2 pseudo-mask fallback in `_segment_crop_for_reid`.
- `reid/appearance_manager.py` — Issue 3 upright admission gate.
- `yolo_tracker.py` — tracker defaults for the new attrs.
- `test/…` — new tests per issue.
- `readme.md` Changelog + `DEV_NOTES.md`.
