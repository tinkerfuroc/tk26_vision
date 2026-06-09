# Passive recovery + bg parity + upright gallery — implementation plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD. Phase = one commit.

**Spec:** `docs/superpowers/specs/2026-06-09-passive-recovery-bg-parity-upright-gallery.md`
**Build:** `bash ./tkbuild tk26_vision --packages-select vision_track`
**Test:** `cd /home/tinker/tk25_ws/src/tk26_vision && source .venv-vision-main/bin/activate && python -m pytest src/vision_track/test/ -q`
**Precision invariants:** no existing threshold changed except behind the new gated/param paths. The Issue-1 relaxation applies ONLY when `in_needs_help` (latched) AND `num_candidates==1`. Background neutralization stays at 128×256 (perf invariant). Gallery upright gate is admission-only.

Phases A and B touch DISJOINT files (A: `reid/reid.py`; B: `core/tracking_pipeline.py` + `person_track_node.py`) and may be implemented in parallel. Phase C touches `person_track_node.py` (params) so it runs AFTER Phase B.

---

## Phase A — Issue 2: pseudo-mask fallback in `_segment_crop_for_reid` (one commit)

**Files:** `src/vision_track/vision_track/reid/reid.py` (`_segment_crop_for_reid` ~207-258). Test: `src/vision_track/test/test_segment_crop_fallback.py` (new). Doc: `readme.md`.

- [ ] **Step 1 — failing test.** `test_segment_crop_fallback.py`: construct an `AppearanceExtractor` (or call the method on a minimal instance if construction is heavy — prefer constructing it; if it loads OSNet, instead bind `_segment_crop_for_reid` onto a stub since it's pure numpy/cv2). Assert: with `mask_crop=None` and with an all-zero mask, `_segment_crop_for_reid(crop, mask)` returns a `(256,128,3)` array whose background is neutralized (NOT byte-equal to a plain resize of the raw crop) — e.g. compare against the raw-resize and assert they differ, and assert the center (person-prior) region is closer to the original than the corners. With a real (nonzero) mask, behavior unchanged (still segments to mask). Apache header.
- [ ] **Step 2 — run, verify FAIL** (current code returns the raw crop for mask-None).
- [ ] **Step 3 — implement.** In `_segment_crop_for_reid`, replace the mask-None/empty early-returns (`reid.py:235-239`) with a synthesized **bbox-inscribed ellipse pseudo-mask** the size of `crop` (ellipse centered, axes ~0.5*w and ~0.5*h, an upright-person prior), then fall through to the SAME dilate→tight-crop→resize(128×256)→GaussianBlur→`0.15*fg+0.85*blur` path used for a real mask. Keep all work at the 128×256 size (no full-res blur). Use `cv2.ellipse` on a zeros `uint8` array.
- [ ] **Step 4 — run, verify PASS** + full suite (no regressions; no new flake8 in touched lines).
- [ ] **Step 5 — README changelog** (same commit): mask-None query crops are now background-neutralized via an ellipse pseudo-mask, so OSNet input has the same background processing as gallery views (background-independent matching).
- [ ] **Step 6 — build + commit** explicit paths: `reid/reid.py` + the test + `readme.md`.
  `feat(vision_track): pseudo-mask fallback for maskless ReID crops (query/gallery bg parity)`

---

## Phase B — Issue 1: relaxed lone-candidate recovery in latched NEEDS_HELP (one commit)

**Files:** `core/tracking_pipeline.py` (`_confirm_reid_candidate`), `person_track_node.py` (set `tracker.in_needs_help`, declare/read params), `yolo_tracker.py` (defaults). Tests: `test/test_needs_help_recovery.py` (new). Doc: `readme.md`.

- [ ] **Step 1 — failing test.** `test_needs_help_recovery.py` (SimpleNamespace stub-tracker like `test/test_lookalike_pursuit.py`; monkeypatch `ReIDMatcher.compute_similarity` to feed a controlled sim). Drive `_confirm_reid_candidate` (and/or `reidentify_target`) for a LONE candidate at sim 0.60:
  - `in_needs_help=True, num_candidates==1` → after `needs_help_confirm_frames` (5) frames it ARMS and COMMITS (swaps `target_track_id`); assert the swap happened.
  - `in_needs_help=False, num_candidates==1` → does NOT commit at 0.60 (strict 0.72 bar; `target_track_id` unchanged).
  - `in_needs_help=True, num_candidates==2` → does NOT use the relaxed bar (multi stays at `reid_threshold` path; assert no relaxed commit at a sub-threshold sim).
  Stub attrs: `in_needs_help`, `single_person_commit_bar_help=0.55`, `needs_help_confirm_frames=5`, plus the existing Phase-3 attrs.
- [ ] **Step 2 — run, verify FAIL.**
- [ ] **Step 3 — implement.** In `_confirm_reid_candidate` (`tracking_pipeline.py`):
  ```python
  in_help = bool(getattr(tracker, 'in_needs_help', False)) and num_candidates == 1
  if in_help:
      commit_bar = getattr(tracker, 'single_person_commit_bar_help', 0.55)
      required_confirmation = getattr(tracker, 'needs_help_confirm_frames', 5)
  else:
      commit_bar = (tracker.single_person_commit_bar if num_candidates == 1
                    else tracker.reid_threshold)
      required_confirmation = tracker.reid_confirmation_frames + post_shake_extra
  ```
  (Keep `is_hit = match_similarity >= commit_bar`, the N-of-M window, and the arming `sum(window) >= reid_preconfirm_frames` exactly as Phase 3 — only `commit_bar`/`required_confirmation` are overridden in the gate.) Everything downstream (commit → swap → caller's `committed_swap` → `fsm.start` present=True) is unchanged so the help-latch clears on the real re-lock.
- [ ] **Step 4 — node wiring + params.** In `person_track_node.py`: declare/read `single_person_commit_bar_help` (0.55) and `needs_help_confirm_frames` (5); set them on the tracker. Set `self.tracker.in_needs_help = self._help_latched` once per tracking iteration BEFORE `self.tracker.update(...)` is called (find the main loop update site; the latched flag is stable so no oscillation). Add tracker defaults in `yolo_tracker.py` (`in_needs_help=False`, `single_person_commit_bar_help=0.55`, `needs_help_confirm_frames=5`).
- [ ] **Step 5 — run full suite green** (existing passive-reacq/lookalike/lock tests must stay green — the strict path is unchanged when `in_needs_help` is False).
- [ ] **Step 6 — README changelog** (same commit): post-NEEDS_HELP passive recovery — while latched in NEEDS_HELP with exactly one person visible, the lone commit bar relaxes to `single_person_commit_bar_help` (0.55, == wave bar) over `needs_help_confirm_frames` (5) sustained hits, so a returning operator auto-re-locks without a wave; multi-person + in-window passive stay strict; N-of-M preserved.
- [ ] **Step 7 — build + commit** explicit paths.
  `feat(vision_track): relax lone passive recovery to the wave bar while latched in NEEDS_HELP`

---

## Phase C — Issue 3: upright-only gallery admission (one commit; AFTER Phase B)

**Files:** `reid/appearance_manager.py` (admission gate), `person_track_node.py` (param), `yolo_tracker.py` (default). Test: `test/test_gallery_upright_gate.py` (new). Doc: `readme.md`.

- [ ] **Step 1 — failing test.** `test_gallery_upright_gate.py`: test the uprightness gate (extract a small pure helper, e.g. `_is_upright(bbox, min_aspect)`, or test `update_appearance`/`crop_quality_ok` admission with a stub). Assert: bbox h/w = 2.5 with `min_aspect=1.5` → admit (True); h/w = 1.0 → reject (False); boundary h/w == 1.5 → admit (>=). Apache header.
- [ ] **Step 2 — run, verify FAIL.**
- [ ] **Step 3 — implement.** In `reid/appearance_manager.py` add the uprightness check to the gallery-admission path (`update_appearance`, before `_update_feature_history`/`maybe_add`, alongside `crop_quality_ok` ~104-145): compute `h = y2-y1`, `w = x2-x1`; if `w <= 0 or (h / w) < tracker.gallery_min_aspect_ratio`: skip admission (do not enrich the gallery this frame). Tracking/lock unaffected. Prefer a small pure helper `_is_upright(bbox, min_aspect)` for testability.
- [ ] **Step 4 — node param + default.** Declare/read `gallery_min_aspect_ratio` (1.5) in `person_track_node.py`, set on tracker; default `1.5` in `yolo_tracker.py`.
- [ ] **Step 5 — run full suite green.**
- [ ] **Step 6 — README changelog** (same commit): gallery now admits only upright crops (bbox h/w >= `gallery_min_aspect_ratio`, default 1.5) since the operator stands throughout — keeps the multi-view gallery clean; admission-only, never affects the live lock or matching.
- [ ] **Step 7 — build + commit** explicit paths.
  `feat(vision_track): admit only upright crops into the ReID gallery`

---

## Final
- [ ] Full suite once more; DEV_NOTES entry (2026-06-09) summarizing all three + operator checks (returns auto-recover post-NEEDS_HELP without a wave; matching robust to background; gallery upright-only). Commit `docs(tk26_vision): DEV_NOTES — passive recovery + bg parity + upright gallery`.
