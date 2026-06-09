# Passive recovery + bg parity + mask-fill gallery + larger seg model — implementation plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD. Phase = one commit.

**Spec:** `docs/superpowers/specs/2026-06-09-passive-recovery-bg-parity-upright-gallery.md`
**Build:** `bash ./tkbuild tk26_vision --packages-select vision_track` (from `/home/tinker/tk25_ws`)
**Test:** `cd /home/tinker/tk25_ws/src/tk26_vision && source .venv-vision-main/bin/activate && python -m pytest src/vision_track/test/ -q`
**Precision invariants:** no existing threshold changed except behind the new gated/param paths. The Issue-1 relaxation applies ONLY when `in_needs_help` (latched) AND `num_candidates==1`, at bar 0.62 over 12-of-16 frames. Background neutralization stays at 128×256 (perf invariant). The gallery gate is admission-only.

Phase B (Issue 1) lands first; Phase C (Issue 3) follows it (both touch `person_track_node.py` + `yolo_tracker.py`); Phase D (Issue 4) is operational and runs last.

---

## Phase A — Issue 2: DROPPED (no code change)

Pseudo-mask fallback **reverted** per operator review (2026-06-10): the OSNet
query already uses the real seg mask (plumbing verified — `result_parser`
populates `DetectionResult.mask`; every query call site passes it). `reid/reid.py`
is unchanged from `main`. No tests, no params, no commit. Fewer maskless frames
come from the better seg model (Phase D), not a synthesized mask. See spec
§"Issue 2 — RESOLVED, NO CODE CHANGE".

---

## Phase B — Issue 1: relaxed lone-candidate recovery in latched NEEDS_HELP (one commit)

**Files:** `core/tracking_pipeline.py` (`_confirm_reid_candidate`), `person_track_node.py` (set `tracker.in_needs_help`, declare/read params), `yolo_tracker.py` (defaults). Tests: `test/test_needs_help_recovery.py` (new). Doc: `readme.md`.

- [ ] **Step 1 — failing test.** `test_needs_help_recovery.py` (SimpleNamespace stub-tracker like `test/test_lookalike_pursuit.py`; monkeypatch `ReIDMatcher.compute_similarity` — or whatever `_confirm_reid_candidate` calls — to feed a controlled sim). Drive `_confirm_reid_candidate` for a LONE candidate:
  - `in_needs_help=True, num_candidates==1`, sim 0.65 (≥ 0.62) → after 12 hits within a 16-frame window it ARMS (at `reid_preconfirm_frames`=3) and COMMITS (swaps `target_track_id`); assert the swap happened.
  - `in_needs_help=True, num_candidates==1`, sim 0.60 (< 0.62) → never a hit → does NOT commit.
  - `in_needs_help=False, num_candidates==1`, sim 0.65 → does NOT commit (strict 0.72 bar; `target_track_id` unchanged).
  - `in_needs_help=True, num_candidates==2`, sim 0.65 → does NOT use the relaxed bar (multi stays `reid_threshold` path; assert no relaxed commit at a sub-0.72 sim).
  Stub attrs: `in_needs_help`, `single_person_commit_bar_help=0.62`, `needs_help_confirm_frames=12`, `needs_help_commit_window=16`, plus the existing Phase-3 attrs (`single_person_commit_bar=0.72`, `reid_threshold=0.55`, `reid_preconfirm_frames=3`, `reid_confirmation_frames`, `provisional_commit_window=18`, `reid_confirm_window=[]`, ...).
- [ ] **Step 2 — run, verify FAIL.**
- [ ] **Step 3 — implement.** In `_confirm_reid_candidate` (`tracking_pipeline.py`), BEFORE the `window_m`/`_push_window` definitions, compute the gate and override the three locals:
  ```python
  in_help = bool(getattr(tracker, "in_needs_help", False)) and num_candidates == 1
  if in_help:
      commit_bar = getattr(tracker, "single_person_commit_bar_help", 0.62)
      required_confirmation = int(getattr(tracker, "needs_help_confirm_frames", 12))
      window_m = int(getattr(tracker, "needs_help_commit_window", 16))
  else:
      required_confirmation = tracker.reid_confirmation_frames + post_shake_extra
      commit_bar = (getattr(tracker, "single_person_commit_bar", 0.72)
                    if num_candidates == 1 else tracker.reid_threshold)
      window_m = getattr(tracker, "provisional_commit_window", 18)
  is_hit = match_similarity >= commit_bar
  ```
  (Replace the existing `required_confirmation`/`commit_bar`/`is_hit`/`window_m` assignments at ~626–634 with this block so `window_m` is set before `_push_window` closes over it. Keep the N-of-M window, the arming `sum(window) >= reid_preconfirm_frames`, and the commit `sum(window) >= required_confirmation` EXACTLY as Phase 3 — only the three locals change in the gate.) Everything downstream (commit → swap → caller's `committed_swap` → `fsm.start` present=True) is unchanged so the help-latch clears on the real re-lock.
- [ ] **Step 4 — node wiring + params.** In `person_track_node.py`: `declare_parameter('single_person_commit_bar_help', 0.62)`, `declare_parameter('needs_help_confirm_frames', 12)`, `declare_parameter('needs_help_commit_window', 16)`; read them into `self.*` and set on `self.tracker` next to where `single_person_commit_bar` / `provisional_commit_window` are set (~447–448). Set `self.tracker.in_needs_help = self._help_latched` once per tracking iteration BEFORE the tracker's `update(...)` is called (the latched flag is stable so no oscillation). Add tracker defaults in `yolo_tracker.py` (`in_needs_help=False`, `single_person_commit_bar_help=0.62`, `needs_help_confirm_frames=12`, `needs_help_commit_window=16`).
- [ ] **Step 4b — WALL-CLOCK passive-recovery window (~5 s, NOT frames).** Operator constraint: frame counts are unreliable in a tournament. Convert NEEDS_HELP escalation to wall-clock: (i) `core/reacq_state.py` → `reacq_state(tracked, time_since_lost, help_after_sec)` (NEEDS_HELP when `time_since_lost >= help_after_sec`; `<=0` immediate; update docstring); (ii) `person_track_node.py` — replace param `active_help_after_frames` with `active_help_after_sec` (default 5.0); add `self._last_confirmed_time` anchor (init at tracking start, reset per goal, refresh only on `feedback.target_lost == False`); pass `time_since_lost = time.time() - self._last_confirmed_time` to both `reacq_state` telemetry calls (~1254, ~1421) and to `_is_awaiting_help`; change `_is_awaiting_help` to latch at `time_since_lost >= active_help_after_sec` and disable at `active_help_after_sec <= 0` (keep the time-based `active_help_timeout_sec` bound; keep latch-clear-on-true-relock); update the warn log to seconds; (iii) `core/debug_state.py` — rename field `active_help_after_frames`→`active_help_after_sec` (float) and node call sites (~1529, ~1586); (iv) grep `track_web`/dashboard assets for `active_help_after_frames` and update any display to seconds. Update tests: `test/test_reacq_state.py`, `test/test_active_help_hold.py`, `test/test_debug_state.py`, `test/test_active_reid_interfaces.py` (any that reference the old name).
- [ ] **Step 5 — run full suite green** (existing passive-reacq/lookalike/lock tests must stay green — the strict path is unchanged when `in_needs_help` is False).
- [ ] **Step 6 — README changelog** (same commit): (a) post-NEEDS_HELP passive recovery — while latched in NEEDS_HELP with exactly one person visible, the lone commit bar relaxes to `single_person_commit_bar_help` (0.62) over `needs_help_confirm_frames`/`needs_help_commit_window` (12-of-16) sustained hits, so a returning operator auto-re-locks without a wave; multi-person + in-window passive stay strict 0.72; N-of-M preserved and strengthened. (b) passive-recovery window extended to ~5 s (`active_help_after_frames` 45→150) before escalating to NEEDS_HELP.
- [ ] **Step 7 — build + commit** explicit paths (`core/tracking_pipeline.py`, `person_track_node.py`, `yolo_tracker.py`, the test, `readme.md`).
  `feat(vision_track): relax lone passive recovery (0.62, 12-of-16) + extend passive window to 5s`

---

## Phase C — Issue 3: mask-fill gallery admission gate (one commit; AFTER Phase B)

**Files:** `reid/appearance_manager.py` (gate dict from tracker attrs), `person_track_node.py` (params), `yolo_tracker.py` (defaults). Test: `test/test_gallery_mask_fill_gate.py` (new). Doc: `readme.md`. (`reid/quality.py` needs NO change — `crop_quality_ok` already takes `min_mask_coverage`/`max_aspect_ratio` as kwargs.)

- [ ] **Step 1 — failing test.** `test_gallery_mask_fill_gate.py`: import `crop_quality_ok` from `vision_track.reid.quality`. With `min_crop_h=80, min_blur_var=50.0` fixed and `min_mask_coverage=0.35, max_aspect_ratio=2.0` (the new defaults):
  - square-but-clean: `crop_h=200, crop_w=200, mask_coverage=0.45, blur_var=100, aspect_ratio=1.0` → True (REGRESSION GUARD: under the old `max_aspect_ratio=0.9` this was False).
  - low fill: `mask_coverage=0.20` (others clean) → False.
  - None coverage: `mask_coverage=None` (others clean) → True (not rejected on fill when no mask).
  - boundary: `mask_coverage=0.35` → False (gate is `<=`); `mask_coverage=0.36` → True.
  Also add a test that `update_appearance` builds the gate from tracker attrs: a SimpleNamespace tracker with `gallery_min_mask_fill=0.35`, `gallery_max_aspect_ratio=2.0` results in those values reaching `crop_quality_ok` (monkeypatch `crop_quality_ok` in the appearance_manager module to capture kwargs, feed a stub `appearance_extractor.extract_features` returning a `mask_coverage` feature). Apache header.
- [ ] **Step 2 — run, verify FAIL** (the `update_appearance` capture test fails: it currently passes `**DEFAULT_GATE`, i.e. 0.4 / 0.9, not the tracker values).
- [ ] **Step 3 — implement.** In `reid/appearance_manager.py` `update_appearance`, replace `**DEFAULT_GATE` in the `crop_quality_ok(...)` call with a gate dict assembled from tracker attrs, falling back to `DEFAULT_GATE` for untouched keys:
  ```python
  gate = dict(DEFAULT_GATE)
  gate["min_mask_coverage"] = float(getattr(tracker, "gallery_min_mask_fill", DEFAULT_GATE["min_mask_coverage"]))
  gate["max_aspect_ratio"] = float(getattr(tracker, "gallery_max_aspect_ratio", DEFAULT_GATE["max_aspect_ratio"]))
  ... crop_quality_ok(crop_h=..., crop_w=..., mask_coverage=..., blur_var=..., aspect_ratio=..., **gate)
  ```
  Do NOT change `crop_quality_ok` or `DEFAULT_GATE` defaults (other call sites/tests rely on them; the new behavior comes from the per-call override).
- [ ] **Step 4 — node params + defaults.** In `person_track_node.py`: `declare_parameter('gallery_min_mask_fill', 0.35)`, `declare_parameter('gallery_max_aspect_ratio', 2.0)`; read into `self.*` and set on `self.tracker`. Add defaults in `yolo_tracker.py` (`gallery_min_mask_fill=0.35`, `gallery_max_aspect_ratio=2.0`).
- [ ] **Step 5 — run full suite green** (existing quality-gate tests must stay green — they call `crop_quality_ok` directly with explicit kwargs, unaffected).
- [ ] **Step 6 — README changelog** (same commit): gallery admission now gates on mask-fill (`gallery_min_mask_fill`, default 0.35 = mask_pixels/bbox_area) instead of bbox aspect ratio (`gallery_max_aspect_ratio` relaxed to 2.0) — a square-but-clean upright operator in a crowd is no longer rejected for box shape, while merged/garbage boxes (low fill) still are. Admission-only; never affects the live lock or matching. Both configurable at launch.
- [ ] **Step 7 — build + commit** explicit paths (`reid/appearance_manager.py`, `person_track_node.py`, `yolo_tracker.py`, the test, `readme.md`).
  `feat(vision_track): gate ReID gallery on mask-fill, not bbox aspect ratio`

---

## Phase D — Issue 4: larger YOLO-seg model + TensorRT engine (operational; AFTER C)

**Files:** `scripts/export_yolo_trt.py` (already exists — use as-is), `DEV_NOTES.md` + `readme.md` (benchmark + recommended launch), optionally a launch arg. No tracker logic change.

- [ ] **Step 1 — provision + export.** Confirm `tensorrt` is importable in the export venv (`.venv-vision-main` does NOT have it per the script docstring; check `.venv-fs`, else `pip install` a TensorRT build matching the RTX 5070 Ti / CUDA on the box). Then: `python src/vision_track/scripts/export_yolo_trt.py --model yolo11m-seg.pt --imgsz 736 --out <abs>/yolo11m-seg.engine`. (If TRT cannot be provisioned here, record that in DEV_NOTES and leave the recommended-launch doc + the `.pt` fallback; do not block.)
- [ ] **Step 2 — benchmark.** Run the tracker with the engine (`-p model_path:=<abs>/yolo11m-seg.engine -p inference_size:=736`) on live cameras for ~2 min; capture YOLO inference ms + end-to-end Hz from `[perf]` logs vs the current `yolo11s-seg.pt`. Confirm ≥ ~25–30 Hz and within the 10 s/call budget. If `m` is too slow, fall back to an `s`-seg engine (still a TRT win); if ample headroom, optionally try `l`-seg.
- [ ] **Step 3 — document, do NOT change the default.** Keep `model_path` default `yolo11s-seg.pt`. Add the recommended production launch override to `readme.md` (and a commented opt-in launch arg if the launch file is the deploy entrypoint). Record the benchmark numbers + chosen model in `DEV_NOTES.md`.
- [ ] **Step 4 — commit** explicit paths.
  `docs(vision_track): benchmark + recommend yolo11m-seg TensorRT engine for production`

---

## Final
- [ ] Full suite once more; DEV_NOTES entry (2026-06-09) summarizing all four + operator checks (returns auto-recover post-NEEDS_HELP without a wave at 0.62/12-of-16; matching robust to background; gallery gated on mask-fill not aspect; larger seg model benchmarked). Commit `docs(tk26_vision): DEV_NOTES — passive recovery + bg parity + mask-fill gallery + seg model`.
