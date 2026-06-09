# Passive recovery + bg parity + mask-fill gallery + larger seg model — implementation plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. TDD. Phase = one commit.

**Spec:** `docs/superpowers/specs/2026-06-09-passive-recovery-bg-parity-upright-gallery.md`
**Build:** `bash ./tkbuild tk26_vision --packages-select vision_track` (from `/home/tinker/tk25_ws`)
**Test:** `cd /home/tinker/tk25_ws/src/tk26_vision && source .venv-vision-main/bin/activate && python -m pytest src/vision_track/test/ -q`
**Precision invariants:** no existing threshold changed except behind the new gated/param paths. The Issue-1 relaxation applies ONLY when `in_needs_help` (latched) AND `num_candidates==1`, at bar 0.62 over 12-of-16 frames. Background neutralization stays at 128×256 (perf invariant). The gallery gate is admission-only.

Phases A and B touch DISJOINT files (A: `reid/reid.py`; B: `core/tracking_pipeline.py` + `person_track_node.py` + `yolo_tracker.py`) and may be implemented in parallel. Phase C touches `person_track_node.py` + `yolo_tracker.py` (params/defaults) so it runs AFTER Phase B. Phase D is operational (export + benchmark + docs) and runs last.

---

## Phase A — Issue 2: pseudo-mask fallback in `_segment_crop_for_reid` (one commit)

**Files:** `src/vision_track/vision_track/reid/reid.py` (`_segment_crop_for_reid`). Test: `src/vision_track/test/test_segment_crop_fallback.py` (new). Doc: `readme.md`.

- [ ] **Step 1 — failing test.** `test_segment_crop_fallback.py`: `_segment_crop_for_reid` is pure numpy/cv2 — bind it onto a stub (do NOT construct `AppearanceExtractor`, which loads OSNet): e.g. `from vision_track.reid.reid import AppearanceExtractor` then call `AppearanceExtractor._segment_crop_for_reid(object.__new__(AppearanceExtractor), crop, mask)` OR extract the method's logic if `self` is unused (check — it appears to use no instance state). Construct a 200×100×3 uint8 crop with a distinct bright center and dark corners. Assert: (a) with `mask_crop=None` AND with an all-zero `mask`, the result is shape `(256,128,3)` and is NOT byte-equal to `cv2.resize(crop,(128,256))` (background was neutralized); (b) the center region (person prior) is closer to the resized original than the corner regions (corners blurred toward background); (c) with a real nonzero mask the output still segments to that mask (unchanged path — compare to current behavior). Apache header.
- [ ] **Step 2 — run, verify FAIL** (current code returns the raw crop for mask-None/empty/all-zero).
- [ ] **Step 3 — implement.** In `_segment_crop_for_reid`, replace the early-returns (`if mask_crop is None or mask_crop.size == 0: return crop` and the all-zero `m.sum()==0` guard) with: synthesize a **bbox-inscribed ellipse pseudo-mask** `uint8` array the size of `crop` — `m = np.zeros(crop.shape[:2], np.uint8); cv2.ellipse(m, center=(w//2,h//2), axes=(int(w*0.5), int(h*0.5)), angle=0, startAngle=0, endAngle=360, color=1, thickness=-1)` — then fall through to the SAME dilate→tight-crop→resize(128×256)→GaussianBlur→`0.15*fg+0.85*blur` path. Keep all work at the 128×256 size (no full-res blur). Guard `w>0 and h>0` (degenerate crop → still return a resize, never crash).
- [ ] **Step 4 — run, verify PASS** + full suite (no regressions; no new flake8 in touched lines).
- [ ] **Step 5 — README changelog** (same commit): mask-None/empty query crops are now background-neutralized via an ellipse pseudo-mask, so OSNet input has the same background processing as gallery views (background-independent matching).
- [ ] **Step 6 — build + commit** explicit paths: `src/vision_track/vision_track/reid/reid.py` + `src/vision_track/test/test_segment_crop_fallback.py` + `src/vision_track/readme.md`.
  `feat(vision_track): pseudo-mask fallback for maskless ReID crops (query/gallery bg parity)`

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
- [ ] **Step 5 — run full suite green** (existing passive-reacq/lookalike/lock tests must stay green — the strict path is unchanged when `in_needs_help` is False).
- [ ] **Step 6 — README changelog** (same commit): post-NEEDS_HELP passive recovery — while latched in NEEDS_HELP with exactly one person visible, the lone commit bar relaxes to `single_person_commit_bar_help` (0.62) over `needs_help_confirm_frames`/`needs_help_commit_window` (12-of-16) sustained hits, so a returning operator auto-re-locks without a wave; multi-person + in-window passive stay strict 0.72; N-of-M preserved and strengthened.
- [ ] **Step 7 — build + commit** explicit paths (`core/tracking_pipeline.py`, `person_track_node.py`, `yolo_tracker.py`, the test, `readme.md`).
  `feat(vision_track): relax lone passive recovery (0.62, 12-of-16) while latched in NEEDS_HELP`

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
