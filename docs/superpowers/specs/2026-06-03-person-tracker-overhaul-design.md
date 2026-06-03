# Person-tracker accuracy + latency overhaul — design

- **Date:** 2026-06-03
- **Status:** Approved (design); plans pending
- **Scope:** `src/vision_track` (the `/track_person` action) + `benchmarks/person_tracker` (ptbench)
- **Related:** `benchmarks/person_tracker/PLAN.md` (benchmark contract), memory `vision-track-tracker-gap-analysis`, `person-tracker-benchmark-strategy`

## 1. Problem

A static, four-analyst evaluation of the `vision_track` person tracker against the ptbench
arena gates predicts an **overall FAIL**, dominated by `wrong_lock_episodes`. Two findings
were verified directly in-code; the rest are high-confidence static predictions pending real
Orbbec arena recordings.

### 1.1 Predicted scorecard (no arena bags yet)

| Gate | PASS bar | Predicted | Dominant driver |
|---|---|---|---|
| `correct_lock_rate` | ≥0.92 | WARN→FAIL | back-to-camera trips the raw-cosine floor; partial-occlusion frames dropped by the `conf=0.5` prefilter; lookalike hard-rejects drop the true operator |
| `wrong_lock_episodes` | ==0 | **FAIL (highest risk)** | untrained ReID head → identity decided by color + spatial proximity → locks crossers/lookalikes |
| `reacquire_latency_s` | ≤1.0 | WARN→FAIL | re-entry gets a new ByteTrack ID; 12-frame confirm + 0.72 single-person floor before recovery commits |
| `pos_error_lateral_m` | ≤0.25 | WARN (occlusion FAIL-risk) | `mean` over x/y + silent bbox fallback — and the number is currently untrustworthy (§1.2 #4) |
| `false_target_rate` | ≤0.05 | WARN→FAIL-risk | `allow_indefinite_recovery` never declares hard-lost; provisional matches published while operator absent |
| `throughput_hz` | ≥12 | WARN | 15 Hz self-cap; multi-person ReID storms dip to 8–12 Hz even on the RTX 5070 Ti |

### 1.2 Six root causes (deduped across analysts; ★ verified in-code)

1. ★ **The deep ReID embedding is untrained.** `reid.py:53-76` constructs `channel_attention`,
   `bottleneck`, and four `part_bottlenecks` as fresh `torch.nn` modules; there is **no
   `load_state_dict`/`torch.load`/checkpoint anywhere in `reid/`**. Only the ResNet50 *backbone*
   is ImageNet-pretrained. The 0.55-weighted "deep" term is therefore a random projection of
   ImageNet features — it retains some signal but is far weaker than a ReID-trained head, so the
   system leans on color histograms (0.45 combined) and **spatial proximity**, which is exactly
   what locks onto the person crossing between robot and operator. This is the master cause.
2. ★ **Operator identity is never specified.** The node locks `results[0]` (whatever person YOLO
   lists first) at goal start (`yolo_tracker.py:385-389`, called from `person_track_node.py:622`);
   `TrackPerson.action` has no select field. A bystander enumerated first corrupts every gate from
   frame 0 — and the offline benchmark inherits the nondeterminism.
3. **ByteTrack low-confidence recovery is disabled.** YOLO is called with `conf=0.5`
   (`yolo_tracker.py:298`), stripping boxes before ByteTrack runs, so its two-stage association
   (`track_low_thresh=0.1`) operates on an empty bin. A partially-occluded operator (conf
   0.3–0.45) is dropped → new ID on re-entry → inflated reacquire + lost correct-locks. No project
   `bytetrack.yaml` exists (stock Ultralytics config in use).
4. **The benchmark itself has a blocking fidelity defect.** Offline GT centroids are computed
   bbox-only (`labeler/label_io.py:138`, `mask=None`) while predictions are mask-filtered
   (`replay/runner.py:120-125`) — same function, divergent args. The measured lateral "error" is
   partly a mask-vs-bbox artifact, not tracker error. Also: the offline runner does not replicate
   the deployed tracker config; `pos_error_range_m` is computed but **ungraded**
   (`scoreboard.py` `GateConfig:20-30`); offline scores every frame while the live node drops
   frames under load (`score_cli.py:66` defaults to the offline backend).
5. **Throughput is gated by a self-throttle and ReID storms, not YOLO.** A hard `tracking_rate=15`
   (`person_track_node.py:137,539,576-578`) caps the node below the 30 Hz camera; the real cliff is
   5–7 redundant FP32 ResNet50 passes/frame during multi-person re-ID (the same crop is re-embedded
   up to 4×/frame).
6. **Latent crashers/hazards (verified):** `reid_mode:='native'` → `ImportError`
   (`person_track_node.py:195`; `track_yolo_native.py` absent); `/target_points` goes stale during
   loss (publish gated on `not target_lost` at `person_track_node.py:680-685`, never republished —
   a hazard for real nav consumers, though ptbench reads the feedback flag so it is scored
   correctly); `max_time_lost` hardcodes `frame_rate=30` (`byte_tracker.py:279`) → ~2 s buffer at
   15 Hz; target velocity uses wall-clock `dt` not frame stamps (`yolo_tracker.py:672-689`).

## 2. Benchmark gates (reference)

The ptbench gate thresholds the overhaul targets (from `ptbench/common/scoreboard.py`):
`correct_lock_rate` PASS ≥0.92 / WARN ≥0.85; `wrong_lock_episodes` PASS iff ==0;
`reacquire_latency_s` (median) PASS ≤1.0 / WARN ≤2.0; `pos_error_lateral_m` (median) PASS ≤0.25 /
WARN ≤0.40; `false_target_rate` PASS ≤0.05 / WARN ≤0.10; `throughput_hz` PASS ≥12.0 / WARN ≥8.0.
Scenarios: `cml_crossing`, `occlusion_reentry`, `lookalike_distractors`, `back_to_camera`,
`range_lighting`.

## 3. Locked design decisions

1. **Operator init — heuristic only.** At goal start, among class-`person` detections pick the
   candidate maximizing a combined *centeredness* (bbox-center proximity to image center) and
   *nearness* (smaller median depth) score, tie-broken by detection confidence. No `.action`
   change; works with the existing (partly-broken) BT callers. Assumes the operator starts roughly
   centered/near — true for "follow me" framing. (`yolo_tracker.py:385-389`)
2. **ReID backbone — OSNet-AIN, parametrized.** New ROS param `reid_backbone`, default
   `osnet_ain_x1_0`, switchable to `osnet_x0_25`. Adds `torchreid` as a `.venv-vision-main`
   dependency with a cached weight resolver. The embedding-dim change is absorbed by the existing
   `appearance_manager` history-clear-on-dim-mismatch path.
3. **Benchmark GT — dual, gate on field.** Each GT frame carries two centroids: `centroid_field`
   (seg mask + robust median = best available estimate) and `centroid_track` (node-identical math).
   The gate scores against `centroid_field`; `centroid_track` is reported as a diagnostic only.
   Schema version bumps `1.0 → 1.1` with a back-compatible loader.

## 4. Deliverable structure

- **This design spec** (the rationale + decisions + per-phase contracts).
- **Four implementation plans** under `docs/superpowers/plans/`, one per phase, written next via
  the writing-plans skill. Each phase is independently mergeable.
- **Branch:** `feat/person-tracker-overhaul` off `dev`; TDD per change; phase-by-phase commits.
- **Validation philosophy:** each phase plan states **now-testable acceptance** (unit tests, the
  existing 173-test ptbench suite, synthetic fixtures, offline Occluded-REID) separately from
  **arena acceptance (deferred)** until recordings exist. Per `person-tracker-benchmark-strategy`,
  academic ReID sets are tuning knobs, never gates.

## 5. Phase 0 — Make the benchmark trustworthy + safe quick wins

*Goal: every later phase is measured against a correct ruler; land zero-risk wins; kill the latent crashers.*

### Components

1. **Dual-GT fidelity fix.**
   - `ptbench/common/schema.py`: add `centroid_field` and `centroid_track` to `GtFrame`; bump
     `schema_version` to `1.1`; loader accepts `1.0` (maps its single `centroid_3d` → both fields)
     and `1.1`.
   - `labeler/label_io.py` `build_gt_clip`: compute `centroid_field` (mask + robust median) and
     `centroid_track` (node-identical mean-x/y + median-z, no mask) per frame.
   - `common/metrics.py`: score correctness/lateral/range against `centroid_field`; emit a
     `centroid_track`-based diagnostic block.
   - `common/scoreboard.py`: add a `pos_error_range_m` gate to `GateConfig` (proposed PASS ≤0.30 /
     WARN ≤0.50 — confirmed during plan-writing against the depth-noise budget).
   - `replay/score_cli.py`: make the `action` backend the acceptance default; document the offline
     backend as approximate (it does not replicate the live frame-dropping loop or deployed config).
2. **Node instrumentation** (param `perf_logging_enabled`, default `false`): per-stage
   `perf_counter` timers in `_run_tracking_loop`; per published frame log `mask_pixel_count`,
   `valid_pixel_count`, `used_mask` (did the `<10`-px bbox fallback at `person_track_node.py:366-368`
   fire), `depth_z_iqr`, both the mask and bbox centroids, and the "alive-but-no-centroid" ticks.
3. **Geometry quick win:** median (not mean) on the lateral x/y axes + z-outlier rejection in
   `_calculate_centroid` (`person_track_node.py:378-381`); mirror in `ptbench/common/geometry.py`
   with a unit test asserting node↔geometry parity so the two never silently desync.
4. **Throughput quick wins:** raise/remove the `tracking_rate=15` cap (let frame-seq dedup gate the
   loop); `imgsz` 1280→736 + `half=True` on the YOLO `model.track` call; ROI-crop the depth
   unproject (`person_track_node.py:290-322,646`) so only the target bbox is unprojected.
5. **Operator-init heuristic** (decision #1).
6. **Association quick win:** periodic-ReID switch margin `0.08 → 0.15` (`tracking_pipeline.py:508`).
7. **Latent crashers:** guard `reid_mode='native'` with a clear `NotImplementedError` (or remove
   the branch) at `person_track_node.py:195`; republish a lost-sentinel on `/target_points` during
   loss; plumb the real loop rate into ByteTrack `frame_rate` (fixes `max_time_lost`); use
   frame-stamp `dt` in the velocity model (`yolo_tracker.py:672-689`).

### Testing (now)
ptbench unit tests extended for schema 1.1 round-trip + 1.0 back-compat, the new range gate, and
node↔geometry centroid parity; node changes exercised by T0/T1 startup tiers; a synthetic dual-GT
fixture proving field-vs-track divergence is measured (a tracker matching `centroid_track` exactly
still shows nonzero field error).

### Acceptance
All Phase-0 unit + parity tests green; existing 173-test suite still green; `reid_mode='native'`
fails loudly not silently; manual T1 confirms the node starts and publishes a lost-sentinel.

## 6. Phase 1 — The accuracy core (ReID + identity gating)

*Goal: drive `wrong_lock_episodes` toward 0 and lift `correct_lock_rate`.*

### Components

1. **ReID backbone abstraction.** Refactor `reid.py` so `PersonReIDModel` wraps a pluggable
   backbone behind the existing `extract_features(crop) → L2-normalized vector` interface (so
   `reid_search`/`appearance_manager` are untouched). Implement the torchreid OSNet path (default
   `osnet_ain_x1_0`, param `osnet_x0_25`) loading genuine pretrained weights via a cached resolver.
   Re-weight fusion now that the deep term is reliable (raise `WEIGHT_REID`, lower the color
   weights, `reid.py:590-595`); recalibrate `REID_THRESHOLD`/raw/color floors
   (`reid.py:608,617,621-623`) offline. The legacy random-head path is removed.
2. **Decouple YOLO conf from ByteTrack.** Pass a low detection conf (~0.15) to `model.track`
   (`yolo_tracker.py:298`); introduce `src/vision_track/config/` with a project `bytetrack.yaml`
   and a `default.yaml` for the now-numerous params (matching `object_detection_new`'s convention);
   keep an explicit higher gate where the custom new-target logic needs it.
3. **Gallery hygiene.** Quality-gate history inserts in `appearance_manager.py:70,93-114` — min crop
   height, `mask_coverage>0.4` (already computed but unused at `reid.py:318-320`), a blur/Laplacian
   floor, and back-view rejection — closing the ungated-append poisoning path.
4. **Identity-gated ambiguity.** Lowe-style ratio test on the deep term in `reid_search.py`
   (`_resolve_ambiguity`); require any spatial-proximity switch to also win the deep term by a
   margin (stop letting proximity override identity at `reid_search.py:264-303`); raise
   `distinctiveness_threshold` 0.03→0.10 (`registry.py:23`) and run it on every multi-person frame.

### Testing
Offline Occluded-REID ROC drives threshold/weight calibration (knob, not gate); synthetic
lookalike + back-to-camera discrimination fixtures assert same-vs-different separation improves over
the random head; ptbench `action`-backend smoke once weights load.
**Arena acceptance (deferred):** `wrong_lock_episodes==0` on cml_crossing + lookalike bags;
`correct_lock_rate` recovered on back_to_camera.

## 7. Phase 2 — Recovery policy + geometry robustness

*Goal: `reacquire_latency` and `false_target_rate` without re-introducing wrong locks.*

### Components

1. **Asymmetric-hysteresis lock-state machine** in `tracking_pipeline.py`: emit a provisional
   position fast *only* when it clears the high single-candidate bar (0.72, `reid_search.py:339`) +
   distinctiveness margin; keep `target_lost=True` during any coast; bound recovery by replacing the
   effectively-infinite `allow_indefinite_recovery` default (`person_track_node.py:126,186-191`)
   with a configurable frame cap.
2. **Depth-gated crosser rejection:** plumb the operator's last depth into the tracker (currently
   node-only, `person_track_node.py:290`) and reject candidates whose depth jumps toward the camera
   beyond a threshold — a crosser passing between robot and operator is geometrically nearer, a cue
   appearance cannot spoof. Applied in `_verify_person_candidate`/`detect_occlusion`.
3. **Geometry robustness:** torso-band sampling (chest-height rows of the mask) + EMA/constant-
   velocity smoothing on the published 3D point, reset on loss.

### Testing
Synthetic occlusion/crossing sequences exercise the state-machine transitions and the depth gate;
EMA/torso-band unit tests. **Arena acceptance (deferred):** `reacquire_latency` median ≤1.0 s on
occlusion_reentry; `false_target_rate` ≤0.05; no new wrong-lock episodes vs Phase 1.

## 8. Phase 3 — Throughput hardening

*Goal: comfortable `throughput_hz` margin under crowds.*

### Components

1. **Batched ReID:** stack K candidate crops into one `[K,3,256,128]` forward pass (numerically
   identical to the current per-crop loop in `reid_search.py`/`reid.py`).
2. **Embedding cache** keyed by `(track_id, frame_seq)` — eliminates the up-to-4×/frame re-embed of
   the same crop across `_score_candidates`/`_verify_person_candidate`/`periodic_reid_validation`/
   `_confirm_reid_candidate`.
3. **fp16 on the ReID forward**; **TensorRT engine export** for YOLO (res/batch-locked, FP16) as the
   optional top-end.

### Testing
Batch-equivalence (batched == sequential within tolerance) and cache-correctness unit tests; the
Phase-0 perf instrumentation confirms the per-stage budget improvement. **Arena acceptance
(deferred):** sustained `throughput_hz` ≥12 in 3–4-person re-ID scenes.

## 9. Cross-cutting concerns

- **New config dir** `src/vision_track/config/` (`default.yaml` + `bytetrack.yaml`) introduced in
  Phase 1, following `object_detection_new`'s pattern; build via the tk26 `build.sh` wrapper so
  install-tree shebangs see the venv.
- **Schema migration:** ptbench schema `1.0 → 1.1` is additive + back-compatible; any existing
  `gt.json` (none recorded yet) still loads.
- **Dependency:** `torchreid` added to `src/vision_track/requirements.txt`; weight cache documented
  in `src/tk26_vision/CLAUDE.md`.
- **Geometry lockstep:** the node's `_calculate_centroid` and `ptbench/common/geometry.py` must stay
  identical for `centroid_track`; enforced by a parity unit test.

## 10. Dependency order & risks

**Order:** 0 → 1 → 2 → 3. Phase 0 is a hard prerequisite (it fixes the ruler). Phase 2's aggressive
recovery depends on Phase 1's reliable identity (else reacquire↔wrong-lock tension bites). Phase 3 is
pure optimization, last.

**Risks:**
- `torchreid` install / weight provenance in the offline venv — mitigated by the cached resolver and
  the `osnet_x0_25` fallback param.
- Heuristic operator-init assumes centered/near start; if violated, an explicit goal field (deferred
  by decision #1) becomes the fallback design.
- Arena gate numbers for Phases 1–3 cannot be confirmed until recordings exist; each plan separates
  now-testable from arena-deferred acceptance so progress is verifiable without bags.
- Lowering `imgsz`/conf trades small-/far-person recall for speed and recall; tuned jointly in
  Phase 0/1 against the range_lighting fixtures.
