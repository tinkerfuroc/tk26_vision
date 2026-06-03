# Person Tracker — Phase 3: Throughput Hardening — Implementation Plan

**For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the multi-person ReID throughput cliff. Under crowds the tracker fires 5–7 redundant FP32 ResNet50/OSNet forward passes per frame — the *same* crop is re-embedded up to 4×/frame across `_score_candidates` → `_verify_person_candidate` → `periodic_reid_validation` → `_confirm_reid_candidate`. Phase 3 collapses these into (a) one batched forward pass over all candidates, (b) a per-frame embedding cache so a crop is embedded at most once, and (c) an fp16 forward, plus (d) an optional TensorRT export path for YOLO. Target: sustained `throughput_hz` ≥ 12 in 3–4-person re-ID scenes (arena-deferred); numerically-equivalent batching and cache correctness validated by unit tests now.

**Architecture:** Phase 3 is *pure optimization* — it must not change tracking decisions. The embedding cache and batch-stacking helper are ROS-free pure logic (no `rclpy`), wrapping the Phase-1 `PersonReIDModel.extract_features` stable interface (OSNet backbone). `extract_features_batch` stacks K crops into one `[K,3,256,128]` forward; the per-`(track_id, frame_seq)` cache lets the four call sites reuse the score-pass embedding within a frame. fp16 toggles the forward dtype while keeping outputs L2-normalized. TensorRT export is a documented, scripted, best-effort top-end with a manual verification step (no hard unit test — it is resolution/batch-locked and hardware-specific).

**Tech Stack:** Python 3.10, PyTorch 2.11 + CUDA 12.8 (`.venv-vision-main`), torchreid OSNet (Phase 1), Ultralytics YOLO (`yolo11s-seg.pt`), numpy, OpenCV, pytest. ROS2 Humble (`vision_track` / `/track_person`). Build via the tk26 `build.sh` wrapper so install-tree shebangs see the venv.

**DEPENDENCY — READ FIRST:** Phase 3 depends on **Phase 1 (OSNet backbone) being merged** into `feat/person-tracker-overhaul`. Phase 1 makes `PersonReIDModel` wrap a pluggable backbone behind the stable `extract_features(crop) → L2-normalized vector` interface, introduces the `reid_backbone` param (default `osnet_ain_x1_0`), and removes the legacy random-head ResNet50 path. **Do not start Task 1 until `reid_backbone`/OSNet are present in `vision_track/vision_track/reid/reid.py`.** Phase 3 also references the Phase-0 `perf_logging_enabled` param for its arena-deferred acceptance (per-stage timings). If, at execution time, Phase 1 has *not* landed, STOP and surface the blocker — the batch/cache must wrap the OSNet backbone, not the legacy head.

> **Worktree (WT):** `/home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker` (branch `feat/person-tracker-overhaul`). All repo-relative paths below are under WT.
> **Venv python (VENV):** `/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python`.
> **Test invocation:** keep new logic ROS-free (NO top-level `rclpy` import). Run pure unit tests from the package dir:
> `cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_<name>.py -v`
> The torch-dependent tests (batch-equivalence, fp16 tolerance) must `pytest.importorskip("torch")` / skip cleanly when torch or OSNet weights are unavailable. The cache-key / eviction / invalidation logic must be testable WITHOUT torch (pure dict/LRU).

---

## File Structure

```
src/vision_track/
├── vision_track/
│   ├── reid/
│   │   ├── reid.py                 # EDIT: add extract_features_batch + fp16 to PersonReIDModel; AppearanceExtractor.extract_features_batch
│   │   ├── reid_search.py          # EDIT: _score_candidates → one batched embed call; read cache via tracker
│   │   ├── embedding_cache.py      # NEW: ROS-free FrameEmbeddingCache (bounded, keyed by (track_id, frame_seq))
│   │   └── appearance_manager.py   # (unchanged — referenced only)
│   ├── core/
│   │   └── tracking_pipeline.py    # EDIT: route _verify_person_candidate / periodic_reid_validation / _confirm_reid_candidate embeds through the cache
│   ├── yolo_tracker.py             # EDIT: own a FrameEmbeddingCache; pass reid_fp16 into AppearanceExtractor; optional .engine load
│   └── person_track_node.py        # EDIT: declare + plumb reid_fp16 param (and optional yolo_engine_path)
├── scripts/
│   └── export_yolo_trt.py          # NEW (Task 4): scripted, documented YOLO→TensorRT FP16 export
├── requirements.txt                # (unchanged — torch/torchvision/ultralytics already present)
└── test/
    ├── test_embedding_cache.py     # NEW (Task 2): pure dict/LRU cache tests, NO torch
    ├── test_reid_batch.py          # NEW (Task 1): crop-stacking shape (no model) + batch-equivalence (torch-gated)
    └── test_reid_fp16.py           # NEW (Task 3): fp16 vs fp32 tolerance + L2-norm (torch-gated)
```

**Cross-cutting design decisions (locked):**

- **`frame_seq` for the cache = `tracker.frame_count`.** `update_tracker` increments `tracker.frame_count` exactly once per update, at the very top (`tracking_pipeline.py:23`), *before* any ReID embedding runs in that frame. It is therefore a monotonic per-frame token shared by all four embed call sites within one `update()`. The node's own `self.frame_seq` (camera counter, `person_track_node.py:99,288`) is NOT used as the cache key — it lives behind the lock and is decoupled from the tracker's processing cadence. The cache invalidates whenever the observed `frame_count` changes (new frame ⇒ drop the previous frame's entries).
- **Cache scope = whole-detection feature dict.** The cached value is the full `Dict[str, np.ndarray]` returned by `AppearanceExtractor.extract_features` (reid vector + color histograms + size, etc.), keyed by `(track_id, frame_seq)`. This is what every call site consumes, so caching the whole dict (not just the `'reid'` vector) eliminates *all* redundant per-crop work (color histograms re-run too), not only the deep forward.
- **Batch shape:** `extract_features_batch(crops: list[np.ndarray]) -> np.ndarray` returns `[K, feature_dim]`, row `i` == `extract_features(crops[i])` within tolerance. Empty list ⇒ `np.zeros((0, feature_dim))`. The deep forward stacks to `[K,3,256,128]`; color/size features stay per-crop in the `AppearanceExtractor` wrapper.
- **fp16 is forward-only.** Model params/inputs cast to half for the `backbone(...)` call; the L2-normalize and `.cpu().numpy()` return float32. Default `reid_fp16=True`. On CPU (no CUDA) the implementation silently falls back to fp32 (half on CPU is slow/unsupported for some ops) — gated on `self.device == "cuda"`.

---

### Task 1 — Batched ReID forward (`extract_features_batch`) + `_score_candidates` refactor

Stack K candidate crops into one `[K,3,256,128]` forward pass; route `_score_candidates` through it. Batched output must be numerically equivalent to the current per-crop loop.

**Files:**
- `src/vision_track/vision_track/reid/reid.py` — `PersonReIDModel.extract_features` (lines 133–192, the single-crop forward to mirror); add `PersonReIDModel.extract_features_batch` after line 192. `AppearanceExtractor.extract_features` (lines 246–322, the per-detection dict builder); add `AppearanceExtractor.extract_features_batch` after line 322.
- `src/vision_track/vision_track/reid/reid_search.py` — `_score_candidates` (lines 108–149) currently loops `tracker.appearance_extractor.extract_features(...)` per candidate (line 120).
- `src/vision_track/test/test_reid_batch.py` — NEW.

> NOTE: After Phase 1, `PersonReIDModel.extract_features` wraps the OSNet backbone; the resize/normalize preamble (lines 146–158) may differ from the ResNet50 listing above. **Read the merged `extract_features` first** and mirror its exact preprocessing in the batch path — the batch helper must reuse whatever preprocessing the merged single-crop path uses, not the pre-Phase-1 ResNet50 code shown here.

- [ ] **Step 1 — Failing test: pure crop-stacking shape (no model).** Add to `test/test_reid_batch.py` a test that the stacking/resize/normalize logic produces a correctly-shaped tensor for K crops of varying sizes, with NO model invocation. Extract the pure tensor-builder as a static helper so it is testable in isolation:

```python
# test/test_reid_batch.py
import numpy as np
import pytest

from vision_track.reid.reid import PersonReIDModel


def _make_crop(h, w, seed):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8))


def test_stack_crops_shape_varying_sizes():
    crops = [_make_crop(200, 90, 1), _make_crop(50, 30, 2), _make_crop(400, 150, 3)]
    tensor = PersonReIDModel._stack_crops(crops)  # CPU torch.Tensor [K,3,256,128]
    assert tuple(tensor.shape) == (3, 3, 256, 128)
    assert tensor.dtype.is_floating_point


def test_stack_crops_empty():
    tensor = PersonReIDModel._stack_crops([])
    assert tuple(tensor.shape) == (0, 3, 256, 128)
```

- [ ] **Step 1 — Run to fail.** `_stack_crops` does not exist yet.
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_reid_batch.py -v
  ```
  Expected: `AttributeError: type object 'PersonReIDModel' has no attribute '_stack_crops'` (collected, FAILED). If torch is unavailable the whole module skips — that is acceptable; record it and proceed.

- [ ] **Step 1 — Minimal impl: extract `_stack_crops` static helper.** In `reid.py`, add a static method that mirrors the merged `extract_features` preprocessing (resize → `(128, 256)`, ImageNet normalize, permute, stack). Refactor `extract_features` to delegate to it for the K=1 case so the two paths share one preprocessing definition (no drift):

```python
# reid.py — inside PersonReIDModel, after extract_features
    # ImageNet normalization constants (mirror extract_features)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @staticmethod
    def _stack_crops(crops: list) -> "torch.Tensor":
        """Resize + ImageNet-normalize K crops into one [K,3,256,128] CPU tensor.

        Numerically identical preprocessing to extract_features. Empty list ->
        a real [0,3,256,128] tensor so downstream stacking/forward is well-defined.
        """
        if not crops:
            return torch.zeros((0, 3, 256, 128), dtype=torch.float32)
        batch = np.empty((len(crops), 256, 128, 3), dtype=np.float32)
        for i, crop in enumerate(crops):
            resized = cv2.resize(crop, (128, 256))
            batch[i] = (resized / 255.0 - PersonReIDModel._MEAN) / PersonReIDModel._STD
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()
        return tensor
```
  (Keep `extract_features`'s public behaviour identical; you MAY refactor its body to call `_stack_crops([crop])` then run the existing forward on `tensor` — but only if the merged Phase-1 forward is preserved exactly. If unsure, leave `extract_features` as-is and just add `_stack_crops`.)

- [ ] **Step 1 — Run to pass.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_reid_batch.py -v
  ```
  Expected: `test_stack_crops_shape_varying_sizes PASSED`, `test_stack_crops_empty PASSED`.

- [ ] **Step 1 — Commit.** `git add -A && git commit -m "feat(vision_track): _stack_crops helper for batched ReID preprocessing"` (+ `Co-Authored-By` trailer).

- [ ] **Step 2 — Failing test: batch == sequential (torch-gated).** Add a test asserting `extract_features_batch` row-equals the per-crop `extract_features` loop within tolerance. Gate on torch + a real model build (skip if the OSNet backbone can't load offline):

```python
# test/test_reid_batch.py (append)
def _model_or_skip():
    torch = pytest.importorskip("torch")
    try:
        m = PersonReIDModel(device="cpu")
    except Exception as e:
        pytest.skip(f"ReID model unavailable: {e}")
    if not getattr(m, "use_deep_features", False) or m.backbone is None:
        pytest.skip("ReID backbone did not load")
    return m


def test_batch_equivalence_matches_sequential():
    m = _model_or_skip()
    crops = [_make_crop(180, 80, k) for k in range(5)]
    seq = np.stack([m.extract_features(c) for c in crops], axis=0)
    batched = m.extract_features_batch(crops)
    assert batched.shape == seq.shape
    # Eval-mode forward is deterministic; allow small fp accumulation differences.
    np.testing.assert_allclose(batched, seq, atol=1e-4, rtol=0)


def test_batch_empty_returns_zero_rows():
    m = _model_or_skip()
    out = m.extract_features_batch([])
    assert out.shape == (0, m.feature_dim)
```

- [ ] **Step 2 — Run to fail.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_reid_batch.py -k equivalence -v
  ```
  Expected: `AttributeError: 'PersonReIDModel' object has no attribute 'extract_features_batch'` (FAILED). If the backbone can't load offline the test SKIPS — note it; the equivalence guarantee then rests on Step 1's shape test + manual run, but do not block.

- [ ] **Step 2 — Minimal impl: `PersonReIDModel.extract_features_batch`.** Mirror `extract_features`'s forward exactly but over a `[K,...]` batch. The post-backbone head ops (channel attention, GAP, part pooling, bottlenecks, concat, L2-normalize) are already batch-safe (they operate on dim 0 generically); the only change is no `unsqueeze(0)` and slicing returns `[K,...]`:

```python
# reid.py — inside PersonReIDModel, after _stack_crops
    def extract_features_batch(self, crops: list) -> np.ndarray:
        """Embed K crops in ONE forward pass. Row i == extract_features(crops[i]).

        Returns [K, feature_dim] float32, each row L2-normalized. Empty -> [0, dim].
        """
        if not self.use_deep_features or self.backbone is None:
            return np.zeros((len(crops), self.feature_dim), dtype=np.float32)
        if not crops:
            return np.zeros((0, self.feature_dim), dtype=np.float32)

        tensor = self._stack_crops(crops).to(self.device)
        with torch.no_grad():
            features = self.backbone(tensor)              # [K, C, h, w]
            if self.channel_attention is not None:
                attn = self.channel_attention(features)
                attn = attn.view(-1, features.shape[1], 1, 1)
                features = features * attn
                global_feat = self.bottleneck(self.gap(features).flatten(1))   # [K, 512]
                part_features = self.part_pool(features)                       # [K, C, 4, 1]
                part_feats = []
                for i in range(4):
                    part_i = part_features[:, :, i, :].flatten(1)
                    part_feats.append(self.part_bottlenecks[i](part_i))       # [K, 128]
                combined = torch.cat([global_feat, torch.cat(part_feats, dim=1)], dim=1)
            else:
                combined = self.gap(features).flatten(1)
            combined = torch.nn.functional.normalize(combined, p=2, dim=1)
        return combined.cpu().numpy().astype(np.float32)
```
  **IMPORTANT:** if the merged Phase-1 OSNet backbone exposes a different head (e.g. torchreid returns a single embedding directly with no part bottlenecks), replace the head block above with the *exact* head used by the merged `extract_features` — the contract is row-equivalence, so copy the merged forward verbatim, only stripping `unsqueeze(0)` and keeping dim-0 as K.

- [ ] **Step 2 — Run to pass.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_reid_batch.py -v
  ```
  Expected: equivalence + empty tests PASS (or SKIP if backbone unavailable); the Step-1 shape tests still PASS.

- [ ] **Step 2 — Commit.** `git add -A && git commit -m "feat(vision_track): extract_features_batch — one forward pass over K crops"`.

- [ ] **Step 3 — Failing test: `AppearanceExtractor.extract_features_batch` builds K dicts with one deep forward.** This wrapper crops K detections, calls `PersonReIDModel.extract_features_batch` once for the `'reid'` vectors, and fills color/size per crop. Test that it returns the same per-detection dicts as looping `extract_features`:

```python
# test/test_reid_batch.py (append)
def test_appearance_extractor_batch_matches_loop():
    pytest.importorskip("torch")
    from vision_track.reid.reid import AppearanceExtractor
    try:
        ae = AppearanceExtractor(device="cpu")
    except Exception as e:
        pytest.skip(f"AppearanceExtractor unavailable: {e}")
    if not getattr(ae.person_reid, "use_deep_features", False):
        pytest.skip("ReID backbone did not load")
    frame = _make_crop(480, 640, 99)
    bboxes = [(10, 10, 90, 210), (200, 20, 280, 220), (400, 30, 470, 230)]
    looped = [ae.extract_features(frame, b, None, class_id=0) for b in bboxes]
    batched = ae.extract_features_batch(frame, bboxes, [None] * 3, [0] * 3)
    assert len(batched) == len(looped)
    for got, exp in zip(batched, looped):
        assert set(got.keys()) == set(exp.keys())
        np.testing.assert_allclose(got["reid"], exp["reid"], atol=1e-4, rtol=0)
        np.testing.assert_allclose(got["body_color"], exp["body_color"], atol=1e-6, rtol=0)
```

- [ ] **Step 3 — Run to fail.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_reid_batch.py -k appearance -v
  ```
  Expected: `AttributeError: 'AppearanceExtractor' object has no attribute 'extract_features_batch'` (FAILED or SKIP if model absent).

- [ ] **Step 3 — Minimal impl: `AppearanceExtractor.extract_features_batch`.** Refactor `extract_features` (lines 246–322) so its crop-clamp + dict-building logic (everything except the single `self.person_reid.extract_features(crop)` call) is reusable per-detection; then add the batch wrapper that does ONE deep forward and stitches the deep vector into each dict:

```python
# reid.py — inside AppearanceExtractor, after extract_features
    def extract_features_batch(self, frame, bboxes, masks, class_ids) -> list:
        """Vectorize the deep ReID forward across N detections.

        Returns list[dict] aligned to bboxes; each dict == extract_features(...)
        for that detection. Non-person / invalid-bbox entries are filled by the
        per-crop path (no deep batch slot consumed).
        """
        n = len(bboxes)
        masks = masks if masks is not None else [None] * n
        class_ids = class_ids if class_ids is not None else [-1] * n

        # Build per-detection dicts WITHOUT the person deep vector first, and
        # collect the person crops that need a deep embedding.
        out = [None] * n
        person_idx = []
        person_crops = []
        for i in range(n):
            d = self.extract_features(frame, bboxes[i], masks[i], class_ids[i])
            out[i] = d
            if class_ids[i] == self.PERSON_CLASS_ID and d:
                # crop already validated inside extract_features; recompute the
                # same clamped crop for the batch forward (cheap; numpy slice).
                x1, y1, x2, y2 = bboxes[i]
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    person_idx.append(i)
                    person_crops.append(frame[y1:y2, x1:x2].copy())

        if person_crops:
            deep = self.person_reid.extract_features_batch(person_crops)  # [P, dim]
            for slot, i in enumerate(person_idx):
                if out[i] is not None and "reid" in out[i]:
                    out[i]["reid"] = deep[slot]
        return out
```
  **Equivalence note:** the simplest correct implementation reuses `extract_features` per detection for *all* the color/size work and *overwrites only* the `'reid'` vector with the batched result. That guarantees byte-identical color features and a within-tolerance deep vector while still collapsing the K deep forwards into one. (A future optimization can also batch the color histograms; not required for the throughput target — the deep forward is the cliff.)

- [ ] **Step 3 — Run to pass.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_reid_batch.py -v
  ```
  Expected: all Task-1 tests PASS (or SKIP without backbone).

- [ ] **Step 3 — Commit.** `git add -A && git commit -m "feat(vision_track): AppearanceExtractor.extract_features_batch — single deep forward per frame"`.

- [ ] **Step 4 — Refactor `_score_candidates` to one batched embed call.** Replace the per-candidate `extract_features` loop (`reid_search.py:119-123`) with a single `extract_features_batch` over all candidate bboxes, then iterate the returned dicts for scoring. Preserve the exact downstream behaviour (skip empty-dict candidates, debug logging, `compute_similarity`). The cache (Task 2) is wired in Step 5 — this step is just the batch call:

```python
# reid_search.py — _score_candidates body, replacing the loop at lines 119-149
    bboxes = [r.bbox for r in candidates]
    masks = [r.mask for r in candidates]
    class_ids = [r.class_id for r in candidates]
    feature_dicts = tracker.appearance_extractor.extract_features_batch(
        frame, bboxes, masks, class_ids
    )

    for result, features in zip(candidates, feature_dicts):
        if not features:
            logger.debug(f"ID {result.track_id}: No features extracted")
            continue
        if is_person and "reid" in features and target_reid is not None:
            if target_reid.shape[0] == features["reid"].shape[0]:
                raw_cosine = ReIDMatcher._cosine_similarity(target_reid, features["reid"])
                logger.debug(f"ID {result.track_id}: raw ReID cosine={raw_cosine:.3f}")
            else:
                logger.debug(
                    f"ID {result.track_id}: feature dim mismatch "
                    f"({target_reid.shape[0]} vs {features['reid'].shape[0]})"
                )
        if is_person and "body_color" in features:
            target_body = tracker.target_appearance.get_body_color()
            if target_body is not None:
                body_sim = ReIDMatcher._histogram_similarity(target_body, features["body_color"])
                logger.debug(f"ID {result.track_id}: body color similarity={body_sim:.3f}")
        similarity = ReIDMatcher.compute_similarity(
            tracker.target_appearance, features, result.bbox, current_time, is_person=is_person,
        )
        candidate_scores.append((result, similarity, features))

    return candidate_scores
```

- [ ] **Step 4 — Run to pass (regression).** No new test; rely on Task-1 equivalence + existing tests. Confirm import-level health and that the existing flake8/pep257 tests are unaffected:
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "from vision_track.reid import reid_search; print('import OK')"
  ```
  Expected: `import OK` (no top-level rclpy import is pulled in).

- [ ] **Step 4 — Commit.** `git add -A && git commit -m "perf(vision_track): _score_candidates embeds all candidates in one batched forward"`.

---

### Task 2 — Per-frame embedding cache keyed by `(track_id, frame_seq)`

A pure, ROS-free, bounded cache so the four embed call sites within one frame reuse the score-pass feature dict instead of re-embedding the same crop up to 4×.

**Files:**
- `src/vision_track/vision_track/reid/embedding_cache.py` — NEW, no rclpy / no torch import.
- `src/vision_track/test/test_embedding_cache.py` — NEW, NO torch.
- `src/vision_track/vision_track/yolo_tracker.py` — owns the cache instance (`self._init_reid_settings`, lines 122–133); `frame_seq` source is `self.frame_count`.
- `src/vision_track/vision_track/reid/reid_search.py` — `_score_candidates` populates the cache.
- `src/vision_track/vision_track/core/tracking_pipeline.py` — `_verify_person_candidate` (line 165), `periodic_reid_validation` (lines 488-490), `_confirm_reid_candidate` (line 328), and `_handle_occlusion_state` (line 124) read the cache before embedding.

- [ ] **Step 1 — Failing test: cache hit/miss/eviction/invalidate (NO torch).** Write `test_embedding_cache.py` covering get/put, the bounded-size LRU eviction, and the per-`frame_seq` invalidation (a new frame_seq drops the previous frame's entries):

```python
# test/test_embedding_cache.py
import pytest

from vision_track.reid.embedding_cache import FrameEmbeddingCache


def test_miss_then_hit():
    c = FrameEmbeddingCache(max_entries=8)
    assert c.get(track_id=3, frame_seq=10) is None
    c.put(track_id=3, frame_seq=10, features={"reid": [1.0]})
    assert c.get(track_id=3, frame_seq=10) == {"reid": [1.0]}


def test_new_frame_seq_invalidates_old_frame():
    c = FrameEmbeddingCache(max_entries=8)
    c.put(track_id=3, frame_seq=10, features={"reid": [1.0]})
    # Touching frame 11 drops everything from frame 10.
    c.begin_frame(11)
    assert c.get(track_id=3, frame_seq=10) is None
    assert c.get(track_id=3, frame_seq=11) is None
    c.put(track_id=3, frame_seq=11, features={"reid": [2.0]})
    assert c.get(track_id=3, frame_seq=11) == {"reid": [2.0]}


def test_get_with_stale_frame_seq_returns_none():
    c = FrameEmbeddingCache(max_entries=8)
    c.begin_frame(5)
    c.put(track_id=1, frame_seq=5, features={"reid": [9.0]})
    # A read tagged with a different (older) frame_seq must miss, not return stale.
    assert c.get(track_id=1, frame_seq=4) is None


def test_bounded_lru_eviction_within_frame():
    c = FrameEmbeddingCache(max_entries=2)
    c.begin_frame(7)
    c.put(track_id=1, frame_seq=7, features={"a": 1})
    c.put(track_id=2, frame_seq=7, features={"b": 2})
    c.get(track_id=1, frame_seq=7)            # touch 1 -> 2 is now LRU
    c.put(track_id=3, frame_seq=7, features={"c": 3})  # evicts track 2
    assert c.get(track_id=2, frame_seq=7) is None
    assert c.get(track_id=1, frame_seq=7) == {"a": 1}
    assert c.get(track_id=3, frame_seq=7) == {"c": 3}


def test_clear():
    c = FrameEmbeddingCache(max_entries=4)
    c.put(track_id=1, frame_seq=1, features={"x": 0})
    c.clear()
    assert c.get(track_id=1, frame_seq=1) is None
```

- [ ] **Step 1 — Run to fail.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_embedding_cache.py -v
  ```
  Expected: `ModuleNotFoundError: No module named 'vision_track.reid.embedding_cache'` (collection error → FAILED).

- [ ] **Step 1 — Minimal impl: `FrameEmbeddingCache`.** Pure stdlib (`collections.OrderedDict`), no torch/rclpy. Single-frame scope with LRU bound; `begin_frame(seq)` invalidates when the seq changes:

```python
# vision_track/reid/embedding_cache.py
"""ROS-free, torch-free per-frame embedding cache.

Eliminates the up-to-4x/frame re-embedding of the same person crop across
_score_candidates / _verify_person_candidate / periodic_reid_validation /
_confirm_reid_candidate. Scoped to a single frame_seq: when the tracker advances
to a new frame, the previous frame's entries are dropped (appearances are not
reused across frames — only within the one update() call). Bounded LRU so a
crowd of stale track_ids cannot grow it without limit.
"""
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class FrameEmbeddingCache:
    def __init__(self, max_entries: int = 32):
        self._max = max(1, int(max_entries))
        self._store: "OrderedDict[Tuple[int, int], Dict[str, Any]]" = OrderedDict()
        self._frame_seq: Optional[int] = None

    def begin_frame(self, frame_seq: int) -> None:
        """Mark the start of processing for frame_seq; drop prior-frame entries."""
        if frame_seq != self._frame_seq:
            self._store.clear()
            self._frame_seq = frame_seq

    def get(self, track_id: int, frame_seq: int) -> Optional[Dict[str, Any]]:
        if frame_seq != self._frame_seq:
            return None
        key = (track_id, frame_seq)
        val = self._store.get(key)
        if val is not None:
            self._store.move_to_end(key)  # mark MRU
        return val

    def put(self, track_id: int, frame_seq: int, features: Dict[str, Any]) -> None:
        # Auto-begin a frame on first put so callers may skip begin_frame.
        if frame_seq != self._frame_seq:
            self.begin_frame(frame_seq)
        key = (track_id, frame_seq)
        self._store[key] = features
        self._store.move_to_end(key)
        while len(self._store) > self._max:
            self._store.popitem(last=False)  # evict LRU

    def clear(self) -> None:
        self._store.clear()
        self._frame_seq = None
```

- [ ] **Step 1 — Run to pass.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_embedding_cache.py -v
  ```
  Expected: all 5 cache tests PASS. (Note: `test_miss_then_hit` exercises `put` without a prior `begin_frame`; the auto-begin in `put` covers it. `test_bounded_lru_eviction_within_frame` exercises the LRU bound + `move_to_end` touch.)

- [ ] **Step 1 — Commit.** `git add -A && git commit -m "feat(vision_track): FrameEmbeddingCache — ROS-free per-frame embedding cache"`.

- [ ] **Step 2 — Tracker owns a cache; `_score_candidates` populates it.** In `yolo_tracker.py` `_init_reid_settings` (after line 132), add `self.embedding_cache = FrameEmbeddingCache(max_entries=32)` (import it at top alongside the other reid imports, line ~22). In `reid_search.py` `_score_candidates`, after building each candidate's `features` dict, store it: `tracker.embedding_cache.put(result.track_id, tracker.frame_count, features)`. Place `tracker.embedding_cache.begin_frame(tracker.frame_count)` at the start of `_score_candidates` (defensive — `put` also auto-begins). No behaviour change yet; this just primes the cache.

```python
# yolo_tracker.py — top imports
from .reid.embedding_cache import FrameEmbeddingCache
# yolo_tracker.py — _init_reid_settings, after self.feature_refresh_interval line
        self.embedding_cache = FrameEmbeddingCache(max_entries=32)
```
```python
# reid_search.py — _score_candidates, just before the candidate loop
    tracker.embedding_cache.begin_frame(tracker.frame_count)
    ...
    # inside the loop, after `features = ...` and the empty-dict skip:
        tracker.embedding_cache.put(result.track_id, tracker.frame_count, features)
```

- [ ] **Step 2 — Run to pass (smoke import).**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "from vision_track import yolo_tracker; from vision_track.reid import reid_search; print('OK')"
  ```
  Expected: `OK`.

- [ ] **Step 2 — Commit.** `git add -A && git commit -m "feat(vision_track): tracker owns FrameEmbeddingCache; _score_candidates primes it"`.

- [ ] **Step 3 — Failing test: pipeline reuses cached features (no re-embed).** Add a focused test in `test_embedding_cache.py` that a fake tracker whose `appearance_extractor.extract_features` counts calls does NOT re-embed when the cache already holds the entry for `(track_id, frame_count)`. Use a minimal duck-typed tracker so the test stays ROS-free and torch-free:

```python
# test/test_embedding_cache.py (append)
class _CountingExtractor:
    def __init__(self):
        self.calls = 0
    def extract_features(self, frame, bbox, mask, class_id):
        self.calls += 1
        return {"reid": [0.0], "body_color": [0.0]}


def test_cached_features_skip_reembed():
    from vision_track.reid.embedding_cache import FrameEmbeddingCache
    from vision_track.core.tracking_pipeline import _get_or_extract_features

    cache = FrameEmbeddingCache(max_entries=8)
    ex = _CountingExtractor()

    class _T:  # duck-typed tracker
        frame_count = 42
        embedding_cache = cache
        appearance_extractor = ex

    t = _T()
    f1 = _get_or_extract_features(t, frame=None, track_id=7, bbox=(0, 0, 1, 1), mask=None, class_id=0)
    f2 = _get_or_extract_features(t, frame=None, track_id=7, bbox=(0, 0, 1, 1), mask=None, class_id=0)
    assert ex.calls == 1            # second call served from cache
    assert f1 == f2
```

- [ ] **Step 3 — Run to fail.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_embedding_cache.py -k reembed -v
  ```
  Expected: `ImportError: cannot import name '_get_or_extract_features'` (FAILED). (Note: `tracking_pipeline` imports numpy + reid only — no rclpy — so this import is safe in a torch-free, ROS-free test as long as torch is importable; if torch is missing the module still imports because reid's torch import is module-level — guard with `pytest.importorskip("torch")` at the top of this test if the offline venv lacks torch.)

- [ ] **Step 3 — Minimal impl: `_get_or_extract_features` helper + route the 4 call sites.** Add a single cache-aware helper to `tracking_pipeline.py` and replace the four bare `tracker.appearance_extractor.extract_features(...)` calls with it:

```python
# core/tracking_pipeline.py — module-level helper
def _get_or_extract_features(tracker, frame, track_id, bbox, mask, class_id):
    """Cache-aware single-detection feature extraction.

    Reuses the score-pass embedding for (track_id, frame_count) within one frame,
    eliminating the up-to-4x/frame re-embed. track_id < 0 (unstable) is never
    cached (it collides across detections); falls through to a direct extract.
    """
    cache = getattr(tracker, "embedding_cache", None)
    seq = getattr(tracker, "frame_count", None)
    if cache is not None and seq is not None and track_id is not None and track_id >= 0:
        hit = cache.get(track_id, seq)
        if hit is not None:
            return hit
        features = tracker.appearance_extractor.extract_features(frame, bbox, mask, class_id=class_id)
        if features:
            cache.put(track_id, seq, features)
        return features
    return tracker.appearance_extractor.extract_features(frame, bbox, mask, class_id=class_id)
```
  Then replace, preserving each existing call's args exactly:
  - `_handle_occlusion_state` line 124: `features = _get_or_extract_features(tracker, frame, result.track_id, result.bbox, result.mask, 0)`
  - `_verify_person_candidate` line 165: `features = _get_or_extract_features(tracker, frame, result.track_id, result.bbox, result.mask, 0)`
  - `_confirm_reid_candidate` line 328: `features = _get_or_extract_features(tracker, frame, reid_match.track_id, reid_match.bbox, reid_match.mask, reid_match.class_id)`
  - `periodic_reid_validation` lines 488-490: `features_cur = _get_or_extract_features(tracker, frame, current_result.track_id, current_result.bbox, current_result.mask, current_result.class_id)`

  Leave `verify_post_occlusion` (line 563, compares against `pre_occlusion_appearance`) and `register_other_persons` (line 408, embeds OTHER persons under negated temp IDs) using the direct `extract_features` — they embed different crops/contexts and caching them would be incorrect or pointless (negated temp IDs aren't ByteTrack IDs).

- [ ] **Step 3 — Run to pass.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_embedding_cache.py -v
  ```
  Expected: all cache tests PASS including `test_cached_features_skip_reembed`.

- [ ] **Step 3 — Commit.** `git add -A && git commit -m "perf(vision_track): route ReID embeds through FrameEmbeddingCache (kill 4x/frame re-embed)"`.

- [ ] **Step 4 — Invalidate cache per new frame at the top of `update_tracker`.** Add `tracker.embedding_cache.begin_frame(tracker.frame_count)` immediately after `tracker.frame_count += 1` (`tracking_pipeline.py:23`). This guarantees a clean cache for every processed frame even if no candidate is scored, and makes `frame_count` the single source of truth for `frame_seq`. Also clear the cache in `YOLOTracker.reset()` (after line 841 `self.frame_count = 0`): `self.embedding_cache.clear()`.

```python
# core/tracking_pipeline.py — update_tracker, after line 23
    tracker.frame_count += 1
    if getattr(tracker, "embedding_cache", None) is not None:
        tracker.embedding_cache.begin_frame(tracker.frame_count)
```
```python
# yolo_tracker.py — reset(), after self.frame_count = 0
        self.embedding_cache.clear()
```

- [ ] **Step 4 — Run to pass (full cache suite + imports).**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_embedding_cache.py test/test_reid_batch.py -v && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "from vision_track import yolo_tracker; from vision_track.core import tracking_pipeline; print('OK')"
  ```
  Expected: cache + batch tests PASS (batch may SKIP without backbone); `OK`.

- [ ] **Step 4 — Commit.** `git add -A && git commit -m "perf(vision_track): begin_frame at update_tracker top; clear cache on reset"`.

---

### Task 3 — fp16 ReID forward (`reid_fp16`, default True)

Half-precision forward on CUDA; outputs stay L2-normalized float32 and within tolerance of fp32.

**Files:**
- `src/vision_track/vision_track/reid/reid.py` — `PersonReIDModel.__init__` (lines 25–37), `_load_reid_model` (`.to(self.device)` block, lines 84–90), `extract_features` (forward, lines 160–192), `extract_features_batch` (Task 1).
- `src/vision_track/vision_track/yolo_tracker.py` — `__init__` signature (lines 63–73) + `AppearanceExtractor(self.device)` construction (line 113).
- `src/vision_track/vision_track/reid/reid.py` — `AppearanceExtractor.__init__` (lines 208–221) passes the flag to `PersonReIDModel`.
- `src/vision_track/vision_track/person_track_node.py` — declare `reid_fp16` (after line 129), load it (after line 157), pass to `YOLOTracker` (after line 212).
- `src/vision_track/test/test_reid_fp16.py` — NEW (torch-gated).

- [ ] **Step 1 — Failing test: fp16 output is L2-normalized + close to fp32 (torch-gated).** Skip cleanly without torch/CUDA/backbone:

```python
# test/test_reid_fp16.py
import numpy as np
import pytest


def _crop(h, w, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def _model(fp16):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("fp16 path requires CUDA")
    from vision_track.reid.reid import PersonReIDModel
    try:
        m = PersonReIDModel(device="cuda", fp16=fp16)
    except Exception as e:
        pytest.skip(f"ReID model unavailable: {e}")
    if not getattr(m, "use_deep_features", False) or m.backbone is None:
        pytest.skip("ReID backbone did not load")
    return m


def test_fp16_output_is_l2_normalized_float32():
    m = _model(fp16=True)
    v = m.extract_features(_crop(200, 90, 1))
    assert v.dtype == np.float32
    assert abs(np.linalg.norm(v) - 1.0) < 1e-3


def test_fp16_close_to_fp32():
    m16 = _model(fp16=True)
    m32 = _model(fp16=False)
    crop = _crop(220, 100, 7)
    v16 = m16.extract_features(crop)
    v32 = m32.extract_features(crop)
    # cosine similarity between fp16 and fp32 embeddings of the same crop
    cos = float(np.dot(v16, v32) / (np.linalg.norm(v16) * np.linalg.norm(v32) + 1e-9))
    assert cos > 0.999
```

- [ ] **Step 1 — Run to fail.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_reid_fp16.py -v
  ```
  Expected: `TypeError: __init__() got an unexpected keyword argument 'fp16'` (FAILED), or SKIP if no CUDA/backbone.

- [ ] **Step 1 — Minimal impl: thread `fp16` through the model.** Add `fp16: bool = False` to `PersonReIDModel.__init__` (store `self.fp16 = fp16 and device == "cuda"`). After moving modules to device in `_load_reid_model` (lines 84–90), if `self.fp16`, call `.half()` on each module (`backbone`, `channel_attention`, `bottleneck`, `part_bottlenecks`, plus the fallback path's modules). In `extract_features` and `extract_features_batch`, cast the input tensor to half when `self.fp16` (`tensor = tensor.half()` after `.to(self.device)`), and cast `combined = combined.float()` before `normalize`/`.cpu().numpy()` so the return is float32:

```python
# reid.py — PersonReIDModel.__init__
    def __init__(self, device: str = "cpu", fp16: bool = False):
        self.device = device
        self.fp16 = bool(fp16) and device == "cuda"
        self.feature_dim = 512
        ...
```
```python
# reid.py — _load_reid_model, after the .to(self.device) block (lines 85-90)
            if self.fp16:
                self.backbone.half()
                self.channel_attention.half()
                self.bottleneck.half()
                self.part_bottlenecks.half()
```
```python
# reid.py — extract_features / extract_features_batch, after `.to(self.device)`
            if self.fp16:
                tensor = tensor.half()
            ...
            # before normalize:
            combined = combined.float()
            combined = torch.nn.functional.normalize(combined, p=2, dim=1)
```
  (BatchNorm in eval mode runs fine in half on CUDA; if a specific op rejects half, keep that op in float by upcasting locally — but the standard backbone+linear+BN path is half-safe. For the fallback ResNet18 path, half its modules too.)

- [ ] **Step 1 — Run to pass.**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_reid_fp16.py -v
  ```
  Expected: both fp16 tests PASS (or SKIP without CUDA/backbone).

- [ ] **Step 1 — Commit.** `git add -A && git commit -m "feat(vision_track): fp16 ReID forward (L2-normalized float32 output)"`.

- [ ] **Step 2 — Plumb `reid_fp16` through AppearanceExtractor → YOLOTracker → node param.** Thread the flag end-to-end with default `True`:

```python
# reid.py — AppearanceExtractor.__init__
    def __init__(self, device: str = "cpu", reid_fp16: bool = False):
        self.device = device
        self.person_reid = PersonReIDModel(device, fp16=reid_fp16)
        self._load_general_feature_extractor()
```
```python
# yolo_tracker.py — __init__ signature: add reid_fp16: bool = True; store self.reid_fp16
# yolo_tracker.py — line 113 construction:
            self.appearance_extractor = AppearanceExtractor(self.device, reid_fp16=self.reid_fp16)
```
```python
# person_track_node.py — _declare_parameters, after reid_mode (line 129)
        self.declare_parameter('reid_fp16', True)  # half-precision ReID forward (CUDA only)
# person_track_node.py — _load_parameters, after reid_mode (line 157)
        self.reid_fp16 = self.get_parameter('reid_fp16').value
# person_track_node.py — _init_tracker, custom-ReID branch (after line 212 reid_verification_interval=...)
                    reid_fp16=self.reid_fp16,
```

- [ ] **Step 2 — Run to pass (smoke).**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "from vision_track import yolo_tracker; from vision_track.reid.reid import AppearanceExtractor; import inspect; assert 'reid_fp16' in inspect.signature(AppearanceExtractor.__init__).parameters; assert 'reid_fp16' in inspect.signature(yolo_tracker.YOLOTracker.__init__).parameters; print('OK')"
  ```
  Expected: `OK`.

- [ ] **Step 2 — Build + manual node smoke (run-to-pass, hardware-best-effort).** Build via the wrapper so install-tree shebangs see the venv, then confirm the node declares the param and starts:
  ```
  /home/tinker/tk25_ws/src/tk26_vision/scripts/build.sh --packages-select vision_track
  ```
  Then (T1-style, no cameras needed for param check): `ros2 run vision_track person_track_server --ros-args -p reid_fp16:=true` should log `Person Track Node initialized successfully` and not error on the param. Record the result; this is best-effort (depends on a working ROS env), not a hard gate.

- [ ] **Step 2 — Commit.** `git add -A && git commit -m "feat(vision_track): reid_fp16 ROS param (default True) plumbed to AppearanceExtractor"`.

---

### Task 4 — Optional: TensorRT engine export for YOLO (best-effort top-end)

A documented, scripted export path (`yolo11s-seg.pt` → `.engine`, FP16) plus a node param to load the engine. Clearly marked optional/best-effort — TensorRT engines are resolution/batch-locked and hardware-specific, so there is **no hard unit test**; verification is a manual step.

**Files:**
- `src/vision_track/scripts/export_yolo_trt.py` — NEW.
- `src/vision_track/vision_track/person_track_node.py` — `model_path` already accepts an arbitrary path (line 120 / `resolve_weights` line 183); Ultralytics `YOLO(...)` (`yolo_tracker.py:210`) loads `.engine` transparently — document that pointing `model_path` at a `.engine` is the load mechanism, no code change needed beyond a guard note.

- [ ] **Step 1 — Write the export script.** Pure CLI, no ROS. Exports an FP16 TensorRT engine at a fixed `imgsz` so the runtime resolution must match:

```python
# scripts/export_yolo_trt.py
#!/usr/bin/env python3
"""Export a YOLO seg model to a FP16 TensorRT engine (OPTIONAL top-end speedup).

The engine is RESOLUTION- and BATCH-LOCKED to the imgsz used here; the live node
MUST run YOLO at the same imgsz (person_track_node `inference_size` param). This
is hardware-specific (built for THIS GPU/TensorRT version) and is not portable —
re-export on each deployment box. Best-effort: if TensorRT is absent, the .pt
model continues to work unchanged.

Usage:
    export_yolo_trt.py --model yolo11s-seg.pt --imgsz 736 --out yolo11s-seg.engine
Verify (manual):
    ros2 run vision_track person_track_server --ros-args \
        -p model_path:=/abs/path/yolo11s-seg.engine -p inference_size:=736
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=736)
    ap.add_argument("--out", default=None, help="optional rename for the produced .engine")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.model)
    engine_path = model.export(format="engine", half=True, imgsz=args.imgsz, device=0)
    print(f"Exported TensorRT engine: {engine_path}")
    if args.out and args.out != str(engine_path):
        import shutil
        shutil.copyfile(engine_path, args.out)
        print(f"Copied to: {args.out}")
    print(
        "RUN: ros2 run vision_track person_track_server --ros-args "
        f"-p model_path:=<abs>/{args.out or engine_path} -p inference_size:={args.imgsz}"
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 1 — Smoke (no engine build) — argparse only.** Building a real engine needs TensorRT + the target GPU and takes minutes; the unit-level check is just that the script parses and imports cleanly:
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python scripts/export_yolo_trt.py --help
  ```
  Expected: argparse usage text printed, exit 0. (Do NOT run a full `--model ... ` export in CI — it is slow + hardware-bound.)

- [ ] **Step 2 — Document the load path + manual verification.** Add a short note to `src/tk26_vision/CLAUDE.md` (vision_track Configuration bullet) and `src/tk26_vision/DEV_NOTES.md`: pointing `model_path` at a `.engine` makes Ultralytics load TensorRT; the engine is imgsz/batch-locked so `inference_size` must match the export `--imgsz`; best-effort and per-box. State explicitly that this is **optional** and the `.pt` path is the default/fallback. No code change to `_load_model`/`YOLO(...)` is required (Ultralytics handles `.engine`); add a one-line guard log in `_load_model` (`yolo_tracker.py:209`) noting when a `.engine` is loaded so misuse (engine without matching imgsz) is diagnosable:
  ```python
  # yolo_tracker.py — _load_model, after logger.info(f"Loading YOLO model: {model_path}")
              if str(model_path).endswith(".engine"):
                  logger.warning(
                      "Loading a TensorRT engine — runtime imgsz MUST match the "
                      "export imgsz (resolution/batch-locked)."
                  )
  ```

- [ ] **Step 2 — Run to pass (import smoke).**
  ```
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "from vision_track import yolo_tracker; print('OK')"
  ```
  Expected: `OK`.

- [ ] **Step 2 — Commit.** `git add -A && git commit -m "feat(vision_track): optional TensorRT YOLO export script + .engine load note (best-effort)"`.

- [ ] **Step 3 — (Manual, hardware) Build + verify an engine.** Operator-in-the-loop, not CI: run `export_yolo_trt.py --model yolo11s-seg.pt --imgsz 736`, then start the node with `model_path:=<abs>.engine inference_size:=736` against live cameras and confirm detections appear + measure `throughput_hz` via the Phase-0 `perf_logging_enabled` per-stage timing. Record the before/after Hz in `DEV_NOTES.md`. This step has no automated assertion.

---

## Acceptance

### Now-testable (must pass before merge)

- **Batch-equivalence (Task 1):** `test/test_reid_batch.py` — `extract_features_batch` row-equals the per-crop `extract_features` loop within `atol=1e-4` (torch-gated; SKIPs cleanly if torch/OSNet weights unavailable). The pure crop-stacking shape tests (`_stack_crops`) pass without a model. `AppearanceExtractor.extract_features_batch` returns dicts identical to the per-detection loop (color byte-identical, deep within tolerance).
- **Cache correctness (Task 2):** `test/test_embedding_cache.py` — hit/miss, per-`frame_seq` invalidation (new frame drops old entries), stale-frame_seq read misses, bounded LRU eviction, `clear`, and the pipeline-level "cached features skip re-embed" test (call-count == 1). All pure dict/LRU, **NO torch required**.
- **fp16 tolerance (Task 3):** `test/test_reid_fp16.py` — fp16 output is float32 + L2-normalized (‖v‖≈1, `atol=1e-3`); fp16 vs fp32 cosine > 0.999 (torch+CUDA-gated; SKIPs without CUDA/backbone).
- **No regressions / ROS-free invariant:** the new modules (`embedding_cache.py`, the batch/cache helpers) import with NO top-level `rclpy`; existing `test_flake8.py` / `test_pep257.py` / `test_copyright.py` still pass; `vision_track` builds via `scripts/build.sh --packages-select vision_track`.
- **TensorRT export script (Task 4):** `export_yolo_trt.py --help` parses and exits 0 (no engine build in CI).

Full run command (gated tests SKIP cleanly where deps absent):
```
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_embedding_cache.py test/test_reid_batch.py test/test_reid_fp16.py -v
```

### Arena-deferred (cannot confirm until Orbbec recordings exist)

- **Sustained `throughput_hz` ≥ 12** in 3–4-person re-ID scenes (PASS bar; WARN ≥ 8), measured on the deployment GPU (RTX 5070 Ti) via the Phase-0 `perf_logging_enabled` per-stage timers — the ReID-embed stage budget should drop from 5–7 forward passes/frame to ≤1 batched forward/frame. This is the headline Phase-3 goal and is gated by the ptbench `throughput_hz` metric (`ptbench/common/scoreboard.py`) once arena bags are recorded.
- **TensorRT top-end (optional):** measured before/after Hz with a matched-imgsz `.engine`, recorded in `DEV_NOTES.md`. Best-effort; not required for the ≥12 Hz target if batching+cache+fp16 already clear it.

> **Invariant reminder:** Phase 3 is pure optimization. None of these changes may alter tracking *decisions* — the batched embed must be numerically equivalent to the per-crop loop, the cache must return exactly what a fresh embed would, and fp16 must stay within cosine 0.999 of fp32. If any arena wrong-lock / correct-lock metric regresses versus Phase 2, treat it as a Phase-3 correctness bug (likely a cache key collision on reused/negative track IDs or an fp16 op that silently changed a threshold comparison), not a tuning issue.
