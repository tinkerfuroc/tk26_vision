# Person Tracker — Phase 1: ReID Backbone + Identity Gating — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the untrained random ReID head with a genuinely pretrained OSNet backbone and add identity-gated association so `wrong_lock_episodes` drops toward 0 and `correct_lock_rate` rises — without changing the `/track_person` action contract.

**Architecture:** `PersonReIDModel` is refactored to wrap a pluggable backbone behind the unchanged `extract_features(crop) → L2-normalized np.ndarray` interface; a torchreid OSNet path (`osnet_ain_x1_0` default, `osnet_x0_25` alt) loads real pretrained weights via a cached resolver, and the random-head path is deleted. Fusion weights/thresholds are recalibrated for the now-reliable deep term, YOLO detection conf is decoupled from ByteTrack's two-stage recovery via a project `bytetrack.yaml`/`default.yaml`, gallery inserts are quality-gated, and a Lowe-style ratio test plus a stronger distinctiveness margin stop spatial proximity from overriding identity.

**Tech Stack:** Python 3.10, pytest, numpy, torch, torchreid, ROS2 Humble (rclpy), ultralytics.

> **Phase 0 dependency (hard prerequisite).** This plan assumes Phase 0 (`docs/superpowers/plans/2026-06-03-person-tracker-phase0-*.md`) is **merged**. Phase 0 introduced: `centroid_field`/`centroid_track` dual-GT + schema 1.1, the `pos_error_range_m` gate in `ptbench/common/scoreboard.py`, the `perf_logging_enabled` node param, median-axis `_calculate_centroid`, the operator-init heuristic in `yolo_tracker.py:385-389`, and the `reid_mode='native'` loud-guard. Do **not** re-introduce or redefine those names here. The ptbench `action`-backend smoke and the existing test suite must be green before starting Phase 1.

> **Environment.** Venv python (absolute) is `/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python` (call it **VENV** below). Worktree root (**WT**) is `/home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker` on branch `feat/person-tracker-overhaul`. Pure unit tests are ROS-free (no top-level `rclpy` import) and run with `cd WT/src/vision_track && VENV -m pytest test/test_<name>.py -v`. Model-constructing / network-touching tests **must** be gated with `pytest.importorskip("torchreid")` / `skipif` so the suite stays green offline.

---

## File Structure

```
WT/
  docs/superpowers/plans/
    2026-06-03-person-tracker-phase1-reid-identity.md   # this plan
  src/vision_track/
    requirements.txt                                    # EDIT: add torchreid + gdown
    config/                                             # NEW dir (Task 2)
      default.yaml                                      # NEW: vision_track ROS params
      bytetrack.yaml                                    # NEW: project ByteTrack config
    setup.py                                            # EDIT: install config/*.yaml to share/
    vision_track/
      reid/
        reid.py                # EDIT: pluggable backbone, OSNet path, fusion re-weight, thresholds (Task 1)
        reid_backbone.py       # NEW: OSNet wrapper + cached weight resolver (Task 1)
        reid_search.py         # EDIT: Lowe ratio + deep-margin spatial gate (Task 4)
        appearance_manager.py  # EDIT: quality-gated gallery inserts (Task 3)
      core/
        registry.py            # EDIT: distinctiveness_threshold 0.03 -> 0.10 (Task 4)
      yolo_tracker.py          # EDIT: yolo_track_conf -> model.track, project bytetrack.yaml (Task 2)
      person_track_node.py     # EDIT: declare reid_backbone + yolo_track_conf params (Tasks 1,2)
    test/
      test_reid_backbone.py    # NEW: torch-gated integration (shape + L2-norm) (Task 1)
      test_fusion_weights.py   # NEW: pure-logic weight normalization (Task 1)
      test_appearance_quality_gate.py  # NEW: pure-logic gallery-insert gate (Task 3)
      test_reid_ratio_gate.py  # NEW: pure-logic Lowe ratio + deep-margin (Task 4)
      test_distinctiveness_margin.py   # NEW: pure-logic registry margin (Task 4)
  src/tk26_vision/CLAUDE.md    # EDIT: document torchreid weight cache + freeze-lock note (Task 1)
```

**Notes on conventions verified in-code:**
- `vision_track/setup.py` currently has **no** `config/` entry and uses `from glob import glob` (no `import os`). `object_detection_new/setup.py` is the template: `(os.path.join('share', package_name, 'config'), glob('config/*.yaml'))` — Task 2 mirrors it (adds `import os`).
- `YOLOTracker.__init__` (`yolo_tracker.py:63-73`) signature: `confidence_threshold=0.5`, `inference_size`, `reid_verification_interval`, no backbone/track-conf params yet. `self.confidence_threshold` is passed straight to `model.track(conf=...)` at `yolo_tracker.py:298`.
- `AppearanceExtractor.__init__(device)` (`reid.py:208`) constructs `PersonReIDModel(device)` (`reid.py:218`). `PersonReIDModel.__init__(device='cpu')` (`reid.py:25`) — Task 1 adds a `backbone_name` arg threaded from the node param.
- `self.reid_threshold = ReIDMatcher.REID_THRESHOLD` (`yolo_tracker.py:126`); `set_reid_threshold` exists at `yolo_tracker.py:855-863`.
- Venv check (confirmed at plan time): `torch` present, `torchreid` **absent**, `gdown` **absent**. Freeze-lock convention exists: `.venv-da3/freeze.lock.txt`. There is no `.venv-vision-main/freeze.lock.txt` yet; Task 1 establishes one for this venv.

---

### Task 1 — ReID backbone abstraction (OSNet) + fusion re-weight + threshold recalibration

**Files:**
- `WT/src/vision_track/requirements.txt` (append after `torchvision>=0.15.0`, current line ~14)
- `WT/src/vision_track/vision_track/reid/reid_backbone.py` (NEW)
- `WT/src/vision_track/vision_track/reid/reid.py` (`PersonReIDModel` __init__ `25-37`, `_load_reid_model` `39-104`, `_load_fallback_model` `106-131`, `extract_features` `133-192`; fusion weights `590-595`; thresholds `608,617,621-623`)
- `WT/src/vision_track/vision_track/person_track_node.py` (param block `120-129`, read block `150-157`, tracker construct `207-213`)
- `WT/src/vision_track/test/test_reid_backbone.py` (NEW, torch-gated)
- `WT/src/vision_track/test/test_fusion_weights.py` (NEW, pure-logic)
- `WT/src/tk26_vision/CLAUDE.md` (Environment section — weight-cache + freeze-lock note)

### Step 1.1 — Install torchreid into the venv + record the freeze-lock

- [ ] **Step 1** Install torchreid (and `gdown`, which torchreid uses to pull OSNet weights from Google Drive) into VENV. torchreid is published on PyPI as `torchreid` (the `deep-person-reid` project). Run:
  ```bash
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pip install torchreid gdown
  ```
  Expected: resolves and installs `torchreid` + `gdown` against the existing `torch`/`torchvision`/`numpy==2.2.6` in the venv. If torchreid's metadata pins `numpy<2` or an incompatible torch, install with `--no-deps` and hand-install only the genuinely-missing transitive deps (torchreid's runtime needs are `numpy`, `Pillow`, `six`, `scipy`, `gdown`, `Cython` — all but `gdown` already present) — record the deviation in the CLAUDE.md note (Step 6). Do **not** let the install downgrade `numpy` (the rest of the vision tree depends on the 2.x ABI per `src/tk26_vision/CLAUDE.md`).
- [ ] **Step 2** Verify the import and that OSNet is constructible offline-or-cached:
  ```bash
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -c "import torchreid; from torchreid.reid.models import build_model; print('OK', torchreid.__version__)" 2>/dev/null
  ```
  Expected: `OK <version>`. (Newer torchreid exposes `torchreid.models.build_model`; older exposes `torchreid.reid.models.build_model`. Note which path resolves — `reid_backbone.py` will try both.)
- [ ] **Step 3** Record the freeze-lock so future installs diff against a known-good state (matches the `.venv-da3/freeze.lock.txt` convention):
  ```bash
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pip freeze > /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/freeze.lock.txt
  ```
  This file is git-ignored (venv dir), so it is a local source-of-truth, not a committed artifact.
- [ ] **Step 4** Append to `WT/src/vision_track/requirements.txt` (after the `torchvision>=0.15.0` line):
  ```
  # Person ReID backbone (OSNet) — pretrained weights via torchreid
  torchreid>=1.4.0
  gdown>=4.0.0
  ```
- [ ] **Step 5** Commit:
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/requirements.txt
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "build(vision_track): add torchreid + gdown for OSNet ReID backbone

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

### Step 1.2 — Cached OSNet backbone wrapper (`reid_backbone.py`)

- [ ] **Step 1 (failing test — torch-gated integration)** Create `WT/src/vision_track/test/test_reid_backbone.py`. The test gates on torchreid so the suite stays green offline, and asserts the embedding contract: a single `[D]` float32 vector with L2 norm == 1 for a non-empty crop, and a zero vector of the right dim for a degenerate crop.
  ```python
  import numpy as np
  import pytest

  torchreid = pytest.importorskip("torchreid")  # skip offline / if not installed

  from vision_track.reid.reid_backbone import OSNetBackbone, build_reid_backbone


  @pytest.fixture(scope="module")
  def backbone():
      # osnet_x0_25 is the smallest variant — fastest to construct/download in CI.
      return build_reid_backbone("osnet_x0_25", device="cpu")

  def test_extract_features_shape_and_l2_norm(backbone):
      crop = (np.random.rand(200, 80, 3) * 255).astype(np.uint8)
      feat = backbone.extract_features(crop)
      assert feat.ndim == 1
      assert feat.dtype == np.float32
      assert feat.shape[0] == backbone.feature_dim
      assert backbone.feature_dim > 0
      assert abs(float(np.linalg.norm(feat)) - 1.0) < 1e-4

  def test_feature_dim_is_stable_across_calls(backbone):
      a = backbone.extract_features((np.random.rand(150, 60, 3) * 255).astype(np.uint8))
      b = backbone.extract_features((np.random.rand(300, 120, 3) * 255).astype(np.uint8))
      assert a.shape == b.shape == (backbone.feature_dim,)

  def test_build_unknown_backbone_raises():
      with pytest.raises(ValueError):
          build_reid_backbone("not_a_real_backbone", device="cpu")

  def test_osnet_backbone_type(backbone):
      assert isinstance(backbone, OSNetBackbone)
  ```
- [ ] **Step 2 (run-to-fail)** Run:
  ```bash
  cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest test/test_reid_backbone.py -v 2>/dev/null
  ```
  Expected: collection error / `ModuleNotFoundError: No module named 'vision_track.reid.reid_backbone'` (the module does not exist yet). If torchreid is somehow absent, the file would `skip` at `importorskip` — but Step 1.1 installed it, so expect the import-failure on `reid_backbone`.
- [ ] **Step 3 (minimal impl)** Create `WT/src/vision_track/vision_track/reid/reid_backbone.py`:
  ```python
  """Pluggable ReID feature backbones for PersonReIDModel.

  The OSNet path loads genuine pretrained Re-ID weights via torchreid's
  pretrained-model cache (downloaded once to ~/.cache/torch/checkpoints or the
  torchreid default, then reused). This replaces the legacy random-head path in
  reid.py whose deep term was an untrained projection of ImageNet features.
  """
  import logging
  from typing import Protocol

  import cv2
  import numpy as np
  import torch

  logger = logging.getLogger(__name__)

  # torchreid input convention for person ReID: HxW = 256x128 (h>w).
  _REID_H, _REID_W = 256, 128
  _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
  _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

  # Supported OSNet variants and their published embedding dims.
  _OSNET_DIMS = {
      "osnet_ain_x1_0": 512,
      "osnet_x1_0": 512,
      "osnet_x0_75": 512,
      "osnet_x0_5": 512,
      "osnet_x0_25": 512,
  }


  class ReIDBackbone(Protocol):
      feature_dim: int
      def extract_features(self, crop: np.ndarray) -> np.ndarray: ...


  def _resolve_build_model():
      """torchreid moved build_model between versions; try both paths."""
      try:
          from torchreid.reid.models import build_model  # >=1.4 layout
          return build_model
      except Exception:  # pragma: no cover - version-dependent
          from torchreid.models import build_model  # older layout
          return build_model


  class OSNetBackbone:
      """OSNet feature extractor returning L2-normalized embeddings."""

      def __init__(self, backbone_name: str, device: str = "cpu"):
          if backbone_name not in _OSNET_DIMS:
              raise ValueError(
                  f"Unknown ReID backbone '{backbone_name}'. "
                  f"Supported: {sorted(_OSNET_DIMS)}"
              )
          self.backbone_name = backbone_name
          self.device = device
          self.feature_dim = _OSNET_DIMS[backbone_name]

          build_model = _resolve_build_model()
          # num_classes is irrelevant for feature extraction; pretrained=True
          # triggers the cached weight download (torchreid pulls OSNet weights
          # from its hosted Google-Drive mirror via gdown on first use).
          model = build_model(
              name=backbone_name,
              num_classes=1,
              pretrained=True,
          )
          model.eval()
          model.to(self.device)
          self.model = model

      def extract_features(self, crop: np.ndarray) -> np.ndarray:
          if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
              return np.zeros(self.feature_dim, dtype=np.float32)

          resized = cv2.resize(crop, (_REID_W, _REID_H))
          norm = (resized.astype(np.float32) / 255.0 - _IMAGENET_MEAN) / _IMAGENET_STD
          tensor = torch.from_numpy(norm).permute(2, 0, 1).unsqueeze(0).to(self.device)

          with torch.no_grad():
              feat = self.model(tensor)            # [1, feature_dim]
              feat = torch.nn.functional.normalize(feat, p=2, dim=1)
          return feat.cpu().numpy().flatten().astype(np.float32)


  def build_reid_backbone(backbone_name: str, device: str = "cpu") -> "ReIDBackbone":
      """Factory for the configured ReID backbone. Currently OSNet only."""
      return OSNetBackbone(backbone_name, device=device)
  ```
- [ ] **Step 4 (run-to-pass)** Run the same command from Step 2. Expected: the 4 tests pass (first run downloads OSNet weights — allow network; subsequent runs use cache). If the environment is fully offline and weights are not cached, the tests will error on download, **not** skip — note this in the run output and fetch weights once on a connected host; the cached weights persist in the torchreid checkpoint dir.
- [ ] **Step 5 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/reid/reid_backbone.py src/vision_track/test/test_reid_backbone.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): OSNet ReID backbone with cached pretrained weights

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

### Step 1.3 — Rewire `PersonReIDModel` onto the backbone; remove the random head

- [ ] **Step 1 (failing test — torch-gated)** Append to `WT/src/vision_track/test/test_reid_backbone.py` a test that constructs `PersonReIDModel` with the alt backbone param and asserts `extract_features` still returns an L2-normalized 1-D vector matching `feature_dim` (this is the contract `reid_search`/`appearance_manager` depend on):
  ```python
  def test_person_reid_model_uses_backbone():
      from vision_track.reid.reid import PersonReIDModel
      m = PersonReIDModel(device="cpu", backbone_name="osnet_x0_25")
      crop = (np.random.rand(180, 70, 3) * 255).astype(np.uint8)
      feat = m.extract_features(crop)
      assert feat.shape == (m.feature_dim,)
      assert abs(float(np.linalg.norm(feat)) - 1.0) < 1e-4
  ```
- [ ] **Step 2 (run-to-fail)** `cd .../src/vision_track && VENV -m pytest test/test_reid_backbone.py::test_person_reid_model_uses_backbone -v 2>/dev/null`. Expected: `TypeError` — `PersonReIDModel.__init__()` got an unexpected keyword `backbone_name` (current signature is `__init__(self, device="cpu")` at `reid.py:25`).
- [ ] **Step 3 (minimal impl)** Edit `reid.py`:
  - `PersonReIDModel.__init__` (`reid.py:25-37`) — add `backbone_name: str = "osnet_ain_x1_0"`, store it, drop `self.use_deep_features`/`self.feature_dim = 512` guesswork; delegate to the backbone:
    ```python
    def __init__(self, device: str = "cpu", backbone_name: str = "osnet_ain_x1_0"):
        self.device = device
        self.backbone_name = backbone_name
        from .reid_backbone import build_reid_backbone
        self.backbone = build_reid_backbone(backbone_name, device=device)
        self.feature_dim = self.backbone.feature_dim
    ```
  - **Delete** `_load_reid_model` (`39-104`) and `_load_fallback_model` (`106-131`) entirely (random-head removal — the master root cause).
  - Replace `extract_features` (`133-192`) body with a thin delegate:
    ```python
    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        """Extract an L2-normalized ReID embedding from a person crop (RGB)."""
        return self.backbone.extract_features(crop)
    ```
  - In `AppearanceExtractor.__init__` (`reid.py:208`), thread the backbone name through: change the signature to `def __init__(self, device: str = "cpu", reid_backbone: str = "osnet_ain_x1_0"):` and the construction at `reid.py:218` to `self.person_reid = PersonReIDModel(device, backbone_name=reid_backbone)`.
- [ ] **Step 4 (run-to-pass)** Re-run Step 2's command. Expected: pass. Then run the full module `VENV -m pytest test/test_reid_backbone.py -v 2>/dev/null` — all pass.
- [ ] **Step 5 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/reid/reid.py src/vision_track/test/test_reid_backbone.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "refactor(vision_track): PersonReIDModel wraps pluggable backbone; remove random head

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

### Step 1.4 — Thread `reid_backbone` ROS param through the node

- [ ] **Step 1 (impl — no separate unit test; covered by T1 startup + Task-2 default.yaml)** In `person_track_node.py`:
  - In `_declare_parameters` (after `reid_mode` at line `129`): `self.declare_parameter('reid_backbone', 'osnet_ain_x1_0')`.
  - In the read block (after `self.reid_mode = ...` at line `157`): `self.reid_backbone = self.get_parameter('reid_backbone').value`.
  - Pass it into the custom-tracker construction (`person_track_node.py:207-213`): add `reid_backbone=self.reid_backbone` to the `YOLOTracker(...)` kwargs.
- [ ] **Step 2 (thread through YOLOTracker → AppearanceExtractor)** In `yolo_tracker.py`:
  - `__init__` (`63-73`): add `reid_backbone: str = "osnet_ain_x1_0"` param; store `self.reid_backbone = reid_backbone`.
  - Construction at `yolo_tracker.py:113`: change to `self.appearance_extractor = AppearanceExtractor(self.device, reid_backbone=self.reid_backbone)`.
- [ ] **Step 3 (verify import-clean)** `cd .../src/vision_track && VENV -c "import vision_track.yolo_tracker" 2>/dev/null` — expect no error (no top-level rclpy in yolo_tracker; person_track_node imports rclpy so test it under ROS env in T1, not here).
- [ ] **Step 4 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/yolo_tracker.py src/vision_track/vision_track/person_track_node.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): plumb reid_backbone param node -> tracker -> extractor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

### Step 1.5 — Fusion-weight normalization + recalibrated thresholds (pure-logic TDD)

The fusion math at `reid.py:781-822` normalizes the active weight subset (`weights / np.sum(weights)`), so re-weighting is safe even when some feature terms are absent. The pure test below pins the **normalization invariant** and the **new weight ordering** (deep dominates) without touching the model.

- [ ] **Step 1 (failing test — pure-logic)** Create `WT/src/vision_track/test/test_fusion_weights.py`:
  ```python
  import numpy as np
  from vision_track.reid.reid import ReIDMatcher


  def test_person_weights_sum_to_one_after_normalization():
      w = np.array([
          ReIDMatcher.WEIGHT_REID,
          ReIDMatcher.WEIGHT_BODY_COLOR,
          ReIDMatcher.WEIGHT_COLOR,
          ReIDMatcher.WEIGHT_UPPER,
          ReIDMatcher.WEIGHT_LOWER,
      ])
      assert abs(float(np.sum(w / np.sum(w))) - 1.0) < 1e-9

  def test_deep_term_dominates_after_reweight():
      # Phase 1: with a trained backbone the deep term must dominate color.
      assert ReIDMatcher.WEIGHT_REID >= 0.70
      color_total = (
          ReIDMatcher.WEIGHT_BODY_COLOR
          + ReIDMatcher.WEIGHT_COLOR
          + ReIDMatcher.WEIGHT_UPPER
          + ReIDMatcher.WEIGHT_LOWER
      )
      assert ReIDMatcher.WEIGHT_REID > color_total

  def test_raw_reid_floor_raised_for_trained_backbone():
      # Trained OSNet separates same/different far better than the random head,
      # so the raw-cosine floor can be raised from the legacy 0.60.
      assert ReIDMatcher.MIN_REID_SIMILARITY_RAW >= 0.30
      assert ReIDMatcher.MIN_REID_SIMILARITY_RAW <= 0.55

  def test_color_hard_floors_relaxed():
      # Color is now a backup cue, not a gate — its hard floors must not
      # hard-reject a true match on lighting/clothing variation.
      assert ReIDMatcher.MIN_BODY_COLOR_SIMILARITY <= 0.45
      assert ReIDMatcher.MIN_UPPER_SIMILARITY <= 0.45
      assert ReIDMatcher.MIN_LOWER_SIMILARITY <= 0.45
  ```
- [ ] **Step 2 (run-to-fail)** `cd .../src/vision_track && VENV -m pytest test/test_fusion_weights.py -v 2>/dev/null`. Expected failures: `test_deep_term_dominates_after_reweight` (current `WEIGHT_REID=0.55 < 0.70`), `test_raw_reid_floor_raised...` (current `0.60 > 0.55`), `test_color_hard_floors_relaxed` (current `0.60 > 0.45`). `test_person_weights_sum_to_one...` already passes (normalization is dynamic).
- [ ] **Step 3 (minimal impl)** Edit `reid.py` class-level constants (`590-595`, `617`, `621-623`). Replace with the recalibrated values (deep dominates; color demoted to backup; hard color floors relaxed so they no longer gate; raw floor lowered to OSNet's same/different operating point — final value tuned by the offline Occluded-REID ROC in Step 5, these are the starting points):
  ```python
  WEIGHT_REID = 0.75        # trained OSNet deep features now dominate
  WEIGHT_BODY_COLOR = 0.13  # clothing backup
  WEIGHT_COLOR = 0.05
  WEIGHT_UPPER = 0.04
  WEIGHT_LOWER = 0.03
  WEIGHT_SIZE = 0.0
  ...
  REID_THRESHOLD = 0.55     # combined floor; OSNet shifts the operating point — retune in Step 5
  ...
  MIN_REID_SIMILARITY_RAW = 0.40   # OSNet raw-cosine same/different gap is wide; retune in Step 5
  ...
  MIN_BODY_COLOR_SIMILARITY = 0.40 # backup cue, must not hard-reject true matches
  MIN_UPPER_SIMILARITY = 0.40
  MIN_LOWER_SIMILARITY = 0.40
  ```
  Note: `self.reid_threshold` reads `ReIDMatcher.REID_THRESHOLD` at construction (`yolo_tracker.py:126`), so changing the class constant propagates. The single-person guard `0.72` in `reid_search.py:339` is Phase-2 scope (recovery policy) — leave it for now.
- [ ] **Step 4 (run-to-pass)** Re-run Step 2's command. Expected: all 4 pass.
- [ ] **Step 5 (offline Occluded-REID calibration — knob, NOT a gate)** Drive a same-vs-different cosine ROC on the Occluded-REID dataset (or any held-out person-crop pairs) with the OSNet backbone to choose the final `MIN_REID_SIMILARITY_RAW`, `REID_THRESHOLD`, and `REID_MARGIN`. Procedure: extract embeddings for N same-identity pairs and N cross-identity pairs via `OSNetBackbone.extract_features`, compute cosine, pick `MIN_REID_SIMILARITY_RAW` at the cross-identity 95th percentile and `REID_THRESHOLD` where same-identity recall ≥ 0.9 at false-match ≤ 0.05. Record the chosen numbers + ROC plot path in the commit body. **This calibration informs the constants but is never a CI gate** (per `person-tracker-benchmark-strategy`: academic ReID sets are tuning knobs only). If no dataset is reachable offline, ship the Step-3 starting values and flag the retune as arena-deferred.
- [ ] **Step 6 (CLAUDE.md note)** Edit `WT/src/tk26_vision/CLAUDE.md` — in the Environment section, add a short subsection documenting: (a) `torchreid` + `gdown` are now `.venv-vision-main` deps (`pip install -r src/vision_track/requirements.txt`); (b) OSNet pretrained weights are fetched once via torchreid's gdown mirror and cached under the torchreid checkpoint dir (`~/.cache/torch/checkpoints` or torchreid default) — pre-warm on a connected host before offline runs; (c) the new `.venv-vision-main/freeze.lock.txt` is the diff-target for future installs (same convention as `.venv-da3/freeze.lock.txt`); (d) any `--no-deps` deviation from Step 1.1.
- [ ] **Step 7 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/reid/reid.py src/vision_track/test/test_fusion_weights.py src/tk26_vision/CLAUDE.md
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): re-weight fusion for trained OSNet; recalibrate ReID floors

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

### Task 2 — Decouple YOLO detection conf from ByteTrack two-stage recovery

**Root cause (spec §1.2 #3):** `model.track(conf=0.5)` (`yolo_tracker.py:298`) strips boxes *before* ByteTrack's two-stage association runs, so its `track_low_thresh=0.1` low-conf recovery bin is always empty. A partially-occluded operator (conf 0.3–0.45) is dropped → new ID on re-entry. Fix: feed a **low** detection conf (`yolo_track_conf ~0.15`) to `model.track`, and supply a **project** `bytetrack.yaml` (so the high/low thresholds and buffer are ours, not stock Ultralytics). The custom new-target IoU logic in `initialize_tracking` still selects the operator from the (now larger) candidate set — no separate high gate is needed there because operator-init uses the Phase-0 centeredness/nearness heuristic over class-person detections, which is robust to the extra low-conf boxes; if a downstream consumer needs a stricter detection gate it reads `confidence_threshold` separately.

**Files:**
- `WT/src/vision_track/config/bytetrack.yaml` (NEW)
- `WT/src/vision_track/config/default.yaml` (NEW)
- `WT/src/vision_track/setup.py` (add `import os`; add config glob to `data_files`)
- `WT/src/vision_track/vision_track/yolo_tracker.py` (`__init__` `63-73`; `track` `297-311`; add tracker-config resolution)
- `WT/src/vision_track/vision_track/person_track_node.py` (declare/read `yolo_track_conf`; pass through)
- `WT/src/vision_track/test/test_bytetrack_config.py` (NEW, pure-logic YAML/shape assertions)

### Step 2.1 — Project `bytetrack.yaml` + `default.yaml`

- [ ] **Step 1 (failing test — pure-logic)** Create `WT/src/vision_track/test/test_bytetrack_config.py`. It locates the in-source config dir (the tests run from `src/vision_track/`, so `config/` is a sibling of `test/`) and asserts the low-conf recovery is actually enabled:
  ```python
  import os
  import pytest

  yaml = pytest.importorskip("yaml")

  CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config")


  def _load(name):
      with open(os.path.join(CFG_DIR, name)) as f:
          return yaml.safe_load(f)

  def test_bytetrack_yaml_exists_and_is_bytetrack():
      cfg = _load("bytetrack.yaml")
      assert cfg["tracker_type"] == "bytetrack"

  def test_bytetrack_low_conf_recovery_enabled():
      cfg = _load("bytetrack.yaml")
      # The low bin must sit below the detection conf we pass to model.track (0.15),
      # otherwise the two-stage recovery has nothing to recover.
      assert cfg["track_low_thresh"] <= 0.15
      assert cfg["track_high_thresh"] >= 0.2
      assert cfg["new_track_thresh"] >= cfg["track_high_thresh"]
      assert cfg["track_buffer"] >= 30

  def test_default_yaml_has_phase1_params():
      cfg = _load("default.yaml")["/**"]["ros__parameters"]
      assert cfg["yolo_track_conf"] <= 0.2
      assert cfg["reid_backbone"] in ("osnet_ain_x1_0", "osnet_x0_25")
  ```
- [ ] **Step 2 (run-to-fail)** `cd .../src/vision_track && VENV -m pytest test/test_bytetrack_config.py -v 2>/dev/null`. Expected: `FileNotFoundError` on `config/bytetrack.yaml` (dir does not exist yet).
- [ ] **Step 3 (minimal impl — create config files)** Create `WT/src/vision_track/config/bytetrack.yaml` (project ByteTrack config; keys match Ultralytics' `bytetrack.yaml` schema):
  ```yaml
  # Project ByteTrack config for vision_track person tracking.
  # Decouples detection conf (passed to model.track) from ByteTrack's two-stage
  # association so low-confidence (partially-occluded) operator boxes survive into
  # the second association stage instead of being prefiltered away.
  tracker_type: bytetrack
  track_high_thresh: 0.2    # first-association (high-conf) gate
  track_low_thresh: 0.1     # second-association (low-conf recovery) gate
  new_track_thresh: 0.25    # min conf to spawn a brand-new track
  track_buffer: 45          # frames a lost track is retained before deletion
  match_thresh: 0.8         # IoU match threshold for association
  fuse_score: true          # fuse detection conf into association cost
  ```
  Create `WT/src/vision_track/config/default.yaml` (mirrors `object_detection_new/config/default.yaml` style; this is the canonical param surface for the now-numerous vision_track params — wired into the launch/run convention, not auto-loaded by the node):
  ```yaml
  /**:
    ros__parameters:
      # Model
      model_path: 'yolo11s-seg.pt'
      confidence_threshold: 0.5     # detection conf for non-tracking detect() calls / downstream gates
      yolo_track_conf: 0.15         # LOW conf into model.track so ByteTrack low-conf recovery runs
      enable_reid: true
      inference_size: 1280
      reid_verification_interval: 5

      # ReID backbone (Phase 1)
      reid_backbone: 'osnet_ain_x1_0'   # alt: 'osnet_x0_25'
      reid_mode: 'custom'               # 'custom' or 'native' (native loud-guarded in Phase 0)

      # Loss / recovery
      max_frames_lost: 600
      allow_indefinite_recovery: true
      lost_timeout: 300.0

      # Camera topics
      image_topic: '/camera/color/image_raw'
      depth_topic: '/camera/depth/image_raw'
      camera_info_topic: '/camera/color/camera_info'

      # Loop / output
      tracking_rate: 15.0
      target_point_topic: '/target_points'

      # Logging
      vision_logging_enabled: true
      vision_log_folder: 'vision_log'
      perf_logging_enabled: false       # introduced in Phase 0
  ```
- [ ] **Step 4 (run-to-pass)** Re-run Step 2's command. Expected: all 3 pass.
- [ ] **Step 5 (install config to share/ — setup.py)** Edit `WT/src/vision_track/setup.py`:
  - Change the imports to include `os` (currently only `from glob import glob` + `from setuptools import ...`): add `import os`.
  - In `data_files`, after the `package.xml` line and before the models glob, insert:
    ```python
    (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ```
- [ ] **Step 6 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/config/bytetrack.yaml src/vision_track/config/default.yaml src/vision_track/setup.py src/vision_track/test/test_bytetrack_config.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): project bytetrack.yaml + default.yaml config dir

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

### Step 2.2 — Pass `yolo_track_conf` + project tracker config to `model.track`

- [ ] **Step 1 (impl — YOLOTracker)** Edit `yolo_tracker.py`:
  - `__init__` (`63-73`): add `yolo_track_conf: float = 0.15` param; store `self.yolo_track_conf = yolo_track_conf`.
  - Add a tracker-config resolver: a small method `_resolve_tracker_cfg()` that returns the installed project `bytetrack.yaml` path if found (via `ament_index_python.get_package_share_directory('vision_track')/config/bytetrack.yaml`, guarded by `try/except` so non-ROS unit imports don't break), else falls back to the string `"bytetrack.yaml"` (stock Ultralytics). Store the result on `self.tracker_cfg` in `__init__` after device setup.
    ```python
    def _resolve_tracker_cfg(self) -> str:
        try:
            from ament_index_python.packages import get_package_share_directory
            cfg = os.path.join(get_package_share_directory("vision_track"), "config", "bytetrack.yaml")
            if os.path.exists(cfg):
                return cfg
        except Exception:
            pass
        return "bytetrack.yaml"
    ```
    (add `import os` at top of `yolo_tracker.py` if not already present — it currently imports `logging, time, typing, cv2, numpy, torch`; `os` is **not** imported, so add it.)
  - `track` (`297-311`): change `conf=self.confidence_threshold` → `conf=self.yolo_track_conf`, and `tracker="bytetrack.yaml"` → `tracker=self.tracker_cfg`.
- [ ] **Step 2 (impl — node plumbing)** Edit `person_track_node.py`:
  - `_declare_parameters` (after `confidence_threshold` at line `121`): `self.declare_parameter('yolo_track_conf', 0.15)`.
  - Read block (after `self.confidence_threshold = ...` at line `151`): `self.yolo_track_conf = self.get_parameter('yolo_track_conf').value`.
  - Custom-tracker construction (`207-213`): add `yolo_track_conf=self.yolo_track_conf` to the `YOLOTracker(...)` kwargs. (Leave `confidence_threshold` passed as-is — it remains the detect()-path / downstream gate.)
- [ ] **Step 3 (verify import-clean)** `cd .../src/vision_track && VENV -c "import vision_track.yolo_tracker" 2>/dev/null` — expect no error (the `ament_index` import is inside a guarded method, not module-level).
- [ ] **Step 4 (run existing tests to confirm no regression)** `cd .../src/vision_track && VENV -m pytest test/test_bytetrack_config.py test/test_fusion_weights.py -v 2>/dev/null` — expect all green.
- [ ] **Step 5 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/yolo_tracker.py src/vision_track/vision_track/person_track_node.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): low yolo_track_conf + project bytetrack cfg into model.track

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

### Task 3 — Gallery hygiene: quality-gate appearance-history inserts

**Root cause (spec §1.2 #1, §6.3):** `appearance_manager.py` appends every observation's features into the history deques (`_update_feature_history` `56-78`, `_update_color_histories` `81-114`) with **no quality gate** — a tiny, blurry, or back-view crop poisons the gallery and drags `get_average_feature()`/`get_body_color()` toward garbage. `mask_coverage` is already computed at `reid.py:318-320` but never consumed. Fix: a pure-logic `crop_quality_ok(...)` gate (min crop height, `mask_coverage > 0.4`, Laplacian-variance blur floor, back-view rejection) evaluated in `update_appearance` **before** any history append; failing crops skip the insert (anchor seeding still allowed on the first valid crop only).

**Design of the gate (testable in isolation, no model):** a standalone function in a new module `reid/quality.py` taking primitives so it unit-tests with synthetic numpy:
```
crop_quality_ok(crop_h, crop_w, mask_coverage, blur_var, *, aspect_ratio,
                min_crop_h, min_mask_coverage, min_blur_var, max_aspect_ratio) -> bool
```
Back-view rejection is approximated geometrically here: an unusually wide/short bbox (high aspect_ratio) is the cheap proxy available without a pose model; a true front/back classifier is out of Phase-1 scope, so the gate rejects only degenerate aspect ratios and leaves richer view-direction handling to Phase 2's geometry work. The Laplacian blur variance is computed in `update_appearance` from the actual crop (`cv2.Laplacian(gray, cv2.CV_64F).var()`).

**Files:**
- `WT/src/vision_track/vision_track/reid/quality.py` (NEW — pure function)
- `WT/src/vision_track/vision_track/reid/appearance_manager.py` (`update_appearance` `11-53`; gate before `_update_feature_history`/`_update_color_histories`)
- `WT/src/vision_track/test/test_appearance_quality_gate.py` (NEW, pure-logic)

### Step 3.1 — Pure-logic quality gate

- [ ] **Step 1 (failing test — pure-logic)** Create `WT/src/vision_track/test/test_appearance_quality_gate.py`:
  ```python
  from vision_track.reid.quality import crop_quality_ok, DEFAULT_GATE


  def _ok(**over):
      kw = dict(
          crop_h=180, crop_w=70, mask_coverage=0.6, blur_var=200.0,
          aspect_ratio=70 / 180,
      )
      kw.update(over)
      return crop_quality_ok(**kw, **DEFAULT_GATE)

  def test_good_crop_passes():
      assert _ok() is True

  def test_too_short_crop_rejected():
      assert _ok(crop_h=40) is False

  def test_low_mask_coverage_rejected():
      # spec: mask_coverage must exceed 0.4
      assert _ok(mask_coverage=0.3) is False
      assert _ok(mask_coverage=0.41) is True

  def test_blurry_crop_rejected():
      assert _ok(blur_var=10.0) is False

  def test_back_view_proxy_wide_short_rejected():
      # degenerate wide/short bbox (proxy for non-standing/back-lean) rejected
      assert _ok(aspect_ratio=1.2) is False

  def test_missing_mask_coverage_does_not_hard_fail():
      # mask_coverage=None (no mask) must not crash and must not reject on coverage alone
      assert crop_quality_ok(
          crop_h=180, crop_w=70, mask_coverage=None, blur_var=200.0,
          aspect_ratio=70 / 180, **DEFAULT_GATE
      ) is True
  ```
- [ ] **Step 2 (run-to-fail)** `cd .../src/vision_track && VENV -m pytest test/test_appearance_quality_gate.py -v 2>/dev/null`. Expected: `ModuleNotFoundError: No module named 'vision_track.reid.quality'`.
- [ ] **Step 3 (minimal impl)** Create `WT/src/vision_track/vision_track/reid/quality.py`:
  ```python
  """Quality gate for appearance-history inserts (gallery hygiene).

  Pure functions so they unit-test with synthetic primitives (no model/crop I/O).
  Rejects crops that would poison the ReID gallery: too small, poorly segmented,
  blurry, or degenerate aspect (a cheap back-view / non-standing proxy).
  """
  from typing import Optional

  # Default thresholds; overridable from the node param surface later if needed.
  DEFAULT_GATE = dict(
      min_crop_h=80,          # px; reject far/tiny detections
      min_mask_coverage=0.4,  # spec: mask_coverage > 0.4
      min_blur_var=50.0,      # Laplacian variance floor (sharpness)
      max_aspect_ratio=0.9,   # w/h; standing person is tall (<~0.6); >0.9 is degenerate
  )


  def crop_quality_ok(
      crop_h: int,
      crop_w: int,
      mask_coverage: Optional[float],
      blur_var: float,
      *,
      aspect_ratio: float,
      min_crop_h: int,
      min_mask_coverage: float,
      min_blur_var: float,
      max_aspect_ratio: float,
  ) -> bool:
      if crop_h < min_crop_h or crop_w < 2:
          return False
      # mask_coverage is None when no seg mask is available — don't reject on it then.
      if mask_coverage is not None and mask_coverage <= min_mask_coverage:
          return False
      if blur_var < min_blur_var:
          return False
      if aspect_ratio > max_aspect_ratio:
          return False
      return True
  ```
- [ ] **Step 4 (run-to-pass)** Re-run Step 2's command. Expected: all 6 pass.
- [ ] **Step 5 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/reid/quality.py src/vision_track/test/test_appearance_quality_gate.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): pure-logic gallery-insert quality gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

### Step 3.2 — Wire the gate into `update_appearance`

- [ ] **Step 1 (impl)** Edit `appearance_manager.py` `update_appearance` (`11-53`). After features are extracted and confirmed non-empty (`33-34`), compute the gate inputs from `result` + `frame` and short-circuit when the crop fails — **before** `_update_feature_history`/`_update_color_histories` (line `48-49`). Insert between the `current_time = time.time()` block and the `_update_feature_history` call:
  ```python
  import cv2  # add to module imports at top

  # --- gallery hygiene: skip poisoning inserts ---------------------------------
  x1, y1, x2, y2 = result.bbox
  crop_h, crop_w = max(0, y2 - y1), max(0, x2 - x1)
  aspect_ratio = crop_w / max(crop_h, 1e-6)
  mask_coverage = None
  if "mask_coverage" in features and features["mask_coverage"].size:
      mask_coverage = float(features["mask_coverage"][0])
  blur_var = 0.0
  h, w = frame.shape[:2]
  cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
  if cx2 > cx1 and cy2 > cy1:
      gray = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_RGB2GRAY)
      blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

  from .quality import crop_quality_ok, DEFAULT_GATE
  if tracker.target_appearance is not None and not crop_quality_ok(
      crop_h=crop_h, crop_w=crop_w, mask_coverage=mask_coverage,
      blur_var=blur_var, aspect_ratio=aspect_ratio, **DEFAULT_GATE,
  ):
      logger.debug(
          f"Gallery insert skipped (low quality): h={crop_h} cov={mask_coverage} "
          f"blur={blur_var:.0f} ar={aspect_ratio:.2f}"
      )
      # still refresh motion so velocity/last_seen stay current
      _update_motion(appearance=tracker.target_appearance, result=result, current_time=current_time)
      if tracker.original_track_id is not None and result.class_id == 0:
          tracker.person_registry.update_person(tracker.original_track_id, tracker.target_appearance)
      return
  ```
  **Important ordering note:** the gate is applied **only when `tracker.target_appearance is not None`** (i.e. after the first lock). The very first observation (which *creates* `target_appearance` at `appearance_manager.py:45-46`) must still seed the gallery even if marginal, otherwise the operator never gets an anchor. The Phase-0 operator-init heuristic already picks a good first crop, so the first insert is trusted.
- [ ] **Step 2 (verify import-clean + no regression)** `cd .../src/vision_track && VENV -c "import vision_track.reid.appearance_manager" 2>/dev/null` (expect no error), then re-run the Task-3.1 + Task-1.5 pure tests to confirm nothing regressed: `VENV -m pytest test/test_appearance_quality_gate.py test/test_fusion_weights.py -v 2>/dev/null`.
- [ ] **Step 3 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/reid/appearance_manager.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): apply quality gate before gallery inserts

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

### Task 4 — Identity-gated ambiguity (Lowe ratio + deep-margin spatial gate + distinctiveness)

**Root cause (spec §1.2 #1, §6.4):** when the deep term was untrained, `_resolve_ambiguity` (`reid_search.py:152-203`) leaned on **spatial proximity** (`_resolve_with_spatial_gate` `264-303`) to break ties — so a crosser passing nearer the last-known center could *steal the lock* purely on position, with no identity check. And `PersonRegistry.distinctiveness_threshold = 0.03` (`registry.py:23`) is so loose that a near-duplicate lookalike clears it. Phase 1, with a trained backbone, adds two identity gates:

1. **Lowe-style ratio test on the DEEP term.** Before accepting the top candidate, require `deep_second / deep_best <= ratio_max` (the raw OSNet cosine of the runner-up must be clearly *worse* than the best). A high ratio means two candidates are deep-indistinguishable → ambiguous → refuse rather than guess.
2. **Deep-margin guard on any spatial-proximity switch.** `_resolve_with_spatial_gate` may only switch to the spatially-closer runner-up if that runner-up *also* wins (or ties within a small margin) the deep term — proximity can no longer override identity.
3. **Distinctiveness threshold 0.03 → 0.10** (`registry.py:23`), run on every multi-person frame (already invoked via `_passes_distinctiveness` `306-331` whenever `len(candidate_scores) > 1`).

Both #1 and #2 are pure-logic helpers (operate on raw cosines / scalars) so they unit-test with synthetic numbers; the wiring into `reid_search.py` is mechanical.

**Files:**
- `WT/src/vision_track/vision_track/reid/identity_gate.py` (NEW — pure helpers)
- `WT/src/vision_track/vision_track/reid/reid_search.py` (`_resolve_ambiguity` `152-203`, `_resolve_with_spatial_gate` `264-303`, plus a deep-cosine accessor — needs raw reid cosine per candidate)
- `WT/src/vision_track/vision_track/core/registry.py` (`distinctiveness_threshold` `23`)
- `WT/src/vision_track/test/test_reid_ratio_gate.py` (NEW, pure-logic)
- `WT/src/vision_track/test/test_distinctiveness_margin.py` (NEW, pure-logic)

### Step 4.1 — Pure-logic identity gates

- [ ] **Step 1 (failing test — pure-logic)** Create `WT/src/vision_track/test/test_reid_ratio_gate.py`:
  ```python
  from vision_track.reid.identity_gate import (
      deep_ratio_ambiguous,
      spatial_switch_allowed,
      DEFAULT_RATIO_MAX,
      DEFAULT_DEEP_SWITCH_MARGIN,
  )


  def test_clear_winner_not_ambiguous():
      # best 0.85, second 0.55 -> ratio 0.65 well under the cap
      assert deep_ratio_ambiguous(0.85, 0.55, ratio_max=DEFAULT_RATIO_MAX) is False

  def test_deep_indistinguishable_is_ambiguous():
      # best 0.80, second 0.79 -> ratio ~0.99 -> ambiguous
      assert deep_ratio_ambiguous(0.80, 0.79, ratio_max=DEFAULT_RATIO_MAX) is True

  def test_zero_or_negative_best_is_ambiguous():
      assert deep_ratio_ambiguous(0.0, 0.0, ratio_max=DEFAULT_RATIO_MAX) is True
      assert deep_ratio_ambiguous(-0.1, -0.2, ratio_max=DEFAULT_RATIO_MAX) is True

  def test_spatial_switch_blocked_when_runner_up_loses_deep():
      # spatially closer runner-up but its deep cosine is much worse -> block switch
      assert spatial_switch_allowed(
          deep_best=0.82, deep_candidate=0.55, margin=DEFAULT_DEEP_SWITCH_MARGIN
      ) is False

  def test_spatial_switch_allowed_when_runner_up_ties_or_wins_deep():
      # runner-up at least ties deep within margin -> proximity may break the tie
      assert spatial_switch_allowed(
          deep_best=0.80, deep_candidate=0.78, margin=DEFAULT_DEEP_SWITCH_MARGIN
      ) is True
      assert spatial_switch_allowed(
          deep_best=0.75, deep_candidate=0.85, margin=DEFAULT_DEEP_SWITCH_MARGIN
      ) is True
  ```
- [ ] **Step 2 (run-to-fail)** `cd .../src/vision_track && VENV -m pytest test/test_reid_ratio_gate.py -v 2>/dev/null`. Expected: `ModuleNotFoundError: No module named 'vision_track.reid.identity_gate'`.
- [ ] **Step 3 (minimal impl)** Create `WT/src/vision_track/vision_track/reid/identity_gate.py`:
  ```python
  """Pure-logic identity gates for ReID association.

  Operate on raw deep cosines / scalars so they unit-test without a model.
  Used by reid_search to stop spatial proximity from overriding identity.
  """

  # Lowe ratio cap on the DEEP term: if second/best > this, the two top
  # candidates are deep-indistinguishable and the match is ambiguous.
  DEFAULT_RATIO_MAX = 0.92

  # A spatially-closer runner-up may only steal the lock if its deep cosine is
  # within this margin of (or better than) the best candidate's deep cosine.
  DEFAULT_DEEP_SWITCH_MARGIN = 0.05


  def deep_ratio_ambiguous(deep_best: float, deep_second: float, *, ratio_max: float) -> bool:
      """True if the runner-up is too close to the best on the deep term."""
      if deep_best <= 1e-6:
          return True  # no usable deep signal -> treat as ambiguous
      ratio = deep_second / deep_best
      return ratio > ratio_max


  def spatial_switch_allowed(deep_best: float, deep_candidate: float, *, margin: float) -> bool:
      """True if a spatially-closer candidate also wins/ties the deep term.

      Proximity may break the tie only when the candidate's identity evidence is
      at least as strong (within `margin`) as the current best's.
      """
      return deep_candidate >= deep_best - margin
  ```
- [ ] **Step 4 (run-to-pass)** Re-run Step 2's command. Expected: all 5 pass.
- [ ] **Step 5 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/reid/identity_gate.py src/vision_track/test/test_reid_ratio_gate.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): pure-logic Lowe-ratio + deep-margin identity gates

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

### Step 4.2 — Wire the ratio test into `_resolve_ambiguity` and the deep-margin into the spatial gate

The gates need the **raw deep cosine per candidate**, which `_score_candidates` (`reid_search.py:108-149`) currently computes only for logging (`raw_cosine` at `127`, discarded). Thread it through so `_resolve_ambiguity` can apply the gates.

- [ ] **Step 1 (impl — carry raw deep cosine on each candidate)** Edit `_score_candidates` (`reid_search.py:108-149`): change the per-candidate tuple from `(result, similarity, features)` to `(result, similarity, features, raw_cosine)`. Compute `raw_cosine` (default `0.0`) for person candidates where reid dims match (reuse the existing block at `125-132`). Update the type hint and the `candidate_scores.append(...)` at `147`. Then update **all** consumers of the 3-tuple unpacking:
  - `find_best_match_reid` (`reid_search.py:50`): `best_match, best_similarity, best_features, best_deep = candidate_scores[0]`.
  - `_update_candidate_consistency` (`353-356`): unpack `for match, similarity, _, _ in candidates:`.
  - the `candidate_scores.sort(...)` (`44`) is unaffected (sorts by index 1).
  - the `logger.info` list comprehension at `46-48` (`for r, s, _`) → `for r, s, _, _`.
- [ ] **Step 2 (impl — ratio test in `_resolve_ambiguity`)** Edit `_resolve_ambiguity` (`reid_search.py:152-203`). It now receives the full `candidate_scores` (it already does). Right after computing `margin` (`167`), before the camera-motion / spatial branches, apply the Lowe ratio test on the deep cosines of the top two:
  ```python
  from .identity_gate import deep_ratio_ambiguous, spatial_switch_allowed, DEFAULT_RATIO_MAX, DEFAULT_DEEP_SWITCH_MARGIN

  best_deep = candidate_scores[0][3]
  second_deep = candidate_scores[1][3]
  if deep_ratio_ambiguous(best_deep, second_deep, ratio_max=DEFAULT_RATIO_MAX):
      logger.info(
          f"ReID FAILED (deep ratio): best_deep={best_deep:.3f} second_deep={second_deep:.3f} "
          f"ratio>{DEFAULT_RATIO_MAX} — identities not separable"
      )
      return None, 0.0, {}
  ```
  (The existing `margin >= REID_MARGIN` early-accept at `169-170` stays; the ratio test runs first so a clear deep winner with a small *combined* margin can still proceed to the spatial/motion resolvers, but a deep-indistinguishable pair is refused outright.)
- [ ] **Step 3 (impl — deep-margin guard in spatial gate)** `_resolve_with_spatial_gate` (`reid_search.py:264-303`) currently switches to the spatially-closer runner-up on distance alone (`286-291`). It must also receive the two deep cosines and gate the switch. Change its signature to accept `best_deep, second_deep` and replace the switch condition at `286`:
  ```python
  if (
      dist_second < dist_best - spatial_threshold
      and second_best_similarity > tracker.reid_threshold
      and spatial_switch_allowed(best_deep, second_deep, margin=DEFAULT_DEEP_SWITCH_MARGIN)
  ):
      ...preferring closer ID (now identity-gated)...
      return second_best_match, second_best_similarity, None
  ```
  Update the call site in `_resolve_ambiguity` (`186-194`) to pass `best_deep`/`second_deep`. Apply the **same** deep-margin guard to the camera-motion resolver's position-based switches (`_resolve_with_camera_motion` `206-261`, the `dist_second < dist_best - prediction_threshold` switch at `234`): thread `best_deep`/`second_deep` in and require `spatial_switch_allowed(...)` there too, so neither motion nor spatial proximity can override identity. (The relative-position and consistency switches at `245`/`250` are weaker cues — also gate them with `spatial_switch_allowed` for consistency.)
- [ ] **Step 4 (verify import-clean + run pure tests)** `cd .../src/vision_track && VENV -c "import vision_track.reid.reid_search" 2>/dev/null` (expect no error). Re-run `VENV -m pytest test/test_reid_ratio_gate.py -v 2>/dev/null` — still green (the helper module is unchanged; this confirms the import wiring did not break it).
- [ ] **Step 5 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/reid/reid_search.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): identity-gate ambiguity — Lowe ratio + deep-margin spatial switch

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

### Step 4.3 — Raise distinctiveness threshold 0.03 → 0.10

- [ ] **Step 1 (failing test — pure-logic)** Create `WT/src/vision_track/test/test_distinctiveness_margin.py`. It drives `PersonRegistry.check_distinctiveness` with a stub similarity function and asserts a lookalike (other person scores close to the target) is now rejected at the tighter margin:
  ```python
  import numpy as np
  from vision_track.core.registry import PersonRegistry
  from vision_track.core.tracking_types import TargetAppearance


  def _registry_with_two():
      reg = PersonRegistry()
      reg.register_person(0, TargetAppearance(class_id=0, class_name="person"))   # target
      reg.register_person(1, TargetAppearance(class_id=0, class_name="person"))   # other
      return reg

  def test_threshold_raised_to_0_10():
      assert abs(PersonRegistry().distinctiveness_threshold - 0.10) < 1e-9

  def test_lookalike_rejected_at_tight_margin():
      reg = _registry_with_two()
      # other person scores 0.78 vs target candidate score 0.83 -> margin 0.05 < 0.10
      sim_func = lambda appearance, feats: 0.78
      assert reg.check_distinctiveness(0, {"reid": np.zeros(4)}, 0.83, sim_func) is False

  def test_distinct_candidate_accepted():
      reg = _registry_with_two()
      sim_func = lambda appearance, feats: 0.55   # other much worse
      assert reg.check_distinctiveness(0, {"reid": np.zeros(4)}, 0.83, sim_func) is True

  def test_no_other_persons_always_distinct():
      reg = PersonRegistry()
      reg.register_person(0, TargetAppearance(class_id=0, class_name="person"))
      assert reg.check_distinctiveness(0, {"reid": np.zeros(4)}, 0.5, lambda a, f: 0.99) is True
  ```
  (Confirm `TargetAppearance(class_id=..., class_name=...)` is constructible — its dataclass fields default the histories; `class_id`/`class_name` are accepted kwargs per `appearance_manager.py:46`.)
- [ ] **Step 2 (run-to-fail)** `cd .../src/vision_track && VENV -m pytest test/test_distinctiveness_margin.py -v 2>/dev/null`. Expected failures: `test_threshold_raised_to_0_10` (current `0.03 != 0.10`) and `test_lookalike_rejected_at_tight_margin` (margin 0.05 > current 0.03 → currently *accepted*).
- [ ] **Step 3 (minimal impl)** Edit `registry.py:23`: `self.distinctiveness_threshold = 0.10`.
- [ ] **Step 4 (run-to-pass)** Re-run Step 2's command. Expected: all 4 pass. The "run on every multi-person frame" requirement is already satisfied — `_passes_distinctiveness` (`reid_search.py:306-331`) fires whenever `is_person and original_track_id is not None and len(candidate_scores) > 1`, which is every multi-person frame; no extra wiring needed beyond the threshold bump.
- [ ] **Step 5 (commit)**
  ```bash
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker add src/vision_track/vision_track/core/registry.py src/vision_track/test/test_distinctiveness_margin.py
  git -C /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker commit -m "feat(vision_track): raise distinctiveness margin 0.03 -> 0.10

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
  ```

---

## Acceptance

### Now-testable (no arena bags required)

**Pure-logic unit tests (run offline, must be green):**
```bash
cd /home/tinker/tk25_ws/.worktrees/tk26_vision-person-tracker/src/vision_track && \
  /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python -m pytest \
  test/test_fusion_weights.py test/test_bytetrack_config.py \
  test/test_appearance_quality_gate.py test/test_reid_ratio_gate.py \
  test/test_distinctiveness_margin.py -v 2>/dev/null
```
- Fusion weights normalize to 1; deep term dominates (`WEIGHT_REID ≥ 0.70 > Σcolor`); raw/color floors recalibrated.
- `bytetrack.yaml` enables low-conf recovery (`track_low_thresh ≤ 0.15 < yolo_track_conf` is not required, but `track_low_thresh ≤ 0.15`, `track_high_thresh ≥ 0.2`); `default.yaml` carries `yolo_track_conf` + `reid_backbone`.
- Quality gate rejects short/low-coverage/blurry/degenerate crops; tolerates a missing mask.
- Lowe ratio refuses deep-indistinguishable pairs; spatial/motion switch blocked unless the runner-up wins the deep term within margin.
- Distinctiveness threshold is `0.10` and rejects a 0.05-margin lookalike.

**Torch-gated integration test (runs where torchreid + cached OSNet weights are present; auto-skips offline):**
```bash
cd .../src/vision_track && VENV -m pytest test/test_reid_backbone.py -v 2>/dev/null
```
- `OSNetBackbone.extract_features` returns a 1-D float32 vector of `feature_dim` with L2 norm == 1.
- `PersonReIDModel(backbone_name='osnet_x0_25')` preserves the `extract_features → L2-normalized np.ndarray` contract that `reid_search`/`appearance_manager` depend on.

**Regression / integration:**
- Full existing suite green: `cd .../src/vision_track && VENV -m pytest test/ -v 2>/dev/null` (the `test_copyright/flake8/pep257` ament tests still pass; new code must be flake8/pep257-clean).
- Build via the tk26 wrapper so the new `config/*.yaml` install to `share/vision_track/config/` and entry-point shebangs see the venv: `./src/tk26_vision/scripts/build.sh --packages-select vision_track`.
- T0/T1 startup tiers (`src/tk26_vision/scripts/tests/`) confirm `person_track_server` starts with `reid_backbone` + `yolo_track_conf` declared and advertises `/track_person`; `reid_mode='native'` still loud-guards (Phase 0).

**Offline Occluded-REID calibration (knob, NEVER a gate):** the ROC procedure in Task 1 Step 1.5 sets the final `MIN_REID_SIMILARITY_RAW` / `REID_THRESHOLD` / `REID_MARGIN` and the Lowe `DEFAULT_RATIO_MAX`. Per `person-tracker-benchmark-strategy`, academic ReID datasets tune thresholds but are never a pass/fail gate. Synthetic lookalike + back-to-camera discrimination fixtures (the pure-logic gates above) assert same-vs-different separation improves over the random head.

### Arena-deferred (cannot be confirmed until Orbbec arena recordings exist)

- `wrong_lock_episodes == 0` on `cml_crossing` and `lookalike_distractors` bags (the highest-risk gate; the master fix of this phase).
- `correct_lock_rate` recovered (≥0.92) on `back_to_camera` and `range_lighting` bags (depends on the low-conf recovery + relaxed color floors not dropping the operator).
- ptbench `action`-backend smoke once OSNet weights load on the robot, then full ptbench scoreboard against recorded bags — measured with the Phase-0-fixed ruler (dual-GT `centroid_field` gate, `pos_error_range_m` graded).
