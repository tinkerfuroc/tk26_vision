# MediaPipe 0.10.9 → 1.0.1 Tasks-API port for waving detection — design

**Date:** 2026-08-22
**Status:** approved in brainstorming; implementation plan to follow
**Scope class:** parity port — no behavior change on the default (`vlm`) path

## Problem

`waving_person_server` estimates pose per YOLO person ROI with the legacy
MediaPipe **Solutions** API (`mp.solutions.pose.Pose(static_image_mode=True,
min_detection_confidence=0.5)`, default `model_complexity=1`), pinned at
`mediapipe==0.10.9` (Dec 2023) in `.venv-vision-main`. Since 2026-07-04 this
path runs only in `waving_detector='hybrid'|'mediapipe'` or when the VLM chain
has no key; it is the offline / no-key fallback and the sub-second fast path.

Upstream facts (verified against the wheels, not just docs):

- Latest release is **1.0.1 (2026-08-14)**. From **0.10.30 (2025-12)** the
  wheel is `py3-none-manylinux_2_28` (a ctypes shim over one 114 MB
  `libmediapipe.so`) and **`mp.solutions` no longer exists**. 0.10.21 is the
  last release that still ships the Solutions API. Google ended Solutions
  support in March 2023.
- 1.0.1 declares only `absl-py~=2.3 flatbuffers~=25.9 numpy sounddevice~=0.5
  opencv-contrib-python matplotlib certifi` — no protobuf, no jax. It imports
  and runs under this venv's Python 3.10 / numpy 1.26.4.
- No models are bundled; `.task` bundles are downloaded explicitly. That
  replaces today's silent ~30 s first-call `.tflite` download.
- The Tasks API exposes `BaseOptions(delegate=GPU)` ("limited to Ubuntu").

Measured on the tinker box (RTX 2080 Ti shared with the sim, 6 YOLO person
crops, per-crop median): 0.10.9 Solutions full **90 ms**; 1.0.1 Tasks full CPU
**60 ms**; 1.0.1 Tasks full **GPU 7.8 ms** (lite 6.5, heavy 11). GPU init is a
one-time 3–6 s; VRAM after warm inference is **60 MiB** for full (lite 53,
heavy 82), freed on `close()`. CPU landmarks from 1.0.1 are numerically
identical to 0.10.9 (same BlazePose weights); the fp16 GPU path differs by up
to ~0.04 in normalized `y` and never flipped a verdict on the sample crops.

Bumping the version without a port is not an option: `mp.solutions` raises
`AttributeError` at node start on anything ≥ 0.10.30.

## Goals

1. Move the waving pose pass to the Tasks `PoseLandmarker` on mediapipe 1.0.1,
   GPU delegate first with automatic CPU fallback.
2. **Identical `is_waving` verdicts** to 0.10.9 on a frozen fixture, enforced
   by a pytest that needs no ROS or camera.
3. No change to any other node, venv, or to the default `waving_detector`.

## Non-goals (explicitly out of scope)

- The rgb8/bgr8 colour-order normalizer for `waving_person_server`
  (DEV_NOTES flagged follow-up) — lands separately so benchmarks attribute.
- Changing the default `waving_detector` from `vlm`.
- `lite` / `heavy` model tuning (reachable via param, not defaulted).
- Replacing MediaPipe with YOLO11-pose (needs a labelled GT set to retune).
- Lifting the `protobuf==3.20.3` pin (still required by `tensorboard 2.11.2`,
  a runtime import dep of torchreid) or removing the jax/jaxlib orphans 0.10.9
  leaves behind.

## Rejected alternatives

- **Pin to 0.10.21** (last Solutions release): same CPU graph, no GPU
  delegate, still lazy-downloads; a dead end.
- **YOLO11-pose in the existing YOLO pass**: different keypoint set (17 COCO,
  no per-joint visibility semantics), so thresholds would need re-tuning —
  a behavior change, not a parity port. Candidate for a later round.

## Design

### 1. Dependency change (`.venv-vision-main` only)

- `.venv-vision-main.uv-project/pyproject.toml`: `mediapipe==0.10.9` →
  `mediapipe==1.0.1`; re-lock `uv.lock`; update root `requirements.txt`
  line; refresh `.venv-vision-main/freeze.lock.txt`.
- All of 1.0.1's declared deps are already satisfied in the venv (absl 2.4,
  flatbuffers 25.12, numpy 1.26.4, opencv-contrib 4.10, matplotlib 3.10,
  sounddevice 0.5.5), so the install must change **only** the `mediapipe`
  package. Verification: `pip check` clean and a `pip freeze` diff
  before/after showing exactly one changed line. Any other package moving is
  a failure to investigate, not accept.
- `protobuf` stays 3.20.3. `.venv-da3`, `.venv-fs`, `.venv-calib` untouched.
- Note: the `.venv-*` directories are git-ignored and live only in the main
  checkout (`/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main`); the
  venv change is performed there even when code is edited in a worktree.

### 2. Pose backend adapter — `tk_vision_specialized/_pose_backend.py`

Import-light (mediapipe, numpy, cv2; no rclpy), same convention as
`_waving_vlm.py`.

- `PoseLandmarkIdx(IntEnum)`: `NOSE=0, LEFT_SHOULDER=11, RIGHT_SHOULDER=12,
  LEFT_ELBOW=13, RIGHT_ELBOW=14, LEFT_WRIST=15, RIGHT_WRIST=16` (the names
  `is_waving` reads today) and `POSE_CONNECTIONS` (the 33-landmark edge list
  from the Tasks drawing constants) for the overlay.
- `Landmark`: tiny dataclass `x, y, z, visibility` so callers keep writing
  `landmarks[PL.RIGHT_WRIST].visibility`.
- `PoseBackend(model_path: str, delegate: str = 'gpu',
  min_detection_confidence: float = 0.5)`:
  - Builds `vision.PoseLandmarker` in `RunningMode.IMAGE`, `num_poses=1`
    (the node already isolates one person per YOLO ROI; this is the Tasks
    equivalent of `static_image_mode=True`).
  - `delegate='gpu'`: create with `BaseOptions.Delegate.GPU`, then run one
    warm-up `detect()` on a blank 256×256 frame. Any exception from either
    step → rebuild on CPU; record `self.active_delegate` (`'gpu'|'cpu'`) and
    `self.fallback_reason` so the node can log them once. `delegate='cpu'`
    skips the GPU attempt.
  - `process(rgb_roi: np.ndarray) -> list[Landmark] | None`: wraps the array
    in `mp.Image(SRGB)`, runs `detect`, returns the first pose's normalized
    landmarks, or `None` when none — preserving today's
    `pose_landmarks is None` contract.
  - `close()`: releases the landmarker (frees the GL context / VRAM).
- `draw_pose(bgr_roi, landmarks, connections=POSE_CONNECTIONS)`: cv2-only
  replacement for `mp_draw.draw_landmarks` (circles at joints, lines on
  connections, in place).

### 3. Node and script changes

`waving_person_server.py`:
- Remove `import mediapipe as mp`, `self.mp_pose`, `self.mp_draw`.
- New ROS params: `pose_model_path` (default `'pose_landmarker_full.task'`)
  and `pose_delegate` (default `'gpu'`; `'cpu'` forces CPU).
- `pose_model_path` is resolved with a new
  `vision_util.weights_cache.find_cached(name) -> Path | None`: absolute
  path → as-is if it exists; bare filename → the same `$TK26_MODEL_CACHE` /
  `~/.cache/tk26_vision/weights/` search order as `resolve_weights`, but
  **without** the Ultralytics auto-download (which would try to load a
  `.task` file as a YOLO model and fail confusingly). `None` → the
  `RuntimeError` below. `resolve_weights` itself is unchanged.
- `self.pose = PoseBackend(...)`; log `active_delegate` and, if it fell back,
  `fallback_reason` at WARN once at init.
- Call site `self.pose.process(cv2.cvtColor(person_roi, COLOR_BGR2RGB))`
  unchanged in shape; `landmarks = pose_results` (the adapter returns the
  list directly). `is_waving`: `PL = PoseLandmarkIdx`, body unchanged.
- Both `_annotate_*` drawing sites call `draw_pose(roi, landmarks)`.
- Missing `.task` file → `RuntimeError` at init naming
  `scripts/download_models.py` (same pattern as missing YOLO weights).
- `close()` the backend in the node's destroy path.

`check_waving_inference.py`, `scripts/tests/debug_waving_pipeline.py`: same
adapter; `debug_waving_pipeline.py` keeps its legacy/fixed predicate pair but
indexes via `PoseLandmarkIdx`.

`scripts/download_models.py`: replace `warm_mediapipe()` with
`fetch_pose_landmarker()` that downloads
`https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task`
into `weights_cache._writable_cache()` (idempotent; skip if present; download
to a temp name and rename so a partial file never satisfies `find_cached`).
Flag renamed to `--skip-pose` (keep `--skip-mediapipe` as an alias).

Model variant: **full** only by default (matches `model_complexity=1`).

### 4. Regression gate — frozen-fixture parity

Recorded **before** the venv is upgraded, while 0.10.9 is importable, by a
one-off `scripts/tests/record_pose_fixture.py`:

1. `src/tk_vision_specialized/test/fixtures/pose_parity/*.png`: person ROIs
   cropped by the node's YOLO11m-seg (conf 0.4, CPU is fine) from the two
   ultralytics sample images plus live-camera frames if the Orbbec is up
   (waving and non-waving poses — note in the fixture README which is which
   and whether live frames were obtained). Target 10–20 crops, a few hundred
   KB total.
2. `expected_0.10.9.json`: per crop — detected flag, all 33
   `(x, y, z, visibility)`, and the `is_waving` verdict computed by importing
   the node's predicate (not a copy). Header records mediapipe version and
   the Solutions options used.

`src/tk_vision_specialized/test/test_pose_parity.py` (plain pytest, no ROS,
no camera; skips cleanly if the `.task` file is absent so CI without weights
doesn't fail spuriously — but T0 must have the weights so the skip never
hides a regression on the robot):

- **Verdict parity (hard, CPU):** `PoseBackend(delegate='cpu')` →
  `is_waving` must equal the fixture boolean for every crop; detected flag
  must match.
- **Landmark parity (hard, CPU):** for the 7 joints `is_waving` reads,
  `|Δy| ≤ 0.01`, `|Δvisibility| ≤ 0.05` (measured CPU drift is ~0.001).
- **GPU parity (soft skip, hard verdict):** same with `delegate='gpu'`;
  `pytest.skip` if `active_delegate != 'gpu'`; landmark tolerance widened to
  `|Δy| ≤ 0.05`; verdict equality still hard.
- **Fallback:** monkeypatch the GPU create to raise → backend comes up with
  `active_delegate == 'cpu'` and still passes verdict parity.
- **Drawing smoke:** `draw_pose` on a blank ROI with fixture landmarks does
  not raise and modifies the image.

Stack-wide checks after the venv swap, run and reported verbatim:
`pip check`; `pip freeze` diff; `scripts/tests/t0_static.sh`;
`scripts/tests/t1_startup.sh` (all nodes start, waving node log shows which
delegate came up); existing `tk_vision_specialized` pytest suite; one live
`t2` goal with `waving_detector:=mediapipe` if cameras are running.

### 5. Rollout order

1. Record the fixture under 0.10.9; commit it.
2. Write `_pose_backend.py` + `test_pose_parity.py` (fails only because 1.0.1
   is not yet installed).
3. Upgrade the venv; run `pip check`, freeze diff, T0, T1.
4. Wire the node + three scripts; run parity, unit suite, T0, T1.
5. Live T2 in `mediapipe` mode if cameras are available; record numbers.

### 6. Docs

- `CLAUDE.md` `waving_person_server` entry: add `pose_model_path` /
  `pose_delegate`; drop the "first call silently downloads ~30 s" caveat in
  the bench spec reference where it mentions MediaPipe.
- `src/tk_vision_specialized/README.md`: dependency line (`mediapipe>=1.0`),
  pipeline step 3 wording, changelog entry.
- `DEV_NOTES.md`: dated entry with the benchmark table, VRAM numbers, the
  parity-gate description, and the explicit non-goals above.

## Acceptance

- Parity pytest green on CPU; GPU variant green or skipped with the reason in
  the log, never failed on verdicts.
- `pip freeze` diff shows only `mediapipe 0.10.9 → 1.0.1`; `pip check` clean.
- T0 and T1 pass; T1 log shows `pose delegate: gpu` on the robot.
- Per-crop pose latency in `mediapipe` mode ≤ 20 ms on GPU (vs ~90 ms today),
  measured by `check_waving_inference`'s existing `mediapipe_ms` field.
