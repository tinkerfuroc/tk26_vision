# CLAUDE.md

Guidance for Claude Code when working in `src/tk26_vision/`.

## Project Overview

ROS2 Humble vision module for Tinker 2026: object detection, person tracking with ReID, pan-tilt head control, LLM-backed person features / grocery categorization (OpenRouter), and utility services (door detection, point-cloud relay). Python 3.10 venv at `src/tk26_vision/.venv-vision-main/`.

## Build

Use the wrapper — it sources the venv + ROS, runs `colcon build --symlink-install`, and patches install-tree shebangs so `ros2 run` picks up the venv python (openai, dotenv, ultralytics, pyserial live only in the venv):

```bash
./src/tk26_vision/scripts/build.sh [colcon args...]
```

Plain `colcon build` produces `#!/usr/bin/python3` shebangs that can't see the venv. If you must run it manually, follow up with `./src/tk26_vision/scripts/fix_venv_shebangs.sh` (idempotent; covers all tk26 packages).

If a build errors on stale symlinks, `rm -rf build/<pkg> install/<pkg>` and rebuild that package.

**`monocular_depth` builds under a different venv.** That package depends on `depth_anything_3`, which pins `numpy<2`. The shared `.venv-vision-main` has `numpy==2.2.6` (torch 2.11 / scipy / ultralytics / opencv-python depend on the 2.x ABI), so DA3 lives in `src/tk26_vision/.venv-da3/`. Use the dedicated wrapper:

```bash
./src/tk26_vision/scripts/build_monocular_depth.sh [colcon args...]
```

The wrapper sources `.venv-da3`, runs `colcon build --packages-select monocular_depth` (or the args you pass), then re-shebangs the entry-point script to `.venv-da3/bin/python3`. Don't pass `monocular_depth` to the main `build.sh` — it'll resolve `depth_anything_3` against the wrong venv and the entry-point script will start under `.venv-vision-main`'s python.

**`foundation_stereo` builds under a third venv.** `torch==2.8.0+cu128` +
`tensorrt==10.16.1.11` conflict with both the shared `.venv-vision-main`
and `.venv-da3`. Use the dedicated wrapper:

```bash
./src/tk26_vision/scripts/build_foundation_stereo.sh [colcon args...]
```

The wrapper sources `.venv-fs/`, runs `colcon build --packages-select
foundation_stereo` (or the args you pass), then re-shebangs the entry-point
script. Provisioning the venv once: see `src/foundation_stereo/README.md`.

## Environment

### Python deps

Per-package `requirements.txt` install into the shared venv:

```bash
source src/tk26_vision/.venv-vision-main/bin/activate
pip install -r src/tk26_vision/src/kimi_api/requirements.txt
pip install -r src/tk26_vision/src/pan_tilt/requirements.txt
pip install -r src/tk26_vision/src/vision_track/requirements.txt
```

### Second venv: `.venv-da3` for `monocular_depth`

`depth_anything_3` (vendored at `thirdparty/depth-anything-3/`) pins `numpy<2`, so `monocular_depth` runs under a separate venv at `src/tk26_vision/.venv-da3/`. Provision once:

```bash
cd src/tk26_vision
python3.10 -m venv .venv-da3 --system-site-packages --symlinks
source .venv-da3/bin/activate
pip install --upgrade pip wheel
pip install "numpy==1.23.4"
pip install -e thirdparty/depth-anything-3 --no-deps
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r src/monocular_depth/requirements.txt
pip freeze > .venv-da3/freeze.lock.txt
```

The `requirements.txt` skips DA3's heavy export-pipeline deps (`pycolmap`, `open3d`, `moviepy`, `trimesh`, `plyfile`, `pillow_heif`, `xformers`, `uvicorn`, `fastapi`, `typer`); to drop those, `thirdparty/depth-anything-3/src/depth_anything_3/api.py` carries a tk26 patch that defers `from depth_anything_3.utils.export import export` from module-load to `_export_results` call time. That patch is the only modification to the vendored DA3 tree (search `tk26_vision patch:` to find it). The lock file at `.venv-da3/freeze.lock.txt` is the source of truth — diff against it before any further pip install in that venv.

### OpenRouter API key (kimi_api)

`feature_recognition`, `feature_matching`, `grocery_categorize` require `OPENROUTER_API_KEY`. Copy `src/tk26_vision/src/kimi_api/.env.example` to the workspace-root `.env` and fill in the key — `python-dotenv` auto-loads `.env` from CWD upward at node startup. Missing key ⇒ `RuntimeError` at node init.

Optional: `OPENROUTER_BASE_URL` (default `https://openrouter.ai/api/v1`), `LLM_MODEL` (default `google/gemini-2.5-pro`, also `-p llm_model:=…`).

## Running Nodes

```bash
source install/setup.bash

# Object detection
ros2 run object_detection_new yolo_seg_node                 # /object_detection_yolo (specialist, excludes 'person')
ros2 run object_detection_new yolo_seg_default_node         # /object_detection (pretrained COCO, backward-compat)
ros2 run object_detection_generalist generalist_node        # /object_detection_generalist (pretrained YOLO + YOLO-World/MobileSAM fallback; flip to Gemini via -p enable_vlm:=true)

# Tracking / shelves
ros2 run vision_track person_track_server            # action /track_person
ros2 run tk_vision_specialized spot_on_shelf_server  # action /spot_on_shelf

# Pan-tilt (servo on /dev/ttyUSB0; see src/pan_tilt/README.md)
ros2 launch pan_tilt pan_tilt.launch.py device:=/dev/ttyUSB0
ros2 run pan_tilt controller --ros-args -p device:=/dev/ttyUSB0
ros2 run pan_tilt state_publisher
ros2 run pan_tilt follow_head                        # /follow_head_action + /follow_head_service

# LLM-backed (kimi_api)
ros2 run kimi_api feature_recognition                # /feature_extraction_service, /seat_recommend_service
ros2 run kimi_api feature_matching                   # /feature_matching_service
ros2 run kimi_api grocery_categorize                 # action /grocery_categorize

# Utilities
ros2 run vision_util door_detection                  # /door_detection_srv
ros2 run vision_util get_point_cloud                 # /get_point_cloud_service
ros2 run vision_util get_orbbec_pc                   # /get_orbbec_pc — CUDA-deprojected Orbbec PC (requires CUDA; bypasses SDK colored-PC bottleneck under iGPU workaround)

# DA3-fused PC (separate venv .venv-da3 + dedicated package; see scripts/build_monocular_depth.sh)
ros2 run monocular_depth monocular_depth_pc          # action /monocular_depth_pc — DA3 + RealSense/Orbbec fusion (numpy<2 isolated venv)

# FoundationStereo (separate venv .venv-fs)
ros2 launch foundation_stereo foundation_stereo.launch.py
ros2 launch foundation_stereo foundation_stereo.launch.py \
    stream_enabled:=true stream_align_to_color:=true
```

## Architecture

```
src/tk26_vision/src/
├── tinker_vision_msgs_26/         # canonical vision interfaces — all msgs/srvs/actions live here (absorbed tk23's tinker_vision_msgs)
├── object_detection_new/          # YOLO-seg: specialist (yolo_seg_node, excludes 'person') + default (yolo_seg_default_node, pretrained COCO)
├── object_detection_generalist/   # Pretrained YOLO + YOLO-World (default fallback) or Gemini 2.5 Flash (enable_vlm) + MobileSAM mask, tk26 srv
├── vision_track/                  # ByteTrack + ResNet50 ReID (custom) or YOLO BoT-SORT (native)
├── tk_vision_specialized/         # SpotOnShelf action + waving detector
├── pan_tilt/                      # controller + state_publisher + URDF TF + follow_head (closed-loop absolute targeting in a pan-tilt-rooted frame; feedback-gated settle; sticky ID + EMA)
├── kimi_api/                      # OpenRouter LLM services; _env.py centralizes key loading
├── vision_util/                   # door_detection (Orbbec 20x20 depth heuristic), get_point_cloud (cached relay), get_orbbec_pc (CUDA-deprojected Orbbec PC, sidesteps SDK colored-PC iGPU bottleneck); shared `_pc_utils.py` reused by monocular_depth
├── monocular_depth/               # DA3-fused PC action server; lives in its own venv `.venv-da3` (numpy==1.23.4) because `depth_anything_3` requires numpy<2 — isolation prevents cascade-breaking the rest of the vision tree
└── foundation_stereo/             # FoundationStereo + Fast-FoundationStereo service/action + streaming depth publisher; lives in its own venv `.venv-fs` (torch 2.8 + cu128 + tensorrt 10.16) because those versions conflict with the shared `.venv-vision-main`
```

**Notes:**
- Three object-detection services now coexist. Canonical targets going forward:
  - `/object_detection_yolo` (specialist, custom-trained model, `excluded_classes=['person']`) — arena/competition items.
  - `/object_detection_generalist` (new, `tinker_vision_msgs_26/srv/ObjectDetectionGeneralist` with boolean flags) — the recommended target for open-vocabulary / non-arena callers (person detection, seat-rec helpers, any class YOLO doesn't recognize).
  - `/object_detection` (pretrained COCO, `tinker_vision_msgs_26/srv/ObjectDetection` — the legacy string-flag schema, kept under its original name) — back-compat for tk25_decision BT nodes that still hard-code this name. Prefer the generalist for new code.
- The specialist (`yolo_seg_node`) now silently drops the `'person'` class regardless of what the (future) custom model emits. To detect people, use the generalist or the default node.
- `kimi_api` calls `object_detection_generalist` via the `detection_service` ROS param — retargetable without rebuilding.
- **All vision interfaces live in `tinker_vision_msgs_26`.** The legacy `tinker_vision_msgs` package (tk23) has been retired and the whole `src/tk23_vision/` tree is `COLCON_IGNORE`d. The two srv names (`ObjectDetection` = legacy string-flag schema, `ObjectDetectionGeneralist` = new boolean-flag schema) coexist in the same package so both specialist and generalist servers can be served without name collisions.
- Cameras: RealSense = aligned depth-to-color Image + pinhole intrinsics (ROS convention: x=fwd, y=left, z=up); Orbbec = PointCloud2 reprojected to image grid (standard ROS convention from the cloud).

## Models

YOLO `.pt` files pre-bundled under `object_detection_new/models/` / `vision_track/` or auto-downloaded by Ultralytics. Supported: `yolo11{n,s,m,l,x}-seg.pt`, `yolov8{n,s,m,l,x}-seg.pt`.

## Configuration

Key ROS2 parameters:
- `object_detection_new`: `service_name`, `model_path`, camera topics, `sort_mode`
- `pan_tilt/controller`: `device`, startup/feedback timing, limits, invert/trim, default speed/accel
- `pan_tilt/state_publisher`: `state_topic`, `joint_state_topic`, joint names, stale timeout
- `pan_tilt/follow_head`: full param surface in `src/pan_tilt/config/pan_tilt.yaml`. Highlights: `yolo_model`, `command_topic`/`state_topic`, `home_pan_deg`/`home_tilt_deg`, `pan_deadband_deg`/`tilt_deadband_deg`, `min_command_change_deg` (chatter suppression), `min_detection_interval_sec` (YOLO cap), `max_settle_timeout_sec` + `steady_{pan,tilt}_eps_deg` + `steady_velocity_eps_deg_per_sec` + `steady_sample_count` (feedback-gated settle), `ema_alpha` + `target_ttl_sec` + `reassoc_dist_m` (smoothing + identity lock), `command_speed_raw_{small,large}` + `small_error_deg` + `command_accel_raw` (motion profile). Defaults are biased for **responsiveness over smoothness** — turn `ema_alpha` down and `steady_*_eps_deg` tighter if you want calmer motion.
- `kimi_api/*`: `llm_model`, `detection_service`, `log_prompts`
- `monocular_depth/monocular_depth_pc`: `da3_model` (default `depth-anything/DA3-SMALL`, swap to `depth-anything/DA3-BASE` via `-p`), `fill_mode` (`holes_only`|`full_override`, default `holes_only`), `align_min_overlap_pixels` (2000), `align_trim_frac` (0.05), `output_frame_id` (override; default = depth msg frame), `debug_pc_topic` (default `~/debug_points`, SensorDataQoS). The action result is a single 32FC1 depth image at source RGB resolution (pixel-aligned to color); the goal's `stride` field subsamples **only** the debug PointCloud, which is published on `debug_pc_topic` only when `debug_publish=true`. DA3 weights via the `depth_anything_3` library's HuggingFace cache (`~/.cache/huggingface/hub`); `weights_cache.resolve_weights` is **not** used here. The node lives in its own ROS package + venv (`.venv-da3`) because `depth_anything_3` pins `numpy<2`. Build via `tkbuild tk26_vision --packages-select monocular_depth` (or `./scripts/build_monocular_depth.sh`), run via `ros2 run monocular_depth monocular_depth_pc`.
- `vision_logging_enabled` (default `true` everywhere except `follow_head`, where both the code default and the yaml override default to `false` because the ~30-40 ms synchronous disk IO at 10 Hz detection stalls the action loop) + `vision_log_folder` (default `'vision_log'`) on the bbox/seg/centroid-producing nodes plus the kimi_api VLM services: `yolo_seg_{node,default_node}`, `generalist_node`, `person_track_node`, `waving_person_server`, `follow_head`, `feature_matching`, `feature_recognition` (covers both `feature_extraction_service` and `seat_recommend_service`), `seat_recommend_bbox`. **All vision nodes in one robot session share a single `vision_log/<YYYYmmdd_HHMMSS>/` subdir.** Resolution order on first write: (1) `$TINKER_VISION_SESSION_TS` env var (must match `YYYYmmdd_HHMMSS`) — exported defensively from every `master_*.sh` and `tmux_*.sh` under `src/tk25_basic/src/scripts/`; (2) newest existing `<base>/<YYYYmmdd_HHMMSS>/` subdir by mtime — lets late-spawned standalone nodes join the active session; (3) fresh `strftime` cold-start. Per-call filenames carry the producing node + branch: `<node_name>_<branch>_{orig,overlay,req}_<YYYYmmdd_HHMMSS_mmm>.{jpg,json}` (e.g. `yolo_seg_node_yolo_orig_…jpg`, `feature_recognition_node_feature_extraction_orig_…jpg`). Tracker logs only on lost/reclaim transitions; follow_head logs at its detection tick when re-enabled for debugging; `feature_matching` additionally dumps `…_feature_matching_ref<i>_<ts>.jpg` for each reference image; `feature_recognition.feature_extraction` additionally dumps `…_feature_extraction_crop_<ts>.jpg` of the chosen person; the legacy `visualization=True` debug PNGs (`yolo_seg_node`) now live in the same run_dir as `…_yolo_detection[_all]_<ts>.png`. Pass `-p vision_logging_enabled:=<bool>` to override.

## Third-party drivers

Vendored under `src/tk26_vision/thirdparty/` (plain clones, no submodules, `.git` stripped):
- `librealsense` v2.57.7 — built from source, installed system-wide (`sudo cmake --install build`)
- `realsense-ros` 4.57.7 — finds librealsense via `find_package`, builds in colcon
- `OrbbecSDK_ROS2` v2.7.6 — bundles OrbbecSDK v2, builds in colcon

See `thirdparty/README.md` for build/udev/kernel-patch steps.

## Camera bringup

**Do not run the vendored camera launches bare** — they drop to ~3 Hz together on this workstation due to three compounding misconfigurations (RealSense USB enumeration, realsense-ros `TRANSIENT_LOCAL` QoS default, kernel UDP buffer limit). [`CAMERA_BRINGUP.md`](./CAMERA_BRINGUP.md) has the canonical launch sequence, the config files under `config/`, the required `FASTRTPS_DEFAULT_PROFILES_FILE` env var, and the full root-cause writeup. Rates expected after the fix: ~30 Hz ±5 ms on every color/depth topic.

## Testing

Integration smoke suite at `scripts/tests/`, four tiers each gated by the previous:

| Tier | Script | Scope | Needs |
|---|---|---|---|
| T0 | `t0_static.sh` | shebangs, venv deps, ROS interfaces, entry-point imports, `.env` sanity | venv only |
| T1 | `t1_startup.sh` | all 11 nodes start + advertise + SIGTERM clean; pan_tilt serial pos/neg; kimi_api key pos/neg | venv + (opt) servo |
| T2 | `t2_live.sh` | one call per node with live cameras (empty scene OK) | orbbec + realsense running |
| T3 | `t3_interaction.sh` | cross-node: feature_matching↔yolo, spot_on_shelf↔yolo, controller↔state_publisher↔follow_head TF | T2 + servo |
| T4 | `t4_hardware.sh {servo_motion\|servo_tracking\|shelf_scene\|person\|all}` | hardware-in-the-loop, staged scenes | operator |

## Pan-tilt / head camera extrinsic calibration

Two-phase solver with xArm FK as the ground-truth anchor and a ChArUco board on the EE as the observation target.

**Hardware note.** The camera is mounted at roughly 90° to the tilt arm — at firmware `tilt = 45°` (arm pointing straight up) the optical axis is horizontal; at firmware `tilt = 0°` (servo-zero set via `T:502`) it points ~45° down. This means T_B has a **large non-identity rotation** (~π/2 about X in tilt_link coordinates), not a small mount-tolerance correction. The calibration handles this by warm-starting T_B from the Phase-1 reference pose rather than from the URDF's stale rpy.

Parameters fit:

| Block | DOF | Init | Phase-2 fit? | Notes |
|---|---|---|---|---|
| T_A trans (base_link→pan axis) | 3 | URDF xyz | yes | rotation locked identity |
| T_B trans (tilt_end→camera_link body) | 3 | from warm-start | yes | |
| T_B rotation (rotvec) | 3 | from warm-start | **no** (Phase-2) / yes (polish) | Y-component is degenerate with θ_t_offset; unlock only in joint polish where Phase-1 data breaks the degeneracy |
| T_ee_marker | 6 | identity | Phase-1 only | Phase-1 hand-eye then frozen |
| θ_t_offset | 1 | −π/4 | yes | absorbs servo-zero-set noise |

Total Phase-2 DOF: 7 (or 8 with `--fit-pan-offset`). Polish phase raises to 13–14.

### Procedure

1. **Generate the board.** `python -m pan_tilt.calibration.charuco_generate --out ~/calib/charuco_5x7` → PDF + PNG + JSON spec. Print on A4 matte at 100% scale, mount on 3 mm aluminum composite, re-measure square size with calipers. (Default 5×7 40 mm squares = 200×280 mm on A4; shrink to `--square-len 0.035 --marker-len 0.026` if your printer can't handle 5 mm edge margins.)
2. **Fill config.** Edit `src/pan_tilt/config/calibration.yaml` — replace the placeholder xArm joint waypoints with 12–15 hand-eye poses (Phase 1) and 2–3 grid-anchor poses (Phase 2). Pre-validate each in RViz with the full URDF loaded. The node enforces a software Z-floor + mast exclusion cylinder but does no general collision checking.
3. **Collect.**
   ```bash
   ros2 run pan_tilt calibrate_collect --ros-args \
     -p config:=$(ros2 pkg prefix pan_tilt)/share/pan_tilt/config/calibration.yaml \
     -p out_dir:=$PWD/calib_out -p phase:=both
   ```
   Produces `phase1_handeye.json`, `phase2_chain.json`, `sanity.json`.
4. **(Optional) Calibrate intrinsics.** If reprojection RMSE > 0.5 px during Phase 1, collect ~20 ChArUco shots and run `python -m pan_tilt.calibration.run_calibration intrinsic <images_dir> --out calib_out`.
5. **Solve.**
   ```bash
   python -m pan_tilt.calibration.run_calibration handeye calib_out/phase1_handeye.json --out calib_out
   python -m pan_tilt.calibration.run_calibration chain  calib_out/phase2_chain.json --handeye calib_out/handeye.json --fit-pan-offset --out calib_out
   python -m pan_tilt.calibration.run_calibration validate calib_out
   ```
   The chain step warm-starts T_B from the Phase-1 `Z₀`, which handles the ~90° (about Y) mount rotation automatically. T_B rotation is **locked by default** through the chain fit to avoid the `T_B(Y) ↔ θ_t_offset` degeneracy; pass `--unlock-tb-rotation` only for debugging or comparison runs. The chain solver auto-tries **two warm-start basins** (`θ_p_offset ∈ {0, π}`) and saves the lower-rot-RMSE result — fixes the silent wrong-basin failure on hardware whose pan firmware sign is opposite the FK assumption (symptom: locked-T_B chain rot RMSE stuck at ~20°). The chosen basin is printed alongside residuals. To run chain against the custom-park solve instead of the canonical one, swap `--handeye calib_out/handeye_custom.json`.

   Optional polish (unlocks T_B rotation; auto-rejects MAD-sigma outliers like handeye does):
   ```bash
   python -m pan_tilt.calibration.run_calibration polish \
     --phase1 calib_out/phase1_handeye.json \
     --phase2 calib_out/phase2_chain.json \
     --seed calib_out/chain.json --unlock-tb-rotation --out calib_out
   ```
   Pass multiple `--phase1` files to concatenate datasets collected at different park poses — the extra EE-rotation diversity helps the joint fit, and is the recommended polish input when both `phase1_handeye.json` and `phase1_handeye_custom.json` exist:
   ```bash
   python -m pan_tilt.calibration.run_calibration polish \
     --phase1 calib_out/phase1_handeye.json calib_out/phase1_handeye_custom.json \
     --phase2 calib_out/phase2_chain.json \
     --seed calib_out/chain.json --unlock-tb-rotation --out calib_out
   ```
   Polish flags:
   - `--phase1 PATH [PATH ...]` (required) — one or more phase-1 sample JSONs concatenated in argument order. `--exclude-indices` indexes into this concatenated array (phase1 first, then phase2).
   - `--phase2 PATH` (required).
   - `--exclude-indices N [N ...]` — drop manually-known-bad samples up front. Use this to propagate handeye's `rejected_sample_indices` across phases.
   - `--reject-sigma` (default 3.0), `--max-reject-frac` (default 0.10) — control the iterative MAD-sigma rejection loop.
   - `--no-reject` — skip auto rejection entirely; manual `--exclude-indices` still applies.
6. **Emit URDF diff.** The patcher auto-detects both xacro layouts: the `tk25_basic` macro form at `src/tk25_basic/src/tinker_urdf/src/pan_tilt.urdf.xacro` (the authoritative URDF the main robot bringup loads — patches `attach_xyz` default + `camera_mount_joint` origin) and the `tk26_vision` standalone form at `src/pan_tilt/urdf/pan_tilt.urdf.xacro` (used by `pan_tilt.launch.py` for dev bringup). Run `python -m pan_tilt.calibration.apply_to_urdf --results calib_out/chain.json --xacro <path>` against **both** so RViz and the live robot stay consistent, and apply the diffs manually once reviewed.

### Phase gates

- Intrinsic RMSE < 0.5 px
- Hand-eye trans RMSE < 3 mm, rot RMSE < 0.5°
- Chain held-out trans RMSE < 3 mm, rot RMSE < 0.4°
- Sanity-pose bracket (start vs end) < 2 mm / 0.2°
- **Phase 4 end-to-end** (recommended after polish, before `apply_to_urdf`): self-consistency trans RMSE < 5 mm / rot RMSE < 0.5° (PASS), 10 mm / 1° (WARN). xArm-independent: place a ChArUco board anywhere stationary in `base_link` (tripod, fixture, taped to a wall), sweep the pan-tilt over N held-out `(θ_p, θ_t)`, and check that the base-frame marker pose is consistent across views. `python -m pan_tilt.calibration.run_calibration validate --phase4 phase4_validation.json --params polish.json --out <session>` — see `src/pan_tilt/pan_tilt/calibration/readme.md § Phase 4`.

> **Don't move the board between phase-1 collects.** `T_ee_marker` is the rigid pose of the marker on the EE flange — both `handeye.json` (canonical 45°) and `handeye_custom.json` (operator-chosen park) describe the *same* physical board, so the two solves must agree. The handeye solver cross-checks them and refuses to write if they disagree by more than 5 mm / 1°. Recovery: re-collect *both* phase-1 datasets in one sitting without touching the board, the EE, or the xArm zero. If you intentionally remounted the board (e.g. swapping marker prints for evaluation), pass `--allow-t-ee-marker-mismatch` on the handeye CLI to bypass the gate.

### Robustness measures baked in

- Per-axis backlash mitigation (overshoot-return per cell)
- Servo settle check (feedback_ok + |cur − tgt| < 0.3° held 0.5 s)
- MAD outlier rejection over 10-frame average per cell
- Image-vs-state timestamp skew gate (≤ 20 ms)
- SE(3) log residual (proper manifold metric) with `soft_l1` loss
- 80/20 train/val split at the chain phase

### Synthetic-data regression test

`pytest src/pan_tilt/test/test_calibration.py` fabricates samples from a known ground-truth, runs every solver, and asserts recovery. Run this before touching `optimize.py` or `pan_tilt_model.py`.

## Pan-tilt refactor notes

The old monolithic `pan_tilt/ctrl` path is gone on purpose.

- `ros2 run pan_tilt ctrl` no longer exists.
- `/pan_tilt_ctrl` and `/pan_tilt_ctrl_modify` are not used by current runtime nodes.
- `PanTiltCtrl` still exists as an interface artifact, but the current
  `pan_tilt` package does not subscribe to it.
- Runtime TF comes from `/joint_states` plus `robot_state_publisher`, not from
  the serial driver.
- `config/specs.json` is retained only as reference data; runtime geometry
  now lives in `tk25_basic/tinker_urdf` as a `pan_tilt_macro`
  (`src/tk25_basic/src/tinker_urdf/src/pan_tilt.urdf.xacro`) plus a
  standalone wrapper (`pan_tilt_standalone.urdf.xacro`). The macro is
  reused by `tracer_mini_manipulator.urdf.xacro` with `parent="base_link"`
  so the combined `mobile_manipulator` URDF loaded by MoveIt (via
  `grasp_bringup`) contains `pan_joint` + `tilt_joint` — this kills the
  `move_group` "Joint 'pan_joint' not found in model 'mobile_manipulator'"
  log spam. When running `pan_tilt.launch.py` alongside `grasp_bringup`,
  pass `launch_robot_state_publisher:=false` so only the xArm RSP owns
  `/robot_description` and `/tf` for the pan/tilt chain — the flag is
  wired through `IfCondition` in `pan_tilt.launch.py`, gating the second
  `robot_state_publisher` on the boolean. For the live servo angles to
  reach the grasp RSP, the vendored `xarm_moveit_config/.../_robot_moveit_realmove_cumotion.launch.py`
  carries a tk25 patch that appends `/pan_tilt/joint_states` to the
  `joint_state_publisher` `source_list`; without it the JSP zero-clobbers
  `pan_joint`/`tilt_joint` every cycle and `base_link → camera_link`
  flickers between live and home pose. The geometry was placed in `tinker_urdf` (not
  `pan_tilt`) to keep dependencies flowing `tk26_vision → tk25_basic`
  only — `tinker_urdf` must not depend on `pan_tilt`.

See `scripts/tests/README.md` for env vars and skip conditions. Logs in `scripts/tests/logs/`.

**Suite invariants:**
- `ObjectDetection`: `status=1, objects=[]` on empty scene is expected, not a failure.
- `FeatureMatching` propagates `ObjectDetection` status — `status=1, centroids=[]` is the empty-scene response.
- `get_point_cloud` uses `ApproximateTimeSynchronizer(queue_size=3, slop=0.05)`. Below ~10 Hz camera rate, color+depth can drift past 50 ms and the sync won't fire ⇒ `status=1, error_msg='No camera data for …'`. Not a node bug — check `ros2 topic hz` first, then see [`CAMERA_BRINGUP.md`](./CAMERA_BRINGUP.md) for the canonical launch that sustains ~30 Hz.
- kimi_api loads `.env` via `load_dotenv()` from CWD upward at startup. Negative "no key" tests must move `.env` aside.

Per-run results and operator-in-the-loop matrix in [`DEV_NOTES.md`](./DEV_NOTES.md).

## Known follow-ups

Actionable work that remains open. Full context, rationale, and prioritization in [`DEV_NOTES.md § Follow-ups`](./DEV_NOTES.md#follow-ups--ordered-roughly-by-impact). Items 1–5 and 9 from the previous follow-up list were addressed in the 2026-04-22 "Follow-up wave" session — see that DEV_NOTES entry.

1. **Specialist model training.** The custom-trained competition YOLO does not exist yet — `yolo_seg_node` currently serves pretrained `yolo11m-seg.pt`. `excluded_classes=['person']` is belt-and-suspenders for the future retrain.
2. **VLM latency** (5–10 s/call on Gemini 2.5 Flash) is the dominant cost when `enable_vlm=True`. Default fallback is now YOLO-World (~150–400 ms/call locally) — only flip `enable_vlm` on if YOLO-World can't recognise the target class.
3. **Triple-subscription of camera streams** when specialist + default + generalist all run together. Not urgent, but worth factoring the input half of `YOLOSegmentationNode` into a shared node if we keep the three-service split.
4. **`BtNode_TrackPerson` / `BtNode_ScanForWavingPerson` / `BtNode_FindPointedLuggage` rearchitect** — these BT nodes were migrated to the tk26 generalist srv mechanically in Wave 2.1, but their *semantics* depend on tk23-only response fields the tk26 detection nodes never populate (`result.person_id`, `Object.being_pointed`). The full catalog of broken nodes, live-task blast radius, and recommended fix per node lives alongside the code in [`src/tk25_decision/CLAUDE.md § Known issues & broken nodes`](../../src/tk25_decision/CLAUDE.md#known-issues--broken-nodes).
