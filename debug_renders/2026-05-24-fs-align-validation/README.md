# FS aligned-depth validation — 2026-05-24

Live D435 capture + foundation_stereo aligned-to-color depth, rendered as
a colored point cloud from 5 angles. Confirms the color↔depth alignment
introduced in `fix(foundation_stereo): use latched QoS for the realsense
Extrinsics subscription` (commit `87a3f01`).

## Files

| File | What |
|---|---|
| `color.jpg` | The source color frame the validation ran against (D435 left RGB). |
| `depth_viz.jpg` | FoundationStereo depth (turbo-colormap, in the color frame). Sparse — about 11% of pixels carry valid depth (~46k of 407k), because the IR1-grid depth was forward-projected into the higher-resolution color grid and we don't hole-fill. |
| `cloud.ply` | Colored point cloud (46k pts). Open in MeshLab / CloudCompare / Open3D viewer to rotate freely. |
| `view_front.png` | Camera POV (the angle the depth was captured from). |
| `view_left_30.png` | Side view, ~30° to the left. |
| `view_right_30.png` | Side view, ~30° to the right. |
| `view_top_45.png` | Bird's-eye, ~45° up. |
| `view_bottom_30.png` | Underside, ~30° below. |
| `fs_validate.py` | The script that produced everything in this dir. |

## Service-call stats

```
status=0  fast_trt / output_two_stage  forward_ms=27.4  load_s=0.00
end_to_end_s=0.06  depth=480×848 (32FC1, m), frame_id=xarm_camera_color_optical_frame
valid pixels: 46760 / 407040 (11.5%)
depth range: 0.067 m .. 2.556 m
K_color: fx=606.7  fy=606.8  cx=429.6  cy=235.5
```

## How to reproduce

```bash
# Terminal 1: D435 + IR pair
export FASTRTPS_DEFAULT_PROFILES_FILE=/home/tinker/tk25_ws/src/tk26_vision/config/fastdds_shm.xml
ros2 launch realsense2_camera rs_launch.py \
    camera_name:=xarm_camera \
    align_depth.enable:=true \
    enable_infra1:=true enable_infra2:=true \
    config_file:=/home/tinker/tk25_ws/src/tk26_vision/config/realsense_qos.yaml

# Terminal 2: foundation_stereo
source /home/tinker/tk25_ws/install/setup.bash
ros2 launch foundation_stereo foundation_stereo.launch.py

# Terminal 3: validation
source /home/tinker/tk25_ws/install/setup.bash
/home/tinker/tk25_ws/src/tk26_vision/.venv-fs/bin/python \
    /home/tinker/tk25_ws/src/tk26_vision/debug_renders/2026-05-24-fs-align-validation/fs_validate.py
```

Output lands in `/tmp/fs_views/`. This dir is a snapshot of one such run.

## Notes for future runs

1. **`.venv-fs` must have `numpy==1.26.4`** — the symlink fallback (Task 7)
   pointed at the reference webapp's venv which had numpy 2.2.6, breaking
   the system `cv_bridge` ABI (segfault on `imgmsg_to_cv2`). The current
   `.venv-fs` is a `cp -al` clone of the reference venv with numpy
   downgraded — see the package README for a clean recipe.
2. **Cancel the IR-pair flag** — `rs_launch.py` doesn't enable infra1/infra2
   by default. `enable_infra1:=true enable_infra2:=true` is required.
3. **Sparse 11% coverage** is expected for forward-projected IR→color
   alignment without hole-fill. Downstream consumers that need denser
   output can apply `cv2.medianBlur` or `cv2.dilate` to the depth image.
