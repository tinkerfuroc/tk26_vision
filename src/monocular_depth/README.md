# monocular_depth

ROS2 action server that fuses Depth Anything 3 monocular depth with the live
RealSense or Orbbec sensor depth and returns a deprojected point cloud plus the
fused depth image.

## Why a separate package + venv

`depth_anything_3` pins `numpy<2`. The shared `.venv-vision-main` ships
`numpy==2.2.6` (torch 2.11, scipy, ultralytics, opencv-python depend on the
2.x ABI). Installing DA3 into the shared venv would cascade-break every other
vision node. So this package builds + runs under `src/tk26_vision/.venv-da3/`,
which is provisioned independently.

## Provisioning the venv (one-time)

```bash
cd src/tk26_vision
python3.10 -m venv .venv-da3 --system-site-packages --symlinks
source .venv-da3/bin/activate
pip install --upgrade pip wheel
pip install "numpy==1.23.4"
pip install -e thirdparty/depth-anything-3 --no-deps
pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r src/monocular_depth/requirements.txt
pip freeze > .venv-da3/freeze.lock.txt
```

## DA3 vendor patch

`thirdparty/depth-anything-3/src/depth_anything_3/api.py` carries a tk26
patch that defers `from depth_anything_3.utils.export import export` from
module-load time to `_export_results` call time. This drops the hard
dependency on `moviepy`, `open3d`, `pycolmap`, `trimesh`, `plyfile`,
`pillow_heif` for callers that never run `model.inference(..., export_dir=...)`.
The patch is a 2-line change marked with `tk26_vision patch:` comments.

## Build + run

```bash
# Either of these builds the package under .venv-da3:
./src/tk26_vision/scripts/build_monocular_depth.sh
tkbuild tk26_vision --packages-select monocular_depth

source install/setup.bash
ros2 run monocular_depth monocular_depth_pc

# Send a goal — depth_image only:
ros2 action send_goal /monocular_depth_pc \
    tinker_vision_msgs_26/action/MonocularDepthPC \
    "{camera: 'realsense', stride: 1, debug_publish: false}" --feedback

# Send a goal — also publish the debug PointCloud:
ros2 action send_goal /monocular_depth_pc \
    tinker_vision_msgs_26/action/MonocularDepthPC \
    "{camera: 'realsense', stride: 2, debug_publish: true}" --feedback

# Watch the debug cloud:
ros2 topic echo /monocular_depth_pc/debug_points --once
```

The action result is the fused 32FC1 depth image (metres) at the source RGB
resolution — pixel-aligned with the source color frame. The debug PointCloud
is published on `~/debug_points` (default `/monocular_depth_pc/debug_points`,
SensorDataQoS) **only** when `debug_publish=true`. `stride` subsamples the
debug cloud only; the depth image is always full source resolution.

## Parameters

See the action server module docstring for the full list. Defaults:

- `da3_model = depth-anything/DA3-SMALL` — swap to `depth-anything/DA3-BASE`
  via `-p da3_model:=…`. Path also accepts a local checkpoint dir.
- `fill_mode = holes_only` — alternative `full_override`.
- `align_min_overlap_pixels = 2000`, `align_trim_frac = 0.05`.
- `output_frame_id = ''` (defaults to depth msg's frame).
- `debug_pc_topic = ~/debug_points` — topic for the debug PointCloud2 publisher
  (SensorDataQoS). Tilde expands under the action node namespace.
