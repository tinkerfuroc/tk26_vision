# foundation_stereo

ROS2 node serving FoundationStereo + Fast-FoundationStereo over two modes:

- **On-request**: `/foundation_stereo/get_depth` service and
  `/foundation_stereo/infer_depth` action (per-call overrides for model,
  scale, iters, TRT variant; the action surfaces per-stage feedback +
  cancellation).
- **Streaming**: optional depth publisher that mimics the realsense driver's
  `aligned_depth_to_color/image_rect_raw` topic shape (`16UC1 mm` by
  default, `SensorDataQoS`).

## Why a separate venv

`torch==2.8.0+cu128` + `tensorrt==10.16.1.11` conflict with the versions in
`.venv-vision-main`. This package builds + runs under
`src/tk26_vision/.venv-fs/`, provisioned manually (see below).

## Provisioning `.venv-fs` (one-time)

```bash
cd src/tk26_vision
python3.10 -m venv .venv-fs --system-site-packages --symlinks
source .venv-fs/bin/activate
pip install --upgrade pip wheel
pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r src/foundation_stereo/requirements.txt
pip install tensorrt==10.16.1.11
pip freeze > .venv-fs/freeze.lock.txt
```

## Build + run

```bash
./src/tk26_vision/scripts/build_foundation_stereo.sh
source install/setup.bash

# Default: srv + action only, no streaming.
ros2 launch foundation_stereo foundation_stereo.launch.py

# Streaming, aligned to color (recommended for D435).
ros2 launch foundation_stereo foundation_stereo.launch.py \
    stream_enabled:=true stream_align_to_color:=true
```

## D435 frame-name caveat

`realsense2_camera` publishes the left IR optical frame as
`xarm_camera_infra1_optical_frame`. The xarm URDF declares it as
`xarm_camera_left_ir_optical_frame`. No static_transform_publisher
bridges them. Two clean options when consuming this node's raw-mode
depth:

1. Stream aligned-to-color (the default). Output frame becomes
   `xarm_camera_color_optical_frame`, which agrees between driver and URDF.
2. Set `output_frame_id:=xarm_camera_left_ir_optical_frame` so the published
   depth carries the URDF name. Geometrically equivalent — same sensor.
3. Or run a bridging static TF once at bringup:
   ```bash
   ros2 run tf2_ros static_transform_publisher \
       0 0 0  0 0 0 \
       xarm_camera_left_ir_optical_frame \
       xarm_camera_infra1_optical_frame
   ```

## Spec + plan

- Design: [`docs/superpowers/specs/2026-05-24-foundation-stereo-design.md`](../../docs/superpowers/specs/2026-05-24-foundation-stereo-design.md)
- Implementation plan: [`docs/superpowers/plans/2026-05-24-foundation-stereo.md`](../../docs/superpowers/plans/2026-05-24-foundation-stereo.md)
