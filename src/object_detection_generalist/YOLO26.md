# YOLO26 checkpoint segmentation server

Run from `tkmr:/home/tinker/tk25_ws` after building:

```bash
./src/tk26_vision/scripts/build.sh --packages-select object_detection_new object_detection_generalist
source install/setup.bash
ros2 run object_detection_generalist yolo26_seg_node
```

Service: `/object_detection_yolo26`, type
`tinker_vision_msgs_26/srv/ObjectDetectionGeneralist`.
The default checkpoint is `yolo26s-seg.pt`. Change it with
`--ros-args -p model_path:=yolo26m-seg.pt`, or use an absolute path to a
custom segmentation checkpoint. Labels are read directly from `model.names`,
including custom class IDs; no separate hardcoded COCO list is used.
Bare filenames use the shared `TK26_MODEL_CACHE` / `~/.cache/tk26_vision/weights`
cache and download on a cache miss.

```bash
ros2 param get /yolo26_seg_node labels
ros2 service call /object_detection_yolo26 tinker_vision_msgs_26/srv/ObjectDetectionGeneralist "{camera: orbbec, prompt: '', sort_closest: true}"
ros2 service call /object_detection_yolo26 tinker_vision_msgs_26/srv/ObjectDetectionGeneralist "{camera: orbbec, prompt: 'cup . bottle', sort_closest: true, return_segments: true}"
```

- Empty/whitespace prompt detects all checkpoint labels.
- Nonempty prompt selects exact labels, case-insensitively. Multiple labels use
  ` . `, commas (including Chinese commas), semicolons, pipes or newlines.
  Spaces within a label are preserved, e.g. `dining table`.
- Unknown labels reject the request with `status=1` and an explicit
  `error_msg`; there is no open-vocabulary fallback. Both legacy VLM flags
  are accepted for ABI compatibility but detection always uses the checkpoint.
- Generalist camera intake, 3D centroid calculation, TF, closest/highest sorting,
  mask ordering, RGB/depth payloads and the default 1 m RealSense range gate
  are reused. Closest is camera-to-centroid Euclidean distance and takes
  precedence when both sort flags are true. With both false, model order is kept.
- `return_segments` returns one 8UC1 mask per object, in the same order.
  Orbbec response depth uses the latest independent depth intake, as in generalist.
- `status=0` / `detection_source='yolo'` indicates detections;
  empty/error responses use `status=1` / `detection_source='none'`.
- `confidence_threshold` defaults to 0.25 and applies during inference.
  Existing `excluded_classes`, camera/depth and logging parameters remain available.

For an existing caller to use its original endpoint, stop the other server first
and start this node with `--ros-args -p service_name:=object_detection_generalist`.
Do not run two servers on the same service name.

Regression tests (ROS setup and vision venv required):

```bash
source /opt/ros/humble/setup.bash
source /home/tinker/tk25_ws/install/setup.bash
src/tk26_vision/.venv-vision-main/bin/python -m pytest \
  src/tk26_vision/src/object_detection_generalist/test/test_yolo26_seg_node.py \
  src/tk26_vision/src/object_detection_generalist/test/test_yolo26_sam_node.py -q
```

## YOLO26 bbox + MobileSAM variant

`yolo26_sam_node` uses the same
`tinker_vision_msgs_26/srv/ObjectDetectionGeneralist` service type and request /
response behavior, but loads bbox-only YOLO26 weights and uses each detected box
as a MobileSAM prompt:

```bash
ros2 run object_detection_generalist yolo26_sam_node
ros2 service call /object_detection_yolo26_sam \
  tinker_vision_msgs_26/srv/ObjectDetectionGeneralist \
  "{camera: orbbec, prompt: '', sort_closest: true, return_segments: true}"
```

Defaults are `model_path:=yolo26s.pt`, `sam_weights:=mobile_sam.pt`, and
`service_name:=object_detection_yolo26_sam`. `model_path` must be a detection
checkpoint (`model.task == 'detect'`); segmentation checkpoints are rejected at
startup so the two node variants cannot be accidentally interchanged. Empty and
multi-label prompts, exact checkpoint-label matching, depth/TF geometry, sorting,
payload flags, RealSense range gating, and error behavior match
`yolo26_seg_node`. The response reports `detection_source='yolo'` because YOLO
produces the detections; MobileSAM supplies their masks.

