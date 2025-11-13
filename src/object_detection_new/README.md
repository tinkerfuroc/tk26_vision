# Object Detection New

Simplified ROS2 object detection node using YOLOv11 segmentation.

## Features

- Object detection and segmentation using YOLOv11-seg
- 3D position estimation from RGB-D data
- Support for RealSense and Kinect cameras
- ROS2 service interface for on-demand detection
- Optional continuous detection publishing
- Segmentation mask extraction

## Installation

1. Place your fine-tuned YOLOv11-seg model in `models/` directory
2. Build the package:
```bash
cd /home/cindy/Documents/tk25_ws
colcon build --packages-select object_detection_new
source install/setup.bash
```

## Usage

### Running the Node

```bash
ros2 run object_detection_new yolo_seg_node
```

### With Parameters

```bash
ros2 run object_detection_new yolo_seg_node --ros-args \
  --params-file src/object_detection_new/config/default.yaml
```

### Service Call Example

```bash
# Basic detection
ros2 service call /object_detection_yolo tinker_vision_msgs/srv/ObjectDetection \
  "{flags: []}"

# With segmentation masks and images
ros2 service call /object_detection_yolo tinker_vision_msgs/srv/ObjectDetection \
  "{flags: ['request_segments', 'request_image']}"
```

## Parameters

- `camera_type`: 'realsense' or 'kinect' (default: 'realsense')
- `model_path`: Path to YOLO model (default: 'yolo11m-seg.pt')
- `image_topic`: RGB image topic (default: '/camera/camera/color/image_raw')
- `depth_topic`: Depth/PointCloud topic (default: '/camera/camera/aligned_depth_to_color/image_raw')
- `camera_info_topic`: Camera info topic (default: '/camera/camera/aligned_depth_to_color/camera_info')
- `confidence_threshold`: Detection confidence threshold (default: 0.5)
- `publish_rate`: Rate for continuous publishing in Hz, 0 to disable (default: 5.0)
- `visualization`: Enable CV visualization (default: false)
- `max_depth`: Maximum valid depth in meters (default: 10.0)
- `min_depth`: Minimum valid depth in meters (default: 0.1)

## Service Interface

### Request
- `flags` (string[]): Optional flags for the request
  - `'request_segments'`: Return segmentation masks
  - `'request_image'`: Return RGB and depth images

### Response
- `header`: Message header with timestamp
- `status`: 0 for success, 1 for no detections or error
- `objects`: Array of detected objects with:
  - `cls`: Class name
  - `conf`: Confidence score
  - `centroid`: 3D position (geometry_msgs/Point)
  - `object_id`: Class ID
  - `id`: Always 0 (no tracking)
- `rgb_image`: RGB image (if requested)
- `depth_image`: Depth image (if requested)
- `segments`: Segmentation masks (if requested)

## Topics

### Subscribed
- RGB image (sensor_msgs/Image)
- Depth image or PointCloud2 (depending on camera_type)
- Camera info (sensor_msgs/CameraInfo, RealSense only)

### Published
- `detections` (tinker_vision_msgs/Objects): Continuous detection results (if publish_rate > 0)

## Testing

```bash
# Run all tests
colcon test --packages-select object_detection_new

# View results
colcon test-result --verbose
```

## Differences from Original

This simplified version:
- **Removes** DeepSort tracking
- **Removes** person registration and matching
- **Removes** pose detection (MediaPipe)
- **Removes** pointing detection
- **Removes** waving detection
- **Keeps** core object detection and segmentation
- **Keeps** 3D position calculation
- **Keeps** service interface compatibility

## Model Support

The node supports:
- YOLOv11-seg models (recommended: yolo11m-seg.pt)
- Fine-tuned YOLO segmentation models
- Any Ultralytics YOLO segmentation model

Place your model in the `models/` directory or specify an absolute path in the configuration.
