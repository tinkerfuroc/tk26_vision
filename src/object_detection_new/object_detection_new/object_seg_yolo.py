import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import cv2
from pathlib import Path

# ROS2 messages
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Header
import geometry_msgs.msg
from tinker_vision_msgs.msg import Object, Objects
from tinker_vision_msgs.srv import ObjectDetection

# Message filters for synchronization
from message_filters import Subscriber, ApproximateTimeSynchronizer

# Computer vision
from ultralytics import YOLO
from cv_bridge import CvBridge, CvBridgeError


class YOLOSegmentationNode(Node):
    """
    Simplified object detection node using YOLOv11-seg.

    Provides object detection and segmentation without tracking or
    advanced features.
    """

    def __init__(self):
        super().__init__('yolo_segmentation_node')

        # Declare parameters
        self._declare_parameters()

        # Load parameters
        self._load_parameters()

        # Initialize components
        self._init_model()
        self._init_subscribers()
        self._init_publishers()
        self._init_service()

        # State variables
        self.bridge = CvBridge()
        self.camera_intrinsic = None
        self.recent_rgb_msg = None
        self.recent_depth_msg = None
        self.recent_rgb_img = None
        self.recent_depth_img = None
        self.recent_points = None
        self.recent_valid_mask = None
        self.last_publish_time = None

        self.get_logger().info('YOLO Segmentation Node initialized successfully')

    def _declare_parameters(self):
        """Declare all ROS2 parameters with defaults."""
        self.declare_parameter('camera_type', 'realsense')  # 'realsense' or 'kinect'
        self.declare_parameter('model_path', 'yolov11m-seg.pt')
        self.declare_parameter(
            'image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter(
            'depth_topic',
            '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter(
            'camera_info_topic',
            '/camera/camera/aligned_depth_to_color/camera_info')
        # Hz, 0 = no continuous publishing
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('visualization', False)
        self.declare_parameter('max_depth', 10.0)  # meters
        self.declare_parameter('min_depth', 0.1)   # meters

    def _load_parameters(self):
        """Load all parameters."""
        self.camera_type = self.get_parameter('camera_type').value
        self.model_path = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.visualization = self.get_parameter('visualization').value
        self.max_depth = self.get_parameter('max_depth').value
        self.min_depth = self.get_parameter('min_depth').value

        self.get_logger().info(f'Camera type: {self.camera_type}')
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')

    def _init_model(self):
        """Initialize YOLO model."""
        try:
            model_file = Path(self.model_path)
            found = False
            
            # If not absolute path, search for it
            if not model_file.is_absolute():
                # Try to find in installed share directory first
                try:
                    from ament_index_python.packages import get_package_share_directory
                    share_dir = Path(get_package_share_directory('object_detection_new'))
                    share_model = share_dir / 'models' / self.model_path
                    self.get_logger().info(f'Checking share directory: {share_model}')
                    if share_model.exists():
                        model_file = share_model
                        found = True
                        self.get_logger().info(f'Found model in share directory')
                except Exception as e:
                    self.get_logger().warn(f'Could not check share directory: {e}')
                
                # Try to find in package source directory
                if not found:
                    pkg_dir = Path(__file__).parent.parent
                    src_model = pkg_dir / 'models' / self.model_path
                    self.get_logger().info(f'Checking source directory: {src_model}')
                    if src_model.exists():
                        model_file = src_model
                        found = True
                        self.get_logger().info(f'Found model in source directory')

            # Check if file exists (for absolute paths)
            if model_file.is_absolute() and model_file.exists():
                found = True

            if not found:
                self.get_logger().warn(
                    f'Model not found, will try to download {self.model_path}'
                )
                model_file = Path(self.model_path)

            self.model = YOLO(str(model_file))
            self.get_logger().info(f'YOLO model loaded from {model_file}')
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            raise

    def _init_subscribers(self):
        """Initialize image and depth subscribers with synchronization."""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Synchronized subscribers
        self.image_sub = Subscriber(
            self, Image, self.image_topic, qos_profile=qos_profile
        )

        if self.camera_type == 'realsense':
            self.depth_sub = Subscriber(
                self, Image, self.depth_topic, qos_profile=qos_profile
            )
        else:  # kinect
            self.depth_sub = Subscriber(
                self, PointCloud2, self.depth_topic, qos_profile=qos_profile
            )

        # Synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.depth_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self._sync_callback)

        # Camera info subscriber (for realsense)
        if self.camera_type == 'realsense':
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self._camera_info_callback,
                qos_profile=10
            )

    def _init_publishers(self):
        """Initialize publishers."""
        if self.publish_rate > 0:
            self.detection_pub = self.create_publisher(
                Objects, 'detections', 10
            )
            # Timer for rate-limited publishing
            self.publish_timer = self.create_timer(
                1.0 / self.publish_rate,
                self._publish_detections
            )

    def _init_service(self):
        """Initialize detection service."""
        service_name = 'object_detection_yolo'
        self.detection_srv = self.create_service(
            ObjectDetection,
            service_name,
            self._detection_service_callback
        )
        self.get_logger().info(f'Detection service created: {service_name}')

    def _camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsic parameters."""
        self.camera_intrinsic = msg

    def _sync_callback(self, rgb_msg: Image, depth_msg):
        """Process synchronized RGB and depth messages."""
        try:
            # Convert RGB image
            rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")

            # Convert depth based on camera type
            if self.camera_type == 'realsense':
                if self.camera_intrinsic is None:
                    return
                depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
                depth_img = depth_img.astype(float) / 1000.0  # mm to meters
                points, valid_mask = self._depth_to_points(depth_img)
            else:  # kinect
                points, valid_mask = self._pointcloud_to_array(depth_msg)

            # Store latest data
            self.recent_rgb_msg = rgb_msg
            self.recent_depth_msg = depth_msg
            self.recent_rgb_img = rgb_img
            self.recent_points = points
            self.recent_valid_mask = valid_mask

        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge error: {e}')
        except Exception as e:
            self.get_logger().error(f'Sync callback error: {e}')

    def _depth_to_points(self, depth_img: np.ndarray) -> tuple:
        """Convert depth image to 3D points using camera intrinsics."""
        H, W = depth_img.shape
        fx = self.camera_intrinsic.k[0]
        fy = self.camera_intrinsic.k[4]
        cx = self.camera_intrinsic.k[2]
        cy = self.camera_intrinsic.k[5]

        # Create coordinate grids
        x_coords = np.arange(W)
        y_coords = np.arange(H)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        # Back-project to 3D
        z = depth_img
        x = (x_grid - cx) * z / fx
        y = (y_grid - cy) * z / fy

        # Stack to get points [H, W, 3]
        points = np.stack([x, y, z], axis=-1)

        # Valid mask
        valid_mask = np.ones_like(depth_img, dtype=bool)
        valid_mask[depth_img > self.max_depth] = False
        valid_mask[depth_img < self.min_depth] = False
        valid_mask[np.isnan(depth_img)] = False

        return points, valid_mask

    def _pointcloud_to_array(self, pc_msg: PointCloud2) -> tuple:
        """Convert PointCloud2 to point array."""
        # Assuming Kinect format: 1536x2048
        h, w = 1536, 2048
        arr = np.frombuffer(pc_msg.data, dtype='<f4')
        arr = arr.reshape((h, w, 8))[:, :, :3]

        # Valid mask
        valid_mask = ~np.any(np.isnan(arr), axis=2)
        valid_mask &= (arr[:, :, 2] < self.max_depth)
        valid_mask &= (arr[:, :, 2] > self.min_depth)

        # Remove NaNs
        arr = np.nan_to_num(arr, nan=0.0)

        return arr, valid_mask

    def _detect_objects(
            self, rgb_img: np.ndarray, points: np.ndarray,
            valid_mask: np.ndarray, header: Header,
            request_segments: bool = False) -> tuple:
        """
        Run object detection and return results.

        Returns
        -------
        tuple
            (Objects, list of segment masks)

        """
        # Pad image to multiple of 32
        h, w = rgb_img.shape[:2]
        h_pad = ((h + 31) // 32) * 32
        w_pad = ((w + 31) // 32) * 32

        if h != h_pad or w != w_pad:
            rgb_padded = cv2.copyMakeBorder(
                rgb_img, 0, h_pad - h, 0, w_pad - w,
                cv2.BORDER_CONSTANT, value=0
            )
        else:
            rgb_padded = rgb_img

        # Run YOLO inference
        results = self.model(rgb_padded, imgsz=(h_pad, w_pad), verbose=False)

        # Prepare response
        objects_msg = Objects()
        objects_msg.header = header
        objects_msg.status = 0
        objects_msg.objects = []

        segments = []

        # Process detections
        for result in results:
            if result.boxes is None or result.masks is None:
                continue

            boxes = result.boxes
            masks = result.masks

            for i in range(len(boxes.cls)):
                conf = float(boxes.conf[i])

                # Filter by confidence
                if conf < self.conf_threshold:
                    continue

                cls_id = int(boxes.cls[i])
                cls_name = self.model.names[cls_id]

                # Get bounding box
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                x1, y1 = int(x1), int(y1)
                x2, y2 = int(x2), int(y2)

                # Clip to original image size
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # Get segmentation mask
                mask = masks[i].data.cpu().numpy().squeeze()
                mask = mask[:h, :w]  # Crop to original size
                mask = (mask > 0.5).astype(bool)

                # Calculate 3D centroid
                centroid = self._calculate_centroid(
                    points, mask, valid_mask, (y1, x1, y2, x2)
                )

                if centroid is None:
                    self.get_logger().debug(
                        f'Skipping {cls_name}: invalid depth'
                    )
                    continue

                # Create Object message
                obj = Object()
                obj.conf = conf
                obj.cls = cls_name
                obj.centroid = centroid
                obj.id = 0  # No tracking
                obj.object_id = cls_id
                obj.similarity = 0.0
                obj.being_pointed = 0

                objects_msg.objects.append(obj)

                if request_segments:
                    segments.append(mask.astype(np.uint8) * 255)

                # Visualization
                if self.visualization:
                    self._visualize_detection(
                        rgb_img, (x1, y1, x2, y2), mask,
                        cls_name, conf, centroid
                    )

        objects_msg.status = 0 if len(objects_msg.objects) > 0 else 1

        return objects_msg, segments

    def _calculate_centroid(
            self, points: np.ndarray, mask: np.ndarray,
            valid_mask: np.ndarray,
            bbox: tuple) -> geometry_msgs.msg.Point:
        """Calculate 3D centroid from segmentation mask and point cloud."""
        y1, x1, y2, x2 = bbox

        # Extract region of interest
        roi_mask = mask[x1:x2, y1:y2]
        roi_valid = valid_mask[x1:x2, y1:y2]
        roi_points = points[x1:x2, y1:y2]

        # Combine masks
        combined_mask = roi_mask & roi_valid

        if combined_mask.sum() < 10:
            return None

        # Calculate weighted average
        masked_points = roi_points * combined_mask[:, :, np.newaxis]
        centroid_3d = masked_points.sum(axis=(0, 1)) / combined_mask.sum()

        # Create Point message (coordinate system depends on camera type)
        point = geometry_msgs.msg.Point()
        if self.camera_type == 'realsense':
            point.x = float(centroid_3d[2])   # z -> x (forward)
            point.y = float(-centroid_3d[0])  # x -> -y (left)
            point.z = float(-centroid_3d[1])  # y -> -z (up)
        else:  # kinect
            point.x = float(centroid_3d[0])
            point.y = float(centroid_3d[1])
            point.z = float(centroid_3d[2])

        return point

    def _visualize_detection(
            self, img: np.ndarray, bbox: tuple,
            mask: np.ndarray, cls_name: str,
            conf: float, centroid: geometry_msgs.msg.Point):
        """Visualize detection on image."""
        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        label = f'{cls_name} {conf:.2f} d={centroid.x:.2f}m'
        cv2.putText(
            img, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        # Draw mask contours
        contours, _ = cv2.findContours(
            (mask * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    def _publish_detections(self):
        """Publish detections at configured rate."""
        if self.recent_rgb_img is None or self.recent_points is None:
            return

        try:
            objects_msg, _ = self._detect_objects(
                self.recent_rgb_img,
                self.recent_points,
                self.recent_valid_mask,
                self.recent_rgb_msg.header,
                request_segments=False
            )

            self.detection_pub.publish(objects_msg)

            if self.visualization and self.recent_rgb_img is not None:
                cv2.imshow('Detections', self.recent_rgb_img)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error publishing detections: {e}')

    def _detection_service_callback(
            self, request: ObjectDetection.Request,
            response: ObjectDetection.Response
    ) -> ObjectDetection.Response:
        """Handle detection service requests."""
        self.get_logger().info('Detection service request received')

        # Check if data is available
        if self.recent_rgb_img is None or self.recent_points is None:
            response.header = Header(stamp=self.get_clock().now().to_msg())
            response.status = 1
            response.objects = []
            response.person_id = 0
            self.get_logger().warn('No image data available')
            return response

        # Parse request flags
        request_segments = 'request_segments' in request.flags
        request_image = 'request_image' in request.flags

        # Run detection
        try:
            objects_msg, segments = self._detect_objects(
                self.recent_rgb_img,
                self.recent_points,
                self.recent_valid_mask,
                self.recent_depth_msg.header,
                request_segments=request_segments
            )

            # Fill response
            response.header = objects_msg.header
            response.status = objects_msg.status
            response.objects = objects_msg.objects
            response.person_id = 0  # No tracking

            # Add RGB image if requested
            if request_image:
                response.rgb_image = self.bridge.cv2_to_imgmsg(
                    self.recent_rgb_img, "bgr8"
                )
                depth_img = self.recent_points[:, :, 2].astype(np.float32)
                response.depth_image = self.bridge.cv2_to_imgmsg(
                    depth_img, "32FC1"
                )
            else:
                response.rgb_image = self.bridge.cv2_to_imgmsg(
                    np.zeros((1, 1, 3), dtype=np.uint8), "bgr8"
                )
                response.depth_image = self.bridge.cv2_to_imgmsg(
                    np.zeros((1, 1), dtype=np.float32), "32FC1"
                )

            # Add segments if requested
            if request_segments:
                response.segments = [
                    self.bridge.cv2_to_imgmsg(seg, "8UC1")
                    for seg in segments
                ]
            else:
                response.segments = []

            self.get_logger().info(
                f'Detected {len(response.objects)} objects'
            )

        except Exception as e:
            self.get_logger().error(f'Detection failed: {e}')
            response.header = Header(stamp=self.get_clock().now().to_msg())
            response.status = 1
            response.objects = []
            response.person_id = 0

        return response


def main(args=None):
    rclpy.init(args=args)
    node = YOLOSegmentationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.visualization:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
