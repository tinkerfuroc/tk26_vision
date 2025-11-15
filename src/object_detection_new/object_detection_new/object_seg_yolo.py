import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import numpy as np
import cv2
from pathlib import Path
import threading
import copy
import os
import torch

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
from cv_bridge import CvBridge


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

        # Thread locks for data protection
        self.lock_msg = threading.Lock()
        self.lock_info = threading.Lock()

        # Camera data storage (per camera type)
        self.camera_intrinsic = {
            'realsense': None,
            'orbbec': None
        }
        self.recent_sync_msg = {
            'realsense': None,
            'orbbec': None
        }
        self.recent_publish_time = {
            'realsense': None,
            'orbbec': None
        }
        self.get_logger().info('YOLO Segmentation Node initialized successfully')

    def _declare_parameters(self):
        """Declare all ROS2 parameters with defaults."""
        self.get_logger().info('Declaring parameters...')
        self.declare_parameter('camera_types', ['realsense', 'orbbec'])
        self.declare_parameter('model_path', 'yolo11m-seg.pt')
        # Realsense topics
        self.declare_parameter(
            'realsense_image_topic', '/camera/xarm_camera/color/image_raw')
        self.declare_parameter(
            'realsense_depth_topic',
            '/camera/xarm_camera/aligned_depth_to_color/image_raw')
        self.declare_parameter(
            'realsense_camera_info_topic',
            '/camera/xarm_camera/aligned_depth_to_color/camera_info')
        # Orbbec topics
        self.declare_parameter(
            'orbbec_image_topic', '/camera/color/image_raw')
        self.declare_parameter(
            'orbbec_depth_topic', '/camera/depth_registered/points')
        self.declare_parameter(
            'orbbec_camera_info_topic', '/camera/color/camera_info')
        # Hz, 0 = no continuous publishing
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('visualization', False)
        self.declare_parameter('max_depth', 10.0)  # meters
        self.declare_parameter('min_depth', 0.1)   # meters

        # vision log folder
        self.declare_parameter('vision_log_folder', 'tmp/vision_log')
        
        # Sorting mode: 'none', 'closest', 'highest'
        self.declare_parameter('sort_mode', 'none')
        
        self.get_logger().info('Parameters declared successfully')

    def _load_parameters(self):
        """Load all parameters."""
        self.get_logger().info('Loading parameters...')
        
        self.camera_types = self.get_parameter('camera_types').value
        self.get_logger().info(f'Camera types: {self.camera_types}')
        
        self.model_path = self.get_parameter('model_path').value
        self.get_logger().info(f'Model path: {self.model_path}')
        
        self.publish_rate = self.get_parameter('publish_rate').value
        self.get_logger().info(f'Publish rate: {self.publish_rate} Hz')
        
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
        
        self.visualization = self.get_parameter('visualization').value
        self.get_logger().info(f'Visualization: {self.visualization}')
        
        self.max_depth = self.get_parameter('max_depth').value
        self.min_depth = self.get_parameter('min_depth').value
        self.get_logger().info(f'Depth range: {self.min_depth}m - {self.max_depth}m')
        
        self.vision_log_folder = self.get_parameter('vision_log_folder').value
        self.get_logger().info(f'Vision log folder: {self.vision_log_folder}')

        self.sort_mode = self.get_parameter('sort_mode').value
        self.get_logger().info(f'Default sort mode: {self.sort_mode}')

        if not os.path.exists(self.vision_log_folder):
            os.makedirs(self.vision_log_folder)
            self.get_logger().info(f'Created vision log folder: {self.vision_log_folder}')

    def _init_model(self):
        """Initialize YOLO model."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')
        try:
            model_file = Path(self.model_path)
            found = False

            # If not absolute path, search for it
            if not model_file.is_absolute():
                # Try to find in installed share directory first
                try:
                    from ament_index_python.packages import get_package_share_directory
                    share_dir = Path(get_package_share_directory('object_detection_new'))
                    self.get_logger().info(f'Package share directory: {share_dir}')
                    share_model = 'models/' + self.model_path
                    file_path = os.path.join(share_dir, share_model)
                    self.get_logger().info(f'Checking share directory: {file_path}')
                    if os.path.exists(file_path):
                        model_file = file_path
                        found = True
                        self.get_logger().info('Found model in share directory')
                except Exception as e:
                    self.get_logger().warn(f'Could not check share directory: {e}')

                # Try to find in package source directory
                if not found:
                    pkg_dir = Path(__file__).parent.parent
                    src_model = pkg_dir / 'models' / self.model_path
                    self.get_logger().info(f'Checking source directory: {src_model}')
                    if os.path.exists(src_model):
                        model_file = src_model
                        found = True
                        self.get_logger().info('Found model in source directory')
            else:
                found = os.path.exists(model_file)

            if not found:
                self.get_logger().warn(
                    f'Model not found, will try to download {self.model_path}'
                )
                model_file = Path(self.model_path)

            self.model = YOLO(str(model_file))
            self.model.to(self.device)
            self.get_logger().info(f'YOLO model loaded from {model_file}')

            # Warm up the model with a dummy inference to compile CUDA kernels
            if self.device == 'cuda':
                self.get_logger().info('Warming up YOLO model on GPU...')
                dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
                _ = self.model(dummy_img, imgsz=(640, 640), verbose=False)
                self.get_logger().info('Model warm-up complete')
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

        # Subscribe to both realsense and orbbec cameras
        if 'realsense' in self.camera_types:
            cb_realsense = MutuallyExclusiveCallbackGroup()

            realsense_image_topic = self.get_parameter('realsense_image_topic').value
            realsense_depth_topic = self.get_parameter('realsense_depth_topic').value
            realsense_camera_info_topic = self.get_parameter('realsense_camera_info_topic').value

            image_sub_realsense = Subscriber(
                self, Image, realsense_image_topic, qos_profile=qos_profile
            )
            depth_sub_realsense = Subscriber(
                self, Image, realsense_depth_topic, qos_profile=qos_profile
            )

            sync_realsense = ApproximateTimeSynchronizer(
                [image_sub_realsense, depth_sub_realsense],
                queue_size=10,
                slop=0.1
            )
            sync_realsense.registerCallback(self._realsense_callback)

            self.camera_info_sub_realsense = self.create_subscription(
                CameraInfo,
                realsense_camera_info_topic,
                self._camera_info_realsense_callback,
                qos_profile=10,
                callback_group=cb_realsense
            )
            self.get_logger().info('Subscribed to realsense camera')

        if 'orbbec' in self.camera_types:
            cb_orbbec = MutuallyExclusiveCallbackGroup()

            orbbec_image_topic = self.get_parameter('orbbec_image_topic').value
            orbbec_depth_topic = self.get_parameter('orbbec_depth_topic').value
            orbbec_camera_info_topic = self.get_parameter('orbbec_camera_info_topic').value

            image_sub_orbbec = Subscriber(
                self, Image, orbbec_image_topic, qos_profile=qos_profile
            )
            depth_sub_orbbec = Subscriber(
                self, PointCloud2, orbbec_depth_topic, qos_profile=qos_profile
            )

            sync_orbbec = ApproximateTimeSynchronizer(
                [image_sub_orbbec, depth_sub_orbbec],
                queue_size=10,
                slop=0.1
            )
            sync_orbbec.registerCallback(self._orbbec_callback)

            self.camera_info_sub_orbbec = self.create_subscription(
                CameraInfo,
                orbbec_camera_info_topic,
                self._camera_info_orbbec_callback,
                qos_profile=10,
                callback_group=cb_orbbec
            )
            self.get_logger().info('Subscribed to orbbec camera')

    def _init_publishers(self):
        """Initialize publishers."""
        pass
        # if self.publish_rate > 0:
        #     self.detection_pub = self.create_publisher(
        #         Objects, 'detections', 10
        #     )
        #     # Timer for rate-limited publishing
        #     self.publish_timer = self.create_timer(
        #         1.0 / self.publish_rate,
        #         self._publish_detections
        #     )

    def _init_service(self):
        """Initialize detection service."""
        service_name = 'object_detection_yolo'
        self.detection_srv = self.create_service(
            ObjectDetection,
            service_name,
            self._detection_service_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.get_logger().info(f'Detection service created: {service_name}')

    def _camera_info_realsense_callback(self, msg: CameraInfo):
        """Store realsense camera intrinsic parameters."""
        self.lock_info.acquire()
        self.camera_intrinsic['realsense'] = msg
        self.lock_info.release()

    def _camera_info_orbbec_callback(self, msg: CameraInfo):
        """Store orbbec camera intrinsic parameters."""
        self.lock_info.acquire()
        self.camera_intrinsic['orbbec'] = msg
        self.lock_info.release()

    def _realsense_callback(self, rgb_msg: Image, depth_msg: Image):
        """Process synchronized realsense RGB and depth messages."""
        self.lock_msg.acquire()
        self.recent_sync_msg['realsense'] = (rgb_msg, depth_msg)
        self.recent_publish_time['realsense'] = self.get_clock().now()
        self.lock_msg.release()

    def _orbbec_callback(self, rgb_msg: Image, depth_msg: PointCloud2):
        """Process synchronized orbbec RGB and depth messages."""
        self.lock_msg.acquire()
        self.recent_sync_msg['orbbec'] = (rgb_msg, depth_msg)
        self.recent_publish_time['orbbec'] = self.get_clock().now()
        self.lock_msg.release()

    def _depth_to_points(self, depth_img: np.ndarray, intrinsic: CameraInfo) -> tuple:
        """Convert depth image to 3D points using camera intrinsics."""
        H, W = depth_img.shape
        fx = intrinsic.k[0]
        fy = intrinsic.k[4]
        cx = intrinsic.k[2]
        cy = intrinsic.k[5]

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

    def _pointcloud_to_array(self, pc_msg: PointCloud2, intrinsic: CameraInfo) -> tuple:
        """
        Convert PointCloud2 to point array (Orbbec format).

        Orbbec outputs unordered point cloud, need to reproject to image grid.
        """
        h, w = 720, 1280
        K = np.array(intrinsic.k).reshape((3, 3))

        # Parse point cloud
        arr = np.frombuffer(pc_msg.data, dtype='<f4')
        N = len(arr) // 5  # x, y, z, rgb (padding to 5 floats)
        points = arr.reshape((N, 5))[:, [0, 1, 2]]

        # Project to image coordinates
        points_homo = points / np.repeat(points[:, 2:3], 3, axis=1)
        coor_homo = (K @ points_homo.T).T
        coor = np.rint(coor_homo[:, :2]).astype(int)

        # Create depth image
        depth_img = np.zeros((h, w, 3))
        valid_coords = (coor[:, 0] >= 0) & (coor[:, 0] < w) & \
                       (coor[:, 1] >= 0) & (coor[:, 1] < h)
        depth_img[coor[valid_coords, 1], coor[valid_coords, 0], :] = points[valid_coords]

        # Valid mask
        valid_mask = (depth_img[:, :, 2] > self.min_depth) & \
                     (depth_img[:, :, 2] < self.max_depth)

        return depth_img, valid_mask

    def _process_realsense_data(self, rgb_msg: Image, depth_msg: Image,
                                intrinsic: CameraInfo) -> tuple:
        """Process realsense RGB-D data into usable format."""
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")
        depth_img = depth_img.astype(float) / 1000.0  # mm to meters

        points, valid_mask = self._depth_to_points(depth_img, intrinsic)

        return rgb_img, points, valid_mask, depth_msg.header

    def _process_orbbec_data(self, rgb_msg: Image, depth_msg: PointCloud2,
                             intrinsic: CameraInfo) -> tuple:
        """Process orbbec RGB-D data into usable format."""
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        points, valid_mask = self._pointcloud_to_array(depth_msg, intrinsic)

        return rgb_img, points, valid_mask, depth_msg.header

    def _sort_objects_and_segments(self, objects: list, segments: list, 
                                     sort_mode: str) -> tuple:
        """
        Sort detected objects and their corresponding segments based on centroid position.
        
        Parameters
        ----------
        objects : list
            List of Object messages
        segments : list
            List of segmentation masks (aligned with objects)
        sort_mode : str
            Sorting mode: 'none', 'closest', 'highest'
            - 'none': No sorting (original detection order)
            - 'closest': Sort by distance (closest first, based on centroid.x)
            - 'highest': Sort by height (highest first, based on centroid.z)
        
        Returns
        -------
        tuple
            (sorted objects list, sorted segments list)
        """
        if sort_mode == 'none' or not objects:
            return objects, segments
        
        # Create list of (index, object) tuples for sorting
        indexed_objects = list(enumerate(objects))
        
        if sort_mode == 'closest':
            # Sort by distance (centroid.x in robot frame = forward distance)
            indexed_objects.sort(key=lambda x: math.sqrt(x[1].centroid.x**2 + x[1].centroid.y**2 + x[1].centroid.z**2))
        
        elif sort_mode == 'highest':
            # Sort by height (centroid.z in robot frame = vertical position)
            # Negative to get highest first (larger z values first)
            indexed_objects.sort(key=lambda x: x[1].centroid.y)
        
        else:
            self.get_logger().warn(f'Unknown sort_mode: {sort_mode}, using none')
            return objects, segments
        
        # Extract sorted indices and objects
        sorted_indices = [idx for idx, _ in indexed_objects]
        sorted_objects = [obj for _, obj in indexed_objects]
        
        # Sort segments using the same indices
        sorted_segments = [segments[idx] for idx in sorted_indices] if segments else []
        
        return sorted_objects, sorted_segments

    def _detect_objects(
            self, rgb_img: np.ndarray, points: np.ndarray,
            target_cls: str,
            valid_mask: np.ndarray, header: Header,
            camera: str = 'realsense',
            request_segments: bool = False,
            sort_mode: str = 'none') -> tuple:
        """
        Run object detection and return results.

        Parameters
        ----------
        rgb_img : np.ndarray
            RGB image
        points : np.ndarray
            3D point cloud array
        target_cls : str
            Target class to detect
        valid_mask : np.ndarray
            Valid depth mask
        header : Header
            ROS message header
        camera : str
            'realsense' or 'orbbec' - determines coordinate transformation
        request_segments : bool
            Whether to return segmentation masks
        sort_mode : str
            Sorting mode: 'none', 'closest', 'highest'

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
        start_time = self.get_clock().now()
        results = self.model(rgb_padded, imgsz=(h_pad, w_pad), verbose=False)
        end_time = self.get_clock().now()
        inference_time = (end_time - start_time).nanoseconds / 1e6  # ms
        self.get_logger().info(f'YOLO inference time: {inference_time:.2f} ms')

        # Prepare response
        objects_msg = Objects()
        objects_msg.header = header
        objects_msg.status = 0
        objects_msg.objects = []

        segments = []
        detection_info = []  # Store info for visualization: (bbox, mask, cls_name, conf, centroid)

        # Process detections
        for result in results:
            
            if result.boxes is None or result.masks is None:
                continue

            boxes = result.boxes
            masks = result.masks

            self.get_logger().info(f'Found {len(boxes.cls)} total detections')

            for i in range(len(boxes.cls)):
                conf = float(boxes.conf[i])

                # Filter by confidence
                if conf < self.conf_threshold:
                    continue

                cls_id = int(boxes.cls[i])
                cls_name = self.model.names[cls_id]
                if cls_name != target_cls:
                    continue

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
                    points, mask, valid_mask, (y1, x1, y2, x2), camera
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
                
                # Store detection info for visualization
                detection_info.append({
                    'bbox': (x1, y1, x2, y2),
                    'mask': mask,
                    'cls_name': cls_name,
                    'conf': conf,
                    'centroid': centroid
                })
        
        self.get_logger().info(f'Detected {len(objects_msg.objects)} objects of class "{target_cls}"')
        
        # Sort objects and segments together if requested
        objects_msg.objects, segments = self._sort_objects_and_segments(
            objects_msg.objects, segments, sort_mode
        )
        
        # Also sort detection_info to match
        if sort_mode != 'none' and detection_info:
            # Create indexed list
            indexed_info = list(enumerate(detection_info))
            
            if sort_mode == 'closest':
                indexed_info.sort(key=lambda x: x[1]['centroid'].x)
            elif sort_mode == 'highest':
                indexed_info.sort(key=lambda x: -x[1]['centroid'].z)
            
            detection_info = [info for _, info in indexed_info]
        
        # Visualize all detections in one image
        if self.visualization and detection_info:
            self._visualize_all_detections(rgb_img, detection_info, header)

        objects_msg.status = 0 if len(objects_msg.objects) > 0 else 1

        return objects_msg, segments

    def _calculate_centroid(
            self, points: np.ndarray, mask: np.ndarray,
            valid_mask: np.ndarray,
            bbox: tuple, camera: str) -> geometry_msgs.msg.Point:
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

        # Calculate median for depth, mean for x/y
        obj_pts = roi_points[combined_mask]

        # Create Point message (coordinate system depends on camera type)
        point = geometry_msgs.msg.Point()
        if camera == 'realsense':
            centroid_3d = np.mean(obj_pts, axis=0)
            centroid_3d[2] = np.median(obj_pts[:, 2])  # Use median for depth

            point.x = float(centroid_3d[2])
            point.y = float(-centroid_3d[0])
            point.z = float(-centroid_3d[1])
        else:  # orbbec
            # Orbbec already in correct frame
            centroid_3d = np.mean(obj_pts, axis=0)
            centroid_3d[2] = np.median(obj_pts[:, 2])  # Use median for depth

            point.x = float(centroid_3d[0])
            point.y = float(centroid_3d[1])
            point.z = float(centroid_3d[2])

        return point

    def _visualize_all_detections(
            self, img: np.ndarray, detection_info: list, header: Header):
        """
        Visualize all detections in a single image.
        
        Parameters
        ----------
        img : np.ndarray
            RGB image to draw on
        detection_info : list
            List of detection dictionaries with keys: 'bbox', 'mask', 'cls_name', 'conf', 'centroid'
        header : Header
            ROS message header (for timestamp in filename)
        """
        # Create a copy to avoid modifying original
        vis_img = img.copy()
        
        # Generate random colors for each detection
        np.random.seed(42)  # For consistent colors
        colors = [tuple(map(int, np.random.randint(0, 255, 3))) for _ in detection_info]
        
        self.get_logger().info(f'Visualizing {len(detection_info)} detections')
        
        # Draw all detections
        for idx, (info, color) in enumerate(zip(detection_info, colors)):
            bbox = info['bbox']
            mask = info['mask']
            cls_name = info['cls_name']
            conf = info['conf']
            centroid = info['centroid']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with index (shows sorting order)
            label = f'#{idx+1} {cls_name} {conf:.2f} d={math.sqrt(centroid.x**2 + centroid.y**2 + centroid.z**2):.2f}m h={centroid.y:.2f}m'

            # Add background for text readability
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                vis_img, (x1, y1 - label_h - 10), (x1 + label_w, y1),
                color, -1
            )
            cv2.putText(
                vis_img, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )
            
            # Draw mask contours
            contours, _ = cv2.findContours(
                (mask * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis_img, contours, -1, color, 2)
            
            # Optional: semi-transparent mask overlay
            mask_overlay = vis_img.copy()
            mask_overlay[mask] = (
                mask_overlay[mask] * 0.5 + np.array(color) * 0.5
            ).astype(np.uint8)
            vis_img = cv2.addWeighted(vis_img, 0.7, mask_overlay, 0.3, 0)
        
        # Save with timestamp
        timestamp = header.stamp.sec * 1000000000 + header.stamp.nanosec
        filename = f'{self.vision_log_folder}/detection_{timestamp}.png'
        cv2.imwrite(filename, vis_img)
        self.get_logger().info(f'Saved visualization to {filename}')

    def _detection_service_callback(
            self, request: ObjectDetection.Request,
            response: ObjectDetection.Response
    ) -> ObjectDetection.Response:
        """Handle detection service requests."""
        self.get_logger().info('Detection service request received')

        # Determine which camera to use
        camera = 'orbbec'  # default
        if 'realsense' in request.camera:
            camera = 'realsense'
        elif 'orbbec' in request.camera:
            camera = 'orbbec'
        else:
            self.get_logger().warn(f'Unknown camera: {request.camera}, using orbbec')

        # Get camera data with thread safety
        self.lock_msg.acquire()
        rec_msg = copy.deepcopy(self.recent_sync_msg.get(camera))
        self.lock_msg.release()

        if rec_msg is None:
            response.header = Header(stamp=self.get_clock().now().to_msg())
            response.status = 1
            response.objects = []
            response.person_id = 0
            self.get_logger().warn(f'No {camera} camera data available')
            return response

        # Get camera intrinsics
        self.lock_info.acquire()
        intrinsic = copy.deepcopy(self.camera_intrinsic.get(camera))
        self.lock_info.release()

        if intrinsic is None:
            response.header = Header(stamp=self.get_clock().now().to_msg())
            response.status = 1
            response.objects = []
            response.person_id = 0
            self.get_logger().warn(f'No {camera} camera intrinsic data')
            return response

        # Process camera data
        try:
            if camera == 'realsense':
                rgb_img, points, valid_mask, header = self._process_realsense_data(
                    rec_msg[0], rec_msg[1], intrinsic
                )
            else:  # orbbec
                rgb_img, points, valid_mask, header = self._process_orbbec_data(
                    rec_msg[0], rec_msg[1], intrinsic
                )
        except Exception as e:
            self.get_logger().error(f'Error processing {camera} data: {e}')
            response.header = Header(stamp=self.get_clock().now().to_msg())
            response.status = 1
            response.objects = []
            response.person_id = 0
            return response

        # Parse request flags
        request_segments = 'request_segments' in request.flags
        request_image = 'request_image' in request.flags
        
        # Parse sorting mode from flags or use default
        sort_mode = self.sort_mode  # default from parameters
        if 'sort_closest' in request.flags:
            sort_mode = 'closest'
        elif 'sort_highest' in request.flags:
            sort_mode = 'highest'
        elif 'sort_none' in request.flags:
            sort_mode = 'none'

        # Run detection
        try:
            objects_msg, segments = self._detect_objects(
                rgb_img,
                points,
                request.prompt,
                valid_mask,
                header,
                camera=camera,
                request_segments=request_segments,
                sort_mode=sort_mode
            )

            # Fill response
            response.header = objects_msg.header
            response.status = objects_msg.status
            response.objects = objects_msg.objects
            response.person_id = 0  # No tracking

            # Add RGB image if requested
            if request_image:
                response.rgb_image = self.bridge.cv2_to_imgmsg(
                    rgb_img, "bgr8"
                )
                depth_img = points[:, :, 2].astype(np.float32)
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
                f'Detected {len(response.objects)} objects using {camera}'
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

    # Use MultiThreadedExecutor for concurrent callback processing
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.visualization:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
