#!/usr/bin/env python3
"""
Person Tracking Action Server Node

This node provides a ROS2 action server for tracking persons using the YOLOTracker.
It receives images from the Orbbec camera and provides continuous feedback with
the tracked person's 3D position.

Author: TinkerFuroc
Date: 2025
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration

import numpy as np
import cv2
import threading
import time
from pathlib import Path
import os

# ROS2 messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

# Action definition
from tinker_vision_msgs_26.action import TrackPerson

# Message filters for synchronization
from message_filters import Subscriber, ApproximateTimeSynchronizer

# CV Bridge
from cv_bridge import CvBridge

# Import YOLOTracker
from vision_track.track_yolo import YOLOTracker, TrackerState, TrackingResult


class PersonTrackNode(Node):
    """
    ROS2 Action Server node for person tracking using YOLO.
    
    This node:
    1. Subscribes to Orbbec camera RGB and depth data
    2. Provides a TrackPerson action server
    3. Tracks a person and provides continuous feedback with 3D position
    """

    def __init__(self):
        super().__init__('person_track_node')
        
        # Declare and load parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize components
        self.bridge = CvBridge()
        self.tracker: YOLOTracker = None
        self.tracking_active = False
        self.goal_handle = None
        self.target_point_pub = None
        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Thread locks
        self.lock_msg = threading.Lock()
        self.lock_info = threading.Lock()
        self.lock_tracker = threading.Lock()
        
        # Camera data storage
        self.camera_intrinsic: CameraInfo = None
        self.recent_sync_msg = None  # (rgb_msg, depth_msg)
        self.recent_msg_time = None
        self.frame_seq = 0  # Frame sequence counter
        self.last_processed_seq = -1  # Last processed frame sequence
        
        # Depth limits
        self.max_depth = 10.0  # meters
        self.min_depth = 0.1   # meters
        
        # Initialize tracker
        self._init_tracker()
        
        # Initialize subscribers
        self._init_subscribers()
        
        # Initialize action server
        self._init_action_server()
        
        self.get_logger().info('Person Track Node initialized successfully')

    def _declare_parameters(self):
        """Declare all ROS2 parameters."""
        # Prefer a stronger default model. If unavailable we will fall back at runtime.
        self.declare_parameter('model_path', 'yolo11s-seg.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('enable_reid', True)
        self.declare_parameter('max_frames_lost', 600)  # ~20 seconds at 30fps
        self.declare_parameter('inference_size', 1280)  # imgsz for YOLO; lower for speed
        self.declare_parameter('reid_verification_interval', 5)  # periodic on-track ReID sanity check
        self.declare_parameter('allow_indefinite_recovery', True)  # if True, never abort for long-term loss
        
        # ReID mode: 'custom' uses our ResNet50-based ReID, 'native' uses YOLO's BoT-SORT ReID
        self.declare_parameter('reid_mode', 'custom')  # 'custom' or 'native'
        
        # Orbbec camera topics
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        
        # Tracking parameters
        self.declare_parameter('tracking_rate', 15.0)  # Hz
        self.declare_parameter('lost_timeout', 300.0)  # seconds before declaring failure
        self.declare_parameter('target_point_topic', '/target_points')  # default PointStamped pub topic

        self.get_logger().info('Parameters declared')

    def _load_parameters(self):
        """Load parameters."""
        self.model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.enable_reid = self.get_parameter('enable_reid').value
        self.max_frames_lost = self.get_parameter('max_frames_lost').value
        self.inference_size = self.get_parameter('inference_size').value
        self.reid_verification_interval = self.get_parameter('reid_verification_interval').value
        self.allow_indefinite_recovery = self.get_parameter('allow_indefinite_recovery').value
        self.reid_mode = self.get_parameter('reid_mode').value
        
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        
        self.tracking_rate = self.get_parameter('tracking_rate').value
        self.lost_timeout = self.get_parameter('lost_timeout').value
        self.default_target_point_topic = self.get_parameter('target_point_topic').value
        
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Enable ReID: {self.enable_reid}')
        self.get_logger().info(f'ReID mode: {self.reid_mode}')
        self.get_logger().info(f'Inference size (imgsz): {self.inference_size}')
        self.get_logger().info(f'ReID verification interval: {self.reid_verification_interval}')
        self.get_logger().info(f'Allow indefinite recovery: {self.allow_indefinite_recovery}')
        self.get_logger().info(f'Tracking rate: {self.tracking_rate} Hz')

    def _init_tracker(self):
        """Initialize the YOLO tracker based on reid_mode parameter."""
        self.get_logger().info('Initializing YOLO Tracker...')
        
        try:
            # Find model path
            model_file = self._find_model_path(self.model_path)
            # Allow loss duration to be governed by time, not fixed frames.
            # Use whichever is larger: explicit max_frames_lost or rate * lost_timeout.
            max_frames_allowed = (
                int(self.tracking_rate * self.lost_timeout)
                if not self.allow_indefinite_recovery
                else int(1e12)  # effectively infinite
            )
            max_frames_allowed = max(max_frames_allowed, int(self.max_frames_lost))
            
            if self.reid_mode == 'native':
                # Use native BoT-SORT ReID from YOLO
                from vision_track.track_yolo_native import YOLOTrackerNative
                self.tracker = YOLOTrackerNative(
                    model_path=str(model_file),
                    confidence_threshold=self.confidence_threshold,
                    appearance_thresh=0.25,  # Lower = stricter ReID matching
                    proximity_thresh=0.5,
                    track_buffer=60,  # 2 seconds at 30fps
                )
                self.tracker.max_frames_lost = max_frames_allowed
                self.get_logger().info(f'YOLO Tracker (NATIVE ReID) initialized with model: {model_file}')
            else:
                # Use custom ResNet50-based ReID (default)
                self.tracker = YOLOTracker(
                    model_path=str(model_file),
                    confidence_threshold=self.confidence_threshold,
                    enable_reid=self.enable_reid,
                    inference_size=self.inference_size,
                    reid_verification_interval=int(self.reid_verification_interval)
                )
                self.tracker.max_frames_lost = max_frames_allowed
                self.get_logger().info(f'YOLO Tracker (CUSTOM ReID) initialized with model: {model_file}')
            
            self.get_logger().info(
                f"Max frames lost set to {self.tracker.max_frames_lost} "
                f"(tracking_rate={self.tracking_rate} Hz, lost_timeout={self.lost_timeout}s, "
                f"param_max_frames_lost={self.max_frames_lost})"
            )
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize tracker: {e}')
            raise

    def _find_model_path(self, model_path: str) -> Path:
        """Find the model file path."""
        model_file = Path(model_path)
        preferred_order = [
            'yolo11x-seg.pt',
            'yolo11l-seg.pt',
            'yolo11m-seg.pt',
            'yolo11s-seg.pt',
            'yolo11n-seg.pt',
        ]
        
        if model_file.is_absolute() and model_file.exists():
            return model_file
        
        # Try package share directory (in models subfolder)
        try:
            from ament_index_python.packages import get_package_share_directory
            share_dir = Path(get_package_share_directory('vision_track'))
            model_dirs = [share_dir / 'models', share_dir]
            for d in model_dirs:
                share_model = d / model_path
                if share_model.exists():
                    self.get_logger().info(f'Found model in share: {share_model}')
                    return share_model
            
            # If requested model missing, pick the best available in share/models
            for d in model_dirs:
                if not d.exists():
                    continue
                for candidate in preferred_order:
                    candidate_path = d / candidate
                    if candidate_path.exists():
                        self.get_logger().warn(
                            f"Requested model '{model_path}' not found; using available model '{candidate_path.name}'"
                        )
                        return candidate_path
        except Exception as e:
            self.get_logger().warn(f'Could not check share directory: {e}')
        
        # Try source directory - package root (where setup.py is)
        pkg_dir = Path(__file__).parent.parent
        src_model = pkg_dir / model_path
        if src_model.exists():
            self.get_logger().info(f'Found model in source directory: {src_model}')
            return src_model
        
        # Try source models subdirectory
        src_model = pkg_dir / 'models' / model_path
        if src_model.exists():
            self.get_logger().info(f'Found model in source/models: {src_model}')
            return src_model
        
        # Try object_detection_new package (has yolo11m-seg.pt)
        try:
            from ament_index_python.packages import get_package_share_directory
            od_share_dir = Path(get_package_share_directory('object_detection_new'))
            od_model = od_share_dir / 'models' / model_path
            if od_model.exists():
                self.get_logger().info(f'Found model in object_detection_new: {od_model}')
                return od_model
            # Try alternative model (yolo11m-seg.pt instead of yolo11n-seg.pt)
            alt_model = 'yolo11m-seg.pt'
            od_model = od_share_dir / 'models' / alt_model
            if od_model.exists():
                self.get_logger().info(f'Using alternative model from object_detection_new: {od_model}')
                return od_model
        except Exception:
            pass
        
        # Try HuggingFace download as a final fallback
        downloaded = self._download_model_from_hf(model_path)
        if downloaded is not None and downloaded.exists():
            return downloaded
        
        # Return as-is (YOLO will try to download if needed)
        self.get_logger().warn(f'Model not found locally and download failed, will try YOLO auto-download: {model_path}')
        return Path(model_path)

    def _download_model_from_hf(self, model_name: str) -> Path:
        """
        Attempt to download the requested model from HuggingFace.
        
        Returns:
            Path to the downloaded file, or None if download not possible.
        """
        try:
            from huggingface_hub import hf_hub_download
        except Exception as exc:
            self.get_logger().warn(f'HuggingFace download unavailable ({exc}); skipping download attempt.')
            return None
        
        # Map common model names to the ultralytics repo filenames
        repo_id = "ultralytics/YOLO11"
        filename = model_name
        # Ensure output directory exists
        cache_dir = Path.home() / ".cache" / "vision_track" / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir
            )
            self.get_logger().info(f"Downloaded model from HuggingFace: {downloaded_path}")
            return Path(downloaded_path)
        except Exception as exc:
            self.get_logger().warn(f"Failed to download model '{model_name}' from HuggingFace: {exc}")
            return None

    def _init_subscribers(self):
        """Initialize camera subscribers with synchronization."""
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        cb_group = MutuallyExclusiveCallbackGroup()

        # Synchronized RGB and depth subscribers
        image_sub = Subscriber(self, Image, self.image_topic, qos_profile=qos_profile)
        depth_sub = Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile)
        
        sync = ApproximateTimeSynchronizer(
            [image_sub, depth_sub],
            queue_size=10,
            slop=0.1
        )
        sync.registerCallback(self._camera_callback)
        
        # Camera info subscriber
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            self.camera_info_topic,
            self._camera_info_callback,
            qos_profile=10,
            callback_group=cb_group
        )
        
        self.get_logger().info(f'Subscribed to camera topics:')
        self.get_logger().info(f'  Image: {self.image_topic}')
        self.get_logger().info(f'  Depth: {self.depth_topic}')
        self.get_logger().info(f'  Info: {self.camera_info_topic}')

    def _init_action_server(self):
        """Initialize the action server."""
        self.action_server = ActionServer(
            self,
            TrackPerson,
            'track_person',
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=ReentrantCallbackGroup()
        )
        self.get_logger().info('Action server created: track_person')

    def _camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsic parameters."""
        with self.lock_info:
            # Log resolution on first camera info received
            if self.camera_intrinsic is None:
                self.get_logger().info(f'Camera info received: resolution {msg.width}x{msg.height}')
            self.camera_intrinsic = msg

    def _camera_callback(self, rgb_msg: Image, depth_msg: Image):
        """Process synchronized RGB and depth messages."""
        with self.lock_msg:
            self.recent_sync_msg = (rgb_msg, depth_msg)
            self.recent_msg_time = self.get_clock().now()
            self.frame_seq += 1

    def _depth_image_to_points(self, depth_msg: Image, intrinsic: CameraInfo) -> tuple:
        """
        Unproject a registered depth image (encoding 16UC1, mm) to per-pixel XYZ.

        Depth is already aligned to color (frame_id=camera_color_optical_frame),
        so color intrinsics apply directly.

        Returns:
            points: np.ndarray of shape (H, W, 3) containing 3D points (meters)
            valid_mask: np.ndarray of shape (H, W) with valid depth mask
        """
        h, w = depth_msg.height, depth_msg.width
        fx, fy = intrinsic.k[0], intrinsic.k[4]
        cx, cy = intrinsic.k[2], intrinsic.k[5]

        # Orbbec Femto Bolt default: 16UC1 depth in millimeters.
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(h, w).astype(np.float32) * 0.001

        valid_mask = (depth > self.min_depth) & (depth < self.max_depth)

        # Cache meshgrid across calls at this resolution.
        cache = getattr(self, '_uv_cache', None)
        if cache is None or cache[0] != (h, w):
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._uv_cache = ((h, w), u, v)
        _, u, v = self._uv_cache

        z = depth
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points = np.stack([x, y, z], axis=-1)

        return points, valid_mask

    def _calculate_centroid(
            self, 
            points: np.ndarray, 
            mask: np.ndarray,
            valid_mask: np.ndarray,
            bbox: tuple
    ) -> Point:
        """
        Calculate 3D centroid from segmentation mask and point cloud.
        
        Args:
            points: 3D point array (H, W, 3)
            mask: Segmentation mask (H, W)
            valid_mask: Valid depth mask (H, W)
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Point message with 3D coordinates, or None if calculation fails
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure bbox is within image bounds
        h, w = points.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract region of interest
        roi_points = points[y1:y2, x1:x2]
        roi_valid = valid_mask[y1:y2, x1:x2]
        
        # Use segmentation mask if available
        if mask is not None and mask.shape[0] > 0 and mask.shape[1] > 0:
            roi_mask = mask[y1:y2, x1:x2]
        else:
            roi_mask = np.ones((y2 - y1, x2 - x1), dtype=np.float32)
        
        # Combine masks
        combined_mask = roi_mask.astype(float) * roi_valid.astype(float)
        
        if combined_mask.sum() < 10:
            # Fallback: use valid depth points without segmentation mask
            combined_mask = roi_valid.astype(float)
        
        if combined_mask.sum() < 10:
            return None
        
        # Get valid points
        obj_pts = roi_points[np.nonzero(combined_mask)]
        if len(obj_pts.shape) != 2 or obj_pts.shape[0] == 0:
            return None
        
        # Calculate centroid (mean for x/y, median for depth)
        centroid_3d = np.mean(obj_pts, axis=0)
        centroid_3d[2] = np.median(obj_pts[:, 2])  # Use median for depth (more robust)
        
        # Create Point message (Orbbec frame convention)
        point = Point()
        point.x = float(centroid_3d[0])
        point.y = float(centroid_3d[1])
        point.z = float(centroid_3d[2])
        
        return point

    def _draw_debug_info(
        self, 
        rgb_img: np.ndarray, 
        all_results: list,
        target_result: 'TrackingResult',
        target_track_id: int
    ) -> np.ndarray:
        """
        Draw debug visualization on the RGB image.
        
        Args:
            rgb_img: BGR image to draw on
            all_results: All tracking results from YOLO
            target_result: The target tracking result (or None)
            target_track_id: The current target YOLO track ID
            
        Returns:
            Annotated BGR image
        """
        debug_img = rgb_img.copy()
        
        # Draw all detected persons
        for result in all_results:
            if result.class_id != 0:  # Skip non-person
                continue
                
            x1, y1, x2, y2 = result.bbox
            track_id = result.track_id
            
            # Determine color based on whether this is the target
            is_target = (target_result is not None and track_id == target_result.track_id)
            is_yolo_target = (track_id == target_track_id)
            
            if is_target:
                color = (0, 255, 0)  # Green for tracked target
                thickness = 3
            elif is_yolo_target:
                color = (0, 255, 255)  # Yellow for YOLO target ID (but not matched)
                thickness = 2
            else:
                color = (255, 0, 0)  # Blue for other detections
                thickness = 1
            
            # Draw bounding box
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, thickness)
            
            # Draw track ID label
            label = f"ID:{track_id}"
            if is_target:
                label += " (TARGET)"
            elif is_yolo_target:
                label += " (YOLO_TARGET)"
            
            # Add confidence
            label += f" {result.confidence:.2f}"
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(debug_img, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
            cv2.putText(debug_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw tracker state info
        state_text = f"Target YOLO ID: {target_track_id}"
        if target_result is not None:
            state_text += f" | Matched: {target_result.track_id}"
        else:
            state_text += " | LOST"
        
        cv2.putText(debug_img, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return debug_img

    def _goal_callback(self, goal_request):
        """Handle incoming goal requests."""
        self.get_logger().info('Received track_person goal request')
        
        # Check if camera data is available
        with self.lock_msg:
            has_data = self.recent_sync_msg is not None
        
        with self.lock_info:
            has_intrinsic = self.camera_intrinsic is not None
        
        if not has_data or not has_intrinsic:
            self.get_logger().warn('No camera data available, rejecting goal')
            return GoalResponse.REJECT
        
        # Check if already tracking
        if self.tracking_active:
            self.get_logger().warn('Already tracking, rejecting new goal')
            return GoalResponse.REJECT
        
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        """Handle cancel requests."""
        self.get_logger().info('Received cancel request')
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        """
        Execute the tracking action.
        
        This is the main tracking loop that:
        1. Initializes tracking on a person
        2. Continuously updates tracking and publishes feedback
        3. Handles target loss and cancellation
        """
        self.get_logger().info('Executing track_person action')
        self.tracking_active = True
        self.goal_handle = goal_handle

        params = {
            "return_rgb_img": goal_handle.request.return_rgb_img,
            "return_depth_img": goal_handle.request.return_depth_img,
            "return_segment": goal_handle.request.return_segment,
            "debug_mode": goal_handle.request.debug,
            "target_frame": goal_handle.request.target_frame.strip() if goal_handle.request.target_frame else '',
        }

        result = TrackPerson.Result()
        feedback = TrackPerson.Feedback()

        topic = (goal_handle.request.target_point_topic or '').strip() or self.default_target_point_topic
        self.target_point_pub = self.create_publisher(PointStamped, topic, 10)
        self.get_logger().info(f'Publishing tracked PointStamped to "{topic}"')

        try:
            return self._run_tracking_loop(goal_handle, feedback, result, params)
        except Exception as e:
            self.get_logger().error(f'Exception during tracking: {e}')
            goal_handle.abort()
            result.status = 1
            result.message = f'Exception: {str(e)}'
            return result
        finally:
            self._cleanup_tracking()

    def _run_tracking_loop(self, goal_handle, feedback, result, params):
        last_seen_time = time.time()
        init_start_time = time.time()
        initialized = False
        rate_period = 1.0 / self.tracking_rate

        while rclpy.ok():
            loop_start = time.time()

            if self._handle_cancel(goal_handle, result):
                return result

            data = self._get_latest_data()
            if data is None:
                time.sleep(0.01)
                continue
            if data is False:
                time.sleep(0.005)
                continue

            rgb_img, rgb_msg, depth_msg, intrinsic = data
            rgb_frame = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

            with self.lock_tracker:
                if not initialized:
                    initialized = self._try_initialize(rgb_frame, init_start_time, goal_handle, result)
                    if not initialized:
                        time.sleep(0.1)
                        continue
                    last_seen_time = time.time()
                track_result = self.tracker.update(rgb_frame)

            if track_result is not None:
                last_seen_time = time.time()
                self._handle_tracked_frame(
                    track_result, rgb_img, rgb_msg, depth_msg, intrinsic, feedback, goal_handle, params
                )
            else:
                if self._handle_lost_frame(last_seen_time, rgb_img, rgb_msg, feedback, goal_handle, params, result):
                    return result

            elapsed = time.time() - loop_start
            if elapsed < rate_period:
                time.sleep(rate_period - elapsed)

        if goal_handle.is_active and not goal_handle.is_cancel_requested:
            result.status = 0
            result.message = 'Tracking completed'
            goal_handle.succeed()
        return result

    def _handle_cancel(self, goal_handle, result) -> bool:
        if not goal_handle.is_cancel_requested:
            return False
        self.get_logger().info('Goal canceled')
        goal_handle.canceled()
        result.status = 2
        result.message = 'Tracking canceled by request'
        return True

    def _get_latest_data(self):
        with self.lock_msg:
            if self.recent_sync_msg is None:
                return None
            current_seq = self.frame_seq
            if current_seq == self.last_processed_seq:
                return False
            rgb_msg, depth_msg = self.recent_sync_msg
            self.last_processed_seq = current_seq

        with self.lock_info:
            intrinsic = self.camera_intrinsic
        if intrinsic is None:
            return None

        try:
            # Avoid CvBridge's extra copy; image is already bgr8 on the wire.
            rgb_img = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(
                rgb_msg.height, rgb_msg.width, 3
            )
        except Exception as e:
            self.get_logger().warn(f'Failed to convert RGB image: {e}')
            return None

        return rgb_img, rgb_msg, depth_msg, intrinsic

    def _try_initialize(self, rgb_frame, init_start_time, goal_handle, result) -> bool:
        success = self.tracker.initialize_tracking(rgb_frame, target_class='person')
        if success:
            self.get_logger().info(f'Tracking initialized on person (ID: {self.tracker.original_track_id})')
            return True

        if time.time() - init_start_time > self.lost_timeout:
            self.get_logger().error('Failed to find person for initialization')
            goal_handle.abort()
            result.status = 1
            result.message = 'No person found for initialization'
        return False

    def _handle_tracked_frame(
        self,
        track_result,
        rgb_img,
        rgb_msg,
        depth_msg,
        intrinsic,
        feedback,
        goal_handle,
        params,
    ):
        try:
            points, valid_mask = self._depth_image_to_points(depth_msg, intrinsic)
        except Exception as e:
            self.get_logger().warn(f'Failed to process pointcloud: {e}')
            points, valid_mask = None, None

        position = None
        if points is not None:
            position = self._calculate_centroid(points, track_result.mask, valid_mask, track_result.bbox)

        feedback.target_lost = False
        feedback.target_track_id = track_result.track_id
        feedback.target_position = PointStamped()
        feedback.is_transformation_successful = False

        if position is not None:
            feedback.target_position.header = rgb_msg.header
            feedback.target_position.point = position
            target_frame = params["target_frame"]
            if target_frame and target_frame.lower() != 'none' and target_frame != rgb_msg.header.frame_id:
                try:
                    transform = self.tf_buffer.lookup_transform(
                        target_frame,
                        rgb_msg.header.frame_id,
                        rclpy.time.Time(),
                        timeout=Duration(seconds=0.2)
                    )
                    transformed = do_transform_point(feedback.target_position, transform)
                    feedback.target_position = transformed
                    feedback.is_transformation_successful = True
                except Exception as ex:
                    self.get_logger().warn(f"TF transform to '{target_frame}' failed ({ex}); keeping camera frame")
            else:
                feedback.is_transformation_successful = True

            if (
                not feedback.target_lost
                and feedback.is_transformation_successful
                and self.target_point_pub is not None
            ):
                self.target_point_pub.publish(feedback.target_position)
        else:
            feedback.target_position.header = rgb_msg.header

        if params["return_rgb_img"] or params["debug_mode"]:
            if params["debug_mode"]:
                debug_img = self._draw_debug_info(
                    rgb_img,
                    self.tracker.last_results,
                    track_result,
                    self.tracker.target_track_id
                )
                feedback.rgb_img = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
                feedback.rgb_img.header = rgb_msg.header
            else:
                feedback.rgb_img = rgb_msg

        if params["return_depth_img"] and points is not None:
            depth_vis = (points[:, :, 2] * 1000).astype(np.uint16)
            feedback.depth_img = self.bridge.cv2_to_imgmsg(depth_vis, encoding='16UC1')
            feedback.depth_img.header = rgb_msg.header

        if params["return_segment"] and track_result.mask is not None:
            mask_img = (track_result.mask * 255).astype(np.uint8)
            feedback.segment_img = self.bridge.cv2_to_imgmsg(mask_img, encoding='mono8')
            feedback.segment_img.header = rgb_msg.header

        goal_handle.publish_feedback(feedback)

    def _handle_lost_frame(
        self,
        last_seen_time: float,
        rgb_img,
        rgb_msg,
        feedback,
        goal_handle,
        params,
        result,
    ) -> bool:
        time_since_seen = time.time() - last_seen_time
        feedback.target_lost = True
        feedback.target_track_id = self.tracker.original_track_id if self.tracker.original_track_id else -1
        feedback.target_position = PointStamped()
        feedback.target_position.header = rgb_msg.header
        feedback.is_transformation_successful = False

        if params["return_rgb_img"] or params["debug_mode"]:
            if params["debug_mode"]:
                debug_img = self._draw_debug_info(
                    rgb_img,
                    self.tracker.last_results,
                    None,
                    self.tracker.target_track_id
                )
                feedback.rgb_img = self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
                feedback.rgb_img.header = rgb_msg.header
            else:
                feedback.rgb_img = rgb_msg

        goal_handle.publish_feedback(feedback)

        if time_since_seen > self.lost_timeout:
            self.get_logger().warn(f'Target lost for {time_since_seen:.1f}s, aborting')
            goal_handle.abort()
            result.status = 1
            result.message = f'Target lost for {time_since_seen:.1f} seconds'
            return True
        return False

    def _cleanup_tracking(self):
        """Clean up tracking state."""
        self.tracking_active = False
        self.goal_handle = None

        if self.target_point_pub is not None:
            self.destroy_publisher(self.target_point_pub)
            self.target_point_pub = None

        with self.lock_tracker:
            if self.tracker is not None:
                self.tracker.reset()

        self.get_logger().info('Tracking cleaned up')


def main(args=None):
    rclpy.init(args=args)
    
    node = PersonTrackNode()
    
    # Use MultiThreadedExecutor for concurrent callback processing
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
