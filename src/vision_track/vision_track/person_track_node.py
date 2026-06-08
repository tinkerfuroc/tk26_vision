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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.duration import Duration

import numpy as np
import cv2
import threading
import time
import json
import base64
from pathlib import Path
import os

# ROS2 messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped, Point
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

# Action definition
from tinker_vision_msgs_26.action import TrackPerson

# Service definition (active re-ID / re-seed)
from tinker_vision_msgs_26.srv import ReseedTarget

# Message filters for synchronization
from message_filters import Subscriber, ApproximateTimeSynchronizer

# CV Bridge
from cv_bridge import CvBridge

# Import YOLOTracker
from vision_track.track_yolo import YOLOTracker, TrackerState, TrackingResult

# Shared logger
from vision_util.vision_logging import VisionLogger
from vision_util.weights_cache import resolve_weights

from vision_track.core.centroid import reduce_centroid
from vision_track.core.color_decode import decode_color_msg
from vision_track.core.depth_roi import roi_window
from vision_track.core.frame_diag import compute_frame_diag
from vision_track.core.reacq_state import reacq_state
from vision_track.core.debug_state import build_debug_state


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

        # Track-state cache for lost/reclaim logging (last successful frame)
        self._last_tracked_rgb = None
        self._last_tracked_detection = None
        self._was_lost = False
        # Active re-ID hold: throttle the "awaiting help" log to once per lost
        # episode (reset on re-track and on cleanup).
        self._active_help_logged = False

        # Phase 2: EMA smoother on the published 3D point; reset on loss so a
        # re-acquired target doesn't lerp from a stale point.
        from vision_track.core.centroid_smooth import PointEMA
        self._point_ema = PointEMA(alpha=self.centroid_ema_alpha)

        self._vision_logger = VisionLogger(
            self, self.vision_logging_enabled, self.vision_log_folder,
        )
        self.target_point_pub = None

        # track_web dashboard telemetry publishers (param-gated; see
        # _publish_debug_outputs). Created unconditionally — they cost nothing
        # until something publishes / subscribes, and the per-frame work is
        # guarded by the (default-False) debug_* flags.
        self.debug_state_pub = self.create_publisher(String, '~/debug_state', 10)
        gallery_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.debug_gallery_pub = self.create_publisher(String, '~/debug_gallery', gallery_qos)
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 1)
        self._last_gallery_version = -1

        # Idle telemetry: between goals the tracking loop isn't running, so a
        # light timer keeps the dashboard alive (camera preview + 'idle' state)
        # when the debug params are on. The tick reads the frame cache WITHOUT
        # consuming it (never touches last_processed_seq) so it cannot race
        # the tracking loop.
        self._idle_last_seq = -1
        if self.debug_state_enabled or self.debug_image_enabled:
            self.idle_debug_timer = self.create_timer(0.1, self._idle_debug_tick)

        self.tf_buffer = Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Thread locks
        self.lock_msg = threading.Lock()
        self.lock_info = threading.Lock()
        self.lock_tracker = threading.Lock()
        # Guards tracking_active + goal_handle so goal_callback can atomically
        # test-and-set without two concurrent ACCEPTs under MultiThreadedExecutor.
        self.lock_lifecycle = threading.Lock()
        
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
        # LOW conf fed to model.track so ByteTrack's two-stage (high/low)
        # association recovery actually runs — kept separate from
        # confidence_threshold, which still gates detect() / downstream consumers.
        self.declare_parameter('yolo_track_conf', 0.15)
        self.declare_parameter('enable_reid', True)
        self.declare_parameter('max_frames_lost', 600)  # ~20 seconds at 30fps
        self.declare_parameter('inference_size', 736)  # imgsz for YOLO; lower for speed
        self.declare_parameter('reid_verification_interval', 5)  # periodic on-track ReID sanity check
        # Phase 2: bound recovery so the tracker eventually declares hard-lost.
        # Replaces the effectively-infinite allow_indefinite_recovery coast.
        self.declare_parameter('max_recovery_frames', 45)
        # Spec B: consecutive frames lost before the published reacquisition_state
        # escalates to NEEDS_HELP, so a BT can debounce active (call-out) re-ID.
        self.declare_parameter('active_help_after_frames', 45)
        # While reacquisition_state==NEEDS_HELP, keep the tracker+gallery alive
        # (coast, no abort/reset) so the operator can wave and be reseeded.
        # <=0 (default) holds INDEFINITELY — only a successful reseed or a cancel
        # ends the hold; a positive value bounds it to that many seconds. Disable
        # active help entirely with active_help_after_frames<=0 (legacy abort on
        # hard-lost).
        self.declare_parameter('active_help_timeout_sec', 0.0)
        # track_web dashboard telemetry (all default OFF; byte-identical to
        # legacy behavior with defaults).
        self.declare_parameter('debug_state_enabled', False)
        self.declare_parameter('gallery_keep_crops', False)
        self.declare_parameter('debug_image_enabled', False)
        self.declare_parameter('provisional_high_bar', 0.72)
        self.declare_parameter('provisional_distinct_margin', 0.10)
        # Phase 2: reject candidates whose median depth jumps this much (m)
        # toward the camera vs the operator's last depth — a geometric crosser.
        self.declare_parameter('crosser_depth_jump_m', 0.6)
        # Phase 2: geometry robustness — torso-band sampling + EMA on the point.
        self.declare_parameter('centroid_ema_alpha', 0.5)
        self.declare_parameter('torso_band_enabled', True)
        self.declare_parameter('torso_band_lo', 0.15)
        self.declare_parameter('torso_band_hi', 0.55)

        # ReID mode: 'custom' uses our OSNet-based ReID, 'native' uses YOLO's BoT-SORT ReID
        self.declare_parameter('reid_mode', 'custom')  # 'custom' or 'native'

        # ReID deep backbone (torchreid OSNet). Default is imagenet-init
        # osnet_ain_x1_0; 'osnet_x0_25' is the lighter alt. To upgrade to a
        # Market/MSMT ReID-trained checkpoint, point reid_weights_path at it
        # (overrides imagenet init — config change only).
        self.declare_parameter('reid_backbone', 'osnet_ain_x1_0')
        self.declare_parameter('reid_weights_path', '')
        # Half-precision ReID forward (CUDA only; silent no-op on CPU). Default
        # True for throughput in multi-person re-ID scenes — output stays
        # float32 + L2-normalized so identity gating is unaffected.
        self.declare_parameter('reid_fp16', True)
        # Multi-view reacquisition gallery (Phase 3). enabled is the kill-switch
        # (False restores exact legacy single-anchor scoring); size = K diverse
        # views kept; novelty_max = max cosine to existing views to admit a new
        # one; score_mode = 'max' | 'top2_mean' (precision fallback).
        self.declare_parameter('reid_gallery_enabled', True)
        self.declare_parameter('reid_gallery_size', 6)
        self.declare_parameter('reid_gallery_novelty_max', 0.85)
        self.declare_parameter('reid_gallery_score_mode', 'max')

        # Orbbec camera topics
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        
        # Tracking parameters
        self.declare_parameter('tracking_rate', 15.0)  # Hz
        self.declare_parameter('lost_timeout', 300.0)  # seconds before declaring failure
        self.declare_parameter('target_point_topic', '/target_points')  # default PointStamped pub topic

        # Vision logging (default-on). Tracker logs only on lost/reclaim
        # transitions — no per-frame artifacts during steady-state tracking.
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')

        # Perf/quality instrumentation (default-off; zero overhead in production).
        self.declare_parameter('perf_logging_enabled', False)

        self.get_logger().info('Parameters declared')

    def _load_parameters(self):
        """Load parameters."""
        self.model_path = self.get_parameter('model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.yolo_track_conf = self.get_parameter('yolo_track_conf').value
        self.enable_reid = self.get_parameter('enable_reid').value
        self.max_frames_lost = self.get_parameter('max_frames_lost').value
        self.inference_size = self.get_parameter('inference_size').value
        self.reid_verification_interval = self.get_parameter('reid_verification_interval').value
        self.max_recovery_frames = self.get_parameter('max_recovery_frames').value
        self.active_help_after_frames = int(self.get_parameter('active_help_after_frames').value)
        self.active_help_timeout_sec = float(self.get_parameter('active_help_timeout_sec').value)
        self.debug_state_enabled = bool(self.get_parameter('debug_state_enabled').value)
        self.gallery_keep_crops = bool(self.get_parameter('gallery_keep_crops').value)
        self.debug_image_enabled = bool(self.get_parameter('debug_image_enabled').value)
        self.provisional_high_bar = self.get_parameter('provisional_high_bar').value
        self.provisional_distinct_margin = self.get_parameter('provisional_distinct_margin').value
        self.crosser_depth_jump_m = self.get_parameter('crosser_depth_jump_m').value
        self.centroid_ema_alpha = self.get_parameter('centroid_ema_alpha').value
        self.torso_band_enabled = self.get_parameter('torso_band_enabled').value
        self.torso_band_lo = self.get_parameter('torso_band_lo').value
        self.torso_band_hi = self.get_parameter('torso_band_hi').value
        self.reid_mode = self.get_parameter('reid_mode').value
        self.reid_backbone = self.get_parameter('reid_backbone').value
        self.reid_weights_path = self.get_parameter('reid_weights_path').value
        self.reid_fp16 = self.get_parameter('reid_fp16').value
        self.reid_gallery_enabled = self.get_parameter('reid_gallery_enabled').value
        self.reid_gallery_size = self.get_parameter('reid_gallery_size').value
        self.reid_gallery_novelty_max = self.get_parameter('reid_gallery_novelty_max').value
        self.reid_gallery_score_mode = self.get_parameter('reid_gallery_score_mode').value

        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        
        self.tracking_rate = self.get_parameter('tracking_rate').value
        self.lost_timeout = self.get_parameter('lost_timeout').value
        self.default_target_point_topic = self.get_parameter('target_point_topic').value

        self.vision_logging_enabled = self.get_parameter('vision_logging_enabled').value
        self.vision_log_folder = self.get_parameter('vision_log_folder').value

        self.perf_logging_enabled = self.get_parameter('perf_logging_enabled').value
        
        self.get_logger().info(f'Model path: {self.model_path}')
        self.get_logger().info(f'Enable ReID: {self.enable_reid}')
        self.get_logger().info(f'ReID mode: {self.reid_mode}')
        self.get_logger().info(f'Inference size (imgsz): {self.inference_size}')
        self.get_logger().info(f'ReID verification interval: {self.reid_verification_interval}')
        self.get_logger().info(f'Max recovery frames: {self.max_recovery_frames}')
        self.get_logger().info(f'Tracking rate: {self.tracking_rate} Hz')

    def _init_tracker(self):
        """Initialize the YOLO tracker based on reid_mode parameter."""
        self.get_logger().info('Initializing YOLO Tracker...')
        
        try:
            model_file = resolve_weights(self.model_path)
            # Allow loss duration to be governed by time, not fixed frames.
            # Use whichever is larger: explicit max_frames_lost or rate * lost_timeout.
            # Bounded by max_recovery_frames; max_frames_lost remains the
            # ByteTrack buffer ceiling. The lock FSM owns hard-lost timing.
            max_frames_allowed = max(int(self.max_frames_lost), int(self.max_recovery_frames))
            
            if self.reid_mode == 'native':
                raise NotImplementedError(
                    "reid_mode='native' is not implemented in tk26 — "
                    "track_yolo_native.YOLOTrackerNative does not exist. "
                    "Use reid_mode='custom' (the default)."
                )
            else:
                # Use custom OSNet-based ReID (default)
                self.tracker = YOLOTracker(
                    model_path=str(model_file),
                    confidence_threshold=self.confidence_threshold,
                    enable_reid=self.enable_reid,
                    inference_size=self.inference_size,
                    reid_verification_interval=int(self.reid_verification_interval),
                    reid_backbone=self.reid_backbone,
                    reid_weights_path=self.reid_weights_path,
                    reid_fp16=self.reid_fp16,
                    reid_gallery_enabled=self.reid_gallery_enabled,
                    reid_gallery_size=int(self.reid_gallery_size),
                    reid_gallery_novelty_max=float(self.reid_gallery_novelty_max),
                    reid_gallery_score_mode=self.reid_gallery_score_mode,
                    keep_gallery_thumbs=self.gallery_keep_crops,
                    yolo_track_conf=self.yolo_track_conf,
                )
                self.tracker.max_frames_lost = max_frames_allowed
                # Communicate the real loop cadence so loss/buffer timing is
                # wall-clock-correct (ByteTrack frame_rate is wired through a
                # project bytetrack.yaml in Phase 1; here we record it on the
                # tracker for max_frames_lost derivation).
                self.tracker.frame_rate = float(self.tracking_rate)
                from vision_track.core.lock_state_machine import LockStateMachine
                self.tracker.max_recovery_frames = int(self.max_recovery_frames)
                self.tracker.provisional_high_bar = float(self.provisional_high_bar)
                self.tracker.provisional_distinct_margin = float(self.provisional_distinct_margin)
                self.tracker.lock_state_machine = LockStateMachine(
                    high_bar=self.tracker.provisional_high_bar,
                    distinct_margin=self.tracker.provisional_distinct_margin,
                    commit_frames=self.tracker.reid_confirmation_frames,
                    max_recovery_frames=self.tracker.max_recovery_frames,
                )
                # Phase 2: crosser-rejection gate threshold (m). The tracker
                # reads operator_last_depth_m + candidate_depths_m (both plumbed
                # from the node) and this jump to reject toward-camera crossers.
                self.tracker.crosser_depth_jump_m = float(self.crosser_depth_jump_m)
                self.get_logger().info(f'YOLO Tracker (CUSTOM ReID) initialized with model: {model_file}')
            
            self.get_logger().info(
                f"Max frames lost set to {self.tracker.max_frames_lost} "
                f"(tracking_rate={self.tracking_rate} Hz, lost_timeout={self.lost_timeout}s, "
                f"param_max_frames_lost={self.max_frames_lost})"
            )
            
        except Exception as e:
            self.get_logger().error(f'Failed to initialize tracker: {e}')
            raise

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

        # Active re-ID: re-lock the tracker on an externally-confirmed bbox
        # (e.g. raise-hand operator) without wiping the multi-view gallery.
        self.reseed_srv = self.create_service(
            ReseedTarget, '~/reseed_target', self._reseed_callback,
            callback_group=ReentrantCallbackGroup())
        self.get_logger().info('Service created: ~/reseed_target')

    def _reseed_callback(self, request, response):
        """Re-lock the tracker on request.bbox, preserving the gallery.

        Runs under lock_tracker (serialized with the tracking loop's
        tracker.update). Uses the latest cached color frame to match the bbox.
        """
        roi = request.bbox
        bbox = (int(roi.x_offset), int(roi.y_offset),
                int(roi.x_offset + roi.width), int(roi.y_offset + roi.height))
        self.get_logger().info(
            f'Reseed requested: bbox={bbox} frame_id={request.frame_id!r}')
        # _get_latest_data() returns (rgb_img, rgb_msg, depth_msg, intrinsic) on
        # success, or one of two falsy sentinels: None (no msg / no intrinsic /
        # decode fail) and False (frame-seq dedup, i.e. nothing new). A success
        # is a 4-tuple (always truthy), so `not data` covers both sentinels.
        data = self._get_latest_data()
        if not data:
            self.get_logger().warn('Reseed failed: no camera frame available')
            response.success = False
            response.target_track_id = -1
            response.message = 'no camera frame available'
            return response
        rgb_img = data[0]
        rgb_msg = data[1]
        if request.frame_id and request.frame_id != rgb_msg.header.frame_id:
            self.get_logger().warn(
                f'Reseed bbox frame_id {request.frame_id!r} != camera frame '
                f'{rgb_msg.header.frame_id!r}; matching against the camera frame anyway')
        # Mirror the live tracking loop: it feeds the tracker a BGR->RGB frame
        # (see _run_tracking_loop). reseed_target must get the SAME convention
        # or detection/ReID degrades. Do NOT pass the raw bgr8 buffer here.
        rgb_frame = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # A raising service callback propagates out of the MultiThreadedExecutor
        # and crashes the node; the YOLO/ReID work inside reseed_target can throw
        # on a bad frame / CUDA OOM, so guard it (mirrors _execute_callback).
        try:
            with self.lock_tracker:
                tid = self.tracker.reseed_target(rgb_frame, bbox, target_class='person')
        except Exception as exc:  # service must never crash the node
            self.get_logger().error(f'Reseed errored: {exc}')
            response.success = False
            response.target_track_id = -1
            response.message = f'reseed error: {exc}'
            return response
        response.success = tid >= 0
        response.target_track_id = int(tid)
        response.message = 'reseeded' if tid >= 0 else 'no detection matched bbox'
        self.get_logger().info(
            f'Reseed result: success={response.success} '
            f'track_id={response.target_track_id} ({response.message})')
        return response

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

    def _depth_image_to_points(self, depth_msg: Image, intrinsic: CameraInfo, bbox: tuple = None) -> tuple:
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

        # Only the target bbox is ever sampled by _calculate_centroid, so
        # restrict the unproject to a padded window around it. Pixels outside
        # the window stay zeroed and invalid.
        x0, y0, x1, y1 = roi_window(bbox, w=w, h=h, pad=16)
        valid_roi = np.zeros_like(valid_mask)
        valid_roi[y0:y1, x0:x1] = valid_mask[y0:y1, x0:x1]
        valid_mask = valid_roi

        # Cache meshgrid across calls at this resolution.
        cache = getattr(self, '_uv_cache', None)
        if cache is None or cache[0] != (h, w):
            u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
            self._uv_cache = ((h, w), u, v)
        _, u, v = self._uv_cache

        points = np.zeros((h, w, 3), dtype=np.float32)
        z_roi = depth[y0:y1, x0:x1]
        u_roi = u[y0:y1, x0:x1]
        v_roi = v[y0:y1, x0:x1]
        points[y0:y1, x0:x1, 0] = (u_roi - cx) * z_roi / fx
        points[y0:y1, x0:x1, 1] = (v_roi - cy) * z_roi / fy
        points[y0:y1, x0:x1, 2] = z_roi

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

        # Phase 2: restrict to the chest band BEFORE the Phase-0 robust reduction
        # so swinging arms/legs/head don't pull the centroid. Layers on top of the
        # robust median-x/y + z-outlier reduction below — does not replace it.
        from vision_track.core.centroid_smooth import torso_band_mask
        if self.torso_band_enabled:
            yb1, yb2 = torso_band_mask((x1, y1, x2, y2),
                                       lo=self.torso_band_lo, hi=self.torso_band_hi)
            y1, y2 = yb1, yb2
            if y2 <= y1:
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
        
        # Robust reduction shared with ptbench geometry: median lateral x/y +
        # z-outlier-rejected median z (vision_track.core.centroid.reduce_centroid).
        cx_m, cy_m, cz_m = reduce_centroid(obj_pts)

        # Create Point message (Orbbec frame convention)
        point = Point()
        point.x = cx_m
        point.y = cy_m
        point.z = cz_m

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

        # Atomic test-and-set: reserve the tracking slot here so a second
        # concurrent goal_callback sees tracking_active=True and rejects.
        # _execute_callback / _cleanup_tracking release the slot under the
        # same lock.
        with self.lock_lifecycle:
            if self.tracking_active:
                self.get_logger().warn('Already tracking, rejecting new goal')
                return GoalResponse.REJECT
            self.tracking_active = True

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
        # tracking_active was already set True under lock_lifecycle in
        # _goal_callback; we just record the goal handle here.
        with self.lock_lifecycle:
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

        # Warm CUDA on THIS executor thread before the lock loop. The __init__
        # warmup ran on the main thread; the first cuDNN call on an action-worker
        # thread pays a ~0.5s one-time init that would otherwise land on the first
        # tracked frame — a freeze right at lock that, under load, drops the
        # just-acquired target into a false loss. Paid here (during init search,
        # before any lock) it is invisible. ByteTrack state is reset after.
        try:
            with self.lock_tracker:
                tk = self.tracker
                dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
                tk.track(dummy, persist=True)            # warm YOLO on this thread
                tk._reset_bytetrack_state()
                ext = getattr(tk, 'appearance_extractor', None)
                if ext is not None:                      # warm OSNet ReID too
                    ext.extract_features(dummy, (0, 0, 100, 100), None)
                    batch = getattr(ext, 'extract_features_batch', None)
                    if batch is not None:
                        batch(dummy, [(0, 0, 100, 100)], [None], [0])
        except Exception as warm_exc:  # never block tracking on a warmup hiccup
            self.get_logger().debug(f'action-thread warmup skipped: {warm_exc}')

        while rclpy.ok():
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

            loop_start = time.time()
            t_track0 = time.perf_counter()
            with self.lock_tracker:
                if not initialized:
                    initialized = self._try_initialize(rgb_frame, init_start_time, goal_handle, result)
                    if not initialized:
                        # Dashboard UX: without this the track_web stale banner
                        # ("NO DATA") shows for the whole init search right
                        # after the operator presses Start.
                        self._publish_phase_debug_state('initializing')
                        self._publish_raw_debug_image(rgb_img)
                        time.sleep(0.1)
                        continue
                    last_seen_time = time.time()
                # Phase 2: median depth per visible person bbox (from the last
                # frame's results against the current depth) so the pipeline's
                # crosser gate can reject toward-camera candidates this frame.
                if self.tracker.last_results and depth_msg is not None:
                    self._refresh_candidate_depths(depth_msg)
                track_result = self.tracker.update(rgb_frame)
            t_track = time.perf_counter() - t_track0

            t_post0 = time.perf_counter()
            if track_result is not None:
                last_seen_time = time.time()
                self._handle_tracked_frame(
                    track_result, rgb_img, rgb_msg, depth_msg, intrinsic, feedback, goal_handle, params
                )
            else:
                if self._handle_lost_frame(last_seen_time, rgb_img, rgb_msg, feedback, goal_handle, params, result):
                    return result
            self._publish_debug_outputs(rgb_img, track_result, feedback, last_seen_time)
            t_post = time.perf_counter() - t_post0

            if self.perf_logging_enabled:
                tk = self.tracker
                self.get_logger().info(
                    f"[perf] track={t_track*1000:.1f}ms "
                    f"(yolo={getattr(tk, '_t_yolo_ms', 0.0):.1f} "
                    f"pipe={getattr(tk, '_t_pipeline_ms', 0.0):.1f}) "
                    f"post={t_post*1000:.1f}ms "
                    f"loop={(time.time()-loop_start)*1000:.1f}ms"
                )

            # No artificial Hz cap: frame-seq dedup in _get_latest_data gates the
            # loop to the camera rate. A 1 ms yield keeps the GIL fair.
            time.sleep(0.001)

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

        # Normalize the wire format (Orbbec = rgb8, others = bgr8) to BGR once,
        # here — every downstream consumer (tracker feed via BGR2RGB, debug
        # draw/publish, vision logger) assumes BGR. Zero-copy for bgr8 (the
        # returned view is read-only; all writers copy first).
        rgb_img, err = decode_color_msg(rgb_msg)
        if rgb_img is None:
            self.get_logger().warn(f'color frame dropped: {err}',
                                   throttle_duration_sec=5.0)
            return None

        return rgb_img, rgb_msg, depth_msg, intrinsic

    def _refresh_candidate_depths(self, depth_msg):
        """Median-depth per visible person bbox, for the crosser gate.

        Reads the previous frame's person results and writes the
        track_id -> median-depth (m) map onto the tracker so the pure depth gate
        in the pipeline can reject toward-camera crossers. The node is the sole
        owner of the depth image; the tracker never touches ROS.
        """
        from vision_track.core.depth_gate import roi_median_depth
        h, w = depth_msg.height, depth_msg.width
        depth = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(h, w)
        depths = {}
        for r in self.tracker.last_results:
            if r.class_id != 0 or r.track_id < 0:
                continue
            m = roi_median_depth(depth, r.bbox, self.min_depth, self.max_depth)
            if m is not None:
                depths[r.track_id] = m
        self.tracker.candidate_depths_m = depths

    def _try_initialize(self, rgb_frame, init_start_time, goal_handle, result) -> bool:
        success = self.tracker.initialize_tracking(rgb_frame, target_class='person')
        if success:
            # Phase 2: arm the lock FSM on the freshly committed id. Without
            # this the FSM stays in 'lost' and step() short-circuits, so no
            # recovery decision (provisional/commit/hard-lost) is ever made.
            if (
                getattr(self.tracker, 'lock_state_machine', None) is not None
                and self.tracker.original_track_id is not None
            ):
                self.tracker.lock_state_machine.start(self.tracker.original_track_id)
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
            points, valid_mask = self._depth_image_to_points(depth_msg, intrinsic, bbox=track_result.bbox)
        except Exception as e:
            self.get_logger().warn(f'Failed to process pointcloud: {e}')
            points, valid_mask = None, None

        position = None
        if points is not None:
            position = self._calculate_centroid(points, track_result.mask, valid_mask, track_result.bbox)

        # Phase 2: EMA-smooth the published 3D point. First sample (or first after
        # a loss reset) passes through; later frames blend. Applied before the
        # depth plumb / feedback so consumers and the crosser gate see the
        # smoothed point.
        if position is not None:
            sx, sy, sz = self._point_ema.update((position.x, position.y, position.z))
            position.x, position.y, position.z = float(sx), float(sy), float(sz)

        # Phase 2: plumb the operator's last known depth (m) into the tracker so
        # the crosser depth gate can reject toward-camera candidates next frame.
        # z is the optical-frame forward axis (depth). Only the node owns depth;
        # use the smoothed z so the gate sees the same point consumers do.
        if position is not None:
            self.tracker.operator_last_depth_m = float(position.z)

        if self.perf_logging_enabled and points is not None:
            diag = compute_frame_diag(points, track_result.mask, valid_mask, track_result.bbox)
            self.get_logger().info(
                f"[diag] mask_px={diag['mask_pixel_count']} "
                f"valid_px={diag['valid_pixel_count']} used_mask={diag['used_mask']} "
                f"z_iqr={diag['depth_z_iqr']:.3f} "
                f"mask_c={diag['mask_centroid']} bbox_c={diag['bbox_centroid']} "
                f"no_centroid={diag['no_centroid']}"
            )

        # The FSM is the publish/target_lost authority. A tracked frame here
        # means the committed id was matched (Stage 1) or a recovery candidate
        # was surfaced (Stage 2). The recovery path (reidentify_target) already
        # stepped the FSM authoritatively and set last_frame_recovery=True; in
        # that case the node MUST defer to last_lock_decision and not re-step
        # present=True — doing so on a partial-confirm recovery frame would flip
        # target_lost=False below the high bar and defeat the asymmetric
        # hysteresis. Only a genuine Stage-1 present-by-id hold (not a recovery
        # frame) re-steps present=True here. The pipeline remains the
        # identity-swap authority — this only drives the publish/target_lost gate.
        fsm = getattr(self.tracker, 'lock_state_machine', None)
        decision = getattr(self.tracker, 'last_lock_decision', None)
        recovery_frame = bool(getattr(self.tracker, 'last_frame_recovery', False))
        target_present = (
            not recovery_frame
            and self.tracker.target_track_id is not None
            and track_result.track_id == self.tracker.original_track_id
            and getattr(self.tracker, 'frames_lost', 0) == 0
        )
        if fsm is not None and target_present:
            decision = fsm.step(
                sim_score=1.0, present=True, frames_since_loss=0,
                num_candidates=1, distinct_margin=float('inf'), depth_consistent=True,
            )
            self.tracker.last_lock_decision = decision
        feedback.target_lost = bool(decision.target_lost) if decision is not None else False
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

        # Derive from the real status: during a provisional-recovery coast
        # track_result is not None but feedback.target_lost is True, so a hardcoded
        # REACQ_TRACKING would contradict target_lost. Report PASSIVE/NEEDS_HELP
        # honestly in that window and REACQ_TRACKING only when fully held.
        feedback.reacquisition_state = reacq_state(
            tracked=not feedback.target_lost,
            frames_lost=int(getattr(self.tracker, 'frames_lost', 0)),
            help_after=self.active_help_after_frames)
        goal_handle.publish_feedback(feedback)

        # Cache the latest good frame for the lost-transition dump, and emit
        # a 'reclaimed' artifact if we're just coming back from a lost state.
        self._last_tracked_rgb = rgb_img.copy()
        self._last_tracked_detection = {
            'bbox': list(track_result.bbox) if track_result.bbox is not None else None,
            'mask': track_result.mask,
            'cls_name': 'person',
            'conf': float(getattr(track_result, 'confidence', 0.0) or 0.0),
            'centroid': [
                float(position.x), float(position.y), float(position.z)
            ] if position is not None else None,
            'track_id': int(track_result.track_id),
        }
        if self._was_lost and self._vision_logger.enabled:
            self._vision_logger.write(
                rgb_img, [self._last_tracked_detection],
                request_ctx={'target_frame': params.get('target_frame')},
                branch='person_track',
                extras={'event': 'reclaimed'},
            )
        self._was_lost = False
        # Re-tracking ends the lost episode: re-arm the active-help log throttle
        # so the next loss logs the hold entry again.
        self._active_help_logged = False

    def _is_awaiting_help(self, frames_lost, time_since_seen):
        """True while coasting (holding) for active re-ID help — wave to resume.

        Escalates after ``active_help_after_frames`` consecutive lost frames,
        then holds INDEFINITELY when ``active_help_timeout_sec <= 0`` (default —
        only a successful reseed or a goal cancel ends it), or up to that many
        seconds otherwise. ``active_help_after_frames <= 0`` disables active help
        (legacy abort on hard-lost).
        """
        if (self.active_help_after_frames <= 0
                or frames_lost < self.active_help_after_frames):
            return False
        if self.active_help_timeout_sec <= 0.0:
            return True
        return time_since_seen <= self.active_help_timeout_sec

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

        # First tick after a TRACKING → LOST transition: dump the last-good
        # frame and the current (failed) frame. Subsequent lost ticks don't
        # log, so a long occlusion produces exactly two artifacts.
        if (not self._was_lost) and self._vision_logger.enabled:
            if self._last_tracked_rgb is not None and self._last_tracked_detection is not None:
                self._vision_logger.write(
                    self._last_tracked_rgb, [self._last_tracked_detection],
                    request_ctx={'target_frame': params.get('target_frame')},
                    branch='person_track',
                    extras={'event': 'lost', 'time_since_seen_s': float(time_since_seen)},
                )
            self._vision_logger.write(
                rgb_img, None,
                request_ctx={'target_frame': params.get('target_frame')},
                branch='person_track',
                extras={'event': 'lost_current', 'time_since_seen_s': float(time_since_seen)},
            )
        # Phase 2: reset the point smoother on the TRACKING → LOST transition so a
        # re-acquired target doesn't lerp from a stale pre-loss point.
        if not self._was_lost:
            self._point_ema.reset()
        self._was_lost = True

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

        feedback.reacquisition_state = reacq_state(
            tracked=False, frames_lost=int(getattr(self.tracker, 'frames_lost', 0)),
            help_after=self.active_help_after_frames)
        goal_handle.publish_feedback(feedback)

        # Republish a lost-sentinel so /target_points consumers see the loss
        # instead of a stale last-good point. NaN coords flag "no target".
        if self.target_point_pub is not None:
            sentinel = PointStamped()
            sentinel.header = rgb_msg.header
            sentinel.point.x = float('nan')
            sentinel.point.y = float('nan')
            sentinel.point.z = float('nan')
            self.target_point_pub.publish(sentinel)

        decision = getattr(self.tracker, 'last_lock_decision', None)
        hard_lost = decision is not None and decision.state == 'lost'
        # Active re-ID hold: once escalated to NEEDS_ACTIVE_HELP, keep the tracker
        # + multi-view gallery alive (coast, no abort/reset) so the BT can call
        # ~/reseed_target and re-lock the self-identified operator preserving
        # identity. Bounded by active_help_timeout_sec; lost_timeout stays the
        # absolute ceiling. Set active_help_timeout_sec<=0 to disable (legacy abort).
        frames_lost = int(getattr(self.tracker, 'frames_lost', 0))
        if self._is_awaiting_help(frames_lost, time_since_seen):
            if not self._active_help_logged:
                bound = ('indefinitely' if self.active_help_timeout_sec <= 0.0
                         else f'up to {self.active_help_timeout_sec:.0f}s')
                self.get_logger().warn(
                    f'Target lost {frames_lost}f; awaiting active re-ID help '
                    f'(holding {bound} for ~/reseed_target — wave to resume)')
                self._active_help_logged = True
            return False
        if hard_lost or time_since_seen > self.lost_timeout:
            reason = 'hard-lost (recovery cap)' if hard_lost else f'lost for {time_since_seen:.1f}s'
            self.get_logger().warn(f'Target {reason}, aborting')
            goal_handle.abort()
            result.status = 1
            result.message = f'Target {reason}'
            return True
        return False

    def _publish_phase_debug_state(self, phase: str):
        """Out-of-tracking telemetry tick: 'initializing' during goal init,
        'idle' between goals.

        Published while the tracking loop isn't producing live telemetry so the
        dashboard shows the current phase instead of the NO-DATA stale banner.
        Param-gated; must never raise into the loop.
        """
        if not self.debug_state_enabled:
            return
        try:
            state = build_debug_state(
                self.tracker, ts=time.time(),
                target_lost=True,
                reacquisition_state=1,  # REACQ_PASSIVE: searching, not locked
                time_since_seen=0.0, awaiting_help=False,
                active_help_after_frames=self.active_help_after_frames,
                active_help_timeout_sec=self.active_help_timeout_sec)
            state["fsm_state"] = phase
            state["candidates"] = []      # may be stale from a previous goal;
            state["best_sim"] = None      # no live click targets outside the
            state["second_sim"] = None    # tracking loop
            self.debug_state_pub.publish(String(data=json.dumps(state)))
        except Exception as exc:
            self.get_logger().warn(f'{phase} debug state failed: {exc}',
                                   throttle_duration_sec=5.0)

    def _publish_raw_debug_image(self, rgb_img):
        """Un-annotated BGR camera frame for the dashboard outside TRACKING."""
        if not (self.debug_image_enabled
                and self.debug_image_pub.get_subscription_count() > 0):
            return
        try:
            self.debug_image_pub.publish(
                self.bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8'))
        except Exception as exc:
            self.get_logger().warn(f'raw debug image failed: {exc}',
                                   throttle_duration_sec=5.0)

    def _idle_debug_tick(self):
        """Dashboard telemetry while NO goal is active (loop not running)."""
        if self.tracking_active:
            return  # the tracking loop owns telemetry during a goal
        self._publish_phase_debug_state('idle')
        if not (self.debug_image_enabled
                and self.debug_image_pub.get_subscription_count() > 0):
            return
        with self.lock_msg:
            pair = self.recent_sync_msg
            seq = self.frame_seq
        if pair is None or seq == self._idle_last_seq:
            return
        rgb_img, err = decode_color_msg(pair[0])
        if rgb_img is None:
            self.get_logger().warn(f'idle frame dropped: {err}',
                                   throttle_duration_sec=5.0)
            return
        self._idle_last_seq = seq
        self._publish_raw_debug_image(rgb_img)

    def _publish_debug_outputs(self, rgb_img, track_result, feedback, last_seen_time):
        """Param-gated dashboard telemetry; must never raise into the loop."""
        try:
            if self.debug_state_enabled:
                tss = time.time() - last_seen_time
                frames_lost = int(getattr(self.tracker, 'frames_lost', 0))
                awaiting = (bool(feedback.target_lost)
                            and self._is_awaiting_help(frames_lost, tss))
                state = build_debug_state(
                    self.tracker, ts=time.time(),
                    target_lost=bool(feedback.target_lost),
                    reacquisition_state=int(feedback.reacquisition_state),
                    time_since_seen=tss, awaiting_help=awaiting,
                    active_help_after_frames=self.active_help_after_frames,
                    active_help_timeout_sec=self.active_help_timeout_sec)
                self.debug_state_pub.publish(String(data=json.dumps(state)))
                if self.gallery_keep_crops:
                    self._maybe_publish_gallery(state["gallery_version"])
            if self.debug_image_enabled and self.debug_image_pub.get_subscription_count() > 0:
                annotated = self._draw_debug_info(
                    rgb_img, self.tracker.last_results, track_result,
                    self.tracker.target_track_id)
                self.debug_image_pub.publish(
                    self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))
        except Exception as exc:  # telemetry must never kill tracking
            self.get_logger().warn(f'debug output failed: {exc}',
                                   throttle_duration_sec=5.0)

    def _maybe_publish_gallery(self, version: int):
        if version == self._last_gallery_version:
            return
        app = getattr(self.tracker, 'target_appearance', None)
        thumbs = list(getattr(getattr(app, 'gallery', None), 'thumbs', []) or []) if app else []
        encoded = []
        for t in thumbs:
            # Per-thumb fault tolerance: one bad crop degrades to a None slot
            # instead of raising — which would leave _last_gallery_version
            # stale and retry (and re-warn) every frame until the gallery
            # changes again.
            enc = None
            if t is not None:
                try:
                    ok, buf = cv2.imencode('.jpg', cv2.cvtColor(t, cv2.COLOR_RGB2BGR),
                                           [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        enc = base64.b64encode(buf).decode('ascii')
                except Exception:
                    enc = None
            encoded.append(enc)
        self.debug_gallery_pub.publish(String(data=json.dumps(
            {'version': version, 'thumbs': encoded})))
        self._last_gallery_version = version

    def _cleanup_tracking(self):
        """Clean up tracking state."""
        with self.lock_lifecycle:
            self.tracking_active = False
            self.goal_handle = None

        self._last_tracked_rgb = None
        self._last_tracked_detection = None
        self._was_lost = False
        self._active_help_logged = False
        self._point_ema.reset()

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
