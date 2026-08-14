import math
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.time import Time
import numpy as np
import cv2
import threading
import copy
import torch
import time
from typing import Tuple

# ROS2 messages
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
import geometry_msgs.msg
from tinker_vision_msgs_26.msg import Object, Objects
from tinker_vision_msgs_26.srv import ObjectDetection

# Computer vision
from ultralytics import YOLO
from cv_bridge import CvBridge

# Shared logger
from vision_util.vision_logging import VisionLogger
from vision_util.mask_utils import largest_connected_component_in_bbox
from vision_util.weights_cache import resolve_weights
from vision_util.camera_intake import (
    CameraIntake,
    IntakeConfig,
    StreamSpec,
    configure_camera_backend,
)
from vision_util.depth_reproject import (
    decode_depth_metres,
    depth_image_to_points,
    realsense_body_axes_points,
)
from vision_util.depth_source import FfsPreferredDepthSource
from vision_util.tf_lookup import TransformHelper


class _CompatibleCameraIntake(CameraIntake):
    """Mirror intake state into the legacy attributes used by subclasses."""

    def __init__(self, node, cfg, callback_group=None, *, bridge=None):
        self._compat_owner = node
        super().__init__(
            node, cfg, callback_group=callback_group, bridge=bridge
        )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        super()._camera_info_callback(msg)
        with self._compat_owner.lock_info:
            self._compat_owner.camera_intrinsic[self.cfg.camera] = msg

    def _store(self, *, color_msg, depth_msg) -> None:
        super()._store(color_msg=color_msg, depth_msg=depth_msg)
        bundle = self.latest()
        with self._compat_owner.lock_msg:
            self._compat_owner.recent_sync_msg[self.cfg.camera] = (
                color_msg, depth_msg
            )
            self._compat_owner.recent_publish_time[self.cfg.camera] = (
                bundle.recv_time if bundle is not None else None
            )

    def _store_provider_result(self, result):
        bundle = super()._store_provider_result(result)
        if bundle is None:
            return None
        with self._compat_owner.lock_msg:
            self._compat_owner.recent_sync_msg[self.cfg.camera] = (
                bundle.color_msg, bundle.depth_msg
            )
            self._compat_owner.recent_publish_time[self.cfg.camera] = (
                Time.from_msg(bundle.header.stamp)
            )
        info = self.camera_info()
        if info is not None:
            with self._compat_owner.lock_info:
                self._compat_owner.camera_intrinsic[self.cfg.camera] = info
        return bundle


class YOLOSegmentationNode(Node):
    """
    Simplified object detection node using YOLOv11-seg.

    Provides object detection and segmentation without tracking or
    advanced features.
    """

    def __init__(self, node_name='yolo_segmentation_node', parameter_overrides=None):
        super().__init__(node_name, parameter_overrides=parameter_overrides or [])

        # Declare parameters
        self._declare_parameters()

        # Load parameters
        self._load_parameters()

        self.bridge = CvBridge()

        # These dictionaries and locks are compatibility attributes consumed
        # by ObjectMatchServer and PlacingLocationServer. CameraIntake owns the
        # authoritative cache and mirrors each update here.
        self.lock_msg = threading.RLock()
        self.lock_info = threading.RLock()
        self.camera_intrinsic = {
            'realsense': None,
            'orbbec': None,
        }
        self.recent_sync_msg = {
            'realsense': None,
            'orbbec': None,
        }
        self.recent_publish_time = {
            'realsense': None,
            'orbbec': None,
        }
        self._camera_intakes = {}

        # FFS captures its own stereo pair, so native depth is supplied only
        # if fallback is selected. Thread-local storage keeps that per call.
        self._native_depth_context = threading.local()
        self._depth_source = FfsPreferredDepthSource(
            self,
            self._native_depth_provider,
            bridge=self.bridge,
        )

        # Keep the public buffer/listener aliases used by inherited servers.
        try:
            self.declare_parameter('camera_backend', 'service')
        except Exception:
            pass
        try:
            self.declare_parameter(
                'transform_provider_endpoint', '/head_camera_server'
            )
        except Exception:
            pass
        camera_backend = self.get_parameter('camera_backend').value
        transform_endpoint = self.get_parameter(
            'transform_provider_endpoint'
        ).value
        self._tf_helper = TransformHelper(
            self,
            cache_time_s=180.0,
            backend=camera_backend,
            provider_endpoint=transform_endpoint,
        )
        self.tf_buffer = self._tf_helper.buffer
        self.tf_listener = self._tf_helper._listener

        # Initialize components
        self._init_model()
        self._init_subscribers()
        self._init_publishers()
        self._init_service()

        # Per-call audit: filled by `_acquire_depth`; consumed by the sidecar
        # JSON writer downstream. 'native' until FFS path runs once.
        self._last_depth_source: str = 'native'

        self.get_logger().info('YOLO Segmentation Node initialized successfully')

    def _declare_parameters(self):
        """Declare all ROS2 parameters with defaults."""
        self.get_logger().info('Declaring parameters...')
        self.declare_parameter('camera_types', ['realsense', 'orbbec'])
        self.declare_parameter('model_path', 'yolo11m-seg.pt')
        self.declare_parameter('service_name', 'object_detection_yolo')
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
            'orbbec_depth_topic', '/camera/depth/image_raw')
        self.declare_parameter(
            'orbbec_camera_info_topic', '/camera/color/camera_info')        
        # Hz, 0 = no continuous publishing
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('confidence_threshold', 0.0)
        self.declare_parameter('visualization', False)
        self.declare_parameter('max_depth', 10.0)  # meters
        self.declare_parameter('min_depth', -10.0)   # meters
        self.declare_parameter('sync_wait_time_limit', 5) # how many 0.1 seconds to wait
        # Max age (seconds) of the most recent synced frame pair before we refuse
        # detection. 0 means "reject anything older than now", which fails even
        # a healthy 30 Hz camera because recent_publish_time is stamped at the
        # sync callback and is always >0 ms behind wall clock. Budget matches
        # 2x the ApproximateTimeSynchronizer slop of 0.1s.
        self.declare_parameter('img_sync_thres', 0.20)

        # Base folder for per-call artifacts; a run-timestamped subdir is
        # created lazily on first write. Resolves relative to CWD when not
        # absolute.
        self.declare_parameter('vision_log_folder', 'vision_log')

        # Per-call artifact dump (req_{ts}.json, orig_{ts}.jpg, overlay_{ts}.jpg).
        # Default-on so every production call leaves an audit trail; pass
        # `-p vision_logging_enabled:=false` to opt out.
        self.declare_parameter('vision_logging_enabled', True)
        
        # Sorting mode: 'none', 'closest', 'highest'
        self.declare_parameter('sort_mode', 'none')

        # Class names to drop before the target-class filter, regardless of prompt.
        # Default empty; specialist entry point overrides to ['person'] so a
        # custom-trained competition model never emits people.
        self.declare_parameter('excluded_classes', [''])

        # FoundationStereo depth fallback. When prefer_ffs=True (default) the
        # node queries the FFS depth service first and falls back to the native
        # camera depth only if the call fails or times out. Set prefer_ffs=False
        # to skip FFS and use native depth unconditionally.
        self.declare_parameter('prefer_ffs', True)
        self.declare_parameter('ffs_service', '/foundation_stereo/get_depth')
        self.declare_parameter('ffs_wait_for_service_s', 0.2)
        self.declare_parameter('ffs_call_timeout_s', 8.0)
        self.declare_parameter('ffs_align_to_color', True)
        self.declare_parameter('ffs_fallback_log_period_s', 30.0)

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

        self.vision_logging_enabled = self.get_parameter('vision_logging_enabled').value
        self.get_logger().info(f'Vision logging enabled: {self.vision_logging_enabled}')

        self._vision_logger = VisionLogger(
            self, self.vision_logging_enabled, self.vision_log_folder
        )

        self.sort_mode = self.get_parameter('sort_mode').value
        self.get_logger().info(f'Default sort mode: {self.sort_mode}')

        self.sync_wait_time_limit = self.get_parameter('sync_wait_time_limit').value
        self.get_logger().info(f'Sync wait time limit: {self.sync_wait_time_limit} x 0.1 times')

        self.img_sync_thres = self.get_parameter('img_sync_thres').value
        self.get_logger().info(f'Image sync threshold: {self.img_sync_thres} seconds')

        raw_excluded = self.get_parameter('excluded_classes').value or []
        self.excluded_classes = {c for c in raw_excluded if c}
        if self.excluded_classes:
            self.get_logger().info(f'Excluded classes: {sorted(self.excluded_classes)}')


    def _init_model(self):
        """Initialize YOLO model."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')
        try:
            model_file = resolve_weights(self.model_path)
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
        """Initialize one synchronized CameraIntake per configured camera."""
        for camera in ('realsense', 'orbbec'):
            if camera not in self.camera_types:
                continue
            cfg = configure_camera_backend(
                self,
                IntakeConfig(
                    camera=camera,
                    color=StreamSpec(
                        self.get_parameter(
                            f'{camera}_image_topic').value,
                        best_effort=True,
                        qos_depth=10,
                    ),
                    depth=StreamSpec(
                        self.get_parameter(
                            f'{camera}_depth_topic').value,
                        best_effort=True,
                        qos_depth=10,
                    ),
                    camera_info=StreamSpec(
                        self.get_parameter(
                            f'{camera}_camera_info_topic'
                        ).value,
                        best_effort=False,
                        qos_depth=10,
                    ),
                    sync_queue=10,
                    sync_slop_s=0.1,
                    age_source='stamp',
                ),
                default_endpoint=(
                    '/wrist_camera_server'
                    if camera == 'realsense'
                    else '/head_camera_server'
                ),
            )
            intake = _CompatibleCameraIntake(
                self,
                cfg,
                callback_group=MutuallyExclusiveCallbackGroup(),
                bridge=self.bridge,
            )
            self._camera_intakes[camera] = intake
            setattr(
                self,
                f'camera_info_sub_{camera}',
                (
                    intake._subscriptions[-1]
                    if intake._subscriptions
                    else None
                ),
            )
            if cfg.backend == 'service':
                self.get_logger().info(
                    f'Using {camera} camera provider at '
                    f'{cfg.provider_endpoint}'
                )
            else:
                self.get_logger().info(f'Subscribed to {camera} camera')

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
        service_name = self.get_parameter('service_name').value
        self.detection_srv = self.create_service(
            ObjectDetection,
            service_name,
            self._detection_service_callback,
            callback_group=MutuallyExclusiveCallbackGroup()
        )
        self.get_logger().info(f'Detection service created: {service_name}')

    def _camera_info_realsense_callback(self, msg: CameraInfo):
        """Compatibility callback forwarding to the RealSense intake."""
        self._camera_intakes['realsense']._camera_info_callback(msg)

    def _camera_info_orbbec_callback(self, msg: CameraInfo):
        """Compatibility callback forwarding to the Orbbec intake."""
        self._camera_intakes['orbbec']._camera_info_callback(msg)

    def _realsense_callback(self, rgb_msg: Image, depth_msg: Image):
        """Compatibility callback forwarding to the RealSense intake."""
        self._camera_intakes['realsense']._sync_callback(rgb_msg, depth_msg)

    def _orbbec_callback(self, rgb_msg: Image, depth_msg: Image):
        """Compatibility callback forwarding to the Orbbec intake."""
        self._camera_intakes['orbbec']._sync_callback(rgb_msg, depth_msg)

    def _orbbec_depth_to_array(self, depth_msg: Image, intrinsic: CameraInfo) -> tuple:
        """Reproject the Orbbec's registered depth Image to a points array.

        Depth is registered to color (depth_registration:=true), so its
        shape and frame always match the live color stream -- at whatever
        resolution the driver is launched with, never a fixed size.
        """
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        depth_m = decode_depth_metres(depth_raw)
        points, valid_mask = depth_image_to_points(
            depth_m,
            intrinsic.k,
            valid_band=(self.min_depth, self.max_depth),
            return_valid_mask=True,
        )

        return points, valid_mask

    def _native_depth_provider(self):
        depth_msg = getattr(self._native_depth_context, 'depth_msg', None)
        if depth_msg is None:
            raise RuntimeError('native RealSense depth is unavailable')
        return depth_msg

    def _acquire_depth(self, depth_msg: Image) -> Tuple[np.ndarray, str]:
        """Acquire a realsense depth image in meters, preferring the
        FoundationStereo service when ``prefer_ffs`` is enabled.

        Returns ``(depth_meters_float32, source)`` where ``source`` is
        ``'ffs'`` (the FFS service answered with status=0) or ``'native'``
        (FFS disabled, unavailable, or failed and we fell back to the
        synced realsense depth message). The native branch handles the
        `16UC1` mm → float meters conversion that previously lived inline
        in `_process_realsense_data`; FFS output is already `32FC1` meters
        and is returned unchanged.

        Param reads are per-call so `ros2 param set prefer_ffs false` /
        `... true` flips take effect on the next service call without a
        node restart.
        """
        self._native_depth_context.depth_msg = depth_msg
        try:
            return self._depth_source.acquire(
                align_to_color=bool(
                    self.get_parameter('ffs_align_to_color').value
                )
            )
        finally:
            del self._native_depth_context.depth_msg

    def _process_realsense_data(self, rgb_msg: Image, depth_msg: Image,
                                intrinsic: CameraInfo) -> tuple:
        """Process realsense RGB-D data into usable format."""

        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        depth_img, depth_source = self._acquire_depth(depth_msg)
        # Stash on the node so the sidecar JSON writer downstream can include
        # which depth source served this call. Per-call, overwritten on every
        # `_process_realsense_data`.
        self._last_depth_source = depth_source

        points, validmask_points = realsense_body_axes_points(
            depth_img,
            intrinsic.k,
            valid_band=(1e-6, 10.0),
            clip=(0.0, 10.0),
        )

        return rgb_img, points, validmask_points, depth_msg.header

    def _process_orbbec_data(self, rgb_msg: Image, depth_msg: Image,
                             intrinsic: CameraInfo) -> tuple:
        """Process orbbec RGB-D data into usable format."""
        rgb_img = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        points, valid_mask = self._orbbec_depth_to_array(depth_msg, intrinsic)

        return rgb_img, points, valid_mask, depth_msg.header

    def _sort_objects_and_segments(self, objects: list, segments: list, 
                                     sort_mode: str, camera: str = 'orbbec',
                                     source_frame: str = 'camera_link',
                                     header: Header = None,
                                     closest_distances: list = None) -> tuple:
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
            - 'closest': Sort by 3D distance from the camera to the centroid.
              When `closest_distances` is provided, those camera-frame distances
              are used even if returned object centroids have been transformed
              into a target frame such as base_link or map.
            - 'highest': Sort by height (highest first, based on centroid.z in map frame)
        camera : str, optional
            Camera type ('realsense' or 'orbbec')
        source_frame : str, optional
            Source frame for transformations (e.g., 'camera_link')
        header : Header, optional
            ROS message header for timestamp
        
        Returns
        -------
        tuple
            (sorted objects list, sorted segments list)
        """
        self.get_logger().info(f"Sorting objects with mode: {sort_mode}, camera: {camera}")
        if sort_mode == 'none' or not objects:
            self.get_logger().info("No sorting applied")
            return objects, segments
        
        # Create list of (index, object) tuples for sorting
        indexed_objects = list(enumerate(objects))
        
        if sort_mode == 'closest':
            # Sort by 3D distance from camera to seen centroid. If the caller
            # passes raw camera-frame distances, prefer those; otherwise fall
            # back to the current centroid frame for older call sites.
            def _dist_sq(obj):
                c = obj.centroid
                return c.x * c.x + c.y * c.y + c.z * c.z
            if closest_distances is not None:
                indexed_objects.sort(key=lambda x: closest_distances[x[0]])
            else:
                indexed_objects.sort(key=lambda x: _dist_sq(x[1]))
            if indexed_objects:
                nearest = indexed_objects[0][1].centroid
                nearest_d = (
                    closest_distances[indexed_objects[0][0]]
                    if closest_distances is not None
                    else _dist_sq(indexed_objects[0][1]) ** 0.5
                )
                self.get_logger().info(
                    f"Sorted by closest ({camera}): nearest at "
                    f"({nearest.x:.2f}, {nearest.y:.2f}, {nearest.z:.2f}) m, "
                    f"camera_distance={nearest_d:.2f} m"
                )
        
        elif sort_mode == 'highest':
            # Try to transform to map frame for proper height sorting
            use_map_frame = False
            transform = None
            
            if source_frame and header:
                transform = self._tf_helper.try_lookup(
                    'map',
                    source_frame,
                    stamp=header.stamp,
                    timeout_s=0.1,
                )
                if transform is not None:
                    use_map_frame = True
                    self.get_logger().info("Using map frame for height sorting")
                else:
                    self.get_logger().warn(
                        f"Failed to get transform from {source_frame} to map frame. "
                        f"Falling back to 'closest' sorting mode."
                    )
                    sort_mode = 'closest'
            else:
                self.get_logger().warn(
                    "No source_frame or header provided for 'highest' mode. "
                    "Falling back to 'closest' sorting mode."
                )
                sort_mode = 'closest'
            
            if use_map_frame:
                # Transform centroids to map frame and sort by z (height)
                transformed_heights = []
                
                for idx, obj in indexed_objects:
                    try:
                        # Create PointStamped message
                        point_stamped = geometry_msgs.msg.PointStamped()
                        point_stamped.header = header
                        point_stamped.header.frame_id = source_frame
                        point_stamped.point = obj.centroid
                        
                        # Transform point
                        transformed_point = self._tf_helper.transform_point(
                            point_stamped, transform
                        )
                        if transformed_point is None:
                            raise RuntimeError('point transform failed')
                        self.get_logger().info(
                            f"Object {idx} original point at {obj.centroid} ({point_stamped.point}), transformed to {transformed_point.point}")
                        height = transformed_point.point.z
                        # Store idx, obj, and transformed height for sorting
                        transformed_heights.append((idx, obj, height))
                        
                    except Exception as e:
                        self.get_logger().warn(
                            f"Failed to transform object {idx}: {e}. Using camera frame z as fallback."
                        )
                        # Fallback: use camera frame z
                        transformed_heights.append((idx, obj, obj.centroid.z))
                
                # Sort by transformed height (negative for highest first)
                transformed_heights.sort(key=lambda x: -x[2])
                # Extract idx and obj, discarding the height value
                indexed_objects = [(idx, obj) for idx, obj, _ in transformed_heights]

                if transformed_heights:
                    self.get_logger().info(
                        f"Sorted by height in map frame: highest at z={transformed_heights[0][2]:.2f}m" + \
                        f"All points: {[f'z={item[2]:.2f}m' for item in transformed_heights]}"
                    )
            else:
                # Fallback to closest sorting.
                def _dist_sq_fb(obj):
                    c = obj.centroid
                    return c.x * c.x + c.y * c.y + c.z * c.z
                if closest_distances is not None:
                    indexed_objects.sort(key=lambda x: closest_distances[x[0]])
                else:
                    indexed_objects.sort(key=lambda x: _dist_sq_fb(x[1]))
                if indexed_objects:
                    nearest_d_fb = (
                        closest_distances[indexed_objects[0][0]]
                        if closest_distances is not None
                        else _dist_sq_fb(indexed_objects[0][1]) ** 0.5
                    )
                    self.get_logger().info(
                        f"Fallback: sorted by closest at "
                        f"camera_distance={nearest_d_fb:.2f}m"
                    )
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
            sort_mode: str = 'none',
            target_frame: str = '') -> tuple:
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
        closest_distances = []
        detection_info = []  # Store info for visualization: (bbox, mask, cls_name, conf, centroid)
        detection_info_all = []

        # Look up the source -> target frame transform once for the whole
        # batch. None when no transform is needed; failure aborts cleanly.
        centroid_tf, batch_ok = self._lookup_centroid_transform(
            header.frame_id, target_frame, header.stamp, camera,
        )
        if not batch_ok:
            objects_msg.status = 1
            return objects_msg, []

        # Process detections
        for result in results:
            
            if result.boxes is None or result.masks is None:
                continue

            boxes = result.boxes
            masks = result.masks

            self.get_logger().info(f'Found {len(boxes.cls)} total detections via camera {camera}')

            for i in range(len(boxes.cls)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = self.model.names[cls_id]

                self.get_logger().info(f'Detection {i}: class={cls_name}, conf={conf:.2f}')


                # Filter by confidence
                if conf < self.conf_threshold:
                    self.get_logger().info(f'Skipping detection {i}: low confidence {conf:.2f}')
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
                # Drop disconnected fragments inside the detector bbox.
                # Running CC globally can pick an unrelated fragment outside
                # the bbox, leaving centroid ROI empty.
                mask = largest_connected_component_in_bbox(
                    mask, (x1, y1, x2, y2)
                )

                detection_info_all.append({
                    'bbox': (x1, y1, x2, y2),
                    'mask': mask,
                    'cls_name': cls_name,
                    'conf': conf,
                    'centroid': None
                })

                # Calculate 3D centroid
                centroid = self._calculate_centroid(
                    points, mask, valid_mask, (x1, y1, x2, y2), camera
                )

                if centroid is None:
                    self.get_logger().info(
                        f'Skipping {cls_name}: invalid depth'
                    )
                    continue

                if cls_name in self.excluded_classes:
                    continue

                if cls_name != target_cls:
                    continue

                closest_distance = math.sqrt(
                    centroid.x * centroid.x
                    + centroid.y * centroid.y
                    + centroid.z * centroid.z
                )

                # Express centroid in target_frame; the source->target
                # lookup was hoisted above, so this is just in-memory math.
                if centroid_tf is not None:
                    centroid = self._apply_centroid_transform(
                        centroid, centroid_tf, header.frame_id, header.stamp,
                    )

                # Store detection info for visualization
                detection_info.append({
                    'bbox': (x1, y1, x2, y2),
                    'mask': mask,
                    'cls_name': cls_name,
                    'conf': conf,
                    'centroid': centroid
                })

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
                closest_distances.append(closest_distance)

                if request_segments:
                    segments.append(mask.astype(np.uint8) * 255)
        
        self.get_logger().info(f'Detected {len(objects_msg.objects)} objects of class "{target_cls}"')
        
        # Sort objects and segments together if requested
        # Pass the camera type and source frame from header for TF transformations
        source_frame = header.frame_id
        objects_msg.objects, segments = self._sort_objects_and_segments(
            objects_msg.objects, segments, sort_mode,
            camera=camera,
            source_frame=source_frame,
            header=header,
            closest_distances=closest_distances,
        )
        
        # # Also sort detection_info to match
        # if sort_mode != 'none' and detection_info:
        #     # Create indexed list
        #     indexed_info = list(enumerate(detection_info))
            
        #     if sort_mode == 'closest':
        #         indexed_info.sort(key=lambda x: x[1]['centroid'].x)
        #     elif sort_mode == 'highest':
        #         indexed_info.sort(key=lambda x: -x[1]['centroid'].z)
            
        #     detection_info = [info for _, info in indexed_info]
        
        # Visualize all detections in one image
        if self.visualization:
            self._visualize_all_detections(rgb_img, detection_info)
            self._visualize_all_detections(rgb_img, detection_info_all, displaying_all=True)

        # Stash for the service callback to write debug artifacts if desired.
        # Copied so later tick of the same node can't mutate them mid-write.
        self._last_detection_info = list(detection_info)
        self._last_detection_info_all = list(detection_info_all)
        self._last_rgb_img = rgb_img.copy()

        objects_msg.status = 0 if len(objects_msg.objects) > 0 else 1

        # Centroids were transformed into target_frame above; reflect that
        # on the response header so the (PointStamped header, point) pair
        # the BT consumes is self-consistent.
        if target_frame and self._frame_supports_tf_transform(camera):
            objects_msg.header.frame_id = target_frame

        return objects_msg, segments

    def _calculate_centroid(
            self, points: np.ndarray, mask: np.ndarray,
            valid_mask: np.ndarray,
            bbox: tuple, camera: str) -> geometry_msgs.msg.Point:
        """Calculate 3D centroid from segmentation mask and point cloud.
        
        Args:
            bbox: (x1, y1, x2, y2) where x is column (horizontal), y is row (vertical)
        """
        x1, y1, x2, y2 = bbox

        # Extract region of interest (numpy arrays are [row, col] = [y, x])
        roi_mask = mask[y1:y2, x1:x2]
        roi_valid = valid_mask[y1:y2, x1:x2]
        if np.sum(roi_mask) == 0:
            self.get_logger().warn(
                f'empty mask in bbox={bbox}; falling back to bbox depth'
            )
            roi_mask = np.ones_like(roi_mask)
        roi_points = points[y1:y2, x1:x2]

        # Combine masks using multiplication (works for both bool and float masks)
        # This matches seg_langsam's approach: mask_obj[x1: x2, y1: y2] * validmask_pt[x1: x2, y1: y2]
        combined_mask = roi_mask.astype(float) * roi_valid.astype(float)

        if combined_mask.sum() < 10:
            return None

        # Calculate median for depth, mean for x/y
        # Use np.nonzero to get indices of non-zero elements (matching seg_langsam)
        obj_pts = roi_points[np.nonzero(combined_mask)]
        if len(obj_pts.shape) != 2 or obj_pts.shape[0] == 0:
            return None

        # Create Point message (coordinate system depends on camera type)
        point = geometry_msgs.msg.Point()
        if camera == 'realsense':
            centroid_3d = np.mean(obj_pts, axis=0)
            centroid_3d[2] = np.median(obj_pts[:, 2])  # Use median for depth

            # RealSense camera frame to ROS convention
            # obj_pts contains [x_camera, y_camera, z_camera] where:
            #   x_camera = lateral (from pixel rows)
            #   y_camera = vertical (from pixel columns)  
            #   z_camera = depth (forward)
            # Convert to ROS standard frame:
            #   x = forward (depth)
            #   y = left (negative of vertical pixel position)
            #   z = up (negative of lateral pixel position)
            point.x = float(centroid_3d[2])  # depth -> forward
            point.y = float(-centroid_3d[1])  # -y_camera -> left
            point.z = float(-centroid_3d[0])  # -x_camera -> up
        else:  # orbbec
            # Orbbec already in correct frame
            centroid_3d = np.mean(obj_pts, axis=0)
            centroid_3d[2] = np.median(obj_pts[:, 2])  # Use median for depth

            point.x = float(centroid_3d[0])
            point.y = float(centroid_3d[1])
            point.z = float(centroid_3d[2])

        return point

    # RealSense centroids use hand-rolled body-axis values (x=fwd, y=left,
    # z=up) that disagree with their reported optical header.frame_id;
    # skipping TF preserves existing grasp-service behavior. Orbbec values
    # match their frame_id and transform cleanly.
    _CAMERAS_WITH_UNRELIABLE_FRAME_ID = frozenset({'realsense'})

    def _frame_supports_tf_transform(self, camera: str) -> bool:
        return camera not in self._CAMERAS_WITH_UNRELIABLE_FRAME_ID

    def _lookup_centroid_transform(self, source_frame: str,
                                   target_frame: str, stamp,
                                   camera: str = 'orbbec'):
        """Look up source_frame -> target_frame once per service call.

        Returns (tf_or_None, batch_ok).
          * tf_or_None=None, batch_ok=True  — no transform needed (empty
            target, same frame, or camera flagged unreliable). Loop just
            uses the raw centroids.
          * tf_or_None=<TransformStamped>, batch_ok=True — apply this to
            every centroid; do_transform_point is in-memory math.
          * tf_or_None=None, batch_ok=False — TF lookup failed; caller
            should abort the batch rather than emit mis-framed centroids.

        Hoisted out of the per-detection loop so a 20-person scene with
        an unavailable TF doesn't pay 20 × the lookup_transform timeout.
        """
        if not target_frame or target_frame == source_frame:
            return None, True
        if not self._frame_supports_tf_transform(camera):
            return None, True
        tf = self._tf_helper.try_lookup(
            target_frame,
            source_frame,
            stamp=stamp,
            timeout_s=0.1,
        )
        if tf is None:
            self.get_logger().warn(
                f'TF {source_frame} -> {target_frame} failed; '
                'dropping batch'
            )
            return None, False
        return tf, True

    def _apply_centroid_transform(self, point, tf,
                                  source_frame: str, stamp):
        """Apply a pre-fetched transform to a centroid Point. Cheap; no I/O."""
        ps = geometry_msgs.msg.PointStamped()
        ps.header.frame_id = source_frame
        ps.header.stamp = stamp
        ps.point = point
        transformed = self._tf_helper.transform_point(ps, tf)
        if transformed is None:
            raise RuntimeError('centroid point transform failed')
        return transformed.point

    def _visualize_all_detections(
            self, img: np.ndarray, detection_info: list, displaying_all=False):
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
        
        # self.get_logger().info(f'Visualizing {len(detection_info)} detections')
        
        # Draw all detections
        for idx, (info, color) in enumerate(zip(detection_info, colors)):
            bbox = info['bbox']
            mask = info['mask']
            cls_name = info['cls_name']
            conf = info['conf']
            
            x1, y1, x2, y2 = bbox
            
            # Draw bounding box
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with index (shows sorting order)
            label = ''
            if displaying_all:
                label = f'#{idx+1} {cls_name} {conf:.2f}'
            else:
                centroid = info['centroid']
                label = f'#{idx+1} {cls_name} {conf:.2f} x={centroid.x:.2f}m y={centroid.y:.2f}m z={centroid.z:.2f}m'

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
        
        # Save with timestamp into the shared session run_dir.
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        suffix = 'detection_all' if displaying_all else 'detection'
        filename = self._vision_logger.aux_path(
            timestamp, suffix, 'png', branch='yolo',
        )
        cv2.imwrite(filename, vis_img)
        self.get_logger().info(f'Saved visualization to {filename}')

    def _write_debug_artifacts(self, rgb_img, detections, request_ctx,
                               branch='yolo', vlm_raw=None, timings=None):
        """Dispatch to the shared VisionLogger; preserved as a compat wrapper
        so subclasses (generalist_node) can keep their existing call sites."""
        extras = {'vlm_raw': vlm_raw} if vlm_raw is not None else None
        self._vision_logger.write(
            rgb_img, detections, request_ctx=request_ctx,
            branch=branch, extras=extras, timings=timings,
        )

    def _wait_for_recent_frame(self, camera: str, *, warn: bool = False):
        """Return a current intake pair using its configured backend.

        The subscription path keeps the legacy fixed-call-time polling policy.
        The service path delegates freshness to the provider's header stamps.
        """
        intake = self._camera_intakes.get(camera)
        if intake is None:
            return None
        if intake.cfg.backend == 'service':
            bundle = intake.wait_fresh(
                max_age_s=self.img_sync_thres,
                timeout_s=self.sync_wait_time_limit * 0.1,
                on_timeout='fail',
            )
            if bundle is None:
                if warn:
                    self.get_logger().warn(
                        f'Skipping detection: no recent {camera} provider '
                        'data within sync threshold'
                    )
                return None
            return copy.deepcopy((bundle.color_msg, bundle.depth_msg))

        call_time = self.get_clock().now()
        for _ in range(self.sync_wait_time_limit):
            bundle = intake.latest()
            recent_time = bundle.recv_time if bundle is not None else None
            if recent_time is None or (
                (call_time - recent_time).nanoseconds / 1e9
                > self.img_sync_thres
            ):
                if warn:
                    self.get_logger().warn(
                        f'Skipping detection: no recent {camera} data '
                        f'within sync threshold (called at {call_time}, '
                        f'most recent {recent_time})'
                    )
                time.sleep(0.1)
                continue

            # Match the old second lock/read: a newer pair that arrived after
            # the freshness check is the pair served to the caller.
            latest = intake.latest()
            if latest is not None:
                return copy.deepcopy(
                    (latest.color_msg, latest.depth_msg)
                )
        return None

    def _detection_service_callback(
            self, request: ObjectDetection.Request,
            response: ObjectDetection.Response
    ) -> ObjectDetection.Response:
        """Handle detection service requests."""
        _t0 = time.perf_counter()
        self.get_logger().info('Detection service request received')

        # Determine which camera to use
        camera = 'orbbec'  # default
        if 'realsense' in request.camera:
            camera = 'realsense'
        elif 'orbbec' in request.camera:
            camera = 'orbbec'
        else:
            self.get_logger().warn(f'Unknown camera: {request.camera}, using orbbec')

        rec_msg = self._wait_for_recent_frame(camera, warn=True)

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
        request_segments = True
        # request_segments = 'request_segments' in request.flags
        request_image = True
        # request_image = 'request_image' in request.flags
        
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
                sort_mode=sort_mode,
                target_frame=request.target_frame,
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

            if self._vision_logger.enabled:
                self._write_debug_artifacts(
                    self._last_rgb_img,
                    self._last_detection_info,
                    request_ctx={
                        'service': 'tk23_ObjectDetection',
                        'prompt': request.prompt,
                        'camera': request.camera,
                        'flags': request.flags,
                        'target_frame': request.target_frame,
                        'sort_mode': sort_mode,
                        'n_all_detections': len(self._last_detection_info_all),
                        'depth_source': self._last_depth_source,
                    },
                    branch='yolo',
                    timings={'yolo': time.perf_counter() - _t0},
                )

        except Exception as e:
            self.get_logger().exception(f'Detection failed: {e}')
            response.header = Header(stamp=self.get_clock().now().to_msg())
            response.status = 1
            response.objects = []
            response.person_id = 0
            # Failure-case audit trail: rgb_img is in scope here (raise came
            # from _detect_objects, after _process_*_data succeeded).
            # Pre-rgb early returns above (no camera msg / no intrinsics /
            # camera-data parse error) intentionally don't log — there is
            # no image to render.
            if self._vision_logger.enabled:
                self._vision_logger.write(
                    rgb_img, [],
                    request_ctx={
                        'service': 'tk23_ObjectDetection',
                        'prompt': request.prompt,
                        'camera': request.camera,
                        'flags': request.flags,
                        'target_frame': request.target_frame,
                        'sort_mode': sort_mode,
                        'error': str(e),
                        'depth_source': self._last_depth_source,
                    },
                    branch='error',
                    timings={'yolo': time.perf_counter() - _t0},
                )

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
