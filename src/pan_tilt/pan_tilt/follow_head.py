"""YOLO-based head-following action + service."""

import collections
import math
import os
import threading
import time

import cv2
import numpy as np
import rclpy
import rclpy.executors
import rclpy.time
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tinker_vision_msgs_26.action import FollowHeadAction
from tinker_vision_msgs_26.msg import PanTiltCommand, PanTiltState
from tinker_vision_msgs_26.srv import FollowHead
from ultralytics import YOLO

from pan_tilt.head_tracking_helpers import PersonTracker, WorldTargetEMA
# Shared logger
from vision_util.vision_logging import VisionLogger
from vision_util.weights_cache import resolve_weights


def get_array_from_points(
    points: PointCloud2, cam_K: np.array, image_shape=None,
):
    """Convert a PointCloud2 into an (H, W, 3) xyz grid + valid-pixel mask.

    Dimensions are derived from the PointCloud2 when it is organized
    (`height > 1`); otherwise they come from ``image_shape`` (the companion
    color image) so we do not assume 720×1280. The per-point stride is
    ``point_step // 4`` float32 entries — matches the pattern used by
    object_seg_yolo.
    """
    floats_per_point = max(points.point_step // 4, 3)
    arr = np.frombuffer(points.data, dtype='<f4')
    N = len(arr) // floats_per_point
    pts = arr.reshape((N, floats_per_point))[:, :3]

    if points.height > 1 and points.width > 0:
        h, w = int(points.height), int(points.width)
    elif image_shape is not None:
        h, w = int(image_shape[0]), int(image_shape[1])
    else:
        h, w = 720, 1280

    z_col = pts[:, 2:3]
    # Avoid divide-by-zero: keep rows with positive z only.
    valid_z = z_col[:, 0] > 1e-6
    pts_v = pts[valid_z]
    points_homo = pts_v / np.repeat(pts_v[:, 2:3], 3, axis=1)
    coor_homo = (cam_K @ points_homo.T).T
    coor = np.rint(coor_homo[:, :2]).astype(int)

    in_bounds = (
        (coor[:, 0] >= 0) & (coor[:, 0] < w)
        & (coor[:, 1] >= 0) & (coor[:, 1] < h)
    )
    coor = coor[in_bounds]
    pts_v = pts_v[in_bounds]

    depth_img = np.zeros((h, w, 3), dtype=np.float64)
    depth_img[coor[:, 1], coor[:, 0], :] = pts_v
    mask = (depth_img[:, :, 2] > 1e-3).astype(int)
    return depth_img, mask


class FollowHeadNode(Node):
    def __init__(self):
        super().__init__('follow_head_node')

        self.declare_parameter('yolo_model', 'yolov8s-seg.pt')
        yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        yolo_model = self._resolve_model_path(yolo_model)

        # Vision logging runs at the node's existing 1 Hz YOLO cadence
        # (control_interval=1.0), so no extra throttle state machine is needed.
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self.declare_parameter('command_topic', '/pan_tilt_controller/cmd')
        self.declare_parameter('state_topic', '/pan_tilt_controller/state')
        self.declare_parameter('home_pan_deg', 0.0)
        self.declare_parameter('home_tilt_deg', 45.0)
        self.declare_parameter('pan_deadband_deg', 3.0)
        self.declare_parameter('tilt_deadband_deg', 3.0)
        self.declare_parameter('min_command_change_deg', 1.5)
        # Phase B — feedback-gated settle
        self.declare_parameter('min_detection_interval_sec', 0.2)
        self.declare_parameter('max_settle_timeout_sec', 1.5)
        self.declare_parameter('steady_pan_eps_deg', 0.5)
        self.declare_parameter('steady_tilt_eps_deg', 0.5)
        self.declare_parameter('steady_velocity_eps_deg_per_sec', 10.0)
        self.declare_parameter('steady_sample_count', 2)
        self.declare_parameter('state_stale_timeout_sec', 0.3)
        # Phase C — identity continuity + EMA smoothing on world target
        self.declare_parameter('target_ttl_sec', 0.8)
        self.declare_parameter('ema_alpha', 0.4)
        self.declare_parameter('reassoc_dist_m', 0.4)
        # Phase E — config surface + motion profile
        self.declare_parameter('blur_threshold', 80.0)
        self.declare_parameter('small_error_deg', 10.0)
        self.declare_parameter('command_speed_raw_small', 60)
        self.declare_parameter('command_speed_raw_large', 0)  # 0 = use controller default
        self.declare_parameter('command_accel_raw', 0)  # 0 = use controller default
        self._vision_logger = VisionLogger(
            self,
            self.get_parameter('vision_logging_enabled')
            .get_parameter_value()
            .bool_value,
            self.get_parameter('vision_log_folder').get_parameter_value().string_value,
        )
        command_topic = (
            self.get_parameter('command_topic').get_parameter_value().string_value
        )
        state_topic = (
            self.get_parameter('state_topic').get_parameter_value().string_value
        )
        self.home_pan_deg = (
            self.get_parameter('home_pan_deg').get_parameter_value().double_value
        )
        self.home_tilt_deg = (
            self.get_parameter('home_tilt_deg').get_parameter_value().double_value
        )
        self.pan_deadband_deg = (
            self.get_parameter('pan_deadband_deg').get_parameter_value().double_value
        )
        self.tilt_deadband_deg = (
            self.get_parameter('tilt_deadband_deg').get_parameter_value().double_value
        )
        self.min_command_change_deg = (
            self.get_parameter('min_command_change_deg')
            .get_parameter_value()
            .double_value
        )
        self.min_detection_interval_sec = (
            self.get_parameter('min_detection_interval_sec')
            .get_parameter_value()
            .double_value
        )
        self.max_settle_timeout_sec = (
            self.get_parameter('max_settle_timeout_sec')
            .get_parameter_value()
            .double_value
        )
        self.steady_pan_eps_deg = (
            self.get_parameter('steady_pan_eps_deg')
            .get_parameter_value()
            .double_value
        )
        self.steady_tilt_eps_deg = (
            self.get_parameter('steady_tilt_eps_deg')
            .get_parameter_value()
            .double_value
        )
        self.steady_velocity_eps_deg_per_sec = (
            self.get_parameter('steady_velocity_eps_deg_per_sec')
            .get_parameter_value()
            .double_value
        )
        self.steady_sample_count = int(
            self.get_parameter('steady_sample_count')
            .get_parameter_value()
            .integer_value
        )
        self.state_stale_timeout_sec = (
            self.get_parameter('state_stale_timeout_sec')
            .get_parameter_value()
            .double_value
        )
        self.target_ttl_sec = (
            self.get_parameter('target_ttl_sec')
            .get_parameter_value()
            .double_value
        )
        self.ema_alpha = (
            self.get_parameter('ema_alpha').get_parameter_value().double_value
        )
        self.reassoc_dist_m = (
            self.get_parameter('reassoc_dist_m')
            .get_parameter_value()
            .double_value
        )
        self.blur_threshold = (
            self.get_parameter('blur_threshold')
            .get_parameter_value()
            .double_value
        )
        self.small_error_deg = (
            self.get_parameter('small_error_deg')
            .get_parameter_value()
            .double_value
        )
        self.command_speed_raw_small = int(
            self.get_parameter('command_speed_raw_small')
            .get_parameter_value()
            .integer_value
        )
        self.command_speed_raw_large = int(
            self.get_parameter('command_speed_raw_large')
            .get_parameter_value()
            .integer_value
        )
        self.command_accel_raw = int(
            self.get_parameter('command_accel_raw')
            .get_parameter_value()
            .integer_value
        )

        self._person_tracker = PersonTracker(
            reassoc_dist_m=self.reassoc_dist_m,
            ttl_sec=self.target_ttl_sec,
        )
        self._world_target_ema = WorldTargetEMA(
            alpha=self.ema_alpha,
            ttl_sec=self.target_ttl_sec,
        )

        image_sub = Subscriber(self, Image, '/camera/color/image_raw')
        point_cloud_sub = Subscriber(
            self,
            PointCloud2,
            '/camera/depth_registered/points',
        )
        image_sync_sub = ApproximateTimeSynchronizer(
            [image_sub, point_cloud_sub], queue_size=3, slop=0.05,
        )
        image_sync_sub.registerCallback(self.img_orbbec_callback)

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_orbbec_callback,
            qos_profile=10,
        )

        self.action_server = ActionServer(
            self,
            FollowHeadAction,
            'follow_head_action',
            self.execute_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
            cancel_callback=self.cancel_callback,
        )
        self.service = self.create_service(
            FollowHead,
            'follow_head_service',
            self.follow_head_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.is_canceled = False
        self.recent_img = None
        self.recent_point_cloud = None
        self.recent_header = None
        self.last_used_header = None
        self.lock_msg = threading.Lock()
        self.lock_info = threading.Lock()
        self.lock_state = threading.Lock()

        self.orbbec_K = None
        self.current_pan_deg = None
        self.current_tilt_deg = None
        # Phase B — ring buffer of (monotonic_time, pan_rad, tilt_rad, feedback_ok)
        self._state_history = collections.deque(maxlen=4)
        self._last_commanded_pan_rad = None
        self._last_commanded_tilt_rad = None
        self._last_detection_time = None
        # Phase D — last observability snapshot from follow_head_logic
        self._last_logic_info = {
            'person_visible': False,
            'target_pan_deg': 0.0,
            'target_tilt_deg': 0.0,
        }

        self.model = YOLO(str(resolve_weights(yolo_model)))
        self.bridge = CvBridge()

        self.pan_tilt_cmd_pub = self.create_publisher(PanTiltCommand, command_topic, 1)
        self.pan_tilt_state_sub = self.create_subscription(
            PanTiltState,
            state_topic,
            self.pan_tilt_state_callback,
            10,
        )

        self.last_command_time = None

        self.get_logger().info('Follow Head Node has been started.')

    def _resolve_model_path(self, model_path: str) -> str:
        if os.path.isabs(model_path) and os.path.exists(model_path):
            return model_path

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
        shared_workspace_root = os.path.abspath(os.path.join(repo_root, '../..'))
        candidates = [
            model_path,
            os.path.join(os.getcwd(), model_path),
            os.path.join(repo_root, model_path),
            os.path.join(shared_workspace_root, model_path),
        ]

        for candidate in candidates:
            if os.path.exists(candidate):
                self.get_logger().info(f'Using YOLO weights from: {candidate}')
                return candidate

        self.get_logger().warn(
            f'YOLO weights {model_path!r} were not found locally; '
            'Ultralytics may attempt an online download.',
        )
        return model_path

    def camera_info_orbbec_callback(self, info):
        if self.orbbec_K is not None:
            return
        with self.lock_info:
            self.orbbec_K = np.array(info.k).reshape((3, 3))
        self.get_logger().info('Orbbec camera intrinsic matrix has been set.')

    def pan_tilt_state_callback(self, msg: PanTiltState):
        with self.lock_state:
            self.current_pan_deg = float(np.rad2deg(msg.pan_rad))
            self.current_tilt_deg = float(np.rad2deg(msg.tilt_rad))
            self._state_history.append(
                (
                    time.monotonic(),
                    float(msg.pan_rad),
                    float(msg.tilt_rad),
                    bool(msg.feedback_ok),
                ),
            )

    def img_orbbec_callback(self, image_msg, point_cloud_msg):
        while not self.lock_msg.acquire(timeout=0.1):
            self.get_logger().debug(
                'Waiting for lock to process new image/point-cloud messages...',
            )
        self.recent_img = image_msg
        self.recent_point_cloud = point_cloud_msg
        self.recent_header = image_msg.header
        self.lock_msg.release()

    def follow_head_logic(self):
        self.get_logger().debug('Follow Head logic initiated.')

        while not self.lock_msg.acquire(timeout=0.1):
            self.get_logger().debug('Waiting for lock to process follow head logic...')
        if self.recent_img is None or self.recent_point_cloud is None:
            self.lock_msg.release()
            self.get_logger().warn('No image or point cloud received yet.')
            return None, 'No image or point cloud received yet.'

        recent_img = self.recent_img
        recent_point_cloud = self.recent_point_cloud
        recent_header = self.recent_header
        self.lock_msg.release()

        if self.last_used_header:
            if self.last_used_header == recent_header:
                return None, f'image already used (header: {self.last_used_header})'
            self.get_logger().debug(f'Using new image with header: {recent_header}')

        # Enforce a minimum detection interval (5 Hz cap) so YOLO does not
        # saturate the CPU/GPU at the image-sync rate. Detection runs even
        # while the servo is settling — the settle gate below only blocks
        # COMMAND issuance, not perception. This keeps the tracker + EMA
        # fresh so the next command fires on the latest observation.
        now_mono = time.monotonic()
        if (
            self._last_detection_time is not None
            and (now_mono - self._last_detection_time)
            < self.min_detection_interval_sec
        ):
            self.last_used_header = recent_header
            remaining = (
                self.min_detection_interval_sec
                - (now_mono - self._last_detection_time)
            )
            return None, f'Waiting {remaining:.2f}s for min detection interval.'

        self.last_used_header = recent_header

        color_img = self.bridge.imgmsg_to_cv2(recent_img, desired_encoding='bgr8')
        # Laplacian blur gate — cheap enough to always apply, useful when
        # the camera is genuinely blurred (e.g. large motion, low light).
        if self._is_image_blurred(color_img):
            return None, 'Image blurred, waiting for stable frame.'
        self._last_detection_time = now_mono
        image_shape = color_img.shape
        points, validmask_points = get_array_from_points(
            recent_point_cloud,
            self.orbbec_K,
            image_shape=image_shape,
        )

        h, w, _ = color_img.shape
        H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        if h % 32 != 0 or w % 32 != 0:
            color_img = cv2.copyMakeBorder(
                color_img,
                0,
                H - h,
                0,
                W - w,
                cv2.BORDER_CONSTANT,
                0,
            )
            self.get_logger().warn(
                f'Image shape ({h}, {w}) is not a multiple of 32. '
                'Padded so YOLO does not scale it.',
                throttle_duration_sec=30.0,
            )

        _yolo_t0 = time.perf_counter()
        results = self.model(color_img, imgsz=(H, W))
        _yolo_elapsed = time.perf_counter() - _yolo_t0

        person_centroids_3d = []
        log_detections = []  # image-space bboxes + masks for the overlay dump
        if results[0].masks is not None:
            for i, box in enumerate(results[0].boxes):
                if self.model.names[int(box.cls[0])] == 'person':
                    y1, x1, y2, x2 = results[0].boxes.xyxy[i]
                    bbox = (
                        min(int(x1), h - 1),
                        min(int(y1), w - 1),
                        min(int(x2), h - 1),
                        min(int(y2), w - 1),
                    )
                    x1, y1, x2, y2 = bbox
                    y1 = max(0, y1 - ((y2 - y1) // 3) * 2)

                    mask = results[0].masks[i].data.cpu().numpy().squeeze()
                    mask = mask[:h, :w]

                    mask_pt = mask[x1:x2, y1:y2] * validmask_points[x1:x2, y1:y2]
                    if mask_pt.sum() < 10:
                        self.get_logger().warn(
                            f'Detected {box.cls} with invalid depth info, skipped.',
                        )
                        continue
                    sum_pt = mask_pt.sum()
                    cent_pts = [
                        (points[x1:x2, y1:y2, i] * mask_pt).sum() / sum_pt
                        for i in range(3)
                    ]

                    person_centroid = cent_pts[0], cent_pts[1], cent_pts[2]

                    if person_centroid[2] > 0:
                        person_centroids_3d.append(person_centroid)
                        box_xyxy = [int(v) for v in results[0].boxes.xyxy[i].tolist()]
                        log_detections.append(
                            {
                                'bbox': box_xyxy,
                                'mask': mask.astype(bool),
                                'cls_name': 'person',
                                'conf': (
                                    float(box.conf[0])
                                    if box.conf is not None
                                    else None
                                ),
                                'centroid_3d': [float(c) for c in person_centroid],
                            },
                        )

        if not person_centroids_3d:
            self._last_logic_info['person_visible'] = False
            if self._vision_logger.enabled:
                self._vision_logger.write(
                    color_img[:h, :w], None,
                    request_ctx={}, branch='follow_head',
                    extras={'event': 'no_person'},
                    timings={'yolo': _yolo_elapsed},
                )
            self.get_logger().info('No valid person centroid found.')
            return None, 'No valid person centroid found'

        # Transform every candidate into a pan-tilt-rooted Cartesian frame
        # using only the current servo state (NOT the URDF TF chain — that
        # chain's rpy calibration is unreliable on this robot). A stationary
        # person stays put in this frame as the servo moves, which is what
        # the tracker + EMA need.
        with self.lock_state:
            cur_pan_deg = self.current_pan_deg
            cur_tilt_deg = self.current_tilt_deg
        if cur_pan_deg is None or cur_tilt_deg is None:
            return None, 'Pan-tilt state not yet received; cannot anchor target.'
        cur_pan_rad = math.radians(cur_pan_deg)
        cur_tilt_rad = math.radians(cur_tilt_deg)

        candidates_root = []
        candidates_cam = []
        for cam_xyz in person_centroids_3d:
            xyz_root = self._camera_to_pan_tilt_root(
                cam_xyz, cur_pan_rad, cur_tilt_rad,
            )
            if xyz_root is None:
                continue
            candidates_root.append(xyz_root)
            candidates_cam.append(cam_xyz)

        if not candidates_root:
            return None, 'No candidates with positive depth.'

        now_mono = time.monotonic()
        chosen_root = self._person_tracker.update(candidates_root, now_mono)
        if chosen_root is None:
            self._last_logic_info['person_visible'] = False
            return None, 'PersonTracker returned no lock.'
        target_xyz_root = self._world_target_ema.update(chosen_root, now_mono)

        target_pan_rad, target_tilt_rad = self._pan_tilt_root_to_angles(
            target_xyz_root,
        )
        target_pan_deg = float(np.rad2deg(target_pan_rad))
        target_tilt_deg = float(np.rad2deg(target_tilt_rad))
        self.get_logger().info(
            f'Target in pan-tilt-root: '
            f'xyz=({target_xyz_root[0]:.3f}, {target_xyz_root[1]:.3f}, '
            f'{target_xyz_root[2]:.3f}), '
            f'pan={target_pan_deg:.2f} deg, tilt={target_tilt_deg:.2f} deg '
            f'(cur_pan={cur_pan_deg:.2f}, cur_tilt={cur_tilt_deg:.2f})',
        )

        if self._vision_logger.enabled:
            # Mark which detection the tracker selected by matching back from
            # Match the tracker's chosen root-frame xyz back to its
            # camera-frame centroid for the is_chosen annotation.
            try:
                chosen_idx = min(
                    range(len(candidates_root)),
                    key=lambda i: float(
                        np.linalg.norm(
                            np.asarray(candidates_root[i], dtype=np.float64)
                            - np.asarray(chosen_root, dtype=np.float64),
                        ),
                    ),
                )
                chosen_cam = candidates_cam[chosen_idx]
            except ValueError:
                chosen_cam = None
            for det in log_detections:
                det['is_chosen'] = (
                    chosen_cam is not None
                    and tuple(det['centroid_3d'])
                    == tuple(float(c) for c in chosen_cam)
                )
            self._vision_logger.write(
                color_img[:h, :w], log_detections,
                request_ctx={}, branch='follow_head',
                extras={
                    'chosen_cam_xyz': (
                        [float(c) for c in chosen_cam]
                        if chosen_cam is not None else None
                    ),
                    'chosen_root_xyz': [float(c) for c in chosen_root],
                    'target_root_xyz': [float(c) for c in target_xyz_root],
                    'target_pan_deg': target_pan_deg,
                    'target_tilt_deg': target_tilt_deg,
                    'cur_pan_deg': cur_pan_deg,
                    'cur_tilt_deg': cur_tilt_deg,
                },
                timings={'yolo': _yolo_elapsed},
            )

        # Error-vs-state deadband: if the servo already points near the
        # target, don't re-issue. Falls back to target-magnitude if state is
        # not yet known.
        with self.lock_state:
            current_pan_deg = self.current_pan_deg
            current_tilt_deg = self.current_tilt_deg
        if current_pan_deg is not None and current_tilt_deg is not None:
            pan_err_deg = target_pan_deg - current_pan_deg
            tilt_err_deg = target_tilt_deg - current_tilt_deg
        else:
            pan_err_deg = target_pan_deg
            tilt_err_deg = target_tilt_deg

        self._last_logic_info = {
            'person_visible': True,
            'target_pan_deg': float(target_pan_deg),
            'target_tilt_deg': float(target_tilt_deg),
        }

        if (
            abs(pan_err_deg) < self.pan_deadband_deg
            and abs(tilt_err_deg) < self.tilt_deadband_deg
        ):
            self.get_logger().info(
                'Within deadband: '
                f'pan_err={pan_err_deg:.2f} deg, tilt_err={tilt_err_deg:.2f} deg, '
                'holding position.',
            )
            return (target_pan_deg, target_tilt_deg), ''

        # Feedback-gated settle on the COMMAND only. Detection already ran
        # above so the tracker + EMA stay fresh; we just hold off re-issuing
        # a new command until the servo has converged on the last one.
        settle_state, settle_reason = self._classify_settle_state()
        if settle_state == 'wait':
            self.get_logger().debug(f'Command held: {settle_reason}')
            return (target_pan_deg, target_tilt_deg), ''

        # Anti-chatter: if the new target barely differs from the last
        # commanded position, don't re-issue. This protects the Waveshare
        # firmware from a stream of near-identical commands that each
        # interrupt ongoing motion and turn into jerk.
        if (
            self._last_commanded_pan_rad is not None
            and self._last_commanded_tilt_rad is not None
        ):
            last_pan_deg = math.degrees(self._last_commanded_pan_rad)
            last_tilt_deg = math.degrees(self._last_commanded_tilt_rad)
            if (
                abs(target_pan_deg - last_pan_deg) < self.min_command_change_deg
                and abs(target_tilt_deg - last_tilt_deg)
                < self.min_command_change_deg
            ):
                self.get_logger().debug(
                    'Target within min_command_change_deg of last command; '
                    'holding.',
                )
                return (target_pan_deg, target_tilt_deg), ''

        # Speed scaling: small errors get a slower profile to reduce motion
        # blur; large slews run at the controller default.
        max_err_deg = max(abs(pan_err_deg), abs(tilt_err_deg))
        speed_raw = (
            self.command_speed_raw_small
            if max_err_deg < self.small_error_deg
            else self.command_speed_raw_large
        )
        self._publish_absolute_command(
            target_pan_rad, target_tilt_rad,
            speed_raw=speed_raw, accel_raw=self.command_accel_raw,
        )
        return (target_pan_deg, target_tilt_deg), ''

    def _publish_absolute_command(
        self, target_pan_rad, target_tilt_rad,
        speed_raw=0, accel_raw=0,
    ):
        pan_tilt_msg = PanTiltCommand()
        pan_tilt_msg.header.stamp = self.get_clock().now().to_msg()
        pan_tilt_msg.mode = PanTiltCommand.ABSOLUTE
        pan_tilt_msg.pan_rad = float(target_pan_rad)
        pan_tilt_msg.tilt_rad = float(target_tilt_rad)
        pan_tilt_msg.speed_raw = int(speed_raw)
        pan_tilt_msg.accel_raw = int(accel_raw)
        self.pan_tilt_cmd_pub.publish(pan_tilt_msg)
        self.last_command_time = self.get_clock().now()
        with self.lock_state:
            self._last_commanded_pan_rad = float(target_pan_rad)
            self._last_commanded_tilt_rad = float(target_tilt_rad)

    def follow_head_callback(self, request, response):
        self.get_logger().info('Follow Head Service has been called.')
        _, error_msg = self.follow_head_logic()
        if error_msg:
            response.status = -1
            response.error_msg = error_msg
        else:
            response.status = 0
            response.error_msg = ''
        return response

    async def execute_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        result = FollowHeadAction.Result()

        # Phase C: fresh goal = fresh identity lock + fresh smoothing.
        self._person_tracker.reset()
        self._world_target_ema.reset()
        self._last_detection_time = None

        with self.lock_state:
            current_pan_deg = self.current_pan_deg

        home_pan_rad = float(
            np.deg2rad(
                self.home_pan_deg if current_pan_deg is None else current_pan_deg,
            ),
        )
        home_tilt_rad = float(np.deg2rad(self.home_tilt_deg))
        self._publish_absolute_command(home_pan_rad, home_tilt_rad)

        if not goal_handle.request.start_following:
            goal_handle.abort()
            result = FollowHeadAction.Result()
            result.success = False
            result.message = 'Request to stop following.'
            return result

        feedback_msg = FollowHeadAction.Feedback()
        self.is_canceled = False

        try:
            while (
                rclpy.ok()
                and not goal_handle.is_cancel_requested
                and not self.is_canceled
            ):
                pan_tilt, error_msg = self.follow_head_logic()
                self.get_logger().debug(
                    f'Follow head logic returned: {pan_tilt}, error: {error_msg}',
                )

                if error_msg:
                    self.get_logger().debug(
                        f'follow_head_logic returned error: {error_msg}',
                    )
                    # Short sleep — min_detection_interval_sec already paces
                    # YOLO work from inside follow_head_logic; here we just
                    # yield so the spinner can service state/TF callbacks.
                    time.sleep(0.05)
                    continue

                pan_deg, tilt_deg = pan_tilt
                self._populate_feedback(feedback_msg, pan_deg, tilt_deg)
                goal_handle.publish_feedback(feedback_msg)

                time.sleep(0.05)
        except Exception as e:
            self.get_logger().error(f'Error in loop: {e}')
            goal_handle.abort()
            result = FollowHeadAction.Result()
            result.success = False
            result.message = 'Error in loop.'
            return result

        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            result = FollowHeadAction.Result()
            result.message = 'Goal canceled'
            result.success = True
            return result

        goal_handle.succeed()
        result = FollowHeadAction.Result()
        result.success = True
        result.message = 'Successfully followed head.'
        return result

    async def cancel_callback(self, cancel_request):
        self.is_canceled = True
        self._person_tracker.reset()
        self._world_target_ema.reset()
        return CancelResponse.ACCEPT

    def _populate_feedback(self, feedback_msg, pan_deg, tilt_deg):
        """Fill the FollowHeadAction.Feedback including the Phase D fields.

        `pan_deg` / `tilt_deg` are the target pan/tilt in the robot frame
        (what follow_head_logic just returned). Existing callers read these
        via `getattr(feedback, "pan", None)` so keeping them as the
        semantically-most-useful "target" value is a clean upgrade from the
        old "relative correction" value.
        """
        feedback_msg.pan = float(pan_deg)
        feedback_msg.tilt = float(tilt_deg)
        feedback_msg.target_pan = float(
            self._last_logic_info.get('target_pan_deg', pan_deg),
        )
        feedback_msg.target_tilt = float(
            self._last_logic_info.get('target_tilt_deg', tilt_deg),
        )
        feedback_msg.person_visible = bool(
            self._last_logic_info.get('person_visible', False),
        )
        with self.lock_state:
            cur_pan = self.current_pan_deg
            cur_tilt = self.current_tilt_deg
        feedback_msg.current_pan = float(
            cur_pan if cur_pan is not None else pan_deg,
        )
        feedback_msg.current_tilt = float(
            cur_tilt if cur_tilt is not None else tilt_deg,
        )
        if cur_pan is not None and cur_tilt is not None:
            feedback_msg.error_deg = float(
                math.hypot(pan_deg - cur_pan, tilt_deg - cur_tilt),
            )
        else:
            feedback_msg.error_deg = 0.0

    def _is_image_blurred(self, bgr_img: np.ndarray) -> bool:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def _classify_settle_state(self):
        """Decide whether we can act on a fresh frame given the servo state.

        Returns (state, reason) where state is one of:
          - 'go'             — servo is steady at the last commanded pose
          - 'wait'           — feedback says we are still in motion; skip this tick
          - 'stale_feedback' — feedback is missing/stale; caller falls back
                               to the Laplacian blur gate
          - 'blurred'        — reserved for future expansion
        """
        # No command has been issued yet → nothing to settle against.
        if (
            self._last_commanded_pan_rad is None
            or self._last_commanded_tilt_rad is None
            or self.last_command_time is None
        ):
            return 'go', ''

        with self.lock_state:
            history = list(self._state_history)

        if not history:
            return 'stale_feedback', 'No pan/tilt state feedback received yet.'

        latest_ts, latest_pan, latest_tilt, latest_ok = history[-1]
        now_mono = time.monotonic()
        if (
            not latest_ok
            or (now_mono - latest_ts) > self.state_stale_timeout_sec
        ):
            return (
                'stale_feedback',
                f'Pan/tilt feedback is stale ({now_mono - latest_ts:.2f}s) '
                f'or feedback_ok=false; falling back to blur gate.',
            )

        # Safety watchdog: if we've been waiting for convergence longer than
        # max_settle_timeout_sec, act anyway to avoid deadlock on a lost cmd
        # or stuck servo.
        elapsed_since_cmd = (
            self.get_clock().now() - self.last_command_time
        ).nanoseconds / 1e9
        if elapsed_since_cmd > self.max_settle_timeout_sec:
            self.get_logger().warning(
                f'max_settle_timeout_sec={self.max_settle_timeout_sec:.2f}s '
                f'exceeded (elapsed={elapsed_since_cmd:.2f}s); '
                'advancing without steady-state confirmation.',
                throttle_duration_sec=5.0,
            )
            return 'go', ''

        pan_err_deg = math.degrees(latest_pan - self._last_commanded_pan_rad)
        tilt_err_deg = math.degrees(latest_tilt - self._last_commanded_tilt_rad)
        # Condition 1: state must track the last command (so we're not
        # conflating a slow slew with a steady hold).
        if (
            abs(pan_err_deg) > self.steady_pan_eps_deg
            or abs(tilt_err_deg) > self.steady_tilt_eps_deg
        ):
            return (
                'wait',
                f'Servo settling: pan_err={pan_err_deg:.2f} deg, '
                f'tilt_err={tilt_err_deg:.2f} deg.',
            )

        # Condition 2: velocity ≈ 0 over the last N samples.
        need = max(2, self.steady_sample_count + 1)
        if len(history) < need:
            return (
                'wait',
                f'Need {need} state samples for velocity check, '
                f'have {len(history)}.',
            )
        samples = history[-need:]
        for (t0, p0, tl0, _), (t1, p1, tl1, _) in zip(samples, samples[1:]):
            dt = max(t1 - t0, 1e-6)
            pan_vel_deg = abs(math.degrees(p1 - p0) / dt)
            tilt_vel_deg = abs(math.degrees(tl1 - tl0) / dt)
            if (
                pan_vel_deg > self.steady_velocity_eps_deg_per_sec
                or tilt_vel_deg > self.steady_velocity_eps_deg_per_sec
            ):
                return (
                    'wait',
                    f'Servo still moving: pan_vel={pan_vel_deg:.1f} deg/s, '
                    f'tilt_vel={tilt_vel_deg:.1f} deg/s.',
                )

        return 'go', ''

    def _camera_to_pan_tilt_root(self, xyz_cam, cur_pan_rad, cur_tilt_rad):
        """Map a camera-frame centroid into a pan-tilt-rooted Cartesian frame.

        The "pan-tilt root" is the (conceptual) frame attached to the servo
        base — it does NOT rotate with the pan/tilt joints, so a stationary
        person has the same coordinates across ticks regardless of how the
        servo moves. We construct it purely from (cur_pan, cur_tilt) read
        from /pan_tilt_controller/state + the camera-frame centroid. This
        avoids any dependence on the URDF TF chain (which has non-identity
        rpy mismatches that made base_link unusable on this robot).

        Sign conventions (matches the old working tk23 kinematics):
          - camera optical frame: +x=right, +y=down, +z=forward
          - servo pan: positive = turn right (the URDF pan_joint axis
            "0 0 -1" makes this so)
          - servo tilt: positive = tilt up

        Returns (x, y, z) in the root frame where +x is "forward at
        pan=tilt=0", +y is "left" (away from pan-positive direction), +z
        is "up".
        """
        x_cam, y_cam, z_cam = xyz_cam
        if z_cam <= 0:
            return None
        # Camera-frame angular offsets to the person.
        pan_offset_rad = math.atan2(x_cam, z_cam)
        tilt_offset_rad = math.atan2(-y_cam, z_cam)
        # Pythagorean distance from camera to centroid (≈ distance from pan
        # axis; the camera-to-pan-axis offset is ~cm, negligible at arm's
        # reach).
        distance = math.sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam)
        # Absolute pan/tilt angles that would aim the camera at the person.
        world_pan_rad = cur_pan_rad + pan_offset_rad
        world_tilt_rad = cur_tilt_rad + tilt_offset_rad
        # Convert back to a Cartesian point in the pan-tilt-root frame so the
        # tracker + EMA have a natural metric.
        cos_tilt = math.cos(world_tilt_rad)
        x_root = distance * math.cos(world_pan_rad) * cos_tilt
        y_root = -distance * math.sin(world_pan_rad) * cos_tilt
        z_root = distance * math.sin(world_tilt_rad)
        return (x_root, y_root, z_root)

    def _pan_tilt_root_to_angles(self, xyz_root):
        """Inverse of _camera_to_pan_tilt_root: given a pan-tilt-root
        Cartesian point, return the (pan, tilt) radians that aim the camera
        at it. Sign conventions match the forward map.
        """
        x, y, z = xyz_root
        pan_rad = math.atan2(-y, x)
        tilt_rad = math.atan2(z, math.hypot(x, y))
        return pan_rad, tilt_rad


def main():
    rclpy.init()
    follow_head_node = FollowHeadNode()
    try:
        rclpy.spin(
            follow_head_node,
            executor=rclpy.executors.MultiThreadedExecutor(),
        )
    except KeyboardInterrupt:
        pass
    finally:
        follow_head_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
