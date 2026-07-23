"""YOLO-based head-following action + service."""

import collections
import math
import multiprocessing
import os
import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import rclpy.executors
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from tinker_vision_msgs_26.action import FollowHeadAction
from tinker_vision_msgs_26.msg import PanTiltCommand, PanTiltState
from tinker_vision_msgs_26.srv import FollowHead
from ultralytics import YOLO

from pan_tilt.head_tracking_helpers import PersonTracker, WorldTargetEMA
from vision_util.camera_intake import (
    NO_NEW_FRAME,
    CameraIntake,
    IntakeConfig,
    StreamSpec,
)
from vision_util.depth_reproject import follow_head_optical_points
# Shared logger
from vision_util.vision_logging import VisionLogger
from vision_util.weights_cache import resolve_weights


class FollowHeadNode(Node):
    def __init__(self):
        super().__init__('follow_head_node')

        self.declare_parameter('yolo_model', 'yolov8s-seg.pt')
        yolo_model = self.get_parameter('yolo_model').get_parameter_value().string_value
        yolo_model = self._resolve_model_path(yolo_model)

        # Vision logging runs at the node's existing 1 Hz YOLO cadence
        # (control_interval=1.0), so no extra throttle state machine is needed.
        self.declare_parameter('vision_logging_enabled', False)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self.declare_parameter('command_topic', '/pan_tilt_controller/cmd')
        self.declare_parameter('state_topic', '/pan_tilt_controller/state')
        self.declare_parameter('home_pan_deg', 0.0) # ignore these
        self.declare_parameter('home_tilt_deg', 45.0) # ignore these
        self.declare_parameter('pan_deadband_deg', 3.0)
        self.declare_parameter('tilt_deadband_deg', 3.0)
        self.declare_parameter('min_command_change_deg', 1.5)
        # Phase B — feedback-gated settle
        self.declare_parameter('min_detection_interval_sec', 0.2)
        self.declare_parameter('max_settle_timeout_sec', 1.5)
        # Distance-aware command hold-off after a slew. Suppresses new
        # command issuance for clamp(base + per_deg * slew_deg, base,
        # max_settle_timeout_sec). Detection keeps running so the
        # tracker/EMA stay fresh for the next allowed command.
        self.declare_parameter('settle_base_sec', 0.05)
        self.declare_parameter('settle_per_deg_sec', 0.012)
        self.declare_parameter('steady_pan_eps_deg', 0.5)
        self.declare_parameter('steady_tilt_eps_deg', 0.5)
        self.declare_parameter('steady_velocity_eps_deg_per_sec', 10.0)
        self.declare_parameter('steady_sample_count', 2)
        self.declare_parameter('state_stale_timeout_sec', 0.3)
        # Phase C — identity continuity + EMA smoothing on world target
        self.declare_parameter('target_ttl_sec', 0.8)
        self.declare_parameter('ema_alpha', 0.4)
        self.declare_parameter('reassoc_dist_m', 0.4)
        # Default search radius for goal-supplied seed (XY, m). Overridable
        # per-goal via FollowHeadAction.Goal.seed_radius_m (<= 0 -> default).
        self.declare_parameter('seed_radius_m', 0.5)
        # Pose-target params (YOLO-pose face aiming)
        self.declare_parameter('kp_confidence_threshold', 0.5)
        self.declare_parameter('face_depth_window_px', 11)
        self.declare_parameter('min_triangle_valid_depth_pixels', 10)
        # Distance gate: when the person's body-center depth exceeds
        # this threshold, follow_head aims at the upper-body center
        # (full-bbox depth median, robust to silhouette bleed) instead
        # of the small face/eye window (which loses depth at distance).
        self.declare_parameter('face_target_max_distance_m', 2.5)
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
        self.settle_base_sec = (
            self.get_parameter('settle_base_sec')
            .get_parameter_value()
            .double_value
        )
        self.settle_per_deg_sec = (
            self.get_parameter('settle_per_deg_sec')
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
        self.default_seed_radius_m = (
            self.get_parameter('seed_radius_m')
            .get_parameter_value()
            .double_value
        )
        self.kp_conf_thr = (
            self.get_parameter('kp_confidence_threshold')
            .get_parameter_value()
            .double_value
        )
        self.face_depth_window_px = int(
            self.get_parameter('face_depth_window_px')
            .get_parameter_value()
            .integer_value
        )
        self.min_triangle_valid_depth_pixels = int(
            self.get_parameter('min_triangle_valid_depth_pixels')
            .get_parameter_value()
            .integer_value
        )
        self.face_target_max_distance_m = (
            self.get_parameter('face_target_max_distance_m')
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
        # Goal-time fresh-lock policy. Read once at execute_callback,
        # consumed each tick by follow_head_logic via fresh_lock_costs.
        # _goal_seed_xy holds the BT-supplied seed in base_link XY (Z
        # dropped — base_link XY axes match pan-tilt-root XY by URDF
        # symmetry, so no explicit conversion is needed). When None and
        # _goal_track_centermost is False, the tracker uses its default
        # (closest person).
        self._goal_seed_xy: Optional[Tuple[float, float]] = None
        self._goal_seed_radius_m: float = self.default_seed_radius_m
        self._goal_track_centermost: bool = False
        self._world_target_ema = WorldTargetEMA(
            alpha=self.ema_alpha,
            ttl_sec=self.target_ttl_sec,
        )

        # ReentrantCallbackGroup for the high-rate sensor + state
        # subscribers so they don't serialize on the node default group
        # (which is MutuallyExclusive in Humble). Sharing the default
        # group with all other subs starves the executor of free
        # threads under load — the cancel-service handler ends up
        # queueing for a thread instead of running. Locks
        # inside CameraIntake and lock_state enforce data correctness, so
        # concurrent dispatch on different threads is safe.
        self._sensor_cb_group = ReentrantCallbackGroup()
        self.camera_intake = self._create_camera_intake()

        # ReentrantCallbackGroup so the action server's internal cancel
        # service handler (and our user-level cancel_callback, which it
        # invokes) can run on a different MultiThreadedExecutor thread
        # while execute_callback is sitting in its loop. With
        # MutuallyExclusiveCallbackGroup the cancel handler queues
        # behind execute on the same group mutex; since the execute
        # coroutine never `await`s (it uses time.sleep), the mutex is
        # held until the coroutine returns — which it can't, because
        # is_cancel_requested only flips after cancel_callback runs.
        # Classic deadlock; observed as "the action can't be canceled."
        # Safe here because rclpy's ActionServer enforces single-active-
        # goal semantics by default, so execute_callback is not
        # re-entered for two goals concurrently.
        self.action_server = ActionServer(
            self,
            FollowHeadAction,
            'follow_head_action',
            self.execute_callback,
            callback_group=ReentrantCallbackGroup(),
            cancel_callback=self.cancel_callback,
        )
        self.service = self.create_service(
            FollowHead,
            'follow_head_service',
            self.follow_head_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.is_canceled = False
        self.last_used_seq = None
        self.lock_state = threading.Lock()

        self.current_pan_deg = None
        self.current_tilt_deg = None
        # Phase B — ring buffer of (monotonic_time, pan_rad, tilt_rad, feedback_ok)
        self._state_history = collections.deque(maxlen=4)
        self._last_commanded_pan_rad = None
        self._last_commanded_tilt_rad = None
        self._last_detection_time = None
        # Timing counters (reset every 2 s in follow_head_logic). Tracks
        # callback rate, per-stage durations, and early-return reasons.
        self._perf_window_start = time.monotonic()
        self._perf_window_sec = 2.0
        self._perf_sync_count = 0
        self._perf_last_sync_seq = 0
        self._perf_logic_count = 0
        self._perf_yolo_count = 0
        self._perf_sum = {
            'pc_parse': 0.0, 'yolo': 0.0,
            'extract': 0.0, 'total': 0.0,
        }
        self._perf_early = collections.Counter()
        # Phase D — last observability snapshot from follow_head_logic
        self._last_logic_info = {
            'person_visible': False,
            'target_pan_deg': 0.0,
            'target_tilt_deg': 0.0,
        }

        self.model = YOLO(str(resolve_weights(yolo_model)))

        # Warmup: move the model onto GPU (if available) and run a single
        # dummy inference so the first real action goal doesn't eat the
        # ~2 s cold-start hit (CUDA kernel compile + memory alloc + cuDNN
        # autotune). The dummy frame matches the post-pad shape that
        # follow_head_logic feeds the model on the live Orbbec stream
        # (720 x 1280 padded up to 736 x 1280, mult-of-32). torch is a
        # transitive dep of ultralytics so the import is free.
        try:
            import torch  # noqa: F401  (used by ultralytics; just for cuda check)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(device)
            warmup_img = np.zeros((736, 1280, 3), dtype=np.uint8)
            _warmup_t0 = time.perf_counter()
            self.model(warmup_img, imgsz=(736, 1280), verbose=False)
            warmup_ms = (time.perf_counter() - _warmup_t0) * 1000.0
            self.get_logger().info(
                f'YOLO warmup on {device} took {warmup_ms:.1f} ms '
                f'(model={yolo_model})'
            )
        except Exception as e:
            self.get_logger().warn(
                f'YOLO warmup failed (will warm on first real frame): {e}'
            )

        self.pan_tilt_cmd_pub = self.create_publisher(PanTiltCommand, command_topic, 1)
        self.pan_tilt_state_sub = self.create_subscription(
            PanTiltState,
            state_topic,
            self.pan_tilt_state_callback,
            10,
            callback_group=self._sensor_cb_group,
        )

        self.last_command_time = None
        # Distance-aware command hold-off deadline (monotonic seconds).
        # Set on each _publish_absolute_command; gates command issuance
        # in follow_head_logic. 0.0 = no active hold-off.
        self._settle_until_mono = 0.0

        self.get_logger().info('Follow Head Node has been started.')
        print(
            '[follow_head] started — perf instrumentation active '
            '(stderr, every 2s or 100 ticks)',
            file=sys.stderr,
            flush=True,
        )

    def _create_camera_intake(self):
        """Create the synchronized Orbbec intake with legacy endpoint QoS."""
        return CameraIntake(
            self,
            IntakeConfig(
                camera='orbbec',
                color=StreamSpec(
                    '/camera/color/image_raw',
                    best_effort=True,
                    qos_depth=5,
                ),
                depth=StreamSpec(
                    '/camera/depth/image_raw',
                    best_effort=True,
                    qos_depth=5,
                ),
                camera_info=StreamSpec(
                    '/camera/color/camera_info',
                    best_effort=False,
                    qos_depth=10,
                ),
                sync_queue=10,
                sync_slop_s=0.1,
            ),
            callback_group=self._sensor_cb_group,
        )

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

    def follow_head_logic(self):
        self.get_logger().debug('Follow Head logic initiated.')
        self._perf_logic_count += 1
        self._perf_maybe_flush()

        _total_t0 = time.perf_counter()

        bundle = self.camera_intake.latest_new(self.last_used_seq)
        if bundle is None:
            self._perf_early['no_msg'] += 1
            self.get_logger().warn('No image or depth received yet.')
            return None, 'No image or depth received yet.'
        if bundle is NO_NEW_FRAME:
            self._perf_early['already_used'] += 1
            return None, f'image already used (seq: {self.last_used_seq})'

        self._perf_sync_count += max(
            0, bundle.seq - self._perf_last_sync_seq
        )
        self._perf_last_sync_seq = bundle.seq
        self.get_logger().debug(f'Using new image with seq: {bundle.seq}')

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
            self.last_used_seq = bundle.seq
            remaining = (
                self.min_detection_interval_sec
                - (now_mono - self._last_detection_time)
            )
            self._perf_early['min_interval'] += 1
            return None, f'Waiting {remaining:.2f}s for min detection interval.'

        self.last_used_seq = bundle.seq

        color_img = bundle.color_bgr()
        # Blur gate removed — Orbbec frames are clean enough that a Laplacian
        # variance threshold drops more good frames than bad. YOLO handles
        # mild blur on its own.
        self._last_detection_time = now_mono
        _pc_t0 = time.perf_counter()
        points, validmask_points = follow_head_optical_points(
            bundle.depth_m(),
            bundle.K,
        )
        self._perf_sum['pc_parse'] += time.perf_counter() - _pc_t0

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
        # Plain detection — no ByteTrack persistence. Identity continuity
        # comes from PersonTracker's spatial-sticky logic; the goal-side
        # `target_seed_xyz` / `track_centermost` only affects the FRESH
        # lock at goal start.
        results = self.model(color_img, imgsz=(H, W))
        _yolo_elapsed = time.perf_counter() - _yolo_t0
        self._perf_sum['yolo'] += _yolo_elapsed
        self._perf_yolo_count += 1

        person_centroids_3d: list[tuple[float, float, float]] = []
        # Parallel list of face-anchor pixels matching person_centroids_3d
        # 1:1. Used by the `track_centermost` cost; otherwise unused.
        person_target_pixels: list[tuple[int, int]] = []
        log_detections = []  # image-space bboxes + pose/bbox targets for the overlay/JSON
        # COCO-17 keypoints come from `*-pose.pt` weights. Multi-class
        # det/seg weights (e.g. `yolo11s-seg.pt`) leave `keypoints=None`,
        # in which case we fall back to a bbox-only head proxy so the
        # tracker still gets candidates.
        kps_obj = getattr(results[0], 'keypoints', None)
        has_keypoints = (
            kps_obj is not None
            and getattr(kps_obj, 'xy', None) is not None
            and getattr(kps_obj, 'conf', None) is not None
        )
        kps_xy = kps_obj.xy.cpu().numpy() if has_keypoints else None
        kps_cf = kps_obj.conf.cpu().numpy() if has_keypoints else None
        _extract_t0 = time.perf_counter()
        for i, box in enumerate(results[0].boxes):
            if self.model.names[int(box.cls[0])] != 'person':
                continue
            bbox_xyxy = results[0].boxes.xyxy[i].cpu().numpy()
            xyz_cam, target_px, meta = self._extract_person_target(
                bbox_xyxy,
                kps_xy[i] if has_keypoints else None,
                kps_cf[i] if has_keypoints else None,
                has_keypoints,
                points, validmask_points, (h, w),
            )
            if xyz_cam is None:
                continue

            person_centroids_3d.append(xyz_cam)
            person_target_pixels.append(
                (int(target_px[0]), int(target_px[1]))
            )
            box_xyxy_int = [int(v) for v in bbox_xyxy.tolist()]
            log_detections.append(
                {
                    'bbox': box_xyxy_int,
                    'cls_name': 'person',
                    'conf': (
                        float(box.conf[0])
                        if box.conf is not None
                        else None
                    ),
                    'keypoints': (
                        [
                            [float(kps_xy[i, k, 0]),
                             float(kps_xy[i, k, 1]),
                             float(kps_cf[i, k])]
                            for k in range(kps_xy.shape[1])
                        ] if has_keypoints else []
                    ),
                    'target_pixel': [int(target_px[0]), int(target_px[1])],
                    'depth_region': meta['depth_region'],
                    'region_pixel_count': meta['region_pixel_count'],
                    'centroid_3d': [float(c) for c in xyz_cam],
                },
            )

        self._perf_sum['extract'] += time.perf_counter() - _extract_t0

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
            self._perf_early['no_person'] += 1
            self._perf_sum['total'] += time.perf_counter() - _total_t0
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
            self._perf_early['no_state'] += 1
            self._perf_sum['total'] += time.perf_counter() - _total_t0
            return None, 'Pan-tilt state not yet received; cannot anchor target.'
        cur_pan_rad = math.radians(cur_pan_deg)
        cur_tilt_rad = math.radians(cur_tilt_deg)

        candidates_root: list[tuple[float, float, float]] = []
        candidates_pixel: list[tuple[int, int]] = []
        candidates_cam = []
        for cam_xyz, target_px in zip(
            person_centroids_3d, person_target_pixels,
        ):
            xyz_root = self._camera_to_pan_tilt_root(
                cam_xyz, cur_pan_rad, cur_tilt_rad,
            )
            if xyz_root is None:
                continue
            candidates_root.append(xyz_root)
            candidates_pixel.append(target_px)
            candidates_cam.append(cam_xyz)

        if not candidates_root:
            self._perf_early['no_positive_depth'] += 1
            self._perf_sum['total'] += time.perf_counter() - _total_t0
            return None, 'No candidates with positive depth.'

        # Build the fresh-lock cost array from the current goal mode.
        # Priority: seed > centermost > legacy/closest. Tracker only
        # consults this on a fresh lock; sticky-by-spatial reassoc uses
        # `reassoc_dist_m` against locked_xyz unconditionally.
        fresh_lock_costs: Optional[list[float]] = None
        fresh_lock_mode = 'closest'
        if self._goal_seed_xy is not None:
            seed_x, seed_y = self._goal_seed_xy
            radius = self._goal_seed_radius_m
            fresh_lock_costs = []
            for cand_x, cand_y, _cand_z in candidates_root:
                d = math.hypot(cand_x - seed_x, cand_y - seed_y)
                fresh_lock_costs.append(d if d <= radius else math.inf)
            fresh_lock_mode = 'seed'
        elif self._goal_track_centermost:
            cx, cy = w / 2.0, h / 2.0
            fresh_lock_costs = [
                math.hypot(px - cx, py - cy)
                for (px, py) in candidates_pixel
            ]
            fresh_lock_mode = 'centermost'
        # else: leave None -> tracker default = closest-to-origin.

        now_mono = time.monotonic()
        chosen_root = self._person_tracker.update(
            candidates_root, now_mono, fresh_lock_costs,
        )
        if chosen_root is None:
            self._last_logic_info['person_visible'] = False
            self._perf_early['tracker_no_lock'] += 1
            # Surface why the seed-mode lock failed so the operator can
            # see whether the seed and the live candidates actually
            # agree. Throttled to ~0.5 Hz to avoid spamming the log
            # while still firing on every test run.
            if (
                fresh_lock_mode == 'seed'
                and self._person_tracker.locked_xyz is None
            ):
                seed_xy = self._goal_seed_xy
                radius = self._goal_seed_radius_m
                cand_summary = ', '.join(
                    f'(x={x:.2f}, y={y:.2f}, '
                    f'dist={math.hypot(x - seed_xy[0], y - seed_xy[1]):.2f}m)'
                    for (x, y, _z) in candidates_root
                )
                self.get_logger().warning(
                    f'Seed-mode rejected all {len(candidates_root)} '
                    f'candidate(s): seed_xy=({seed_xy[0]:.2f}, '
                    f'{seed_xy[1]:.2f}), radius={radius:.2f} m, '
                    f'candidates: [{cand_summary}]. Increase '
                    f'seed_radius_m on the goal or tighten the '
                    f'detect-time centroid heuristic.',
                    throttle_duration_sec=2.0,
                )
            self._perf_sum['total'] += time.perf_counter() - _total_t0
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
                    'fresh_lock_mode': fresh_lock_mode,
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
            self._perf_early['deadband'] += 1
            self._perf_sum['total'] += time.perf_counter() - _total_t0
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
                self._perf_early['min_cmd_change'] += 1
                self._perf_sum['total'] += time.perf_counter() - _total_t0
                return (target_pan_deg, target_tilt_deg), ''

        # Distance-aware settle hold-off: after a slew, suppress new
        # command issuance for a window proportional to the commanded
        # distance. Detection above kept running so the next allowed
        # command fires on the latest target.
        now_settle_mono = time.monotonic()
        if now_settle_mono < self._settle_until_mono:
            remaining = self._settle_until_mono - now_settle_mono
            self.get_logger().debug(
                f'Settling for {remaining:.2f}s more before next command.',
            )
            self._perf_early['settling'] += 1
            self._perf_sum['total'] += time.perf_counter() - _total_t0
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
        self._perf_early['cmd_issued'] += 1
        self._perf_sum['total'] += time.perf_counter() - _total_t0
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
            cur_pan_deg = self.current_pan_deg
            cur_tilt_deg = self.current_tilt_deg
            self._last_commanded_pan_rad = float(target_pan_rad)
            self._last_commanded_tilt_rad = float(target_tilt_rad)
        if cur_pan_deg is not None and cur_tilt_deg is not None:
            slew_deg = max(
                abs(math.degrees(target_pan_rad) - cur_pan_deg),
                abs(math.degrees(target_tilt_rad) - cur_tilt_deg),
            )
        else:
            slew_deg = 0.0
        hold_sec = min(
            self.settle_base_sec + self.settle_per_deg_sec * slew_deg,
            self.max_settle_timeout_sec,
        )
        self._settle_until_mono = time.monotonic() + hold_sec

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
        self._settle_until_mono = 0.0

        # Read the goal-time fresh-lock policy. Priority: seed > centermost
        # > legacy/closest. The fields stay set for the lifetime of the
        # goal; the tracker only consults them on a fresh lock.
        seed_pt = goal_handle.request.target_seed_xyz
        if seed_pt.x == 0.0 and seed_pt.y == 0.0 and seed_pt.z == 0.0:
            self._goal_seed_xy = None
            self._goal_seed_radius_m = self.default_seed_radius_m
        else:
            # base_link XY ≡ pan-tilt-root XY (axes aligned, near-zero static
            # offset by URDF symmetry). Z is dropped; we match XY only.
            self._goal_seed_xy = (float(seed_pt.x), float(seed_pt.y))
            radius_raw = float(goal_handle.request.seed_radius_m)
            self._goal_seed_radius_m = (
                radius_raw if radius_raw > 0.0 else self.default_seed_radius_m
            )
        self._goal_track_centermost = bool(
            goal_handle.request.track_centermost
        )
        if self._goal_seed_xy is not None:
            self.get_logger().info(
                f'Fresh-lock mode: seed (xy=({seed_pt.x:.3f}, '
                f'{seed_pt.y:.3f}), radius={self._goal_seed_radius_m:.2f} m).',
            )
        elif self._goal_track_centermost:
            self.get_logger().info(
                'Fresh-lock mode: centermost (image-pixel-distance cost).',
            )
        else:
            self.get_logger().info(
                'Fresh-lock mode: closest (closest-to-origin XY, legacy).',
            )

        with self.lock_state:
            current_pan_deg = self.current_pan_deg

        home_pan_rad = float(
            np.deg2rad(
                self.home_pan_deg if current_pan_deg is None else current_pan_deg,
            ),
        )
        home_tilt_rad = float(np.deg2rad(self.home_tilt_deg))
        # self._publish_absolute_command(home_pan_rad, home_tilt_rad)

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
                    # Yield long enough that this loop does not starve the
                    # image-sync / state-callback threads. At ~30 Hz camera
                    # rate, ~33 ms between frames; 20 ms early-return sleep
                    # gives the executor headroom while still picking up new
                    # frames promptly. min_detection_interval_sec=0.1 caps
                    # YOLO work, so faster polling here yields no benefit.
                    time.sleep(0.02)
                    continue

                pan_deg, tilt_deg = pan_tilt
                self._populate_feedback(feedback_msg, pan_deg, tilt_deg)
                goal_handle.publish_feedback(feedback_msg)

                time.sleep(0.02)
        except Exception as e:
            self.get_logger().error(f'Error in loop: {e}')
            goal_handle.abort()
            result = FollowHeadAction.Result()
            result.success = False
            result.message = 'Error in loop.'
            return result

        # Either the client asked rclpy to cancel (is_cancel_requested) or
        # the cancel_callback set our self.is_canceled flag — treat both
        # as a real cancellation and terminate the goal cleanly.
        if goal_handle.is_cancel_requested or self.is_canceled:
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
        # Reset goal-mode fields so a stale seed doesn't bleed into the
        # next goal if execute_callback never overwrites it (degenerate
        # path, but cheap to be safe).
        self._goal_seed_xy = None
        self._goal_seed_radius_m = self.default_seed_radius_m
        self._goal_track_centermost = False
        self._settle_until_mono = 0.0
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

    def _perf_maybe_flush(self):
        """Emit per-stage timings every ~2 s OR every 100 logic ticks.

        Uses BOTH ``self.get_logger().warn`` and an unconditional
        ``print()`` to stderr. The print bypasses any rclpy log-filter or
        output-redirection that could be swallowing the line when the node
        is launched from a launch file or via composed executables. The
        iteration-count fallback guarantees the line appears even if
        ``time.monotonic()`` has somehow not advanced (e.g. a debugger is
        attached, or the window_start was reset oddly).
        """
        now = time.monotonic()
        elapsed = now - self._perf_window_start
        time_ready = elapsed >= self._perf_window_sec
        count_ready = self._perf_logic_count >= 100
        if not (time_ready or count_ready):
            return
        # Guard: if nothing has happened at all (e.g. node just started and
        # follow_head_logic never ran), skip — we don't want NaN/divide-by-0.
        if self._perf_logic_count == 0:
            return
        n_logic = max(self._perf_logic_count, 1)
        n_yolo = max(self._perf_yolo_count, 1)
        elapsed_safe = max(elapsed, 1e-3)
        avg_ms = {
            'pc_parse': 1000.0 * self._perf_sum['pc_parse'] / n_yolo,
            'yolo': 1000.0 * self._perf_sum['yolo'] / n_yolo,
            'extract': 1000.0 * self._perf_sum['extract'] / n_yolo,
            'total': 1000.0 * self._perf_sum['total'] / n_logic,
        }
        sync_hz = self._perf_sync_count / elapsed_safe
        logic_hz = self._perf_logic_count / elapsed_safe
        yolo_hz = self._perf_yolo_count / elapsed_safe
        early_str = ', '.join(
            f'{k}={v}' for k, v in sorted(self._perf_early.items())
        )
        line = (
            f'[follow_head perf {elapsed:.1f}s] sync={sync_hz:.1f}Hz '
            f'logic={logic_hz:.1f}Hz yolo={yolo_hz:.1f}Hz | '
            f'ms/yolo: pc={avg_ms["pc_parse"]:.1f} '
            f'yolo={avg_ms["yolo"]:.1f} '
            f'extract={avg_ms["extract"]:.1f} total={avg_ms["total"]:.1f} | '
            f'branches: {early_str}'
        )
        try:
            self.get_logger().warn(line)
        except Exception:  # pragma: no cover
            pass
        # Unconditional stderr dump — shows up even under buffered launch
        # outputs and even if the rclpy logger is somehow filtered.
        print(line, file=sys.stderr, flush=True)
        # Reset window.
        self._perf_window_start = now
        self._perf_sync_count = 0
        self._perf_logic_count = 0
        self._perf_yolo_count = 0
        for k in self._perf_sum:
            self._perf_sum[k] = 0.0
        self._perf_early.clear()

    def _is_image_blurred(self, bgr_img: np.ndarray) -> bool:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold

    def _depth_in_mask_median(self, points, region_mask_bool):
        """Median (x, y, z) of valid-depth pixels under a boolean (H, W) mask.

        Returns None if the intersection with z>0 is below
        ``self.min_triangle_valid_depth_pixels``. Median matches the
        tk26_vision standard (e.g. object_seg_yolo.py:813,830).
        """
        if region_mask_bool.sum() < self.min_triangle_valid_depth_pixels:
            return None
        pts_in = points[region_mask_bool]
        pts_in = pts_in[pts_in[:, 2] > 1e-6]
        if pts_in.shape[0] < self.min_triangle_valid_depth_pixels:
            return None
        return np.median(pts_in, axis=0)

    def _extract_face_target(
        self, kxy, kconf, bbox_xyxy, points, validmask, img_hw,
    ):
        """Pick a face/head target + depth region from COCO-17 pose keypoints.

        Tries (in order): triangle(eyes+nose), both-eye window, single-eye
        window, nose window, ear-midpoint window, shoulder+upper-bbox head
        proxy. Returns ``(xyz_cam, (px, py), meta)`` on the first branch whose
        (region ∩ valid-depth) passes the min-pixel floor, else
        ``(None, None, None)``. ``validmask`` is the (H, W) bool mask
        returned by ``_depth_image_to_points``.
        """
        h, w = img_hw
        NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 1, 2, 3, 4, 5, 6

        def ok(idx):
            x, y = float(kxy[idx, 0]), float(kxy[idx, 1])
            c = float(kconf[idx])
            # Ultralytics sometimes emits (0, 0) for undetected keypoints
            # alongside a nonzero conf — guard with both.
            return (
                c >= self.kp_conf_thr
                and (x > 0 or y > 0)
                and 0 <= x < w and 0 <= y < h
            )

        def clip_px(x, y):
            return (
                int(np.clip(round(x), 0, w - 1)),
                int(np.clip(round(y), 0, h - 1)),
            )

        def window_mask(px, py):
            r = self.face_depth_window_px // 2
            m = np.zeros((h, w), dtype=bool)
            y0, y1 = max(0, py - r), min(h, py + r + 1)
            x0, x1 = max(0, px - r), min(w, px + r + 1)
            m[y0:y1, x0:x1] = True
            return m

        valid_bool = validmask if validmask.dtype == bool else validmask.astype(bool)

        def try_region(target_px, region_mask, depth_region):
            combined = region_mask & valid_bool
            xyz = self._depth_in_mask_median(points, combined)
            if xyz is None or xyz[2] <= 0:
                return None
            return (
                (float(xyz[0]), float(xyz[1]), float(xyz[2])),
                target_px,
                {
                    'depth_region': depth_region,
                    'region_pixel_count': int(combined.sum()),
                },
            )

        # 1. Triangle (eyes + nose) — the canonical "stare at mid-eye" case.
        if ok(L_EYE) and ok(R_EYE) and ok(NOSE):
            l_px = clip_px(kxy[L_EYE, 0], kxy[L_EYE, 1])
            r_px = clip_px(kxy[R_EYE, 0], kxy[R_EYE, 1])
            n_px = clip_px(kxy[NOSE, 0], kxy[NOSE, 1])
            target_px = (
                int(round((l_px[0] + r_px[0]) / 2)),
                int(round((l_px[1] + r_px[1]) / 2)),
            )
            tri = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(
                tri,
                np.array([l_px, r_px, n_px], dtype=np.int32),
                1,
            )
            res = try_region(target_px, tri.astype(bool), 'triangle')
            if res is not None:
                return res

        # 2. Both eyes without a usable nose — still aim between eyes.
        if ok(L_EYE) and ok(R_EYE):
            l_px = clip_px(kxy[L_EYE, 0], kxy[L_EYE, 1])
            r_px = clip_px(kxy[R_EYE, 0], kxy[R_EYE, 1])
            target_px = (
                int(round((l_px[0] + r_px[0]) / 2)),
                int(round((l_px[1] + r_px[1]) / 2)),
            )
            res = try_region(
                target_px, window_mask(*target_px), 'eye_window',
            )
            if res is not None:
                return res

        # 3. One eye (prefer left, then right — deterministic).
        for eye_idx, tag in (
            (L_EYE, 'single_eye_left_window'),
            (R_EYE, 'single_eye_right_window'),
        ):
            if ok(eye_idx):
                e_px = clip_px(kxy[eye_idx, 0], kxy[eye_idx, 1])
                res = try_region(e_px, window_mask(*e_px), tag)
                if res is not None:
                    return res

        # 4. Nose alone.
        if ok(NOSE):
            n_px = clip_px(kxy[NOSE, 0], kxy[NOSE, 1])
            res = try_region(n_px, window_mask(*n_px), 'nose_window')
            if res is not None:
                return res

        # 5. Ears — faces away, but still a head-level target.
        ear_pxs = []
        if ok(L_EAR):
            ear_pxs.append(clip_px(kxy[L_EAR, 0], kxy[L_EAR, 1]))
        if ok(R_EAR):
            ear_pxs.append(clip_px(kxy[R_EAR, 0], kxy[R_EAR, 1]))
        if ear_pxs:
            target_px = (
                int(round(sum(p[0] for p in ear_pxs) / len(ear_pxs))),
                int(round(sum(p[1] for p in ear_pxs) / len(ear_pxs))),
            )
            res = try_region(
                target_px, window_mask(*target_px), 'ear_window',
            )
            if res is not None:
                return res

        # 6. Shoulder + bbox-top head proxy — last resort when no face
        # keypoints survive. Aim above the shoulders; sample depth over the
        # bbox upper third (torso/neck region).
        if ok(L_SH) and ok(R_SH):
            l_px = clip_px(kxy[L_SH, 0], kxy[L_SH, 1])
            r_px = clip_px(kxy[R_SH, 0], kxy[R_SH, 1])
            x1b, y1b, x2b, y2b = (float(v) for v in bbox_xyxy)
            bbox_height = max(1.0, y2b - y1b)
            mid_x = (l_px[0] + r_px[0]) / 2.0
            mid_y = (l_px[1] + r_px[1]) / 2.0 - 0.1 * bbox_height
            target_px = clip_px(mid_x, mid_y)
            x1i = int(np.clip(round(x1b), 0, w - 1))
            x2i = int(np.clip(round(x2b), 0, w))
            y1i = int(np.clip(round(y1b), 0, h - 1))
            y2i = int(np.clip(round(y2b), 0, h))
            if x2i > x1i and y2i > y1i:
                upper = np.zeros((h, w), dtype=bool)
                upper[y1i : y1i + max(1, (y2i - y1i) // 3), x1i:x2i] = True
                res = try_region(target_px, upper, 'head_proxy')
                if res is not None:
                    return res

        return (None, None, None)

    def _extract_body_center_target(
        self, bbox_xyxy, points, validmask, img_hw,
    ):
        """Full-bbox body-center target for the at-distance regime.

        Used when the person is far enough that face/head depth is
        unreliable (small bbox, small face, sparse depth pixels under
        the floor). Aims at the bbox horizontal center, ~25% down from
        the bbox top (upper-body / chest) for a natural "looking at
        this person" tilt rather than a stomach-aim. Samples depth
        over the *entire* bbox interior — that's robust against the
        silhouette-edge background bleed that breaks the small head
        window at distance.

        Returns ``(xyz_cam, (px, py), meta)`` or ``(None, None, None)``
        if even the full bbox has no valid depth.
        """
        h, w = img_hw
        x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
        bbox_h = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = y1 + 0.25 * bbox_h
        target_px = (
            int(np.clip(round(cx), 0, w - 1)),
            int(np.clip(round(cy), 0, h - 1)),
        )
        x1i = int(np.clip(round(x1), 0, w - 1))
        x2i = int(np.clip(round(x2), 0, w))
        y1i = int(np.clip(round(y1), 0, h - 1))
        y2i = int(np.clip(round(y2), 0, h))
        if x2i <= x1i or y2i <= y1i:
            return (None, None, None)
        body = np.zeros((h, w), dtype=bool)
        body[y1i:y2i, x1i:x2i] = True
        valid_bool = (
            validmask if validmask.dtype == bool else validmask.astype(bool)
        )
        combined = body & valid_bool
        xyz = self._depth_in_mask_median(points, combined)
        if xyz is None or xyz[2] <= 0:
            return (None, None, None)
        return (
            (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            target_px,
            {
                'depth_region': 'bbox_full_body',
                'region_pixel_count': int(combined.sum()),
            },
        )

    def _extract_person_target(
        self,
        bbox_xyxy,
        kps_xy_i,
        kps_cf_i,
        has_keypoints,
        points,
        validmask,
        img_hw,
    ):
        """Distance-aware person target.

        Always probes body-center depth first — cheap, robust at any
        distance, and tells us whether to bother running the face
        cascade. When the body sits beyond
        ``face_target_max_distance_m`` we use the body target directly
        ("look at this person" — coarse but never fails at distance).
        Within the threshold we run the eye/head cascade for
        receptionist-style precise gaze, falling back to body-center
        only if the cascade also fails (e.g., person facing away with
        no usable face keypoints).
        """
        body_xyz, body_px, body_meta = self._extract_body_center_target(
            bbox_xyxy, points, validmask, img_hw,
        )
        if body_xyz is None:
            return (None, None, None)

        if body_xyz[2] > self.face_target_max_distance_m:
            return (body_xyz, body_px, body_meta)

        if has_keypoints:
            xyz, px, meta = self._extract_face_target(
                kps_xy_i, kps_cf_i, bbox_xyxy,
                points, validmask, img_hw,
            )
        else:
            xyz, px, meta = self._extract_bbox_head_target(
                bbox_xyxy, points, validmask, img_hw,
            )
        if xyz is not None:
            return (xyz, px, meta)
        # Close-range face/head extraction failed (e.g., person facing
        # away, severe occlusion, or extreme blur). Fall back to the
        # body probe we already computed so we still produce a candidate.
        return (body_xyz, body_px, body_meta)

    def _extract_bbox_head_target(
        self, bbox_xyxy, points, validmask, img_hw,
    ):
        """Bbox-only head proxy for non-pose YOLO weights.

        Used when ``results[0].keypoints`` is ``None`` (e.g. the model is
        ``yolo11s-seg.pt`` / ``yolov8s-seg.pt`` instead of a ``-pose``
        variant). Aims at ~15% down from the bbox top, centered, and
        samples depth in a small window around that pixel — narrow
        enough to dodge background bleed at the head silhouette, which
        a full upper-third region would catch (wall / sky behind a
        standing person can pull the median depth several meters out,
        which moves the unprojected XY by >>seed_radius_m and causes
        seed-mode tracker_no_lock).

        Returns ``(xyz_cam, (px, py), meta)`` or ``(None, None, None)``
        if the depth region is too sparse.
        """
        h, w = img_hw
        x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
        bbox_h = max(1.0, y2 - y1)
        bbox_w = max(1.0, x2 - x1)
        # Aim at ~15% down from the bbox top, horizontally centered —
        # approximates head position on a standing/sitting person.
        cx = (x1 + x2) / 2.0
        cy = y1 + 0.15 * bbox_h
        target_px = (
            int(np.clip(round(cx), 0, w - 1)),
            int(np.clip(round(cy), 0, h - 1)),
        )
        # Depth-sampling window: a square centered on the aim point,
        # sized to ~30% of the bbox shorter side but capped at +/- 25 px.
        # The cap prevents enormous boxes (close-up of a person) from
        # sampling well outside the head silhouette; the percentage
        # ensures small/distant boxes still have enough valid depth.
        half = int(max(self.face_depth_window_px // 2,
                       min(25, max(2, int(0.15 * min(bbox_w, bbox_h))))))
        px, py = target_px
        x0i = int(np.clip(px - half, 0, w - 1))
        x1i = int(np.clip(px + half + 1, 0, w))
        y0i = int(np.clip(py - half, 0, h - 1))
        y1i = int(np.clip(py + half + 1, 0, h))
        if x1i <= x0i or y1i <= y0i:
            return (None, None, None)
        win = np.zeros((h, w), dtype=bool)
        win[y0i:y1i, x0i:x1i] = True
        valid_bool = (
            validmask if validmask.dtype == bool else validmask.astype(bool)
        )
        combined = win & valid_bool
        xyz = self._depth_in_mask_median(points, combined)
        if xyz is None or xyz[2] <= 0:
            return (None, None, None)
        return (
            (float(xyz[0]), float(xyz[1]), float(xyz[2])),
            target_px,
            {
                'depth_region': 'bbox_head_window',
                'region_pixel_count': int(combined.sum()),
            },
        )

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
    # Pin the executor's worker pool so the cancel-service handler can
    # always grab a free thread under high-load YAML configs (10 Hz YOLO
    # + 30 Hz camera + 20 Hz state publisher). MultiThreadedExecutor's
    # default num_threads is environment-dependent (cpu_count() can
    # collapse to a small value on constrained hosts) and matches the
    # documented floor used in object_detection_generalist.
    num_threads = max(8, multiprocessing.cpu_count())
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=num_threads)
    follow_head_node.get_logger().info(
        f'Spinning follow_head with MultiThreadedExecutor '
        f'(num_threads={num_threads})'
    )
    try:
        rclpy.spin(follow_head_node, executor=executor)
    except KeyboardInterrupt:
        pass
    finally:
        follow_head_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
