"""YOLO-based head-following action + service."""

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

# Shared logger
from vision_util.vision_logging import VisionLogger


def get_array_from_points(points: PointCloud2, cam_K: np.array):
    """Convert Orbbec PointCloud2 to (H,W,3) ndarray + valid mask.

    Assumes point_step=20 (5 float32 entries: x, y, z, rgb, _pad).
    """
    h, w = 720, 1280
    arr = np.frombuffer(points.data, dtype='<f4')
    N = len(arr) // 5
    pts = arr.reshape((N, 5))[:, [0, 1, 2]]
    points_homo = pts / np.repeat(pts[:, 2:3], 3, axis=1)
    coor_homo = (cam_K @ points_homo.T).T
    coor = np.rint(coor_homo[:, :2]).astype(int)

    depth_img = np.zeros((h, w, 3))
    depth_img[coor[:, 1], coor[:, 0], :] = pts
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

        self.model = YOLO(yolo_model)
        self.bridge = CvBridge()

        self.pan_tilt_cmd_pub = self.create_publisher(PanTiltCommand, command_topic, 1)
        self.pan_tilt_state_sub = self.create_subscription(
            PanTiltState,
            state_topic,
            self.pan_tilt_state_callback,
            10,
        )

        # Throttle control to match pan/tilt capability and wait for stable imagery
        self.control_interval = 1.0
        self.settle_duration = 1.0
        self.last_command_time = None
        self.blur_threshold = 80.0  # Laplacian variance threshold for motion blur

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
        self.get_logger().info('Follow Head logic initiated.')

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

        now = self.get_clock().now()
        if self.last_command_time is not None:
            elapsed = (now - self.last_command_time).nanoseconds / 1e9
            if elapsed < self.control_interval:
                self.last_used_header = recent_header
                remaining = self.control_interval - elapsed
                return None, f'Waiting {remaining:.2f}s for control interval.'

            msg_time = Time.from_msg(recent_header.stamp)
            if msg_time.nanoseconds < (
                self.last_command_time.nanoseconds
                + int(self.settle_duration * 1e9)
            ):
                self.last_used_header = recent_header
                remaining = (
                    self.last_command_time.nanoseconds
                    + int(self.settle_duration * 1e9)
                    - msg_time.nanoseconds
                ) / 1e9
                return (
                    None,
                    f'Image captured during motion, waiting {remaining:.2f}s '
                    'for stable frame.',
                )

        self.last_used_header = recent_header

        color_img = self.bridge.imgmsg_to_cv2(recent_img, desired_encoding='bgr8')
        if self._is_image_blurred(color_img):
            return None, 'Image blurred during motion, waiting for stable frame.'
        points, validmask_points = get_array_from_points(
            recent_point_cloud,
            self.orbbec_K,
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
            if self._vision_logger.enabled:
                self._vision_logger.write(
                    color_img[:h, :w], None,
                    request_ctx={}, branch='follow_head',
                    extras={'event': 'no_person'},
                    timings={'yolo': _yolo_elapsed},
                )
            self.get_logger().info('No valid person centroid found.')
            return None, 'No valid person centroid found'

        closest_centroid = min(person_centroids_3d, key=lambda c: c[2])
        if self._vision_logger.enabled:
            for det in log_detections:
                det['is_chosen'] = (
                    tuple(det['centroid_3d'])
                    == tuple(float(c) for c in closest_centroid)
                )
            self._vision_logger.write(
                color_img[:h, :w], log_detections,
                request_ctx={}, branch='follow_head',
                extras={'chosen_centroid_3d': [float(c) for c in closest_centroid]},
                timings={'yolo': _yolo_elapsed},
            )
        self.get_logger().info(f'Closest person at {closest_centroid}')

        x, y, z = closest_centroid
        if z == 0:
            self.get_logger().warn(
                'Closest person has 0 depth, cannot calculate angles.',
            )
            return None, 'Closest person has 0 depth'

        pan_rad = np.arctan2(x, z)
        tilt_rad = np.arctan2(-y, z)
        pan_deg = np.rad2deg(pan_rad)
        tilt_deg = np.rad2deg(tilt_rad)
        self.get_logger().info(
            f'Calculated pan: {pan_deg:.2f} deg, tilt: {tilt_deg:.2f} deg',
        )

        if (
            abs(pan_deg) < self.pan_deadband_deg
            and abs(tilt_deg) < self.tilt_deadband_deg
        ):
            self.get_logger().info(
                'Pan/Tilt correction is too small: '
                f'pan={pan_deg:.2f}, tilt={tilt_deg:.2f}, skipping.',
            )
            return (pan_deg, tilt_deg), ''

        pan_tilt_msg = PanTiltCommand()
        pan_tilt_msg.header.stamp = self.get_clock().now().to_msg()
        pan_tilt_msg.mode = PanTiltCommand.RELATIVE
        pan_tilt_msg.pan_rad = (
            0.0 if abs(pan_deg) < self.pan_deadband_deg else float(pan_rad)
        )
        pan_tilt_msg.tilt_rad = (
            0.0 if abs(tilt_deg) < self.tilt_deadband_deg else float(tilt_rad)
        )
        pan_tilt_msg.speed_raw = 0
        pan_tilt_msg.accel_raw = 0
        self.pan_tilt_cmd_pub.publish(pan_tilt_msg)
        self.last_command_time = self.get_clock().now()
        return (pan_deg, tilt_deg), ''

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

        with self.lock_state:
            current_pan_deg = self.current_pan_deg

        pan_tilt_msg = PanTiltCommand()
        pan_tilt_msg.header.stamp = self.get_clock().now().to_msg()
        pan_tilt_msg.mode = PanTiltCommand.ABSOLUTE
        pan_tilt_msg.pan_rad = float(
            np.deg2rad(
                self.home_pan_deg if current_pan_deg is None else current_pan_deg,
            ),
        )
        pan_tilt_msg.tilt_rad = float(np.deg2rad(self.home_tilt_deg))
        pan_tilt_msg.speed_raw = 0
        pan_tilt_msg.accel_raw = 0
        self.pan_tilt_cmd_pub.publish(pan_tilt_msg)
        self.last_command_time = self.get_clock().now()

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
                    self.get_logger().warn(f'Error in follow_head_logic: {error_msg}')
                    time.sleep(0.5)
                    continue

                pan_deg, tilt_deg = pan_tilt
                feedback_msg.pan = pan_deg
                feedback_msg.tilt = tilt_deg
                goal_handle.publish_feedback(feedback_msg)

                time.sleep(self.control_interval)
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
        return CancelResponse.ACCEPT

    def _is_image_blurred(self, bgr_img: np.ndarray) -> bool:
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < self.blur_threshold


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
