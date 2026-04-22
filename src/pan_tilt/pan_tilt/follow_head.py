"""YOLO-based head-following action + service.

Ported from tk23 `pan_tilt/follow_head.py`. Detects persons in Orbbec RGB,
computes closest-person 3D centroid from depth point cloud, converts to pan/tilt
angles, and issues `pan_tilt_ctrl_modify` commands at 1 Hz with blur-gating.

Changes from tk23:
- YOLO model selectable via ROS param `yolo_model` (default `yolov8s-seg.pt`).
"""

import math
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
from tinker_vision_msgs_26.msg import PanTiltCtrl
from tinker_vision_msgs_26.srv import FollowHead
from ultralytics import YOLO


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

        image_sub = Subscriber(self, Image, '/camera/color/image_raw')
        point_cloud_sub = Subscriber(self, PointCloud2, '/camera/depth_registered/points')
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

        self.orbbec_K = None

        self.model = YOLO(yolo_model)
        self.bridge = CvBridge()

        self.pan_tilt_abs_pub = self.create_publisher(PanTiltCtrl, 'pan_tilt_ctrl', 1)
        self.pan_tilt_modify_pub = self.create_publisher(PanTiltCtrl, 'pan_tilt_ctrl_modify', 1)

        # Throttle control to match pan/tilt capability and wait for stable imagery
        self.control_interval = 1.0
        self.settle_duration = 1.0
        self.last_command_time = None
        self.blur_threshold = 80.0  # Laplacian variance threshold for motion blur

        self.get_logger().info('Follow Head Node has been started.')

    def camera_info_orbbec_callback(self, info):
        if self.orbbec_K is not None:
            return
        with self.lock_info:
            self.orbbec_K = np.array(info.k).reshape((3, 3))
        self.get_logger().info('Orbbec camera intrinsic matrix has been set.')

    def img_orbbec_callback(self, image_msg, point_cloud_msg):
        while not self.lock_msg.acquire(timeout=0.1):
            self.get_logger().debug('Waiting for lock to process new Image and PointCloud2 messages...')
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
            if msg_time.nanoseconds < self.last_command_time.nanoseconds + int(self.settle_duration * 1e9):
                self.last_used_header = recent_header
                remaining = (
                    self.last_command_time.nanoseconds
                    + int(self.settle_duration * 1e9)
                    - msg_time.nanoseconds
                ) / 1e9
                return None, f'Image captured during motion, waiting {remaining:.2f}s for stable frame.'

        self.last_used_header = recent_header

        color_img = self.bridge.imgmsg_to_cv2(recent_img, desired_encoding='bgr8')
        if self._is_image_blurred(color_img):
            return None, 'Image blurred during motion, waiting for stable frame.'
        points, validmask_points = get_array_from_points(recent_point_cloud, self.orbbec_K)

        h, w, _ = color_img.shape
        H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
        if h % 32 != 0 or w % 32 != 0:
            color_img = cv2.copyMakeBorder(color_img, 0, H - h, 0, W - w, cv2.BORDER_CONSTANT, 0)
            self.get_logger().warn(
                f'Image shape ({h}, {w}) is not a multiple of 32. Padded so YOLO does not scale it.'
            )

        results = self.model(color_img, imgsz=(H, W))

        person_centroids_3d = []
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
                        self.get_logger().warn(f'Detected {box.cls} with invalid depth info, skipped.')
                        continue
                    sum_pt = mask_pt.sum()
                    cent_pts = [(points[x1:x2, y1:y2, i] * mask_pt).sum() / sum_pt for i in range(3)]

                    person_centroid = cent_pts[0], cent_pts[1], cent_pts[2]

                    if person_centroid[2] > 0:
                        person_centroids_3d.append(person_centroid)

        if not person_centroids_3d:
            self.get_logger().info('No valid person centroid found.')
            return None, 'No valid person centroid found'

        closest_centroid = min(person_centroids_3d, key=lambda c: c[2])
        self.get_logger().info(f'Closest person at {closest_centroid}')

        x, y, z = closest_centroid
        if z == 0:
            self.get_logger().warn('Closest person has 0 depth, cannot calculate angles.')
            return None, 'Closest person has 0 depth'

        pan_rad = np.arctan2(x, z)
        tilt_rad = np.arctan2(-y, z)
        pan_deg = np.rad2deg(pan_rad)
        tilt_deg = np.rad2deg(tilt_rad)
        self.get_logger().info(f'Calculated pan: {pan_deg:.2f} deg, tilt: {tilt_deg:.2f} deg')

        pan_tilt_msg = PanTiltCtrl()
        pan_tilt_msg.x = pan_deg
        pan_tilt_msg.y = 0.0
        pan_tilt_msg.speed = 0.0

        if math.fabs(pan_deg) < 3.0:
            self.get_logger().info(f'Pan angle is too small: {pan_deg:.2f}, skipping.')
            return (pan_deg, tilt_deg), ''
        self.pan_tilt_modify_pub.publish(pan_tilt_msg)
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

        pan_tilt_msg = PanTiltCtrl()
        pan_tilt_msg.x = -1000.0
        pan_tilt_msg.y = 45.0
        pan_tilt_msg.speed = 0.0
        self.pan_tilt_abs_pub.publish(pan_tilt_msg)
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
            while rclpy.ok() and not goal_handle.is_cancel_requested and not self.is_canceled:
                pan_tilt, error_msg = self.follow_head_logic()
                self.get_logger().debug(f'Follow head logic returned: {pan_tilt}, error: {error_msg}')

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
    rclpy.spin(follow_head_node, executor=rclpy.executors.MultiThreadedExecutor())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
