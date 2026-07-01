"""Door-state detection service.

Ports tk23 `util/door_detection.py`. Heuristic: average depth at the 20x20
center of the Orbbec depth frame < 1.5 m and at least 5 valid pixels => door
is closed (is_open=0); otherwise is_open=1. Only Orbbec is supported.

Bugfix vs tk23: instantiates `self.bridge = CvBridge()` (tk23 referenced it
without ever creating one; latent because the realsense path returns early).
"""

import copy
import threading

import numpy as np
import rclpy
import rclpy.executors
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tinker_vision_msgs_26.srv import DoorDetection


class DoorDetectionService(Node):
    def __init__(self):
        super().__init__('door_detection_service')

        self.bridge = CvBridge()

        self.ptcloud_sub_orbbec = self.create_subscription(
            PointCloud2,
            '/camera/depth_registered/points',
            self.points_orbbec_callback,
            qos_profile=10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.camera_info_sub_orbbec = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_orbbec_callback,
            qos_profile=10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.lock_img = threading.Lock()
        self.recent_points = None

        self.lock_info = threading.Lock()
        self.recent_intrinsic = None

        self.door_detection_srv = self.create_service(
            DoorDetection,
            'door_detection_srv',
            self.door_detection_srv_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_logger().info('Door detection service initialized.')

    async def camera_info_orbbec_callback(self, info):
        with self.lock_info:
            self.recent_intrinsic = info

    async def points_orbbec_callback(self, depth_msg):
        with self.lock_img:
            self.recent_points = depth_msg

    def img_orbbec_process(self, color_msg, depth_msg, intrinsic_msg):
        color_img = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8') if color_msg is not None else None
        K = np.array(intrinsic_msg.k).reshape((3, 3))

        h, w = 720, 1280
        arr = np.frombuffer(depth_msg.data, dtype='<f4')
        N = len(arr) // 5
        points = arr.reshape((N, 5))[:, [0, 1, 2]]
        points_homo = points / np.repeat(points[:, 2:3], 3, axis=1)
        coor_homo = (K @ points_homo.T).T
        coor = np.rint(coor_homo[:, :2]).astype(int)
        depth_img = np.zeros((h, w, 3))
        depth_img[coor[:, 1], coor[:, 0], :] = points
        validmask = (depth_img[:, :, 2] > 1e-3).astype(int)

        return color_img, depth_img, validmask

    async def door_detection_srv_callback(
        self,
        request: DoorDetection.Request,
        response: DoorDetection.Response,
    ):
        camera: str = request.camera
        if 'realsense' in camera:
            response.status = 1
            response.error_msg = 'Only orbbec camera is supported.'
            return response

        with self.lock_img, self.lock_info:
            depth_msg = copy.deepcopy(self.recent_points)
            intrinsic_msg = copy.deepcopy(self.recent_intrinsic)

        if depth_msg is None or intrinsic_msg is None:
            self.get_logger().warn('No camera data or intrinsic.')
            response.status = 1
            response.error_msg = f'No camera data or intrinsic for {request.camera}.'
            return response

        _, depth_img, validmask = self.img_orbbec_process(None, depth_msg, intrinsic_msg)

        W, H, L = 1280, 720, 10
        x1, x2, y1, y2 = H // 2 - L, H // 2 + L, W // 2 - L, W // 2 + L

        depth_crop = depth_img[x1:x2, y1:y2, 2]
        validmask_crop = validmask[x1:x2, y1:y2]
        valid_sum = validmask_crop.sum()
        avg_depth = (depth_crop * validmask_crop).sum() / (valid_sum + 1e-6)
        self.get_logger().info(
            f'validmask sum: {valid_sum}, depth avg: {avg_depth:.3f}'
        )
        if valid_sum > 5 and avg_depth < 1.5:
            response.is_open = 0
        else:
            response.is_open = 1

        response.status = 0
        response.error_msg = ''
        return response


def main():
    rclpy.init()
    node = DoorDetectionService()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
