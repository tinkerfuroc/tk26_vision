"""Point-cloud relay service.

Extracted from tk23 `util/get_image.py`. Caches the most recent depth frame
(Image for RealSense, PointCloud2 for Orbbec) synchronized with its color
frame, and serves them on demand via `get_point_cloud_service`.

Parity note: the tk23 `GetImage` service is intentionally NOT exposed —
no live caller remained. Only `GetPointCloud` is retained.
"""

import copy
import threading

import rclpy
import rclpy.executors
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tinker_vision_msgs_26.srv import GetPointCloud


class GetPointCloudService(Node):
    def __init__(self):
        super().__init__('get_point_cloud_service')

        self.camera_types = ['realsense', 'orbbec']
        self.image_subs = []

        if 'realsense' in self.camera_types:
            cb_realsense = MutuallyExclusiveCallbackGroup()
            img_realsense_sub = Subscriber(
                self, Image, '/camera/xarm_camera/color/image_raw',
                callback_group=cb_realsense,
            )
            depth_realsense_sub = Subscriber(
                self, Image, '/camera/xarm_camera/aligned_depth_to_color/image_raw',
                callback_group=cb_realsense,
            )
            synced_realsense = ApproximateTimeSynchronizer(
                [img_realsense_sub, depth_realsense_sub], queue_size=3, slop=0.05,
            )
            synced_realsense.registerCallback(self.img_realsense_callback)
            self.image_subs.append(synced_realsense)

            self.camera_info_sub_realsense = self.create_subscription(
                CameraInfo,
                '/camera/xarm_camera/aligned_depth_to_color/camera_info',
                self.camera_info_realsense_callback,
                qos_profile=10,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

        if 'orbbec' in self.camera_types:
            cb_orbbec = MutuallyExclusiveCallbackGroup()
            img_orbbec_sub = Subscriber(
                self, Image, '/camera/color/image_raw', callback_group=cb_orbbec,
            )
            ptcloud_orbbec_sub = Subscriber(
                self, PointCloud2, '/camera/depth_registered/points',
                callback_group=cb_orbbec,
            )
            synced_orbbec = ApproximateTimeSynchronizer(
                [img_orbbec_sub, ptcloud_orbbec_sub], queue_size=3, slop=0.05,
            )
            synced_orbbec.registerCallback(self.img_orbbec_callback)
            self.image_subs.append(synced_orbbec)

            self.camera_info_sub_orbbec = self.create_subscription(
                CameraInfo,
                '/camera/color/camera_info',
                self.camera_info_orbbec_callback,
                qos_profile=10,
                callback_group=MutuallyExclusiveCallbackGroup(),
            )

        self.lock_img = threading.Lock()
        self.recent_img = {'realsense': (None, None), 'orbbec': (None, None)}

        self.lock_info = threading.Lock()
        self.camera_info = {'realsense': None, 'orbbec': None}

        self.point_srv = self.create_service(
            GetPointCloud,
            'get_point_cloud_service',
            self.get_point_cloud_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_logger().info('Point-cloud relay service initialized.')

    async def camera_info_realsense_callback(self, info):
        with self.lock_info:
            self.camera_info['realsense'] = info

    async def camera_info_orbbec_callback(self, info):
        with self.lock_info:
            self.camera_info['orbbec'] = info

    async def img_realsense_callback(self, color_msg, depth_msg):
        with self.lock_img:
            self.recent_img['realsense'] = (color_msg, depth_msg)

    async def img_orbbec_callback(self, color_msg, depth_msg):
        with self.lock_img:
            self.recent_img['orbbec'] = (color_msg, depth_msg)

    async def get_point_cloud_callback(
        self,
        request: GetPointCloud.Request,
        response: GetPointCloud.Response,
    ):
        if request.camera not in self.camera_types:
            self.get_logger().warn(f'No data for camera {request.camera}.')
            response.status = 1
            response.error_msg = f'No camera data for {request.camera}.'
            return response

        with self.lock_img:
            pts = copy.deepcopy(self.recent_img[request.camera][1])

        if pts is None:
            self.get_logger().warn(f'No data for camera {request.camera}.')
            response.status = 1
            response.error_msg = f'No camera data for {request.camera}.'
            return response

        response.status = 0
        response.error_msg = ''
        response.points = pts
        return response


def main():
    rclpy.init()
    node = GetPointCloudService()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
