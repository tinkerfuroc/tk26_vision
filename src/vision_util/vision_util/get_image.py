"""Synchronized color + depth image relay service.

Sibling to `get_point_cloud` — caches the most recent color+depth image
pair for each camera and serves them on demand via `get_image_service`.
Orbbec depth is `depth_registration:=true`-aligned to the color frame at
launch time (see `CAMERA_BRINGUP.md`), so the returned pair can be
unprojected using the color intrinsics directly.

Scope parity with `get_point_cloud`: both cameras are subscribed,
`ApproximateTimeSynchronizer(slop=0.05)` gates the cache.
"""

import copy
import threading

import rclpy
import rclpy.executors
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image
from tinker_vision_msgs_26.srv import GetImage


class GetImageService(Node):
    def __init__(self):
        super().__init__('get_image_service')

        self.camera_types = ['realsense', 'orbbec']

        self.lock_img = threading.Lock()
        self.recent_img = {'realsense': (None, None), 'orbbec': (None, None)}

        self.image_subs = []

        if 'realsense' in self.camera_types:
            cb_realsense = MutuallyExclusiveCallbackGroup()
            color_sub = Subscriber(
                self, Image, '/camera/xarm_camera/color/image_raw',
                callback_group=cb_realsense,
            )
            depth_sub = Subscriber(
                self, Image, '/camera/xarm_camera/aligned_depth_to_color/image_raw',
                callback_group=cb_realsense,
            )
            sync = ApproximateTimeSynchronizer(
                [color_sub, depth_sub], queue_size=3, slop=0.05,
            )
            sync.registerCallback(self.img_realsense_callback)
            self.image_subs.append(sync)

        if 'orbbec' in self.camera_types:
            cb_orbbec = MutuallyExclusiveCallbackGroup()
            color_sub = Subscriber(
                self, Image, '/camera/color/image_raw',
                callback_group=cb_orbbec,
            )
            depth_sub = Subscriber(
                self, Image, '/camera/depth/image_raw',
                callback_group=cb_orbbec,
            )
            sync = ApproximateTimeSynchronizer(
                [color_sub, depth_sub], queue_size=3, slop=0.05,
            )
            sync.registerCallback(self.img_orbbec_callback)
            self.image_subs.append(sync)

        self.image_srv = self.create_service(
            GetImage,
            'get_image_service',
            self.get_image_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_logger().info('Image relay service initialized.')

    async def img_realsense_callback(self, color_msg, depth_msg):
        with self.lock_img:
            self.recent_img['realsense'] = (color_msg, depth_msg)

    async def img_orbbec_callback(self, color_msg, depth_msg):
        with self.lock_img:
            self.recent_img['orbbec'] = (color_msg, depth_msg)

    async def get_image_callback(
        self,
        request: GetImage.Request,
        response: GetImage.Response,
    ):
        if request.camera not in self.camera_types:
            response.status = 1
            response.error_msg = f'Unsupported camera: {request.camera}.'
            return response

        with self.lock_img:
            color_msg, depth_msg = self.recent_img[request.camera]
            color_msg = copy.deepcopy(color_msg)
            depth_msg = copy.deepcopy(depth_msg)

        if color_msg is None or (request.depth and depth_msg is None):
            response.status = 1
            response.error_msg = f'No camera data for {request.camera}.'
            return response

        response.status = 0
        response.error_msg = ''
        response.rgb_image = color_msg
        if request.depth:
            response.depth_image = depth_msg
        return response


def main():
    rclpy.init()
    node = GetImageService()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
