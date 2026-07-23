"""Synchronized color + depth image relay service.

Sibling to `get_point_cloud` — caches the most recent color+depth image
pair for each camera and serves them on demand via `get_image_service`.
By default Orbbec depth comes from the raw depth Image topic. If callers need
depth registered to color, launch the camera with the depth-to-color stream
enabled and override `orbbec_depth_topic:=/camera/depth/image_raw`.

Scope parity with `get_point_cloud`: both cameras use color+depth
`CameraIntake` instances, whose synchronizers gate the caches.
"""

import copy

import rclpy
import rclpy.executors
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from tinker_vision_msgs_26.srv import GetImage
from vision_util.camera_intake import CameraIntake, IntakeConfig, StreamSpec


class GetImageService(Node):
    def __init__(self):
        super().__init__('get_image_service')

        self.declare_parameter(
            'realsense_color_topic', '/camera/xarm_camera/color/image_raw'
        )
        self.declare_parameter(
            'realsense_depth_topic',
            '/camera/xarm_camera/aligned_depth_to_color/image_raw',
        )
        self.declare_parameter('orbbec_color_topic', '/camera/color/image_raw')
        self.declare_parameter('orbbec_depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('sync_queue_size', 10)
        self.declare_parameter('sync_slop', 0.1)

        self.camera_types = ['realsense', 'orbbec']

        self.sync_queue_size = int(self.get_parameter('sync_queue_size').value)
        self.sync_slop = float(self.get_parameter('sync_slop').value)
        self.camera_intakes = {}
        for camera in self.camera_types:
            color_topic = self.get_parameter(
                f'{camera}_color_topic'
            ).value
            depth_topic = self.get_parameter(
                f'{camera}_depth_topic'
            ).value
            self.camera_intakes[camera] = CameraIntake(
                self,
                IntakeConfig(
                    camera=camera,
                    color=StreamSpec(
                        color_topic,
                        best_effort=True,
                        qos_depth=self.sync_queue_size,
                    ),
                    depth=StreamSpec(
                        depth_topic,
                        best_effort=True,
                        qos_depth=self.sync_queue_size,
                    ),
                    sync_queue=self.sync_queue_size,
                    sync_slop_s=self.sync_slop,
                ),
                callback_group=MutuallyExclusiveCallbackGroup(),
            )
            self.get_logger().info(
                f'Subscribed to {camera} images: {color_topic} + '
                f'{depth_topic}'
            )

        self.image_srv = self.create_service(
            GetImage,
            'get_image_service',
            self.get_image_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_logger().info('Image relay service initialized.')

    def get_image_callback(
        self,
        request: GetImage.Request,
        response: GetImage.Response,
    ):
        if request.camera not in self.camera_types:
            response.status = 1
            response.error_msg = f'Unsupported camera: {request.camera}.'
            return response

        bundle = self.camera_intakes[request.camera].latest()
        color_msg = bundle.color_msg if bundle is not None else None
        depth_msg = bundle.depth_msg if bundle is not None else None

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
