"""
Door-state detection service.

Ports tk23 `util/door_detection.py`. Heuristic: average depth at the 20x20
center of the Orbbec depth frame < 1.5 m and at least 5 valid pixels => door
is closed (is_open=0); otherwise is_open=1. Only Orbbec is supported.
"""

import rclpy
import rclpy.executors
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from tinker_vision_msgs_26.srv import DoorDetection
from vision_util.camera_intake import CameraIntake, IntakeConfig, StreamSpec
from vision_util.depth_reproject import depth_image_to_points


class DoorDetectionService(Node):
    def __init__(self):
        super().__init__('door_detection_service')

        self.camera_intake = CameraIntake(
            self,
            IntakeConfig(
                camera='orbbec',
                depth=StreamSpec(
                    '/camera/depth/image_raw',
                    best_effort=False,
                    qos_depth=10,
                ),
                camera_info=StreamSpec(
                    '/camera/color/camera_info',
                    best_effort=False,
                    qos_depth=10,
                ),
            ),
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.door_detection_srv = self.create_service(
            DoorDetection,
            'door_detection_srv',
            self.door_detection_srv_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.get_logger().info('Door detection service initialized.')

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

        bundle = self.camera_intake.latest()
        intrinsic = self.camera_intake.intrinsics()
        if bundle is None or intrinsic is None:
            self.get_logger().warn('No camera data or intrinsic.')
            response.status = 1
            response.error_msg = (
                f'No camera data or intrinsic for {request.camera}.'
            )
            return response

        try:
            depth_img = depth_image_to_points(bundle.depth_m(), intrinsic)
            validmask = (depth_img[:, :, 2] > 1e-3).astype(int)
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to process camera data for {request.camera}: {exc}'
            )
            response.status = 1
            response.error_msg = (
                f'No camera data or intrinsic for {request.camera}.'
            )
            return response

        H, W = depth_img.shape[:2]
        L = 10
        x1, x2, y1, y2 = H // 2 - L, H // 2 + L, W // 2 - L, W // 2 + L

        depth_crop = depth_img[x1:x2, y1:y2, 2]
        validmask_crop = validmask[x1:x2, y1:y2]
        valid_sum = validmask_crop.sum()
        avg_depth = (
            (depth_crop * validmask_crop).sum() / (valid_sum + 1e-6)
        )
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
