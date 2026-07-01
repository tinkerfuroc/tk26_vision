"""Door-state detection service (depth-image based).

Reads the center of the Orbbec depth image (assuming the camera is level and
pointing forward): the door is open (is_open=1) when the center sees far / no
depth return, closed (is_open=0) when it sees a near surface within
open_threshold_m. Only the Orbbec camera is supported. The pure decision math
lives in _door_logic.py.
"""
import threading

import rclpy
import rclpy.executors
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tinker_vision_msgs_26.srv import DoorDetection

from vision_util._door_logic import depth_to_meters, evaluate_door


class DoorDetectionService(Node):
    def __init__(self):
        super().__init__('door_detection_service')

        self.declare_parameter('open_threshold_m', 1.5)
        self.declare_parameter('center_patch_px', 30)
        self.declare_parameter('min_valid_px', 50)
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.recent_depth = None

        depth_topic = self.get_parameter('depth_topic').value
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            qos_profile=qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.srv = self.create_service(
            DoorDetection,
            'door_detection_srv',
            self.door_detection_srv_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(
            f'Door detection service initialized (depth topic: {depth_topic}).'
        )

    def depth_callback(self, msg: Image):
        with self.lock:
            self.recent_depth = msg

    def door_detection_srv_callback(
        self,
        request: DoorDetection.Request,
        response: DoorDetection.Response,
    ):
        if 'realsense' in request.camera:
            response.status = 1
            response.error_msg = 'Only orbbec camera is supported.'
            return response

        with self.lock:
            msg = self.recent_depth

        if msg is None:
            self.get_logger().warn('No depth image received yet.')
            response.status = 1
            response.error_msg = 'No depth image received yet.'
            return response

        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_m = depth_to_meters(depth_raw, msg.encoding)
        except ValueError as exc:
            response.status = 1
            response.error_msg = str(exc)
            return response

        result = evaluate_door(
            depth_m,
            open_threshold_m=self.get_parameter('open_threshold_m').value,
            center_patch_px=self.get_parameter('center_patch_px').value,
            min_valid_px=self.get_parameter('min_valid_px').value,
        )
        self.get_logger().info(
            f'valid={result.valid_count} median={result.median_m:.3f} m '
            f'-> is_open={result.is_open}'
        )
        response.is_open = result.is_open
        response.status = 0
        response.error_msg = ''
        return response


def main():
    rclpy.init()
    node = DoorDetectionService()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
