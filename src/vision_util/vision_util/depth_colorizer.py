import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image


class DepthColorizer(Node):
    def __init__(self):
        super().__init__('depth_colorizer')

        self.declare_parameter('input_topic', '/camera/depth/image_raw')
        self.declare_parameter('output_topic', '~/depth_colorized')
        self.declare_parameter('depth_min_m', 0.3)
        self.declare_parameter('depth_max_m', 3.0)

        in_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.d_min = float(self.get_parameter('depth_min_m').value)
        self.d_max = float(self.get_parameter('depth_max_m').value)
        if self.d_max <= self.d_min:
            raise ValueError(f'depth_max_m ({self.d_max}) must exceed depth_min_m ({self.d_min})')

        self.bridge = CvBridge()
        pub_qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Image, out_topic, pub_qos)
        self.sub = self.create_subscription(
            Image, in_topic, self._cb, qos_profile_sensor_data,
        )
        self.get_logger().info(
            f'colorizing {in_topic} -> {out_topic} '
            f'(yellow={self.d_min:.2f}m, red={self.d_max:.2f}m)'
        )

    def _cb(self, msg: Image) -> None:
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if depth.dtype == np.uint16:
            depth_m = depth.astype(np.float32) * 1e-3
        else:
            depth_m = depth.astype(np.float32)

        valid = np.isfinite(depth_m) & (depth_m > 0.0)
        norm = np.clip((depth_m - self.d_min) / (self.d_max - self.d_min), 0.0, 1.0)
        scaled = (norm * 255.0).astype(np.uint8)

        b = np.zeros_like(scaled)
        g = (255 - scaled).astype(np.uint8)
        r = np.full_like(scaled, 255)
        b[~valid] = 0
        g[~valid] = 0
        r[~valid] = 0
        out = cv2.merge([b, g, r])

        out_msg = self.bridge.cv2_to_imgmsg(out, encoding='bgr8')
        out_msg.header = msg.header
        self.pub.publish(out_msg)


def main():
    rclpy.init()
    node = DepthColorizer()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
