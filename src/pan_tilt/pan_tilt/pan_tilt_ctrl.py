"""Pan-tilt servo controller.

Ported from tk23 `pan_tilt/pan_tilt_ctrl.py`. Listens on `pan_tilt_ctrl` and
`pan_tilt_ctrl_modify` topics, drives a serial-connected pan-tilt head, and
broadcasts base_link -> pan_link -> tilt_link -> camera_link TF chain.

Changes from tk23:
- `specs.json` is loaded from the installed package share dir by default,
  overridable via ROS param `specs_path`. (tk23 hardcoded an absolute path
  under tk23_vision/src/, which broke once the tree was relocated.)
"""

import json
import math
import os
import threading
import time

import rclpy
import rclpy.executors
import serial
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Quaternion, Transform, TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler
from tinker_vision_msgs.msg import PanTiltCtrl


class PanTiltCtrlNode(Node):
    def __init__(self, timer_period=0.1, transform_update_period=0.05, default_speed=360):
        super().__init__('PanTiltCtrl')
        self.declare_parameter('device', '/dev/ttyUSB0')
        self.declare_parameter('specs_path', '')

        self.device = self.get_parameter('device').get_parameter_value().string_value

        specs_path = self.get_parameter('specs_path').get_parameter_value().string_value
        if not specs_path:
            specs_path = os.path.join(
                get_package_share_directory('pan_tilt'),
                'config',
                'specs.json',
            )
        self.get_logger().info(f'Loading specs from: {specs_path}')
        with open(specs_path, 'r') as f:
            self.specs = json.load(f)

        self.offset_x = self.specs['offset_x']
        self.offset_y = self.specs['offset_y']

        self.initial_pos = [0, 20]
        self.tf_broadcaster = TransformBroadcaster(self)

        self.cb_group_tf = MutuallyExclusiveCallbackGroup()
        self.cb_group_server = MutuallyExclusiveCallbackGroup()
        self.ser = serial.Serial(self.device, 115200)
        while not self.ser.is_open:
            self.ser.open()
            time.sleep(1)
            self.get_logger().info(f'Connecting to {self.device}...')

        self.get_logger().info('Resetting pan tilt...')
        self.ser.write(
            f'{{"T":133,"X":{self.initial_pos[0]}, "Y":{self.initial_pos[1]}, "SPD":0, "ACC":0}}\n'.encode()
        )

        self.mutex_transform = threading.Lock()

        self.base_to_pan = TransformStamped()
        self.base_to_pan.header.frame_id = 'base_link'
        self.base_to_pan.child_frame_id = 'pan_link'

        self.pan_to_tilt = TransformStamped()
        self.pan_to_tilt.header.frame_id = 'pan_link'
        self.pan_to_tilt.child_frame_id = 'tilt_link'

        self.tilt_to_camera = TransformStamped()
        self.tilt_to_camera.header.frame_id = 'tilt_link'
        self.tilt_to_camera.child_frame_id = 'camera_link'
        self.tilt_to_camera.transform.translation.x = self.specs['tilt_to_camera']['translation']['x']
        self.tilt_to_camera.transform.translation.y = self.specs['tilt_to_camera']['translation']['y']
        self.tilt_to_camera.transform.translation.z = (
            self.specs['tilt_to_camera']['translation']['z'] + self.specs['tilt_length']
        )
        self.tilt_to_camera.transform.rotation.x = self.specs['tilt_to_camera']['rotation']['x']
        self.tilt_to_camera.transform.rotation.y = self.specs['tilt_to_camera']['rotation']['y']
        self.tilt_to_camera.transform.rotation.z = self.specs['tilt_to_camera']['rotation']['z']
        self.tilt_to_camera.transform.rotation.w = self.specs['tilt_to_camera']['rotation']['w']

        self.sub = self.create_subscription(
            PanTiltCtrl, 'pan_tilt_ctrl', self.pan_tilt_ctrl_callback, 1,
        )
        self.modify_sub = self.create_subscription(
            PanTiltCtrl, 'pan_tilt_ctrl_modify', self.pan_tilt_ctrl_modify_callback, 1,
        )
        self.get_logger().info("Subscribed to 'pan_tilt_ctrl' topic...")

        self.timer_period = timer_period

        self.trasnform_timer = self.create_timer(
            transform_update_period, self.transform_publish, self.cb_group_tf,
        )

        self.current_x = self.initial_pos[0] + self.offset_x
        self.current_y = self.initial_pos[1] + self.offset_y
        self.target_x = self.initial_pos[0]
        self.target_y = self.initial_pos[1]
        self.speed = default_speed

        marked_angles = self.specs['marked_angles']
        self.fixed_transforms = {}
        for marked in marked_angles:
            angle = marked['angle']
            x, y = angle['x'], angle['y']
            transform_base_to_camera = marked['base_to_camera']
            self.fixed_transforms[(x, y)] = transform_base_to_camera

        self.get_logger().info("Service 'pan_tilt_ctrl' is running...")

    async def pan_tilt_ctrl_modify_callback(self, msg):
        self.get_logger().info(f'Modify Received pan_tilt_ctrl message: {msg}')
        self.get_logger().info(f'Current target: {self.target_x}, {self.target_y}')
        self.target_x += msg.x
        self.target_y += msg.y
        self.target_x = min(max(self.target_x, -180.0), 180.0)
        self.target_y = min(max(self.target_y, -29.0), 90.0)
        speed = msg.speed if msg.speed else self.speed

        self.get_logger().info(f'Setting target to: {self.target_x}, {self.target_y}')
        self.get_logger().info(f'Sending command to pan tilt...{self.target_x}, {self.target_y}')
        command = f'{{"T":133,"X":{self.target_x}, "Y":{self.target_y}, "SPD":{speed}, "ACC":0}}\n'
        self.ser.write(command.encode())

        time.sleep(0.15)
        with self.mutex_transform:
            self.current_x = self.target_x + self.offset_x
            self.current_y = self.target_y + self.offset_y
            self.get_logger().info(f'Current position updated to: {self.current_x}, {self.current_y}')
            self.update_transforms()

    async def pan_tilt_ctrl_callback(self, msg):
        self.get_logger().info(f'Control Received pan_tilt_ctrl message: {msg}')
        if not msg.x == -1000.0:
            self.target_x = min(max(msg.x, -180.0), 180.0)
        self.target_y = min(max(msg.y, -29.0), 90.0)
        speed = msg.speed if msg.speed else self.speed

        self.get_logger().info(f'Setting target to: {self.target_x}, {self.target_y}')
        self.get_logger().info(f'Sending command to pan tilt...{self.target_x}, {self.target_y}')
        command = f'{{"T":133,"X":{self.target_x}, "Y":{self.target_y}, "SPD":{speed}, "ACC":0}}\n'
        self.ser.write(command.encode())

        time.sleep(1.0)
        with self.mutex_transform:
            self.current_x = self.target_x + self.offset_x
            self.current_y = self.target_y + self.offset_y
            self.get_logger().info(f'Current position updated to: {self.current_x}, {self.current_y}')
            self.update_transforms()

    def update_transforms(self):
        if (self.current_x, self.current_y) in self.fixed_transforms.keys():
            self.base_to_pan.transform = Transform()

            base_to_camera = self.fixed_transforms[(self.current_x, self.current_y)]
            self.pan_to_tilt.transform.translation.x = base_to_camera['translation']['x']
            self.pan_to_tilt.transform.translation.y = base_to_camera['translation']['y']
            self.pan_to_tilt.transform.translation.z = base_to_camera['translation']['z']

            self.pan_to_tilt.transform.rotation.x = base_to_camera['rotation']['x']
            self.pan_to_tilt.transform.rotation.y = base_to_camera['rotation']['y']
            self.pan_to_tilt.transform.rotation.z = base_to_camera['rotation']['z']
            self.pan_to_tilt.transform.rotation.w = base_to_camera['rotation']['w']
        else:
            self.base_to_pan.transform.translation.x = self.specs['base_to_pan']['translation']['x']
            self.base_to_pan.transform.translation.y = self.specs['base_to_pan']['translation']['y']
            self.base_to_pan.transform.translation.z = self.specs['base_to_pan']['translation']['z']

            pan_rad = math.radians(-self.current_x)
            q_pan = quaternion_from_euler(0, 0, pan_rad)
            q_base_final = q_pan

            self.base_to_pan.transform.rotation = Quaternion(
                x=q_base_final[0], y=q_base_final[1], z=q_base_final[2], w=q_base_final[3],
            )

            self.pan_to_tilt.transform.translation.x = 0.0
            self.pan_to_tilt.transform.translation.y = 0.0
            self.pan_to_tilt.transform.translation.z = self.specs['pan_length']

            tilt_rad = math.radians(self.current_y)
            q_tilt = quaternion_from_euler(0, tilt_rad, 0)
            self.pan_to_tilt.transform.rotation = Quaternion(
                x=q_tilt[0], y=q_tilt[1], z=q_tilt[2], w=q_tilt[3],
            )

    async def transform_publish(self):
        with self.mutex_transform:
            self.update_transforms()
            stamp = self.get_clock().now().to_msg()

            self.base_to_pan.header.stamp = stamp
            self.pan_to_tilt.header.stamp = stamp
            self.tilt_to_camera.header.stamp = stamp

            self.tf_broadcaster.sendTransform([
                self.base_to_pan,
                self.pan_to_tilt,
                self.tilt_to_camera,
            ])


def main():
    rclpy.init()
    executor = rclpy.executors.MultiThreadedExecutor(4)
    node = PanTiltCtrlNode()
    executor.add_node(node)
    executor.spin()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
