#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from tinker_vision_msgs_26.action import DetectWaving
import time


class WavingDetectionClient(Node):
    def __init__(self):
        super().__init__('waving_detection_client')
        self.client = ActionClient(
            self, DetectWaving, '/detect_waving_persons')
        self.detecting = True

        # 等待 action 可用
        while not self.client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Action not available, waiting...')
        self.get_logger().info('Action available! Starting continuous detection...')

    def send_request(self, threshold_meters=5.0, target_frame=""):
        goal = DetectWaving.Goal()
        goal.threshold_meters = threshold_meters
        goal.target_frame = target_frame

        send_future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('✗ Waving detection goal rejected')
            return None

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        response = result_future.result().result

        if response.status == 0:
            self.get_logger().info(f'✓ {response.error_msg}')
            for i, person in enumerate(response.waving_persons):
                self.get_logger().info(
                    f'  Person {i+1}: x={person.point.x:.3f}, y={person.point.y:.3f}, z={person.point.z:.3f} '
                    f'(frame: {person.header.frame_id})'
                )
        elif response.status == 1:
            self.get_logger().debug(f'⚠ {response.error_msg}')
        else:
            self.get_logger().error(f'✗ Error: {response.error_msg}')

        return response

    def continuous_detect(self, threshold_meters=5.0, target_frame="", interval=1.0):
        """
        持续检测挥手人员
        :param threshold_meters: 检测距离阈值（米）
        :param target_frame: 返回坐标系
        :param interval: 检测间隔（秒）
        """
        self.get_logger().info(
            f'Starting continuous detection with interval={interval}s, '
            f'threshold_meters={threshold_meters}, target_frame="{target_frame}"'
        )

        try:
            while rclpy.ok() and self.detecting:
                self.send_request(threshold_meters=threshold_meters, target_frame=target_frame)
                time.sleep(interval)
        except KeyboardInterrupt:
            self.get_logger().info('Detection stopped by user')
            self.detecting = False


def main(args=None):
    rclpy.init(args=args)

    client = WavingDetectionClient()

    # 持续检测，每1秒检测一次
    # 参数1: threshold_meters - 检测距离阈值（米），≤0 表示不限制
    # 参数2: target_frame - 返回结果的目标坐标系，空字符串表示使用原始相机坐标系
    # 参数3: interval - 检测间隔（秒）
    client.continuous_detect(threshold_meters=5.0, target_frame="", interval=1.0)

    rclpy.shutdown()


if __name__ == '__main__':
    main()
