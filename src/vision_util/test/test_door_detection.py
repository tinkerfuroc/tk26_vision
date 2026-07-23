import asyncio
import importlib
import sys
import unittest

import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo
from tinker_vision_msgs_26.srv import DoorDetection

import vision_util


# test_camera_intake loads this module with ROS fakes during package
# collection.
sys.modules.pop('vision_util.camera_intake', None)
vision_util.__dict__.pop('camera_intake', None)
importlib.import_module('vision_util.camera_intake')
sys.modules.pop('vision_util.door_detection', None)
vision_util.__dict__.pop('door_detection', None)
DoorDetectionService = importlib.import_module(
    'vision_util.door_detection'
).DoorDetectionService


class TestDoorDetectionService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.node = DoorDetectionService()
        self.bridge = CvBridge()

    def tearDown(self):
        self.node.destroy_node()

    @staticmethod
    def _camera_info(w, h, fx=500.0, fy=500.0):
        info = CameraInfo()
        info.width = w
        info.height = h
        cx, cy = w / 2.0, h / 2.0
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        return info

    def _publish_depth(self, depth, encoding='16UC1'):
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding=encoding)
        self.node.camera_intake._depth_callback(depth_msg)
        return self.node.camera_intake.latest()

    def _request(self, camera='orbbec'):
        request = DoorDetection.Request()
        request.camera = camera
        return asyncio.run(
            self.node.door_detection_srv_callback(
                request,
                DoorDetection.Response(),
            )
        )

    def test_uses_unsynchronized_reliable_depth_and_info_intake(self):
        intake = self.node.camera_intake
        cfg = intake.cfg

        self.assertIsNone(cfg.color)
        self.assertIsNone(intake._sync)
        self.assertEqual(cfg.depth.topic, '/camera/depth/image_raw')
        self.assertFalse(cfg.depth.best_effort)
        self.assertEqual(cfg.depth.qos_depth, 10)
        self.assertEqual(
            cfg.camera_info.topic,
            '/camera/color/camera_info',
        )
        self.assertFalse(cfg.camera_info.best_effort)
        self.assertEqual(cfg.camera_info.qos_depth, 10)

        w, h = 48, 32
        stale_info = self._camera_info(w, h, fx=0.0)
        self.node.camera_intake._camera_info_callback(stale_info)
        bundle = self._publish_depth(
            np.full((h, w), 1000, dtype=np.uint16)
        )
        latest_info = self._camera_info(w, h)
        self.node.camera_intake._camera_info_callback(latest_info)

        self.assertEqual(bundle.K[0], 0.0)
        self.assertEqual(intake.intrinsics()[0], 500.0)
        self.assertFalse(bundle.depth_m().flags.writeable)
        self.assertFalse(intake.intrinsics().flags.writeable)

        response = self._request(camera='zed')
        self.assertEqual(response.status, 0)
        self.assertEqual(response.error_msg, '')
        self.assertEqual(response.is_open, 0)

    def test_realsense_substring_and_missing_data_keep_failure_semantics(self):
        response = self._request(camera='front_realsense_depth')
        self.assertEqual(response.status, 1)
        self.assertEqual(
            response.error_msg,
            'Only orbbec camera is supported.',
        )

        response = self._request(camera='other')
        self.assertEqual(response.status, 1)
        self.assertEqual(
            response.error_msg,
            'No camera data or intrinsic for other.',
        )

    def test_center_window_requires_more_than_five_strictly_valid_pixels(self):
        w = h = 40
        self.node.camera_intake._camera_info_callback(
            self._camera_info(w, h)
        )
        depth = np.zeros((h, w), dtype=np.uint16)
        depth[10:30, 10:30] = 1
        depth[10, 10:15] = 1000

        self._publish_depth(depth)
        self.assertEqual(self._request().is_open, 1)

        depth[10, 15] = 1000
        self._publish_depth(depth)
        self.assertEqual(self._request().is_open, 0)

    def test_weighted_average_preserves_1_5_metre_epsilon_behavior(self):
        w = h = 40
        self.node.camera_intake._camera_info_callback(
            self._camera_info(w, h)
        )
        depth = np.zeros((h, w), dtype=np.uint16)
        depth[10, 10:16] = 1500

        self._publish_depth(depth)
        response = self._request()

        self.assertEqual(response.status, 0)
        self.assertEqual(response.is_open, 0)

        depth[10, 10:16] = 1501
        self._publish_depth(depth)
        self.assertEqual(self._request().is_open, 1)

    def test_decode_error_returns_failure_and_discards_bad_bundle(self):
        w = h = 40
        self.node.camera_intake._camera_info_callback(
            self._camera_info(w, h)
        )
        self._publish_depth(
            np.zeros((h, w), dtype=np.uint8),
            encoding='8UC1',
        )

        response = self._request(camera='orbbec')

        self.assertEqual(response.status, 1)
        self.assertEqual(
            response.error_msg,
            'No camera data or intrinsic for orbbec.',
        )
        self.assertIsNone(self.node.camera_intake.latest())


if __name__ == '__main__':
    unittest.main()
