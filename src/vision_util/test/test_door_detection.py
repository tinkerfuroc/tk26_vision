import unittest

import numpy as np
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

from vision_util.door_detection import DoorDetectionService


class TestDoorDetectionOrbbecDepth(unittest.TestCase):
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

    def _camera_info(self, w, h, fx=500.0, fy=500.0):
        info = CameraInfo()
        info.width = w
        info.height = h
        cx, cy = w / 2.0, h / 2.0
        info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        return info

    def test_depth_array_matches_input_resolution_not_hardcoded_720x1280(self):
        # Regression: img_orbbec_process used to hardcode h, w = 720, 1280
        # and silently misalign at any other resolution.
        w, h = 1920, 1080
        depth = np.full((h, w), 2000, dtype=np.uint16)  # 2.0 m in mm
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        intrinsic = self._camera_info(w, h)

        _, depth_img, validmask = self.node.img_orbbec_process(
            None, depth_msg, intrinsic
        )

        self.assertEqual(depth_img.shape, (h, w, 3))
        self.assertEqual(validmask.shape, (h, w))

    def test_center_window_reads_live_depth_at_any_resolution(self):
        w, h = 1920, 1080
        depth = np.full((h, w), 1000, dtype=np.uint16)  # 1.0 m
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding='16UC1')
        intrinsic = self._camera_info(w, h)

        _, depth_img, validmask = self.node.img_orbbec_process(
            None, depth_msg, intrinsic
        )

        center_h, center_w = depth_img.shape[0] // 2, depth_img.shape[1] // 2
        crop = depth_img[center_h - 10:center_h + 10,
                         center_w - 10:center_w + 10, 2]
        valid_crop = validmask[center_h - 10:center_h + 10,
                               center_w - 10:center_w + 10]

        self.assertGreater(int(valid_crop.sum()), 5)
        self.assertAlmostEqual(
            float((crop * valid_crop).sum() / valid_crop.sum()), 1.0,
            places=2,
        )


if __name__ == '__main__':
    unittest.main()
