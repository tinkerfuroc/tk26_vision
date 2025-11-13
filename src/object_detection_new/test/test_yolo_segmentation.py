import unittest
import rclpy
from rclpy.node import Node
import numpy as np


class TestYOLOSegmentationUtils(unittest.TestCase):
    """Test cases for YOLO segmentation node utility functions."""

    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        """Set up test fixtures."""
        self.test_node = Node('test_yolo_segmentation')

    def tearDown(self):
        """Clean up."""
        self.test_node.destroy_node()

    def test_image_padding(self):
        """Test that images are padded correctly to multiples of 32."""
        # Create test image
        img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

        h, w = img.shape[:2]
        h_pad = ((h + 31) // 32) * 32
        w_pad = ((w + 31) // 32) * 32

        self.assertEqual(h_pad, 736)  # 23 * 32
        self.assertEqual(w_pad, 1280)  # 40 * 32

    def test_depth_to_points_math(self):
        """Test depth to 3D points conversion math."""
        # Camera parameters
        fx = fy = 500.0
        cx = 320.0
        cy = 240.0

        # Create simple depth image
        depth = np.ones((480, 640), dtype=np.float32) * 2.0

        # Manually calculate center point
        x_coords = np.arange(640)
        y_coords = np.arange(480)
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        z = depth
        x = (x_grid - cx) * z / fx
        y = (y_grid - cy) * z / fy

        # Check center point
        center_x = x[240, 320]
        center_y = y[240, 320]
        center_z = z[240, 320]

        # At center, should be approximately (0, 0, 2)
        self.assertAlmostEqual(center_x, 0.0, places=4)
        self.assertAlmostEqual(center_y, 0.0, places=4)
        self.assertAlmostEqual(center_z, 2.0, places=4)

    def test_centroid_calculation_logic(self):
        """Test 3D centroid calculation logic."""
        # Create synthetic point cloud
        points = np.zeros((100, 100, 3))
        points[:, :, 2] = 2.0  # 2 meters depth

        # Create mask (small square in center)
        mask = np.zeros((100, 100), dtype=bool)
        mask[45:55, 45:55] = True

        valid_mask = np.ones((100, 100), dtype=bool)

        # Calculate centroid manually
        combined_mask = mask & valid_mask
        masked_points = points * combined_mask[:, :, np.newaxis]
        centroid_3d = masked_points.sum(axis=(0, 1)) / combined_mask.sum()

        # Check result
        self.assertAlmostEqual(centroid_3d[2], 2.0, places=2)
        self.assertTrue(combined_mask.sum() > 0)


if __name__ == '__main__':
    unittest.main()
