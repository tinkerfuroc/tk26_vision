import unittest

import numpy as np

from vision_util.depth_reproject import decode_depth_metres, depth_image_to_points


class TestDecodeDepthMetres(unittest.TestCase):
    def test_uint16_millimetres_converts_to_metres(self):
        arr = np.array([[2000, 500], [0, 10000]], dtype=np.uint16)
        result = decode_depth_metres(arr)
        np.testing.assert_allclose(result, [[2.0, 0.5], [0.0, 10.0]])
        self.assertEqual(result.dtype, np.float32)

    def test_float32_metres_passes_through_unchanged(self):
        arr = np.array([[1.5, 2.5]], dtype=np.float32)
        result = decode_depth_metres(arr)
        np.testing.assert_allclose(result, arr)

    def test_unsupported_dtype_raises(self):
        arr = np.zeros((2, 2), dtype=np.uint8)
        with self.assertRaises(ValueError):
            decode_depth_metres(arr)


class TestDepthImageToPoints(unittest.TestCase):
    def test_center_pixel_back_projects_to_zero_xy(self):
        fx = fy = 500.0
        cx, cy = 320.0, 240.0
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        points = depth_image_to_points(depth, k)

        self.assertAlmostEqual(float(points[240, 320, 0]), 0.0, places=4)
        self.assertAlmostEqual(float(points[240, 320, 1]), 0.0, places=4)
        self.assertAlmostEqual(float(points[240, 320, 2]), 2.0, places=4)

    def test_off_center_pixel_matches_pinhole_formula(self):
        fx = fy = 500.0
        cx, cy = 320.0, 240.0
        depth = np.full((480, 640), 4.0, dtype=np.float32)
        k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        points = depth_image_to_points(depth, k)

        # Pixel (row=240, col=420): x = (420-320)*4/500 = 0.8
        self.assertAlmostEqual(float(points[240, 420, 0]), 0.8, places=4)
        # Pixel (row=340, col=320): y = (340-240)*4/500 = 0.8
        self.assertAlmostEqual(float(points[340, 320, 1]), 0.8, places=4)

    def test_output_shape_matches_input_regardless_of_resolution(self):
        # Regression target: object_seg_yolo.py._pointcloud_to_array used to
        # hardcode (720, 1280) and silently clip/crash at any other size.
        fx = fy = 900.0
        cx, cy = 960.0, 540.0
        depth = np.full((1080, 1920), 3.0, dtype=np.float32)
        k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]

        points = depth_image_to_points(depth, k)

        self.assertEqual(points.shape, (1080, 1920, 3))


if __name__ == '__main__':
    unittest.main()
