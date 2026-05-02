import unittest

import numpy as np

from vision_util.mask_utils import largest_connected_component_in_bbox


class TestMaskUtils(unittest.TestCase):
    def test_largest_connected_component_is_bbox_scoped(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[1:10, 1:10] = True
        mask[14:17, 14:17] = True

        result = largest_connected_component_in_bbox(mask, (12, 12, 18, 18))

        self.assertFalse(result[1:10, 1:10].any())
        self.assertTrue(result[14:17, 14:17].all())

    def test_largest_connected_component_empty_bbox_overlap(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[1:10, 1:10] = True

        result = largest_connected_component_in_bbox(mask, (12, 12, 18, 18))

        self.assertFalse(result.any())


if __name__ == '__main__':
    unittest.main()
