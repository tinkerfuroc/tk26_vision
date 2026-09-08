# Copyright 2026 Tinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for ``vision_logging._apply_mask_overlay``.

Regression coverage for the uint8-0/1-mask integer-fancy-index leak: blending
with ``overlay[mask]`` where ``mask`` is a uint8 0/1 array indexes along axis 0
((H, W, W, 3), ~14 GB float32 at 720p) instead of boolean masking. The helper
must coerce the mask to boolean and shape-guard before indexing.
"""
import unittest

import numpy as np

from vision_util.vision_logging import _apply_mask_overlay


# Blend constants mirroring the helper defaults / the original inline blend.
_COLOR = np.array([0, 160, 255], dtype=np.float32)
_ALPHA = 0.5


def _expected_blend(orig_pixel):
    """Reference blend of a single original BGR pixel against the orange color."""
    return (orig_pixel.astype(np.float32) * (1.0 - _ALPHA)
            + _COLOR * _ALPHA).astype(np.uint8)


class TestApplyMaskOverlay(unittest.TestCase):
    def _make_overlay(self):
        # Distinct per-pixel values so an axis-0 fancy-index would be detectable.
        return (np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)).copy()

    def test_uint8_zero_one_mask_blends_only_masked_pixels(self):
        overlay = self._make_overlay()
        before = overlay.copy()

        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1, 2] = 1
        mask[3, 0] = 1

        result = _apply_mask_overlay(overlay, mask)

        # Shape preserved (no (H, W, W, 3) explosion, no axis-0 mangling).
        self.assertEqual(result.shape, (4, 4, 3))
        self.assertEqual(overlay.shape, (4, 4, 3))

        # Masked pixels carry the blend.
        np.testing.assert_array_equal(overlay[1, 2], _expected_blend(before[1, 2]))
        np.testing.assert_array_equal(overlay[3, 0], _expected_blend(before[3, 0]))

        # Every unmasked pixel is untouched.
        unmasked = np.ones((4, 4), dtype=bool)
        unmasked[1, 2] = False
        unmasked[3, 0] = False
        np.testing.assert_array_equal(overlay[unmasked], before[unmasked])

    def test_boolean_mask_matches_uint8_result(self):
        overlay_u8 = self._make_overlay()
        overlay_bool = self._make_overlay()

        u8 = np.zeros((4, 4), dtype=np.uint8)
        u8[0, 0] = 1
        u8[2, 3] = 1
        boolean = u8.astype(bool)

        _apply_mask_overlay(overlay_u8, u8)
        _apply_mask_overlay(overlay_bool, boolean)

        np.testing.assert_array_equal(overlay_u8, overlay_bool)

    def test_no_fancy_index_explosion_on_full_size_mask(self):
        # H != W so an axis-0 integer fancy-index would produce a non-(H,W,3)
        # shape; assert the result stays (H, W, 3) and selection count is right.
        overlay = np.zeros((6, 9, 3), dtype=np.uint8)
        mask = np.zeros((6, 9), dtype=np.uint8)
        mask[2:4, 1:5] = 1  # 2 * 4 = 8 set pixels

        result = _apply_mask_overlay(overlay, mask)

        self.assertEqual(result.shape, (6, 9, 3))
        bool_mask = mask.astype(bool)
        self.assertEqual(overlay[bool_mask].shape[0], int(mask.sum()))

    def test_shape_mismatch_is_noop(self):
        overlay = self._make_overlay()
        before = overlay.copy()

        mask = np.ones((8, 8), dtype=np.uint8)  # wrong H, W

        result = _apply_mask_overlay(overlay, mask)

        self.assertIs(result, overlay)
        np.testing.assert_array_equal(overlay, before)

    def test_empty_mask_is_noop(self):
        overlay = self._make_overlay()
        before = overlay.copy()

        mask = np.zeros((4, 4), dtype=np.uint8)

        result = _apply_mask_overlay(overlay, mask)

        self.assertIs(result, overlay)
        np.testing.assert_array_equal(overlay, before)

    def test_none_mask_is_noop(self):
        overlay = self._make_overlay()
        before = overlay.copy()

        result = _apply_mask_overlay(overlay, None)

        self.assertIs(result, overlay)
        np.testing.assert_array_equal(overlay, before)


if __name__ == '__main__':
    unittest.main()
