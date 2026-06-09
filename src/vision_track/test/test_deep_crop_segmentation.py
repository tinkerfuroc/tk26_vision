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
"""Deep-crop person segmentation helper (_segment_crop_for_reid).

The deep OSNet embedding must be fed a person-only crop: dilate the mask,
tight-crop to the mask bbox, soft-attenuate out-of-mask pixels. mask_crop None
or all-zero is a passthrough.

The helper only touches cv2/np and its two args (no self attributes), so it is
called UNBOUND on object() as self -- this avoids constructing AppearanceExtractor
(which loads heavy torch models).

An L-shaped mask is used for the "real mask" cases: its bbox spans the full L
extent (so the dilated-bbox size is predictable), while the notch of the L stays
background even after the 2-px dilation, giving a clearly-interior pixel AND a
clearly-exterior pixel both inside the tight crop.
"""
import numpy as np
import pytest

AppearanceExtractor = pytest.importorskip(
    "vision_track.reid.reid").AppearanceExtractor
_seg = AppearanceExtractor._segment_crop_for_reid


def _make_crop():
    # 100x60x3 uint8 with a deterministic gradient so every pixel is distinct.
    h, w = 100, 60
    crop = np.empty((h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    crop[..., 0] = (yy * 2) % 256
    crop[..., 1] = (xx * 4) % 256
    crop[..., 2] = ((yy + xx) * 3) % 256
    return crop


# L-shaped mask: a tall left column + a wide bottom row, leaving the top-right
# region as a large background notch inside the bounding box.
# Mask bbox = [Y0:Y1, X0:X1]; notch (background) is the top-right block.
Y0, Y1, X0, X1 = 30, 70, 20, 50
COL_X1 = 30   # left column spans columns [X0:COL_X1]
ROW_Y0 = 55   # bottom row spans rows [ROW_Y0:Y1]


def _make_l_mask(h, w):
    m = np.zeros((h, w), dtype=np.uint8)
    m[Y0:Y1, X0:COL_X1] = 1     # left column
    m[ROW_Y0:Y1, X0:X1] = 1     # bottom row
    return m


def test_none_mask_returns_same_array_identity():
    crop = _make_crop()
    out = _seg(object(), crop, None)
    assert out is crop


def test_all_zero_mask_returns_crop_unchanged():
    crop = _make_crop()
    h, w = crop.shape[:2]
    zero_mask = np.zeros((h, w), dtype=np.uint8)
    out = _seg(object(), crop, zero_mask)
    assert out is crop


def test_real_mask_tight_crops_to_dilated_bbox():
    crop = _make_crop()
    h, w = crop.shape[:2]
    mask = _make_l_mask(h, w)
    out = _seg(object(), crop, mask)

    # Dilation grows the mask bbox by 2 px on each side (iters=2, 3x3 kernel),
    # clamped to the crop bounds. The L bbox is [Y0:Y1, X0:X1], well inside the
    # crop, so no clamping occurs here.
    ey0, ey1 = max(0, Y0 - 2), min(h, Y1 + 2)
    ex0, ex1 = max(0, X0 - 2), min(w, X1 + 2)
    assert out.shape[0] == ey1 - ey0
    assert out.shape[1] == ex1 - ex0
    # Strictly smaller than the full crop (the mask is a sub-region).
    assert out.shape[0] < h
    assert out.shape[1] < w


def test_interior_unchanged_exterior_attenuated():
    crop = _make_crop()
    h, w = crop.shape[:2]
    mask = _make_l_mask(h, w)
    out = _seg(object(), crop, mask)

    # The tight-crop's origin in the original crop coords (dilated bbox).
    ey0, ex0 = max(0, Y0 - 2), max(0, X0 - 2)

    # A pixel clearly INSIDE the mask (deep in the left column, > 2 px from any
    # edge of the column): unchanged vs its source pixel.
    iy, ix = 40, 24  # inside [Y0:Y1, X0:COL_X1], away from the column borders
    src_in = crop[iy, ix]
    got_in = out[iy - ey0, ix - ex0]
    np.testing.assert_array_equal(got_in, src_in)

    # A pixel clearly OUTSIDE the mask: the top-right notch corner, far (> 2 px)
    # from both the left column and the bottom row, so the 2-px dilation cannot
    # reach it -> attenuated toward the blurred copy -> differs from its source.
    oy, ox = Y0 + 2, X1 - 2  # top-right of the bbox, inside the notch
    src_out = crop[oy, ox]
    got_out = out[oy - ey0, ox - ex0]
    assert not np.array_equal(got_out, src_out)
