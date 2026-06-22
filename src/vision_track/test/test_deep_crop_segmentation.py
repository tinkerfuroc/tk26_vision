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
tight-crop to the mask bbox, resize to OSNet's fixed input size (_REID_W x
_REID_H = 128 x 256), then soft-attenuate out-of-mask pixels toward a blurred
copy AT THAT FIXED SIZE. mask_crop None or all-zero is a passthrough.

Perf-regression note: the blur/attenuation moved off the full-res crop onto the
fixed 128x256 resize (constant ~0.3 ms vs area-scaling). The output is therefore
always shape (256, 128, 3) for a real mask, not the dilated-bbox size. Downstream
``_stack_crops`` resizes 128x256 -> 128x256 (a no-op), so this stays equivalent
to the old "blur full-res then resize".

The helper only touches cv2/np and its two args (no self attributes), so it is
called UNBOUND on object() as self -- this avoids constructing AppearanceExtractor
(which loads heavy torch models).

An L-shaped mask is used for the "real mask" cases: its bbox spans the full L
extent, while the notch of the L stays background even after the 2-px dilation.
After the resize to 128x256, the deep-interior of the column maps to clearly-in
pixels and the notch corner maps to clearly-out pixels, giving a robust
interior-vs-exterior assertion against the resized source.
"""
import cv2
import numpy as np
import pytest

_reid = pytest.importorskip("vision_track.reid.reid")
AppearanceExtractor = _reid.AppearanceExtractor
_seg = AppearanceExtractor._segment_crop_for_reid

from vision_track.reid.reid_backbone import _REID_H, _REID_W  # noqa: E402


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


def _resized_source(crop):
    """Reproduce the helper's tight-crop + resize of the *source* pixels.

    Mirrors _segment_crop_for_reid up to (but not including) the blur, so a
    test can compare the helper's output against the resized source per-pixel.
    Returns (resized_crop[256,128,3], resized_dilated_mask[256,128]).
    """
    h, w = crop.shape[:2]
    mask = _make_l_mask(h, w)
    m = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(m, kernel, iterations=2)
    ys, xs = np.where(dil > 0)
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    crop_tc = crop[y1:y2, x1:x2]
    dil_tc = dil[y1:y2, x1:x2]
    resized = cv2.resize(crop_tc, (_REID_W, _REID_H))
    dil_rs = cv2.resize(dil_tc, (_REID_W, _REID_H),
                        interpolation=cv2.INTER_NEAREST)
    return resized, dil_rs


def test_real_mask_output_is_osnet_input_size():
    crop = _make_crop()
    h, w = crop.shape[:2]
    mask = _make_l_mask(h, w)
    out = _seg(object(), crop, mask)

    # The blur/attenuation now happen at OSNet's fixed input size, so the output
    # is always (H, W, 3) = (256, 128, 3) regardless of the person's pixel size.
    assert out.shape == (_REID_H, _REID_W, 3)
    assert out.shape == (256, 128, 3)


def test_interior_unchanged_exterior_attenuated():
    crop = _make_crop()
    h, w = crop.shape[:2]
    mask = _make_l_mask(h, w)
    out = _seg(object(), crop, mask)

    resized, dil_rs = _resized_source(crop)

    # Robust to the resize: every IN-mask pixel must be byte-identical to the
    # resized source (the helper only touches background pixels), and at least
    # one IN-mask pixel must exist to make that assertion meaningful.
    in_mask = dil_rs != 0
    bg_mask = dil_rs == 0
    assert np.any(in_mask)
    assert np.any(bg_mask)
    np.testing.assert_array_equal(out[in_mask], resized[in_mask])

    # The set of background pixels must DIFFER from the resized source (they were
    # attenuated toward the blurred copy). Robust: assert at least one bg pixel
    # changed rather than requiring every single one to (blur of a near-uniform
    # region could leave a pixel coincidentally unchanged).
    bg_changed = np.any(out[bg_mask] != resized[bg_mask])
    assert bg_changed
