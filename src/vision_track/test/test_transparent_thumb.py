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
"""Transparent gallery thumbnails (_make_thumb, Change 3a).

With a mask the thumb is person-only BGRA: alpha=255 inside the mask, 0 outside,
tight-cropped to the mask bbox (so smaller than the full bbox crop when the mask
is a sub-region). With mask=None the thumb is opaque BGRA (alpha all 255).

``_make_thumb`` is a plain module function (no self), so it's imported and called
directly. cv2/np only -- importorskip on the module mirrors repo test style.
"""
import numpy as np
import pytest

_make_thumb = pytest.importorskip(
    "vision_track.reid.appearance_manager")._make_thumb


# A 200x160 frame; the person bbox is a sub-region, and the mask is a sub-region
# of that bbox so we can prove the tight-crop shrinks past the bbox crop.
FH, FW = 200, 160
BX0, BY0, BX1, BY1 = 30, 20, 130, 180          # 100 wide x 160 tall bbox
# Mask: a centered block strictly inside the bbox -> tight crop must shrink.
MX0, MY0, MX1, MY1 = 55, 60, 105, 140          # 50 wide x 80 tall block


def _make_frame():
    # Deterministic gradient so RGB->BGR channel handling is exercised.
    frame = np.empty((FH, FW, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:FH, 0:FW]
    frame[..., 0] = (yy * 2) % 256
    frame[..., 1] = (xx * 3) % 256
    frame[..., 2] = ((yy + xx) * 5) % 256
    return frame


def _make_mask():
    m = np.zeros((FH, FW), dtype=np.uint8)
    m[MY0:MY1, MX0:MX1] = 1
    return m


def test_with_mask_is_bgra_with_correct_alpha():
    frame = _make_frame()
    mask = _make_mask()
    # max_h large enough that no resize happens -> alpha stays a clean 0/255 map
    # aligned 1:1 with the mask, so the per-pixel assertions are exact.
    thumb = _make_thumb(frame, (BX0, BY0, BX1, BY1), mask, max_h=1000)

    assert thumb is not None
    assert thumb.ndim == 3 and thumb.shape[2] == 4           # BGRA

    # Tight-cropped to the mask block size (50 wide x 80 tall), so the alpha is
    # fully opaque inside the block. mask is contiguous block -> all 255.
    assert thumb.shape[0] == MY1 - MY0
    assert thumb.shape[1] == MX1 - MX0
    alpha = thumb[..., 3]
    assert np.all(alpha == 255)                              # mask>0 everywhere


def test_with_mask_alpha_zero_outside_mask():
    # An L-shaped mask leaves a background notch inside the tight-crop bbox, so
    # we can assert alpha==0 where mask==0 AND alpha==255 where mask>0.
    frame = _make_frame()
    mask = np.zeros((FH, FW), dtype=np.uint8)
    # Left column + bottom row of a block -> top-right notch is background.
    mask[MY0:MY1, MX0:MX0 + 15] = 1          # left column
    mask[MY1 - 15:MY1, MX0:MX1] = 1          # bottom row

    thumb = _make_thumb(frame, (BX0, BY0, BX1, BY1), mask, max_h=1000)
    assert thumb is not None and thumb.shape[2] == 4

    alpha = thumb[..., 3]
    # Tight-crop bbox is [MY0:MY1, MX0:MX1] (the L's extent). Coords below are
    # relative to that origin.
    # Inside the left column -> alpha 255.
    assert alpha[5, 5] == 255
    # Top-right notch (far from column + row) -> background -> alpha 0.
    notch_y = 5
    notch_x = (MX1 - MX0) - 5
    assert alpha[notch_y, notch_x] == 0


def test_tight_crop_smaller_than_full_bbox_crop():
    frame = _make_frame()
    mask = _make_mask()
    thumb = _make_thumb(frame, (BX0, BY0, BX1, BY1), mask, max_h=1000)
    full_h, full_w = (BY1 - BY0), (BX1 - BX0)
    assert thumb.shape[0] < full_h
    assert thumb.shape[1] < full_w


def test_none_mask_is_opaque_bgra():
    frame = _make_frame()
    thumb = _make_thumb(frame, (BX0, BY0, BX1, BY1), None, max_h=1000)
    assert thumb is not None
    assert thumb.ndim == 3 and thumb.shape[2] == 4           # BGRA
    # No mask -> full bbox crop, alpha all opaque.
    assert thumb.shape[0] == BY1 - BY0
    assert thumb.shape[1] == BX1 - BX0
    assert np.all(thumb[..., 3] == 255)


def test_resize_preserves_4_channels_and_aspect():
    frame = _make_frame()
    mask = _make_mask()                                       # 50w x 80h block
    thumb = _make_thumb(frame, (BX0, BY0, BX1, BY1), mask, max_h=40)
    assert thumb.shape[2] == 4
    assert thumb.shape[0] == 40                               # resized to max_h
    assert abs(thumb.shape[1] - 25) <= 1                      # aspect 50/80*40
    # After INTER_NEAREST the alpha stays a hard 0/255 map (no intermediate
    # values), so the transparent edge is crisp.
    alpha_vals = np.unique(thumb[..., 3])
    assert set(alpha_vals.tolist()).issubset({0, 255})


def test_degenerate_bbox_returns_none():
    frame = _make_frame()
    assert _make_thumb(frame, (700, 500, 800, 600)) is None   # off-frame
    assert _make_thumb(frame, (50, 50, 50, 120)) is None      # zero width
