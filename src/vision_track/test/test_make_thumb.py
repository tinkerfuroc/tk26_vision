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
"""_make_thumb: clamped, aspect-preserving gallery thumbnails."""
import numpy as np

from vision_track.reid.appearance_manager import _make_thumb


def test_resizes_tall_crop_to_max_height():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = _make_thumb(frame, (100, 50, 200, 450))      # 100x400 crop
    assert t is not None and t.shape[0] == 192
    assert abs(t.shape[1] - 48) <= 1                 # aspect preserved


def test_small_crop_kept_as_is():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = _make_thumb(frame, (10, 10, 60, 110))        # 50x100, under max
    assert t.shape[:2] == (100, 50)


def test_degenerate_bbox_returns_none():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert _make_thumb(frame, (700, 500, 800, 600)) is None   # fully off-frame
    assert _make_thumb(frame, (50, 50, 50, 120)) is None      # zero width


def test_straddling_bbox_clamped_to_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    t = _make_thumb(frame, (600, -20, 700, 150))     # straddles right + top edges
    assert t is not None
    assert t.shape[:2] == (150, 40)                  # clamped to (0..150, 600..640)
