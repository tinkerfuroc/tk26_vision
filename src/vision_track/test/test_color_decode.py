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
"""decode_color_msg: normalize rgb8/bgr8 color Image msgs to BGR ndarrays."""
from types import SimpleNamespace

from vision_track.core.color_decode import decode_color_msg


def _msg(encoding, data, w=2, h=1, step=None):
    return SimpleNamespace(encoding=encoding, width=w, height=h,
                           step=w * 3 if step is None else step,
                           data=bytes(data))


def test_bgr8_passthrough():
    img, err = decode_color_msg(_msg("bgr8", [1, 2, 3, 4, 5, 6]))
    assert err is None
    assert img.shape == (1, 2, 3)
    assert img[0, 0].tolist() == [1, 2, 3]          # untouched


def test_rgb8_channel_swap():
    img, err = decode_color_msg(_msg("rgb8", [10, 20, 30, 40, 50, 60]))
    assert err is None
    assert img[0, 0].tolist() == [30, 20, 10]       # R<->B swapped to BGR
    assert img[0, 1].tolist() == [60, 50, 40]


def test_unsupported_encoding_rejected():
    img, err = decode_color_msg(_msg("yuv422", [0] * 6))
    assert img is None and "yuv422" in err


def test_padded_step_rejected():
    img, err = decode_color_msg(_msg("rgb8", [0] * 8, step=8))
    assert img is None and "step" in err


def test_short_buffer_rejected():
    img, err = decode_color_msg(_msg("bgr8", [1, 2, 3]))
    assert img is None and err
