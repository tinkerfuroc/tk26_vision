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
"""Non-consuming frame read: _get_latest_data(consume=False).

The reseed service runs off the tracking loop; both share last_processed_seq.
The loop consumes nearly every frame, so a consuming read in the reseed would
almost always hit the frame-seq dedup and return False, wrongly rejecting the
reseed even though a frame is cached. consume=False returns the latest cached
frame regardless of seq and never advances last_processed_seq (no race), and
never returns False -- it still returns None when no frame/intrinsic exists.
"""
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

M = pytest.importorskip("vision_track.person_track_node")
PersonTrackNode = M.PersonTrackNode


def _fake_self(frame_seq, last_processed_seq,
               recent_sync_msg='sentinel', camera_intrinsic='intr'):
    if recent_sync_msg == 'sentinel':
        recent_sync_msg = (object(), object())  # (rgb_msg, depth_msg)
    return SimpleNamespace(
        recent_sync_msg=recent_sync_msg,
        frame_seq=frame_seq,
        last_processed_seq=last_processed_seq,
        lock_msg=threading.Lock(),
        lock_info=threading.Lock(),
        camera_intrinsic=camera_intrinsic,
        get_logger=lambda: MagicMock(),
    )


@pytest.fixture(autouse=True)
def _patch_decode(monkeypatch):
    # _get_latest_data calls module-level decode_color_msg(rgb_msg) -> (img, err).
    monkeypatch.setattr(M, 'decode_color_msg', lambda msg: (object(), None))


def test_consume_true_same_seq_returns_false():
    fake = _fake_self(frame_seq=5, last_processed_seq=5)
    result = PersonTrackNode._get_latest_data(fake, consume=True)
    assert result is False
    assert fake.last_processed_seq == 5  # unchanged


def test_consume_true_new_seq_returns_tuple_and_advances():
    fake = _fake_self(frame_seq=6, last_processed_seq=5)
    result = PersonTrackNode._get_latest_data(fake, consume=True)
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert fake.last_processed_seq == 6  # advanced


def test_consume_false_same_seq_returns_tuple_no_advance():
    # The race fix: the loop already consumed seq 5, but a non-consuming read
    # must still hand back the cached frame (NOT False) and leave the token put.
    fake = _fake_self(frame_seq=5, last_processed_seq=5)
    result = PersonTrackNode._get_latest_data(fake, consume=False)
    assert result is not False
    assert isinstance(result, tuple)
    assert len(result) == 4
    assert fake.last_processed_seq == 5  # unchanged


def test_no_msg_returns_none_both_modes():
    fake_consume = _fake_self(frame_seq=5, last_processed_seq=5,
                              recent_sync_msg=None)
    assert PersonTrackNode._get_latest_data(fake_consume, consume=True) is None
    fake_no_consume = _fake_self(frame_seq=5, last_processed_seq=5,
                                 recent_sync_msg=None)
    assert PersonTrackNode._get_latest_data(fake_no_consume, consume=False) is None


def test_no_intrinsic_returns_none():
    fake = _fake_self(frame_seq=6, last_processed_seq=5, camera_intrinsic=None)
    assert PersonTrackNode._get_latest_data(fake, consume=False) is None
