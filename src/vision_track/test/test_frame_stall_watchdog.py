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
"""Frame-starvation watchdog.

The tracker's only frame source is the RGB+depth ApproximateTimeSynchronizer.
If it stops emitting matched pairs, frame_seq freezes and the tracking loop's
``data is False`` branch must NOT busy-wait forever with a frozen dashboard and
no recovery (the diagnosed "stopped getting new camera frames" freeze). The
watchdog classifies the gap and, past thresholds, keeps the dashboard alive and
re-uses the loss/recovery FSM (forever-hold + wave/reseed).
"""
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

PersonTrackNode = pytest.importorskip(
    "vision_track.person_track_node").PersonTrackNode

_classify = PersonTrackNode._classify_frame_stall
_handle = PersonTrackNode._handle_frame_stall


def _cfg(warn=0.5, lost=1.5):
    return SimpleNamespace(frame_stall_warn_sec=warn, frame_stall_lost_sec=lost)


# --- classifier: thresholds must sit above the 30 Hz inter-frame gap ---

def test_classify_ok_below_warn():
    assert _classify(_cfg(), 0.033) == 'ok'   # normal inter-frame gap @ 30 Hz
    assert _classify(_cfg(), 0.49) == 'ok'


def test_classify_warn_band():
    assert _classify(_cfg(), 0.5) == 'warn'
    assert _classify(_cfg(), 1.49) == 'warn'


def test_classify_lost_at_and_above_threshold():
    assert _classify(_cfg(), 1.5) == 'lost'
    assert _classify(_cfg(), 10.0) == 'lost'


# --- handler ---

def _fake_node(warn=0.5, lost=1.5, have_cache=True, abort=False):
    node = SimpleNamespace(
        frame_stall_warn_sec=warn, frame_stall_lost_sec=lost,
        _last_tracked_rgb=object() if have_cache else None,
        _last_tracked_msg=object() if have_cache else None,
    )
    node._classify_frame_stall = types.MethodType(
        PersonTrackNode._classify_frame_stall, node)
    node.get_logger = MagicMock()
    node._publish_phase_debug_state = MagicMock()
    node._publish_raw_debug_image = MagicMock()
    node._handle_lost_frame = MagicMock(return_value=abort)
    return node


def test_ok_gap_is_silent():
    node = _fake_node()
    out = _handle(node, 0.1, 100.0, MagicMock(), MagicMock(), {}, MagicMock())
    assert out is False
    node.get_logger.assert_not_called()
    node._publish_phase_debug_state.assert_not_called()
    node._publish_raw_debug_image.assert_not_called()
    node._handle_lost_frame.assert_not_called()


def test_warn_keeps_dashboard_alive_without_loss_fsm():
    node = _fake_node()
    out = _handle(node, 0.7, 100.0, MagicMock(), MagicMock(), {}, MagicMock())
    assert out is False
    node.get_logger().warn.assert_called_once()
    node._publish_phase_debug_state.assert_called_once_with('camera_stalled')
    node._publish_raw_debug_image.assert_called_once()
    node._handle_lost_frame.assert_not_called()       # warn band: no loss yet


def test_lost_engages_loss_fsm_with_last_good_frame():
    node = _fake_node()
    fb, gh, res = MagicMock(), MagicMock(), MagicMock()
    out = _handle(node, 2.0, 100.0, fb, gh, {'k': 1}, res)
    assert out is False                               # forever-hold: no abort
    node._publish_phase_debug_state.assert_called_once_with('camera_stalled')
    node._handle_lost_frame.assert_called_once_with(
        100.0, node._last_tracked_rgb, node._last_tracked_msg, fb, gh, {'k': 1}, res)


def test_lost_abort_propagates():
    node = _fake_node(abort=True)                      # bounded hold expired
    assert _handle(node, 2.0, 100.0, MagicMock(), MagicMock(), {}, MagicMock()) is True


def test_lost_without_cached_frame_skips_fsm():
    node = _fake_node(have_cache=False)                # stall before first track
    out = _handle(node, 2.0, 100.0, MagicMock(), MagicMock(), {}, MagicMock())
    assert out is False
    node.get_logger().warn.assert_called_once()        # still warns + keeps alive
    node._publish_phase_debug_state.assert_called_once_with('camera_stalled')
    node._publish_raw_debug_image.assert_not_called()  # nothing cached to re-emit
    node._handle_lost_frame.assert_not_called()
