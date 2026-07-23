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
"""Focused regression tests for the person tracker CameraIntake adoption."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

M = pytest.importorskip("vision_track.person_track_node")
PersonTrackNode = M.PersonTrackNode


class FakeBundle:
    def __init__(self, seq=1, color=None):
        self.seq = seq
        self.color_msg = SimpleNamespace(
            header=SimpleNamespace(frame_id='camera'))
        self.depth_msg = object()
        self._color = (
            color
            if color is not None
            else np.zeros((2, 3, 3), dtype=np.uint8)
        )
        self._color.setflags(write=False)

    def color_bgr(self):
        return self._color


class FakeIntake:
    def __init__(self, bundle=None, intrinsic=None):
        self.bundle = bundle
        self.intrinsic = intrinsic
        self.latest_new_calls = []

    def latest(self):
        return self.bundle

    def latest_new(self, last_seq):
        self.latest_new_calls.append(last_seq)
        if self.bundle is None:
            return None
        if last_seq is not None and self.bundle.seq <= last_seq:
            return M.NO_NEW_FRAME
        return self.bundle

    def intrinsics(self):
        return self.intrinsic


def _fake_node(bundle=None, intrinsic=None, last_processed_seq=None):
    return SimpleNamespace(
        camera_intake=FakeIntake(bundle, intrinsic),
        last_processed_seq=last_processed_seq,
        get_logger=lambda: MagicMock(),
    )


def test_init_subscribers_preserves_qos_sync_and_reentrant_group(monkeypatch):
    created = {}

    class FakeCameraIntake:
        def __init__(self, node, cfg, callback_group=None, bridge=None):
            created.update(
                node=node,
                cfg=cfg,
                callback_group=callback_group,
                bridge=bridge,
            )

    monkeypatch.setattr(M, 'CameraIntake', FakeCameraIntake)
    node = SimpleNamespace(
        image_topic='/color',
        depth_topic='/depth',
        camera_info_topic='/info',
        bridge=object(),
        get_logger=lambda: MagicMock(),
    )

    PersonTrackNode._init_subscribers(node)

    cfg = created['cfg']
    assert node.camera_intake.__class__ is FakeCameraIntake
    assert cfg.color.qos_depth == 5
    assert cfg.depth.qos_depth == 5
    assert cfg.color.best_effort is True
    assert cfg.depth.best_effort is True
    assert cfg.camera_info.qos_depth == 10
    assert cfg.camera_info.best_effort is False
    assert cfg.sync_queue == 10
    assert cfg.sync_slop_s == pytest.approx(0.1)
    assert isinstance(created['callback_group'], M.ReentrantCallbackGroup)
    assert created['bridge'] is node.bridge


def test_consuming_read_uses_latest_new_and_advances_sequence():
    bundle = FakeBundle(seq=6)
    intrinsic = np.arange(9, dtype=np.float64)
    node = _fake_node(bundle, intrinsic, last_processed_seq=5)

    result = PersonTrackNode._get_latest_data(node, consume=True)

    assert result[0] is bundle._color
    assert result[1] is bundle.color_msg
    assert result[2] is bundle.depth_msg
    assert result[3] is intrinsic
    assert node.camera_intake.latest_new_calls == [5]
    assert node.last_processed_seq == 6
    assert result[0].flags.writeable is False


def test_consuming_read_returns_false_for_consumed_bundle():
    bundle = FakeBundle(seq=5)
    node = _fake_node(
        bundle,
        np.arange(9, dtype=np.float64),
        last_processed_seq=5,
    )

    assert PersonTrackNode._get_latest_data(node, consume=True) is False
    assert node.last_processed_seq == 5


def test_consume_happens_before_intrinsic_check():
    bundle = FakeBundle(seq=6)
    node = _fake_node(bundle, intrinsic=None, last_processed_seq=5)

    assert PersonTrackNode._get_latest_data(node, consume=True) is None
    assert node.last_processed_seq == 6


def test_non_consuming_read_never_advances_tracking_token():
    bundle = FakeBundle(seq=5)
    intrinsic = np.arange(9, dtype=np.float64)
    node = _fake_node(bundle, intrinsic, last_processed_seq=5)

    result = PersonTrackNode._get_latest_data(node, consume=False)

    assert isinstance(result, tuple)
    assert node.last_processed_seq == 5
    assert node.camera_intake.latest_new_calls == []


def test_missing_bundle_returns_none_in_both_modes():
    node = _fake_node(bundle=None, intrinsic=np.arange(9))

    assert PersonTrackNode._get_latest_data(node, consume=True) is None
    assert PersonTrackNode._get_latest_data(node, consume=False) is None


def test_goal_admission_requires_bundle_and_intrinsics():
    logger = MagicMock()
    node = SimpleNamespace(
        camera_intake=FakeIntake(
            FakeBundle(),
            np.arange(9, dtype=np.float64),
        ),
        lock_lifecycle=MagicMock(),
        tracking_active=False,
        get_logger=lambda: logger,
    )

    assert PersonTrackNode._goal_callback(node, object()) == M.GoalResponse.ACCEPT
    assert node.tracking_active is True

    node.camera_intake.intrinsic = None
    assert PersonTrackNode._goal_callback(node, object()) == M.GoalResponse.REJECT


def test_idle_preview_uses_independent_sequence_and_read_only_source():
    color = np.zeros((3, 4, 3), dtype=np.uint8)
    bundle = FakeBundle(seq=8, color=color)
    node = SimpleNamespace(
        tracking_active=False,
        debug_image_enabled=True,
        debug_image_pub=SimpleNamespace(get_subscription_count=lambda: 1),
        camera_intake=FakeIntake(bundle, np.arange(9)),
        _idle_last_seq=7,
        _publish_phase_debug_state=MagicMock(),
        _publish_raw_debug_image=MagicMock(),
        get_logger=lambda: MagicMock(),
    )

    PersonTrackNode._idle_debug_tick(node)

    node._publish_phase_debug_state.assert_called_once_with('idle')
    node._publish_raw_debug_image.assert_called_once_with(color)
    assert node._idle_last_seq == 8
    assert color.flags.writeable is False


def test_debug_overlay_copies_read_only_intake_image():
    color = np.zeros((40, 60, 3), dtype=np.uint8)
    color.setflags(write=False)
    node = SimpleNamespace(
        tracker=SimpleNamespace(last_lock_decision=None),
    )

    debug = PersonTrackNode._draw_debug_info(
        node,
        color,
        all_results=[],
        target_result=None,
        target_track_id=None,
    )

    assert debug.flags.writeable is True
    assert not np.shares_memory(debug, color)
    assert np.count_nonzero(color) == 0
