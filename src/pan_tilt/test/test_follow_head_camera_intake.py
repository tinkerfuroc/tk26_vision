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
"""Focused regressions for follow-head camera intake migration."""

import collections
import math

import pytest

import pan_tilt.follow_head as follow_head


class _Logger:
    def debug(self, _message):
        pass

    def warn(self, _message):
        pass


class _Intake:
    def __init__(self, bundle):
        self.bundle = bundle
        self.last_seq_args = []

    def latest_new(self, last_seq):
        self.last_seq_args.append(last_seq)
        if self.bundle is None:
            return None
        if last_seq is not None and self.bundle.seq <= last_seq:
            return follow_head.NO_NEW_FRAME
        return self.bundle


class _Bundle:
    def __init__(self, seq):
        self.seq = seq
        self.K = object()
        self.color_decode_calls = 0
        self.depth_decode_calls = 0

    def color_bgr(self):
        self.color_decode_calls += 1
        raise AssertionError('throttled frame must not decode color')

    def depth_m(self):
        self.depth_decode_calls += 1
        raise AssertionError('throttled frame must not decode depth')


def _logic_node(bundle):
    node = object.__new__(follow_head.FollowHeadNode)
    node.camera_intake = _Intake(bundle)
    node.last_used_seq = None
    node._last_detection_time = 9.9
    node.min_detection_interval_sec = 0.2
    node._perf_window_start = 10.0
    node._perf_window_sec = 2.0
    node._perf_sync_count = 0
    node._perf_last_sync_seq = 0
    node._perf_logic_count = 0
    node._perf_yolo_count = 0
    node._perf_sum = {
        'pc_parse': 0.0,
        'yolo': 0.0,
        'extract': 0.0,
        'total': 0.0,
    }
    node._perf_early = collections.Counter()
    node.get_logger = lambda: _Logger()
    return node


def test_camera_intake_preserves_topics_qos_sync_and_callback_group(
    monkeypatch,
):
    captured = {}
    sentinel = object()

    def fake_camera_intake(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return sentinel

    monkeypatch.setattr(follow_head, 'CameraIntake', fake_camera_intake)
    callback_group = object()
    node = object.__new__(follow_head.FollowHeadNode)
    node._sensor_cb_group = callback_group

    intake = node._create_camera_intake()

    assert intake is sentinel
    assert captured['args'][0] is node
    cfg = captured['args'][1]
    assert cfg.camera == 'orbbec'
    assert cfg.color.topic == '/camera/color/image_raw'
    assert cfg.color.best_effort is True
    assert cfg.color.qos_depth == 5
    assert cfg.depth.topic == '/camera/depth/image_raw'
    assert cfg.depth.best_effort is True
    assert cfg.depth.qos_depth == 5
    assert cfg.camera_info.topic == '/camera/color/camera_info'
    assert cfg.camera_info.best_effort is False
    assert cfg.camera_info.qos_depth == 10
    assert cfg.sync_queue == 10
    assert cfg.sync_slop_s == pytest.approx(0.1)
    assert captured['kwargs']['callback_group'] is callback_group


def test_detection_cap_consumes_sequence_without_decoding(monkeypatch):
    bundle = _Bundle(seq=7)
    node = _logic_node(bundle)
    monkeypatch.setattr(follow_head.time, 'monotonic', lambda: 10.0)

    result = node.follow_head_logic()

    assert result[0] is None
    assert result[1] == 'Waiting 0.10s for min detection interval.'
    assert node.last_used_seq == 7
    assert node._perf_sync_count == 7
    assert bundle.color_decode_calls == 0
    assert bundle.depth_decode_calls == 0

    result = node.follow_head_logic()

    assert result == (None, 'image already used (seq: 7)')
    assert node.camera_intake.last_seq_args == [None, 7]
    assert node._perf_early['already_used'] == 1


def test_empty_intake_retains_no_message_result(monkeypatch):
    node = _logic_node(None)
    monkeypatch.setattr(follow_head.time, 'monotonic', lambda: 10.0)

    assert node.follow_head_logic() == (
        None,
        'No image or depth received yet.',
    )
    assert node.last_used_seq is None


def test_analytic_servo_frame_transform_is_unchanged():
    node = object.__new__(follow_head.FollowHeadNode)

    forward = node._camera_to_pan_tilt_root(
        (0.0, 0.0, 2.0),
        cur_pan_rad=math.pi / 2.0,
        cur_tilt_rad=0.0,
    )

    assert forward == pytest.approx((0.0, -2.0, 0.0), abs=1e-12)
    pan, tilt = node._pan_tilt_root_to_angles(forward)
    assert pan == pytest.approx(math.pi / 2.0)
    assert tilt == pytest.approx(0.0)
    assert node._camera_to_pan_tilt_root(
        (0.0, 0.0, -1.0),
        cur_pan_rad=0.0,
        cur_tilt_rad=0.0,
    ) is None
