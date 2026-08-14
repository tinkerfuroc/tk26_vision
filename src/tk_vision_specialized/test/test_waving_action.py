# Copyright 2026 Tinker
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Focused action and shared-intake tests for waving detection."""

from types import SimpleNamespace

import numpy as np
from rclpy.action import CancelResponse
from std_msgs.msg import Header
from tinker_vision_msgs_26.action import DetectWaving
from vision_util.action_queue import QueuedActionGate

import tk_vision_specialized.waving_person_server as waving


def _node(**attributes):
    node = object.__new__(waving.DetectWavingPersonsNode)
    for name, value in attributes.items():
        setattr(node, name, value)
    return node


class _Logger:
    def __init__(self):
        self.errors = []

    def debug(self, _message):
        pass

    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, message):
        self.errors.append(message)


class _GoalHandle:
    def __init__(self, request=None, cancel_states=None):
        self.request = request
        self._cancel_states = list(cancel_states or [])
        self._cancel_requested = False
        self.execute_calls = 0
        self.feedback = []
        self.succeed_calls = 0
        self.abort_calls = 0
        self.canceled_calls = 0

    @property
    def is_cancel_requested(self):
        if self._cancel_states:
            return self._cancel_states.pop(0)
        return self._cancel_requested

    def execute(self):
        self.execute_calls += 1

    def publish_feedback(self, feedback):
        self.feedback.append(feedback)

    def succeed(self):
        self.succeed_calls += 1

    def abort(self):
        self.abort_calls += 1

    def canceled(self):
        self.canceled_calls += 1


class _Gate:
    def __init__(self, cancel=False):
        self.cancel = cancel
        self.finished = []

    def should_cancel(self, _goal_handle):
        return self.cancel

    def notify_finished(self, goal_handle):
        self.finished.append(goal_handle)


class _Frame:
    def __init__(self):
        self.header = Header()
        self.header.frame_id = 'camera'
        self.K = np.array(
            [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0]
        )

    def color_bgr(self):
        return np.zeros((4, 6, 3), dtype=np.uint8)

    def depth_m(self):
        return np.full((4, 6), 2.0, dtype=np.float32)


class _Intake:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def wait_fresh(self, **kwargs):
        self.calls.append(kwargs)
        return self.frame


class _TransformHelper:
    def __init__(self):
        self.lookup_calls = []

    def wait_lookup(self, target, source, **kwargs):
        self.lookup_calls.append((target, source, kwargs))
        return object()

    def transform_point(self, point, _transform):
        return point


class _Yolo:
    names = {}

    def __call__(self, *_args, **_kwargs):
        return [SimpleNamespace(boxes=None, masks=None)]


def _runtime_node(frame=None, gate=None):
    logger = _Logger()
    node = _node(
        _action_gate=gate or _Gate(),
        camera_intake=_Intake(frame),
        transform_helper=_TransformHelper(),
        waving_detector='mediapipe',
        enable_vlm_fallback=False,
        _vlm_chain=[],
        vlm_timeout_s=20.0,
        vlm_max_retries=3,
        min_person_conf=0.4,
        yolo=_Yolo(),
        show_window=False,
        _vision_logger=SimpleNamespace(enabled=False),
    )
    node.get_logger = lambda: logger
    node._publish_debug_image = lambda *_args, **_kwargs: None
    return node


def test_action_server_uses_fifo_callbacks_and_zero_result_timeout(
    monkeypatch,
):
    captured = {}

    def fake_action_server(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return object()

    monkeypatch.setattr(waving, 'ActionServer', fake_action_server)
    callback_group = object()
    node = _node(action_cb_group=callback_group)

    node._create_action_server()

    assert captured['args'] == (
        node,
        DetectWaving,
        'detect_waving_persons',
    )
    assert captured['kwargs']['result_timeout'] == 0
    assert captured['kwargs']['callback_group'] is callback_group
    assert (
        captured['kwargs']['execute_callback']
        == node.detect_waving_execute_callback
    )
    assert (
        captured['kwargs']['cancel_callback']
        == node.detect_waving_cancel_callback
    )


def test_camera_intake_preserves_reliable_rgbd_qos_sync_and_stamp_age(
    monkeypatch,
):
    captured = {}

    def fake_camera_intake(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return object()

    monkeypatch.setattr(waving, 'CameraIntake', fake_camera_intake)
    callback_group = object()
    bridge = object()
    node = _node(
        intake_cb_group=callback_group,
        bridge=bridge,
    )

    node._create_camera_intake('/color', '/depth', '/info', 0.25)

    cfg = captured['args'][1]
    assert cfg.camera == 'orbbec'
    assert (
        cfg.color.topic,
        cfg.depth.topic,
        cfg.camera_info.topic,
    ) == ('/color', '/depth', '/info')
    assert cfg.color.best_effort is False
    assert cfg.depth.best_effort is False
    assert cfg.camera_info.best_effort is False
    assert cfg.color.qos_depth == 10
    assert cfg.depth.qos_depth == 10
    assert cfg.camera_info.qos_depth == 10
    assert cfg.sync_queue == 10
    assert cfg.sync_slop_s == 0.25
    assert cfg.age_source == 'stamp'
    assert captured['kwargs'] == {
        'callback_group': callback_group,
        'bridge': bridge,
    }


def test_handle_accept_and_queued_cancel_delegate_to_gate():
    gate = QueuedActionGate()
    node = _node(_action_gate=gate)
    first = _GoalHandle()
    queued = _GoalHandle()

    node.detect_waving_handle_accepted_callback(first)
    node.detect_waving_handle_accepted_callback(queued)
    response = node.detect_waving_cancel_callback(queued)

    assert first.execute_calls == 1
    assert queued.execute_calls == 0
    assert response == CancelResponse.ACCEPT
    assert gate.should_cancel(queued)

    gate.notify_finished(first)
    assert queued.execute_calls == 1


def test_queued_cancel_waits_for_rclpy_state_and_skips_work(monkeypatch):
    sleeps = []
    monkeypatch.setattr(
        waving.time,
        'sleep',
        lambda seconds: sleeps.append(seconds),
    )
    gate = _Gate(cancel=True)
    node = _runtime_node(gate=gate)
    goal = _GoalHandle(cancel_states=[False, False, True])
    work = []
    node._run_detect_waving = lambda handle: work.append(handle)

    result = node.detect_waving_execute_callback(goal)

    assert work == []
    assert result.status == 1
    assert result.error_msg == 'Waving detection canceled.'
    assert goal.canceled_calls == 1
    assert goal.abort_calls == 0
    assert len(sleeps) == 2
    assert gate.finished == [goal]


def test_pipeline_uses_freshness_tf_and_shared_depth_helper(monkeypatch):
    frame = _Frame()
    node = _runtime_node(frame)
    goal = _GoalHandle(SimpleNamespace(
        threshold_meters=5.0,
        target_frame='map',
        min_waving_persons=0,
    ))
    depth_calls = []

    def fake_depth(depth_m, k):
        depth_calls.append((depth_m, k))
        return (
            np.zeros((4, 6, 3), dtype=np.float32),
            np.ones((4, 6), dtype=bool),
        )

    monkeypatch.setattr(waving, 'waving_optical_points', fake_depth)

    result = node._run_detect_waving(goal)

    assert node.camera_intake.calls == [{
        'max_age_s': 1.0,
        'timeout_s': 2.0,
        'poll_s': 0.05,
        'on_timeout': 'stale',
    }]
    assert node.transform_helper.lookup_calls == [(
        'map',
        'camera',
        {
            'deadline_s': 5.0,
            'latest': False,
            'poll_s': 0.02,
            'stamp': frame.header.stamp,
        },
    )]
    assert len(depth_calls) == 1
    assert depth_calls[0][1] is frame.K
    assert result.status == 1
    assert result.error_msg == 'No waving persons detected'
    assert [feedback.stage for feedback in goal.feedback] == [
        'acquiring_frame',
        'transforming',
        'input_frozen',
        'detecting',
        'judging',
    ]
    assert [feedback.input_frozen for feedback in goal.feedback] == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_payload_failure_succeeds_transport_and_notifies_gate():
    gate = _Gate()
    node = _runtime_node(gate=gate)
    expected = DetectWaving.Result()
    expected.status = -1
    expected.error_msg = 'No image, depth data received yet'
    node._run_detect_waving = lambda _goal: expected
    goal = _GoalHandle()

    result = node.detect_waving_execute_callback(goal)

    assert result is expected
    assert goal.succeed_calls == 1
    assert goal.abort_calls == 0
    assert gate.finished == [goal]
