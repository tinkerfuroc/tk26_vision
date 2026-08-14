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
"""Focused action, cancellation, batching, and intake tests for object_scan."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from rclpy.action import CancelResponse
from std_msgs.msg import Header
from tinker_vision_msgs_26.action import ObjectScan
from vision_util.action_queue import QueuedActionGate

import kimi_api.object_scan as object_scan
from kimi_api._scan_vlm import ScanVlmResult


def _node(**attributes):
    node = object.__new__(object_scan.ObjectScanServer)
    for name, value in attributes.items():
        setattr(node, name, value)
    return node


class _Now:
    def to_msg(self):
        return Time()


class _Clock:
    def now(self):
        return _Now()


class _Logger:
    def __init__(self):
        self.errors = []

    def debug(self, _message):
        pass

    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, message):
        self.errors.append(message)


class _GoalHandle:
    def __init__(self, request=None, cancel_states=None):
        self.request = request
        self._cancel_states = list(cancel_states or [])
        self._is_cancel_requested = False
        self.execute_calls = 0
        self.feedback = []
        self.succeed_calls = 0
        self.abort_calls = 0
        self.canceled_calls = 0

    @property
    def is_cancel_requested(self):
        if self._cancel_states:
            return self._cancel_states.pop(0)
        return self._is_cancel_requested

    @is_cancel_requested.setter
    def is_cancel_requested(self, value):
        self._is_cancel_requested = value

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


class _Intake:
    def __init__(self, frame=None):
        self.frame = frame
        self.max_ages = []

    def latest(self, max_age_s=None):
        self.max_ages.append(max_age_s)
        return self.frame


class _Frame:
    def __init__(self):
        self.header = Header()
        self._bgr = np.zeros((4, 4, 3), dtype=np.uint8)

    def color_bgr(self):
        return self._bgr


def _runtime_node(gate=None, **attributes):
    values = dict(
        _action_gate=gate or _Gate(),
        _provider_chain=[('gemini', 'g')],
        qwen_api_backend='dashscope',
        vlm_timeout_s=20.0,
        vlm_max_retries=3,
        max_workers=1,
        batch_size=2,
        vision_logging=False,
    )
    values.update(attributes)
    node = _node(**values)
    logger = _Logger()
    node.get_logger = lambda: logger
    node.get_clock = lambda: _Clock()
    return node


def test_action_server_uses_queue_callbacks_and_zero_result_timeout(
    monkeypatch,
):
    captured = {}
    sentinel = object()

    def fake_action_server(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return sentinel

    monkeypatch.setattr(object_scan, 'ActionServer', fake_action_server)
    callback_group = object()
    node = _node(action_cb_group=callback_group)

    action = node._create_object_scan_action()

    assert action is sentinel
    assert captured['args'] == (node, ObjectScan, 'object_scan')
    assert captured['kwargs']['result_timeout'] == 0
    assert captured['kwargs']['callback_group'] is callback_group
    assert (
        captured['kwargs']['execute_callback']
        == node.object_scan_execute_callback
    )
    assert (
        captured['kwargs']['cancel_callback']
        == node.object_scan_cancel_callback
    )
    assert (
        captured['kwargs']['handle_accepted_callback']
        == node.object_scan_handle_accepted_callback
    )


def test_two_color_intakes_preserve_topics_reliable_qos_and_group(
    monkeypatch,
):
    captured = []

    def fake_camera_intake(*args, **kwargs):
        captured.append((args, kwargs))
        return object()

    monkeypatch.setattr(object_scan, 'CameraIntake', fake_camera_intake)
    callback_group = object()
    bridge = object()
    node = _node(intake_cb_group=callback_group, bridge=bridge)

    node._create_color_intake('orbbec', '/camera/color/image_raw')
    node._create_color_intake(
        'realsense',
        '/camera/xarm_camera/color/image_raw',
    )

    assert len(captured) == 2
    configs = [args[1] for args, _kwargs in captured]
    assert [cfg.camera for cfg in configs] == ['orbbec', 'realsense']
    assert [cfg.color.topic for cfg in configs] == [
        '/camera/color/image_raw',
        '/camera/xarm_camera/color/image_raw',
    ]
    assert all(cfg.color.best_effort is False for cfg in configs)
    assert all(cfg.color.qos_depth == 10 for cfg in configs)
    assert all(cfg.depth is None for cfg in configs)
    assert all(cfg.camera_info is None for cfg in configs)
    assert all(cfg.backend == 'service' for cfg in configs)
    assert all(cfg.age_source == 'stamp' for cfg in configs)
    assert [cfg.provider_endpoint for cfg in configs] == [
        '/head_camera_server',
        '/wrist_camera_server',
    ]
    assert all(
        kwargs['callback_group'] is callback_group
        for _args, kwargs in captured
    )
    assert all(kwargs['bridge'] is bridge for _args, kwargs in captured)


def test_handle_accept_and_queued_cancel_delegate_to_fifo_gate():
    gate = QueuedActionGate()
    node = _node(_action_gate=gate)
    first = _GoalHandle()
    queued = _GoalHandle()

    node.object_scan_handle_accepted_callback(first)
    node.object_scan_handle_accepted_callback(queued)
    response = node.object_scan_cancel_callback(queued)

    assert first.execute_calls == 1
    assert queued.execute_calls == 0
    assert response == CancelResponse.ACCEPT
    assert gate.should_cancel(queued)

    gate.notify_finished(first)
    assert queued.execute_calls == 1


def test_queued_cancel_waits_for_rclpy_state_and_notifies_finished(
    monkeypatch,
):
    sleeps = []
    monkeypatch.setattr(
        object_scan.time,
        'sleep',
        lambda seconds: sleeps.append(seconds),
    )
    gate = _Gate(cancel=True)
    node = _runtime_node(gate=gate)
    goal = _GoalHandle(cancel_states=[False, False, True])
    work_calls = []
    node._run_object_scan = lambda handle: work_calls.append(handle)

    result = node.object_scan_execute_callback(goal)

    assert work_calls == []
    assert result.status == 1
    assert result.error_msg == 'Object scan canceled.'
    assert result.found_labels == []
    assert goal.canceled_calls == 1
    assert goal.abort_calls == 0
    assert len(sleeps) == 2
    assert all(0.0 <= seconds <= object_scan.CANCEL_STATE_POLL_S
               for seconds in sleeps)
    assert gate.finished == [goal]


def test_cancel_state_timeout_aborts_without_invalid_canceled_transition(
    monkeypatch,
):
    monkeypatch.setattr(object_scan, 'CANCEL_STATE_TIMEOUT_S', 0.0)
    gate = _Gate(cancel=True)
    node = _runtime_node(gate=gate)
    goal = _GoalHandle()

    result = node.object_scan_execute_callback(goal)

    assert result.status == 1
    assert 'cancellation-state error' in result.error_msg
    assert goal.canceled_calls == 0
    assert goal.abort_calls == 1
    assert gate.finished == [goal]


def test_legitimate_failure_payload_succeeds_action_transport():
    gate = _Gate()
    node = _runtime_node(gate=gate)
    goal = _GoalHandle()
    expected = ObjectScan.Result()
    expected.status = 1
    expected.error_msg = 'No orbbec frame within 1.0s'
    expected.found_labels = []
    node._run_object_scan = lambda _goal: expected

    result = node.object_scan_execute_callback(goal)

    assert result is expected
    assert goal.succeed_calls == 1
    assert goal.abort_calls == 0
    assert goal.canceled_calls == 0
    assert gate.finished == [goal]


def test_frame_lookup_uses_exact_receive_age_limit():
    intake = _Intake()
    node = _runtime_node(
        _camera_intakes={
            'orbbec': intake,
            'realsense': _Intake(),
        },
    )
    goal = _GoalHandle(
        SimpleNamespace(camera='orbbec', vocabulary=['cup'])
    )

    result = node._run_object_scan(goal)

    assert result.status == 1
    assert result.error_msg == 'No orbbec frame within 1.0s'
    assert intake.max_ages == [1.0]
    assert [feedback.stage for feedback in goal.feedback] == [
        'acquiring_frame'
    ]


def test_batches_publish_canonical_feedback_and_preserve_vocab_order(
    monkeypatch,
):
    frame = _Frame()
    node = _runtime_node(
        max_workers=2,
        _camera_intakes={
            'orbbec': _Intake(frame),
            'realsense': _Intake(),
        },
    )
    goal = _GoalHandle(SimpleNamespace(
        camera='orbbec',
        vocabulary=['banana', 'apple', 'cup', 'banana'],
    ))
    calls = []

    def fake_scan(_image_url, candidates, **kwargs):
        calls.append((list(candidates), kwargs))
        labels = list(reversed(candidates))
        return ScanVlmResult(labels=labels, provider='gemini')

    monkeypatch.setattr(
        object_scan,
        'request_scan_labels_chain',
        fake_scan,
    )
    monkeypatch.setattr(
        object_scan,
        'encode_to_data_url',
        lambda _bgr: 'data:image',
    )

    result = node._run_object_scan(goal)

    assert result.status == 0
    assert result.error_msg == ''
    assert result.found_labels == ['banana', 'apple', 'cup']
    assert len(calls) == 2
    assert all(call[1]['should_abort'] is not None for call in calls)
    assert [feedback.stage for feedback in goal.feedback] == [
        'acquiring_frame',
        'input_frozen',
        'vlm_call',
        'vlm_call',
        'judging',
    ]
    assert [feedback.message for feedback in goal.feedback[2:4]] == [
        'Scanning batch 1/2.',
        'Scanning batch 2/2.',
    ]
    assert [feedback.input_frozen for feedback in goal.feedback] == [
        False,
        True,
        True,
        True,
        True,
    ]
    assert all(feedback.status == 0 for feedback in goal.feedback)
    assert all(feedback.delay_limit > 0.0 for feedback in goal.feedback)


def test_cancel_after_first_batch_does_not_launch_remaining_batches(
    monkeypatch,
):
    gate = _Gate()
    node = _runtime_node(gate=gate, max_workers=1)
    goal = _GoalHandle()
    launched = []

    def fake_scan(_image_url, candidates, **_kwargs):
        launched.append(list(candidates))
        gate.cancel = True
        return ScanVlmResult(labels=[], provider='gemini')

    monkeypatch.setattr(
        object_scan,
        'request_scan_labels_chain',
        fake_scan,
    )

    with pytest.raises(object_scan._GoalCanceled):
        node._run_batches(
            goal,
            'data:image',
            [['one'], ['two'], ['three']],
        )

    assert launched == [['one']]
    assert [feedback.message for feedback in goal.feedback] == [
        'Scanning batch 1/3.'
    ]
