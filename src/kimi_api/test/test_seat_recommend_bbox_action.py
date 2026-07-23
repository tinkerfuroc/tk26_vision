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
"""Focused action and intake tests for seat_recommend_bbox."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from rclpy.action import CancelResponse
from std_msgs.msg import Header
from tinker_vision_msgs_26.action import SeatRecommendBbox
from vision_util.action_queue import QueuedActionGate

import kimi_api.seat_recommend_bbox as seat_bbox
from kimi_api._seat_bbox_vlm import SeatBboxResult


def _node(**attributes):
    node = object.__new__(seat_bbox.SeatRecommendBboxService)
    for name, value in attributes.items():
        setattr(node, name, value)
    return node


class _GoalHandle:
    def __init__(self, request=None):
        self.request = request
        self.is_cancel_requested = False
        self.execute_calls = 0
        self.feedback = []
        self.succeed_calls = 0
        self.abort_calls = 0
        self.canceled_calls = 0

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


class _DelayedCancelGoalHandle(_GoalHandle):
    def __init__(self, cancel_states):
        self._cancel_states = list(cancel_states)
        super().__init__()

    @property
    def is_cancel_requested(self):
        if self._cancel_states:
            return self._cancel_states.pop(0)
        return self._is_cancel_requested

    @is_cancel_requested.setter
    def is_cancel_requested(self, value):
        self._is_cancel_requested = value


class _Gate:
    def __init__(self, cancel=False):
        self.cancel = cancel
        self.finished = []

    def should_cancel(self, _goal_handle):
        return self.cancel

    def notify_finished(self, goal_handle):
        self.finished.append(goal_handle)


class _Logger:
    def __init__(self):
        self.error_messages = []

    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def warning(self, _message):
        pass

    def error(self, message):
        self.error_messages.append(message)


def test_action_server_uses_queue_callbacks_and_zero_timeout(
    monkeypatch,
):
    captured = {}
    sentinel = object()

    def fake_action_server(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return sentinel

    monkeypatch.setattr(seat_bbox, 'ActionServer', fake_action_server)
    callback_group = object()
    node = _node(action_cb_group=callback_group)

    action = node._create_seat_action()

    assert action is sentinel
    assert captured['args'] == (
        node,
        SeatRecommendBbox,
        'seat_recommend_bbox_service',
    )
    assert captured['kwargs']['result_timeout'] == 0
    assert captured['kwargs']['callback_group'] is callback_group
    assert (
        captured['kwargs']['execute_callback']
        == node.seat_recommend_bbox_execute_callback
    )
    assert (
        captured['kwargs']['cancel_callback']
        == node.seat_recommend_bbox_cancel_callback
    )
    assert (
        captured['kwargs']['handle_accepted_callback']
        == node.seat_recommend_bbox_handle_accepted_callback
    )


def test_camera_intake_preserves_reliable_rgbd_qos_and_sync(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_camera_intake(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return sentinel

    monkeypatch.setattr(seat_bbox, 'CameraIntake', fake_camera_intake)
    callback_group = object()
    bridge = object()
    node = _node(intake_cb_group=callback_group, bridge=bridge)

    intake = node._create_camera_intake('/color', '/depth', '/info')

    assert intake is sentinel
    assert captured['args'][0] is node
    cfg = captured['args'][1]
    assert cfg.camera == 'orbbec'
    assert (cfg.color.topic, cfg.depth.topic, cfg.camera_info.topic) == (
        '/color',
        '/depth',
        '/info',
    )
    assert cfg.color.best_effort is False
    assert cfg.depth.best_effort is False
    assert cfg.camera_info.best_effort is False
    assert cfg.color.qos_depth == 10
    assert cfg.depth.qos_depth == 10
    assert cfg.camera_info.qos_depth == 10
    assert cfg.sync_queue == 3
    assert cfg.sync_slop_s == 0.1
    assert cfg.age_source == 'recv'
    assert captured['kwargs'] == {
        'callback_group': callback_group,
        'bridge': bridge,
    }


def test_handle_accept_and_queued_cancel_delegate_to_fifo_gate():
    gate = QueuedActionGate()
    node = _node(_action_gate=gate)
    first = _GoalHandle()
    queued = _GoalHandle()

    node.seat_recommend_bbox_handle_accepted_callback(first)
    node.seat_recommend_bbox_handle_accepted_callback(queued)
    cancel_response = node.seat_recommend_bbox_cancel_callback(queued)

    assert first.execute_calls == 1
    assert queued.execute_calls == 0
    assert cancel_response == CancelResponse.ACCEPT
    assert gate.should_cancel(queued)

    gate.notify_finished(first)
    assert queued.execute_calls == 1


def test_queued_cancel_waits_with_time_sleep_then_skips_work(monkeypatch):
    sleeps = []
    monkeypatch.setattr(seat_bbox, 'CANCEL_STATE_POLL_S', 0.0)
    monkeypatch.setattr(
        seat_bbox.time,
        'sleep',
        lambda delay: sleeps.append(delay),
    )
    gate = _Gate(cancel=True)
    node = _node(_action_gate=gate)
    goal = _DelayedCancelGoalHandle([False, False, True])
    work_calls = []

    def user_work(_goal):
        work_calls.append(_goal)
        return SeatRecommendBbox.Result()

    node._run_seat_recommend_bbox = user_work

    result = node.seat_recommend_bbox_execute_callback(goal)

    assert work_calls == []
    assert result.status == 1
    assert result.error_msg == 'Seat recommendation canceled.'
    assert goal.canceled_calls == 1
    assert goal.abort_calls == 0
    assert sleeps == [0.0, 0.0]
    assert gate.finished == [goal]


def test_cancel_state_timeout_aborts_without_invalid_transition(monkeypatch):
    monkeypatch.setattr(seat_bbox, 'CANCEL_STATE_TIMEOUT_S', 0.0)
    gate = _Gate(cancel=True)
    logger = _Logger()
    node = _node(_action_gate=gate)
    node.get_logger = lambda: logger
    goal = _GoalHandle()

    result = node.seat_recommend_bbox_execute_callback(goal)

    assert result.status == 1
    assert 'cancellation-state error' in result.error_msg
    assert goal.canceled_calls == 0
    assert goal.abort_calls == 1
    assert logger.error_messages == [result.error_msg]
    assert gate.finished == [goal]


def test_execute_preserves_failure_payload_and_succeeds_transport():
    gate = _Gate()
    node = _node(_action_gate=gate)
    goal = _GoalHandle()
    expected = SeatRecommendBbox.Result()
    expected.status = 1
    expected.error_msg = 'No empty seat detected by VLM.'
    expected.recommendation = 'none'

    node._run_seat_recommend_bbox = lambda _goal: expected

    result = node.seat_recommend_bbox_execute_callback(goal)

    assert result is expected
    assert goal.succeed_calls == 1
    assert goal.abort_calls == 0
    assert goal.canceled_calls == 0
    assert gate.finished == [goal]


def test_internal_exception_aborts_and_notifies_finished():
    gate = _Gate()
    logger = _Logger()
    node = _node(_action_gate=gate)
    node.get_logger = lambda: logger
    goal = _GoalHandle()

    def fail(_goal):
        raise RuntimeError('pipeline exploded')

    node._run_seat_recommend_bbox = fail

    result = node.seat_recommend_bbox_execute_callback(goal)

    assert result.status == 1
    assert result.error_msg == (
        'Seat recommendation failed: pipeline exploded.'
    )
    assert result.recommendation == ''
    assert goal.abort_calls == 1
    assert goal.succeed_calls == 0
    assert logger.error_messages == [
        'Unhandled seat recommendation failure: pipeline exploded'
    ]
    assert gate.finished == [goal]


class _Frame:
    def __init__(self, events):
        self._events = events
        self.K = np.array(
            [10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 1.0]
        )
        header = Header()
        header.frame_id = 'camera'
        header.stamp.sec = 12
        header.stamp.nanosec = 34
        self.depth_msg = SimpleNamespace(header=header)

    def color_bgr(self):
        self._events.append('decode_color')
        return np.zeros((10, 20, 3), dtype=np.uint8)

    def depth_m(self):
        self._events.append('decode_depth')
        return np.full((10, 20), 2.0, dtype=np.float32)


class _Intake:
    def __init__(self, frame, events):
        self._frame = frame
        self._events = events
        self.calls = 0

    def latest(self):
        self.calls += 1
        self._events.append('snapshot')
        return self._frame


class _TransformHelper:
    def __init__(self, events):
        self.events = events
        self.lookup_calls = []

    def try_lookup(self, target, source, stamp, timeout_s):
        self.events.append('tf_lookup')
        self.lookup_calls.append((target, source, stamp, timeout_s))
        return object()

    def transform_point(self, _point, _transform):
        raise AssertionError('same-frame centroid should not be transformed')


class _VisionLogger:
    enabled = False


def _pipeline_node(frame, intake, transform_helper, logger):
    return _node(
        _action_gate=_Gate(),
        _camera_intake=intake,
        _transform_helper=transform_helper,
        _vision_logger=_VisionLogger(),
        camera_types=['orbbec'],
        log_prompts=False,
        llm_model='model',
        vlm_timeout_s=25.0,
        vlm_max_retries=3,
        vlm_strategy='bbox_select',
        _provider_models=[('qwen', 'q')],
        qwen_api_backend='dashscope',
        fewshot_enabled=False,
        max_fewshots=0,
        snap_enabled=False,
        point_bbox_halfsize_px=40,
        sample_depth_halfsize_px=1,
        fallback_depth_m=1.5,
        min_depth_m=0.1,
        max_depth_m=10.0,
    )


def test_pipeline_snapshots_tf_before_vlm_and_preserves_geometry(
    monkeypatch,
):
    events = []
    frame = _Frame(events)
    intake = _Intake(frame, events)
    transform_helper = _TransformHelper(events)
    logger = _Logger()
    node = _pipeline_node(frame, intake, transform_helper, logger)
    node.get_logger = lambda: logger
    request = SimpleNamespace(
        camera='orbbec',
        names=['Ada'],
        features=['blue shirt'],
        target_frame='camera',
        known_seats=['left chair'],
    )
    goal = _GoalHandle(request)

    def fake_vlm(*_args, **kwargs):
        events.append('vlm')
        assert kwargs['should_abort']() is False
        return SeatBboxResult(
            label='left chair',
            box_xyxy=[2, 2, 6, 6],
            seats=[],
            elapsed_s=0.25,
            provider='qwen',
        )

    monkeypatch.setattr(seat_bbox, 'request_seat_bbox_chain', fake_vlm)

    result = node.seat_recommend_bbox_execute_callback(goal)

    assert intake.calls == 1
    assert events == [
        'snapshot',
        'decode_color',
        'tf_lookup',
        'decode_depth',
        'vlm',
    ]
    lookup = transform_helper.lookup_calls[0]
    assert lookup[0:2] == ('camera', 'camera')
    assert lookup[2] is frame.depth_msg.header.stamp
    assert lookup[3] == 1.0
    assert [feedback.stage for feedback in goal.feedback] == [
        'acquiring_frame',
        'transforming',
        'vlm_call',
        'judging',
    ]
    assert all(feedback.status == 0 for feedback in goal.feedback)
    assert result.status == 0
    assert result.error_msg == ''
    assert result.recommendation == 'left chair'
    assert (
        result.bbox.xmin,
        result.bbox.ymin,
        result.bbox.xmax,
        result.bbox.ymax,
    ) == (2, 2, 6, 6)
    assert result.centroid.header.frame_id == 'camera'
    assert result.centroid.point.x == 0.8
    assert result.centroid.point.y == 0.8
    assert result.centroid.point.z == 2.0
    assert goal.succeed_calls == 1


def test_tf_lookup_failure_keeps_status_one_and_skips_vlm(monkeypatch):
    events = []
    frame = _Frame(events)
    intake = _Intake(frame, events)
    transform_helper = _TransformHelper(events)
    transform_helper.try_lookup = lambda *args, **kwargs: None
    logger = _Logger()
    node = _pipeline_node(frame, intake, transform_helper, logger)
    node.get_logger = lambda: logger
    goal = _GoalHandle(SimpleNamespace(
        camera='orbbec',
        names=[],
        features=[],
        target_frame='map',
        known_seats=[],
    ))
    vlm_calls = []
    monkeypatch.setattr(
        seat_bbox,
        'request_seat_bbox_chain',
        lambda *_args, **_kwargs: vlm_calls.append(True),
    )

    result = node.seat_recommend_bbox_execute_callback(goal)

    assert result.status == 1
    assert result.error_msg == 'TF lookup failed for frame map.'
    assert result.recommendation == ''
    assert vlm_calls == []
    assert [feedback.stage for feedback in goal.feedback] == [
        'acquiring_frame',
        'transforming',
    ]
    assert goal.succeed_calls == 1
