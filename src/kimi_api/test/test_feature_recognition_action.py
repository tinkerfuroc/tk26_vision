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
"""Focused action-lifecycle tests for feature_recognition."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import Point
from rclpy.action import CancelResponse
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tinker_vision_msgs_26.action import (
    FeatureExtraction,
    SeatRecommendation,
)
from vision_util.action_queue import QueuedActionGate

import kimi_api.feature_recognition as feature_recognition
from kimi_api._feature_vlm import FeatureVlmResult


def _node(**attributes):
    node = object.__new__(feature_recognition.FeatureService)
    for name, value in attributes.items():
        setattr(node, name, value)
    return node


class _GoalHandle:
    def __init__(self, request=None, cancel_after_reads=None):
        self.request = request
        self._cancel_after_reads = cancel_after_reads
        self._cancel_reads = 0
        self.execute_calls = 0
        self.feedback = []
        self.succeed_calls = 0
        self.abort_calls = 0
        self.canceled_calls = 0

    @property
    def is_cancel_requested(self):
        if self._cancel_after_reads is None:
            return False
        self._cancel_reads += 1
        return self._cancel_reads > self._cancel_after_reads

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


class _Logger:
    def __init__(self):
        self.errors = []

    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, message):
        self.errors.append(message)


class _VisionLogger:
    run_dir = None

    def write(self, **_kwargs):
        return None


def test_both_action_servers_use_explicit_groups_and_zero_result_timeout(
    monkeypatch,
):
    captured = []

    def fake_action_server(*args, **kwargs):
        captured.append((args, kwargs))
        return object()

    monkeypatch.setattr(
        feature_recognition,
        'ActionServer',
        fake_action_server,
    )
    extraction_group = object()
    seat_group = object()
    node = _node(
        extraction_action_cb_group=extraction_group,
        seat_action_cb_group=seat_group,
    )

    node._create_extraction_action()
    node._create_seat_recommend_action()

    extraction_args, extraction_kwargs = captured[0]
    assert extraction_args == (
        node,
        FeatureExtraction,
        'feature_extraction_service',
    )
    assert extraction_kwargs['result_timeout'] == 0
    assert extraction_kwargs['callback_group'] is extraction_group
    assert (
        extraction_kwargs['execute_callback']
        == node.feature_extraction_execute_callback
    )
    assert (
        extraction_kwargs['cancel_callback']
        == node.feature_extraction_cancel_callback
    )

    seat_args, seat_kwargs = captured[1]
    assert seat_args == (
        node,
        SeatRecommendation,
        'seat_recommend_service',
    )
    assert seat_kwargs['result_timeout'] == 0
    assert seat_kwargs['callback_group'] is seat_group
    assert (
        seat_kwargs['execute_callback']
        == node.seat_recommend_execute_callback
    )
    assert (
        seat_kwargs['cancel_callback']
        == node.seat_recommend_cancel_callback
    )


def test_color_only_intake_preserves_reliable_topic_depth_and_group(
    monkeypatch,
):
    captured = {}
    sentinel = object()

    def fake_camera_intake(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return sentinel

    monkeypatch.setattr(
        feature_recognition,
        'CameraIntake',
        fake_camera_intake,
    )
    callback_group = object()
    bridge = object()
    node = _node(intake_cb_group=callback_group, bridge=bridge)

    intake = node._create_seat_color_intake()

    assert intake is sentinel
    assert captured['args'][0] is node
    cfg = captured['args'][1]
    assert cfg.camera == 'orbbec'
    assert cfg.color.topic == '/camera/color/image_raw'
    assert cfg.color.best_effort is False
    assert cfg.color.qos_depth == 10
    assert cfg.depth is None
    assert cfg.camera_info is None
    assert captured['kwargs']['callback_group'] is callback_group
    assert captured['kwargs']['bridge'] is bridge


def test_two_endpoints_share_fifo_gate_and_accept_queued_cancel():
    gate = QueuedActionGate()
    node = _node(_action_gate=gate)
    extraction = _GoalHandle()
    seat = _GoalHandle()

    node.feature_extraction_handle_accepted_callback(extraction)
    node.seat_recommend_handle_accepted_callback(seat)
    cancel_response = node.seat_recommend_cancel_callback(seat)

    assert extraction.execute_calls == 1
    assert seat.execute_calls == 0
    assert cancel_response == CancelResponse.ACCEPT
    assert gate.should_cancel(seat)

    gate.notify_finished(extraction)
    assert seat.execute_calls == 1


def test_queued_cancel_waits_for_rclpy_state_then_skips_user_work():
    gate = _Gate(cancel=True)
    node = _node(_action_gate=gate)
    logger = _Logger()
    node.get_logger = lambda: logger
    goal = _GoalHandle(cancel_after_reads=2)
    user_work_calls = []

    def user_work(_goal):
        user_work_calls.append(_goal)
        return SeatRecommendation.Result()

    node._run_seat_recommend = user_work

    result = node.seat_recommend_execute_callback(goal)

    assert user_work_calls == []
    assert result.status == 1
    assert result.error_msg == 'Seat recommendation canceled.'
    assert result.recommendation == ''
    assert goal.canceled_calls == 1
    assert goal.abort_calls == 0
    assert gate.finished == [goal]


def test_cancel_intent_without_rclpy_state_aborts_as_internal_error(
    monkeypatch,
):
    monkeypatch.setattr(
        feature_recognition,
        'CANCEL_STATE_TIMEOUT_S',
        0.0,
    )
    gate = _Gate(cancel=True)
    node = _node(_action_gate=gate)
    logger = _Logger()
    node.get_logger = lambda: logger
    goal = _GoalHandle()
    user_work_calls = []

    async def user_work(_goal):
        user_work_calls.append(_goal)
        return FeatureExtraction.Result()

    node._run_feature_extraction = user_work

    result = asyncio.run(node.feature_extraction_execute_callback(goal))

    assert user_work_calls == []
    assert result.status == 1
    assert 'Cancellation intent did not transition' in result.error_msg
    assert result.feature == ''
    assert goal.abort_calls == 1
    assert goal.canceled_calls == 0
    assert logger.errors
    assert gate.finished == [goal]


def test_legitimate_failure_payload_succeeds_transport():
    gate = _Gate()
    node = _node(_action_gate=gate)
    goal = _GoalHandle()
    expected = SeatRecommendation.Result()
    expected.status = 1
    expected.error_msg = 'No camera data for orbbec.'
    expected.recommendation = ''
    node._run_seat_recommend = lambda _goal: expected

    result = node.seat_recommend_execute_callback(goal)

    assert result is expected
    assert goal.succeed_calls == 1
    assert goal.abort_calls == 0
    assert goal.canceled_calls == 0
    assert gate.finished == [goal]


class _ImmediateFuture:
    def __init__(self, result):
        self._result = result

    def __await__(self):
        async def done():
            return self

        return done().__await__()

    def result(self):
        return self._result


class _DetectionClient:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def wait_for_service(self, timeout_sec):
        assert timeout_sec == 1.0
        return True

    def call_async(self, request):
        self.requests.append(request)
        return _ImmediateFuture(self.response)


class _Bridge:
    def imgmsg_to_cv2(self, _msg, encoding):
        if encoding == 'bgr8':
            return np.zeros((100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 20:80] = 255
        return mask

    def cv2_to_imgmsg(self, _image, _encoding):
        return Image()


def test_extraction_pipeline_awaits_detection_and_publishes_stages(
    monkeypatch,
):
    header = Header()
    detection_response = SimpleNamespace(
        status=0,
        error_msg='',
        header=header,
        rgb_image=object(),
        segments=[object()],
        objects=[
            SimpleNamespace(
                cls='person',
                centroid=Point(x=0.0, y=0.0, z=1.0),
            )
        ],
    )
    detection_client = _DetectionClient(detection_response)
    node = _node(
        _action_gate=_Gate(),
        _feature_provider_chain=[('gemini', 'model')],
        _vision_logger=_VisionLogger(),
        bridge=_Bridge(),
        detection_cli=detection_client,
        qwen_api_backend='dashscope',
        vlm_max_retries=3,
        vlm_timeout_s=20.0,
    )
    node.get_logger = lambda: _Logger()
    goal = _GoalHandle(SimpleNamespace(camera='orbbec'))
    captured = {}

    def fake_feature_request(*_args, **kwargs):
        captured.update(kwargs)
        return FeatureVlmResult(text='feature text', provider='gemini')

    monkeypatch.setattr(
        feature_recognition,
        'request_feature_description_chain',
        fake_feature_request,
    )
    monkeypatch.setattr(
        feature_recognition,
        'encode_to_data_url',
        lambda _image: 'data:image',
    )

    result = asyncio.run(node._run_feature_extraction(goal))

    assert result.status == 0
    assert result.error_msg == ''
    assert result.feature == 'feature text'
    assert isinstance(result.comparison_image, Image)
    assert len(detection_client.requests) == 1
    assert [feedback.stage for feedback in goal.feedback] == [
        'detecting',
        'input_frozen',
        'vlm_call',
    ]
    assert [feedback.input_frozen for feedback in goal.feedback] == [
        False,
        True,
        True,
    ]
    assert all(feedback.status == 0 for feedback in goal.feedback)
    assert all(feedback.delay_limit > 0.0 for feedback in goal.feedback)
    assert callable(captured['should_abort'])
    assert not captured['should_abort']()


def test_seat_pipeline_uses_latest_cached_frame_and_publishes_stages(
    monkeypatch,
):
    class Intake:
        def __init__(self):
            self.latest_calls = []

        def latest(self, *args, **kwargs):
            self.latest_calls.append((args, kwargs))
            return SimpleNamespace(
                color_bgr=lambda: np.zeros((8, 8, 3), dtype=np.uint8)
            )

    intake = Intake()
    node = _node(
        _action_gate=_Gate(),
        _feature_provider_chain=[('gemini', 'model')],
        _seat_color_intake=intake,
        _vision_logger=_VisionLogger(),
        log_prompts=False,
        qwen_api_backend='dashscope',
        vlm_max_retries=3,
        vlm_timeout_s=20.0,
    )
    node.get_logger = lambda: _Logger()
    goal = _GoalHandle(SimpleNamespace(
        camera='orbbec',
        names=['Alex'],
        features=['wearing a blue shirt'],
    ))
    captured = {}

    def fake_feature_request(*_args, **kwargs):
        captured.update(kwargs)
        return FeatureVlmResult(
            text='Please sit on the left chair.',
            provider='gemini',
        )

    monkeypatch.setattr(
        feature_recognition,
        'request_feature_description_chain',
        fake_feature_request,
    )
    monkeypatch.setattr(
        feature_recognition,
        'encode_to_data_url',
        lambda _image: 'data:image',
    )

    result = node._run_seat_recommend(goal)

    assert result.status == 0
    assert result.error_msg == ''
    assert result.recommendation == 'Please sit on the left chair.'
    assert intake.latest_calls == [((), {})]
    assert [feedback.stage for feedback in goal.feedback] == [
        'acquiring_frame',
        'input_frozen',
        'vlm_call',
    ]
    assert [feedback.input_frozen for feedback in goal.feedback] == [
        False,
        True,
        True,
    ]
    assert all(feedback.status == 0 for feedback in goal.feedback)
    assert all(feedback.delay_limit > 0.0 for feedback in goal.feedback)
    assert callable(captured['should_abort'])
    assert not captured['should_abort']()
