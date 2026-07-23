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
"""Focused action-lifecycle tests for feature_matching."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
from geometry_msgs.msg import Point
from rclpy.action import CancelResponse
from std_msgs.msg import Header
from tinker_vision_msgs_26.action import FeatureMatching
from vision_util.action_queue import QueuedActionGate

import kimi_api.feature_matching as feature_matching
from kimi_api._match_vlm import MatchVlmResult


def _node(**attributes):
    node = object.__new__(feature_matching.FeatureMatchingService)
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


class _CancelAfterChecksGate(_Gate):
    def __init__(self, cancel_after):
        super().__init__()
        self.cancel_after = cancel_after
        self.checks = 0

    def should_cancel(self, _goal_handle):
        self.checks += 1
        return self.checks >= self.cancel_after


def test_action_server_uses_queue_callbacks_and_zero_result_timeout(monkeypatch):
    captured = {}
    sentinel = object()

    def fake_action_server(*args, **kwargs):
        captured['args'] = args
        captured['kwargs'] = kwargs
        return sentinel

    monkeypatch.setattr(feature_matching, 'ActionServer', fake_action_server)
    callback_group = object()
    node = _node(server_cb_group=callback_group)

    action = node._create_matching_action()

    assert action is sentinel
    assert captured['args'] == (
        node,
        FeatureMatching,
        'feature_matching_service',
    )
    assert captured['kwargs']['result_timeout'] == 0
    assert captured['kwargs']['callback_group'] is callback_group
    assert (
        captured['kwargs']['execute_callback']
        == node.feature_matching_execute_callback
    )
    assert (
        captured['kwargs']['cancel_callback']
        == node.feature_matching_cancel_callback
    )
    assert (
        captured['kwargs']['handle_accepted_callback']
        == node.feature_matching_handle_accepted_callback
    )


def test_handle_accept_and_queued_cancel_delegate_to_fifo_gate():
    gate = QueuedActionGate()
    node = _node(_action_gate=gate)
    first = _GoalHandle()
    queued = _GoalHandle()

    node.feature_matching_handle_accepted_callback(first)
    node.feature_matching_handle_accepted_callback(queued)
    cancel_response = node.feature_matching_cancel_callback(queued)

    assert first.execute_calls == 1
    assert queued.execute_calls == 0
    assert cancel_response == CancelResponse.ACCEPT
    assert gate.should_cancel(queued)

    gate.notify_finished(first)
    assert queued.execute_calls == 1


def test_queued_canceled_goal_skips_user_work_and_finishes_canceled():
    gate = _Gate(cancel=True)
    node = _node(_action_gate=gate)
    goal = _GoalHandle()
    goal.is_cancel_requested = True
    user_work_calls = []

    async def user_work(_goal):
        user_work_calls.append(_goal)
        return FeatureMatching.Result()

    node._run_feature_matching = user_work

    result = asyncio.run(node.feature_matching_execute_callback(goal))

    assert user_work_calls == []
    assert result.status == 1
    assert result.error_msg == 'Feature matching canceled.'
    assert result.centroids == []
    assert goal.canceled_calls == 1
    assert goal.succeed_calls == 0
    assert goal.abort_calls == 0
    assert gate.finished == [goal]


def test_queued_cancel_intent_waits_for_cancel_state(monkeypatch):
    sleeps = []
    monkeypatch.setattr(
        feature_matching.time, 'sleep', lambda seconds: sleeps.append(seconds)
    )
    gate = _Gate(cancel=True)
    logger = _Logger()
    node = _node(_action_gate=gate)
    node.get_logger = lambda: logger
    goal = _DelayedCancelGoalHandle([False, False, True])
    user_work_calls = []

    async def user_work(_goal):
        user_work_calls.append(_goal)
        return FeatureMatching.Result()

    node._run_feature_matching = user_work

    result = asyncio.run(node.feature_matching_execute_callback(goal))

    assert user_work_calls == []
    assert result.error_msg == 'Feature matching canceled.'
    assert goal.canceled_calls == 1
    assert goal.abort_calls == 0
    assert logger.error_messages == []
    assert len(sleeps) == 2
    assert all(0.0 <= seconds <= 0.01 for seconds in sleeps)
    assert gate.finished == [goal]


def test_cancel_state_timeout_aborts_without_invalid_transition(monkeypatch):
    monkeypatch.setattr(feature_matching, 'CANCEL_STATE_TIMEOUT_S', 0.0)
    gate = _Gate(cancel=True)
    logger = _Logger()
    node = _node(_action_gate=gate)
    node.get_logger = lambda: logger
    goal = _GoalHandle()

    result = asyncio.run(node.feature_matching_execute_callback(goal))

    assert result.status == 1
    assert 'cancellation-state error' in result.error_msg
    assert goal.canceled_calls == 0
    assert goal.abort_calls == 1
    assert logger.error_messages == [result.error_msg]
    assert gate.finished == [goal]


def test_execute_preserves_result_payload_and_succeeds_transport():
    gate = _Gate()
    node = _node(_action_gate=gate)
    goal = _GoalHandle()
    expected = FeatureMatching.Result()
    expected.status = 1
    expected.error_msg = 'No person detected.'
    expected.centroids = []

    async def user_work(_goal):
        return expected

    node._run_feature_matching = user_work

    result = asyncio.run(node.feature_matching_execute_callback(goal))

    assert result is expected
    assert goal.succeed_calls == 1
    assert goal.abort_calls == 0
    assert goal.canceled_calls == 0
    assert gate.finished == [goal]


def test_internal_exception_aborts_without_secondary_logging_failure():
    gate = _Gate()
    logger = _Logger()
    node = _node(_action_gate=gate)
    node.get_logger = lambda: logger
    goal = _GoalHandle()

    async def user_work(_goal):
        raise RuntimeError('pipeline exploded')

    node._run_feature_matching = user_work

    result = asyncio.run(node.feature_matching_execute_callback(goal))

    assert result.status == 1
    assert result.error_msg == (
        'Feature matching failed: pipeline exploded.'
    )
    assert result.centroids == []
    assert goal.abort_calls == 1
    assert goal.succeed_calls == 0
    assert goal.canceled_calls == 0
    assert logger.error_messages == [
        'Unhandled feature matching failure: pipeline exploded'
    ]
    assert gate.finished == [goal]


def test_active_cancel_after_detection_await_skips_processing():
    gate = _CancelAfterChecksGate(cancel_after=3)
    detection_client = _DetectionClient(object())
    node = _node(
        _action_gate=gate,
        detection_cli=detection_client,
    )
    node.get_logger = lambda: _Logger()
    goal = _GoalHandle(SimpleNamespace(
        camera='orbbec',
        comparison_images=[],
        features=['wearing a blue shirt'],
        max_distance=5.0,
        target_frame='camera',
    ))
    goal.is_cancel_requested = True

    result = asyncio.run(node.feature_matching_execute_callback(goal))

    assert len(detection_client.requests) == 1
    assert [feedback.stage for feedback in goal.feedback] == ['detecting']
    assert result.status == 1
    assert result.error_msg == 'Feature matching canceled.'
    assert goal.canceled_calls == 1
    assert goal.succeed_calls == 0
    assert gate.finished == [goal]


def test_feedback_uses_canonical_fields():
    node = _node(_action_gate=_Gate())
    goal = _GoalHandle()

    node._publish_feedback(
        goal,
        stage='vlm_call',
        message='Matching candidates.',
        delay_limit=25.0,
    )

    assert len(goal.feedback) == 1
    feedback = goal.feedback[0]
    assert feedback.status == 0
    assert feedback.delay_limit == 25.0
    assert feedback.stage == 'vlm_call'
    assert feedback.message == 'Matching candidates.'


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
            return np.zeros((8, 8, 3), dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[1:5, 2:6] = 255
        return mask


class _VisionLogger:
    run_dir = None

    def write(self, **_kwargs):
        return None


class _Logger:
    def __init__(self):
        self.error_messages = []

    def info(self, _message):
        pass

    def warn(self, _message):
        pass

    def error(self, message):
        self.error_messages.append(message)


def test_matching_pipeline_publishes_stages_and_maps_centroid(monkeypatch):
    header = Header()
    header.frame_id = 'camera'
    centroid = Point(x=1.0, y=2.0, z=3.0)
    detection_response = SimpleNamespace(
        status=0,
        error_msg='',
        header=header,
        rgb_image=object(),
        segments=[object()],
        objects=[SimpleNamespace(cls='person', centroid=centroid)],
    )
    detection_client = _DetectionClient(detection_response)
    gate = _Gate()
    logger = _Logger()
    node = _node(
        _action_gate=gate,
        _match_provider_chain=[('gemini', 'model')],
        _vision_logger=_VisionLogger(),
        bridge=_Bridge(),
        detection_cli=detection_client,
        log_prompts=False,
        max_person_per_image=5,
        qwen_api_backend='dashscope',
        vlm_max_retries=3,
        vlm_timeout_s=20.0,
    )
    node.get_logger = lambda: logger
    goal = _GoalHandle(SimpleNamespace(
        camera='orbbec',
        comparison_images=[],
        features=['wearing a blue shirt'],
        max_distance=5.0,
        target_frame='camera',
    ))
    captured = {}

    def fake_match(*_args, **kwargs):
        captured.update(kwargs)
        return MatchVlmResult(indices=[0], provider='gemini')

    monkeypatch.setattr(
        feature_matching, 'request_match_indices_chain', fake_match
    )
    monkeypatch.setattr(
        feature_matching, 'encode_to_data_url', lambda _image: 'data:image'
    )

    result = asyncio.run(node._run_feature_matching(goal))

    assert result.status == 0
    assert result.error_msg == ''
    assert len(result.centroids) == 1
    assert result.centroids[0].header.frame_id == 'camera'
    assert result.centroids[0].point == centroid
    assert [feedback.stage for feedback in goal.feedback] == [
        'detecting',
        'vlm_call',
        'transforming',
    ]
    assert all(feedback.status == 0 for feedback in goal.feedback)
    assert all(feedback.delay_limit > 0.0 for feedback in goal.feedback)
    assert callable(captured['should_abort'])
    assert not captured['should_abort']()


def test_stamped_tf_lookup_degrades_to_detection_frame():
    calls = []

    class Transform:
        def try_lookup(self, target, source, stamp, timeout_s):
            calls.append((target, source, stamp, timeout_s))
            return None

    header = Header()
    header.frame_id = 'camera'
    header.stamp.sec = 12
    point = Point(x=1.0, y=2.0, z=3.0)
    node = _node(_transform_helper=Transform())
    node.get_logger = lambda: _Logger()

    result = node._stamped_in_target_frame(point, header, 'map')

    assert calls == [('map', 'camera', header.stamp, 0.1)]
    assert result.header.frame_id == 'camera'
    assert result.header.stamp == header.stamp
    assert result.point == point
