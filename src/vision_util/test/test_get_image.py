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
"""Focused tests for the synchronized image relay service."""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import sys
import types


def _get_image_module_with_ros_fakes():
    """Import get_image against a deterministic minimal ROS surface."""

    class Logger:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(message)

    class Node:
        def __init__(self, name):
            self.name = name
            self.parameters = {}
            self.logger = Logger()
            self.services = []

        def declare_parameter(self, name, default):
            self.parameters[name] = default

        def get_parameter(self, name):
            return types.SimpleNamespace(value=self.parameters[name])

        def get_logger(self):
            return self.logger

        def create_service(
            self,
            service_type,
            name,
            callback,
            *,
            callback_group,
        ):
            service = types.SimpleNamespace(
                service_type=service_type,
                name=name,
                callback=callback,
                callback_group=callback_group,
            )
            self.services.append(service)
            return service

    class MutuallyExclusiveCallbackGroup:
        pass

    class Image:
        def __init__(self, data=None):
            self.data = list(data or [])

    class GetImage:
        class Request:
            def __init__(self, camera='', depth=False):
                self.camera = camera
                self.depth = depth

        class Response:
            def __init__(self):
                self.status = 0
                self.error_msg = ''
                self.rgb_image = Image()
                self.depth_image = Image()

    @dataclass(frozen=True)
    class StreamSpec:
        topic: str
        best_effort: bool = True
        qos_depth: int = 5

    @dataclass(frozen=True)
    class IntakeConfig:
        camera: str
        color: StreamSpec | None = None
        depth: StreamSpec | None = None
        camera_info: StreamSpec | None = None
        sync_queue: int = 10
        sync_slop_s: float = 0.1
        age_source: str = 'recv'

    class CameraIntake:
        instances = []

        def __init__(self, node, cfg, callback_group=None):
            self.node = node
            self.cfg = cfg
            self.callback_group = callback_group
            self.bundle = None
            self.latest_calls = []
            self.instances.append(self)

        def latest(self, max_age_s=None):
            self.latest_calls.append(max_age_s)
            return self.bundle

    modules = {
        'rclpy': types.ModuleType('rclpy'),
        'rclpy.executors': types.ModuleType('rclpy.executors'),
        'rclpy.callback_groups': types.ModuleType(
            'rclpy.callback_groups'
        ),
        'rclpy.node': types.ModuleType('rclpy.node'),
        'tinker_vision_msgs_26': types.ModuleType(
            'tinker_vision_msgs_26'
        ),
        'tinker_vision_msgs_26.srv': types.ModuleType(
            'tinker_vision_msgs_26.srv'
        ),
        'vision_util.camera_intake': types.ModuleType(
            'vision_util.camera_intake'
        ),
    }
    modules['rclpy'].executors = modules['rclpy.executors']
    modules['rclpy.callback_groups'].MutuallyExclusiveCallbackGroup = (
        MutuallyExclusiveCallbackGroup
    )
    modules['rclpy.node'].Node = Node
    modules['tinker_vision_msgs_26'].srv = modules[
        'tinker_vision_msgs_26.srv'
    ]
    modules['tinker_vision_msgs_26.srv'].GetImage = GetImage
    modules['vision_util.camera_intake'].CameraIntake = CameraIntake
    modules['vision_util.camera_intake'].IntakeConfig = IntakeConfig
    modules['vision_util.camera_intake'].StreamSpec = StreamSpec

    missing = object()
    previous = {
        name: sys.modules.get(name, missing)
        for name in modules
    }
    sys.modules.update(modules)
    sys.modules.pop('vision_util.get_image', None)
    try:
        imported = importlib.import_module('vision_util.get_image')
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module

    fakes = types.SimpleNamespace(
        CameraIntake=CameraIntake,
        GetImage=GetImage,
        Image=Image,
    )
    return imported, fakes


get_image, FAKES = _get_image_module_with_ros_fakes()


def _new_node():
    FAKES.CameraIntake.instances.clear()
    return get_image.GetImageService()


def _intake(node, camera):
    return node.camera_intakes[camera]


def _bundle(color=None, depth=None):
    return types.SimpleNamespace(color_msg=color, depth_msg=depth)


def _call(node, camera, depth):
    request = FAKES.GetImage.Request(camera=camera, depth=depth)
    response = FAKES.GetImage.Response()
    return node.get_image_callback(request, response)


def test_builds_two_color_depth_intakes_with_preserved_configuration():
    node = _new_node()

    assert node.name == 'get_image_service'
    assert node.camera_types == ['realsense', 'orbbec']
    assert node.parameters == {
        'realsense_color_topic': (
            '/camera/xarm_camera/color/image_raw'
        ),
        'realsense_depth_topic': (
            '/camera/xarm_camera/aligned_depth_to_color/image_raw'
        ),
        'orbbec_color_topic': '/camera/color/image_raw',
        'orbbec_depth_topic': '/camera/depth/image_raw',
        'sync_queue_size': 10,
        'sync_slop': 0.1,
    }
    assert set(node.camera_intakes) == {'realsense', 'orbbec'}

    expected_topics = {
        'realsense': (
            '/camera/xarm_camera/color/image_raw',
            '/camera/xarm_camera/aligned_depth_to_color/image_raw',
        ),
        'orbbec': (
            '/camera/color/image_raw',
            '/camera/depth/image_raw',
        ),
    }
    for camera, intake in node.camera_intakes.items():
        cfg = intake.cfg
        assert cfg.camera == camera
        assert (cfg.color.topic, cfg.depth.topic) == expected_topics[camera]
        assert cfg.color.best_effort is True
        assert cfg.depth.best_effort is True
        assert cfg.color.qos_depth == 10
        assert cfg.depth.qos_depth == 10
        assert cfg.sync_queue == 10
        assert cfg.sync_slop_s == 0.1
        assert cfg.color is not None
        assert cfg.depth is not None

    callback_groups = [
        intake.callback_group for intake in node.camera_intakes.values()
    ]
    assert callback_groups[0] is not callback_groups[1]
    assert node.image_srv.name == 'get_image_service'
    assert node.image_srv.callback == node.get_image_callback


def test_reports_unchanged_errors_and_requires_color_for_every_request():
    node = _new_node()

    response = _call(node, 'kinect', depth=False)
    assert response.status == 1
    assert response.error_msg == 'Unsupported camera: kinect.'

    response = _call(node, 'realsense', depth=False)
    assert response.status == 1
    assert response.error_msg == 'No camera data for realsense.'
    assert _intake(node, 'realsense').latest_calls == [None]

    _intake(node, 'orbbec').bundle = _bundle(
        color=None,
        depth=FAKES.Image([9]),
    )
    response = _call(node, 'orbbec', depth=False)
    assert response.status == 1
    assert response.error_msg == 'No camera data for orbbec.'


def test_depth_is_required_only_when_requested():
    node = _new_node()
    color = FAKES.Image([1, 2, 3])
    _intake(node, 'orbbec').bundle = _bundle(color=color, depth=None)

    color_only = _call(node, 'orbbec', depth=False)
    assert color_only.status == 0
    assert color_only.error_msg == ''
    assert color_only.rgb_image.data == [1, 2, 3]
    assert color_only.depth_image.data == []

    with_depth = _call(node, 'orbbec', depth=True)
    assert with_depth.status == 1
    assert with_depth.error_msg == 'No camera data for orbbec.'


def test_response_messages_are_copies_and_cannot_mutate_cached_bundle():
    node = _new_node()
    cached_color = FAKES.Image([1, 2, 3])
    cached_depth = FAKES.Image([4, 5, 6])
    _intake(node, 'realsense').bundle = _bundle(
        color=cached_color,
        depth=cached_depth,
    )

    first = _call(node, 'realsense', depth=True)
    assert first.status == 0
    assert first.rgb_image is not cached_color
    assert first.depth_image is not cached_depth

    first.rgb_image.data[0] = 99
    first.depth_image.data[0] = 88
    assert cached_color.data == [1, 2, 3]
    assert cached_depth.data == [4, 5, 6]

    second = _call(node, 'realsense', depth=True)
    assert second.rgb_image.data == [1, 2, 3]
    assert second.depth_image.data == [4, 5, 6]
    assert second.rgb_image is not first.rgb_image
    assert second.depth_image is not first.depth_image
