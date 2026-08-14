# Copyright 2026 Open Source Robotics Foundation, Inc.
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

"""Focused tests for the generalist node's independent depth intake."""
from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace


_STUBBED_MODULES = []
if 'torch' not in sys.modules and importlib.util.find_spec('torch') is None:
    torch_stub = types.ModuleType('torch')
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
    sys.modules['torch'] = torch_stub
    _STUBBED_MODULES.append('torch')

if (
    'ultralytics' not in sys.modules
    and importlib.util.find_spec('ultralytics') is None
):
    ultralytics_stub = types.ModuleType('ultralytics')
    ultralytics_stub.YOLO = object
    sys.modules['ultralytics'] = ultralytics_stub
    _STUBBED_MODULES.append('ultralytics')

from object_detection_generalist import generalist_node  # noqa: E402

for _module_name in _STUBBED_MODULES:
    sys.modules.pop(_module_name, None)


class _Logger:
    def info(self, _message):
        pass


def test_orbbec_response_depth_is_independent_best_effort_depth_one(
    monkeypatch,
):
    captured = {}

    class FakeIntake:
        def __init__(self, node, cfg, callback_group=None, *, bridge=None):
            captured.update(
                node=node,
                cfg=cfg,
                callback_group=callback_group,
                bridge=bridge,
            )
            self.cfg = cfg
            self._subscriptions = []

    monkeypatch.setattr(generalist_node, 'CameraIntake', FakeIntake)
    monkeypatch.setattr(
        generalist_node,
        'MutuallyExclusiveCallbackGroup',
        lambda: 'response-depth-group',
    )
    node = SimpleNamespace(
        camera_types=['orbbec'],
        bridge=object(),
        _orbbec_response_depth_intake=None,
        get_parameter=lambda name: SimpleNamespace(
            value={
                'orbbec_depth_image_topic':
                    '/camera/depth/image_raw',
            }[name]
        ),
        get_logger=lambda: _Logger(),
    )

    (
        generalist_node.GeneralistDetectionNode
        ._init_orbbec_response_depth_intake(node)
    )

    cfg = captured['cfg']
    assert cfg.camera == 'orbbec_response_depth'
    assert cfg.color is None
    assert cfg.camera_info is None
    assert cfg.depth.topic == '/camera/depth/image_raw'
    assert cfg.depth.best_effort is True
    assert cfg.depth.qos_depth == 1
    assert cfg.backend == 'service'
    assert cfg.age_source == 'stamp'
    assert cfg.provider_endpoint == '/head_camera_server'
    assert captured['callback_group'] == 'response-depth-group'
    assert captured['bridge'] is node.bridge
    assert node._orbbec_depth_image_sub is None


def test_orbbec_response_depth_callback_forwards_to_depth_only_intake():
    received = []
    intake = SimpleNamespace(
        _depth_callback=lambda msg: received.append(msg)
    )
    node = SimpleNamespace(_orbbec_response_depth_intake=intake)
    msg = object()

    generalist_node.GeneralistDetectionNode._orbbec_depth_image_callback(
        node, msg
    )

    assert received == [msg]
