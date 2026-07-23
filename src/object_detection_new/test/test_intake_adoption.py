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

"""Focused regression tests for the detection-node intake refactor."""
from __future__ import annotations

import importlib.util
import sys
import threading
import types
from types import SimpleNamespace

import numpy as np


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

from object_detection_new import object_seg_yolo as yolo  # noqa: E402

for _module_name in _STUBBED_MODULES:
    sys.modules.pop(_module_name, None)


class _Logger:
    def __init__(self):
        self.warnings = []

    def info(self, _message):
        pass

    def warn(self, message):
        self.warnings.append(message)


class _Time:
    def __init__(self, nanoseconds):
        self.nanoseconds = nanoseconds

    def __sub__(self, other):
        return _Time(self.nanoseconds - other.nanoseconds)


def test_subscriber_intakes_preserve_qos_and_ats(monkeypatch):
    captured = []

    class FakeIntake:
        def __init__(self, node, cfg, callback_group=None, *, bridge=None):
            captured.append((node, cfg, callback_group, bridge))
            self._subscriptions = [object(), object(), object()]

    monkeypatch.setattr(yolo, '_CompatibleCameraIntake', FakeIntake)
    monkeypatch.setattr(
        yolo, 'MutuallyExclusiveCallbackGroup', lambda: 'callback-group'
    )

    topics = {
        f'{camera}_{stream}_topic': f'/{camera}/{stream}'
        for camera in ('realsense', 'orbbec')
        for stream in ('image', 'depth', 'camera_info')
    }
    node = SimpleNamespace(
        camera_types=['realsense', 'orbbec'],
        bridge=object(),
        _camera_intakes={},
        get_parameter=lambda name: SimpleNamespace(value=topics[name]),
        get_logger=lambda: _Logger(),
    )

    yolo.YOLOSegmentationNode._init_subscribers(node)

    assert len(captured) == 2
    for _, cfg, callback_group, bridge in captured:
        assert cfg.age_source == 'recv'
        assert cfg.sync_queue == 10
        assert cfg.sync_slop_s == 0.1
        assert cfg.color.best_effort is True
        assert cfg.color.qos_depth == 10
        assert cfg.depth.best_effort is True
        assert cfg.depth.qos_depth == 10
        assert cfg.camera_info.best_effort is False
        assert cfg.camera_info.qos_depth == 10
        assert callback_group == 'callback-group'
        assert bridge is node.bridge


def test_camera_intake_mirrors_legacy_compatibility_state(monkeypatch):
    bundle = SimpleNamespace(recv_time=_Time(123))

    def fake_init(self, _node, cfg, callback_group=None, *, bridge=None):
        self.cfg = cfg

    def fake_store(self, *, color_msg, depth_msg):
        self._bundle = bundle

    monkeypatch.setattr(yolo.CameraIntake, '__init__', fake_init)
    monkeypatch.setattr(
        yolo.CameraIntake, '_camera_info_callback', lambda self, msg: None
    )
    monkeypatch.setattr(yolo.CameraIntake, '_store', fake_store)
    monkeypatch.setattr(
        yolo.CameraIntake, 'latest', lambda self: self._bundle
    )

    owner = SimpleNamespace(
        lock_msg=threading.RLock(),
        lock_info=threading.RLock(),
        camera_intrinsic={'orbbec': None},
        recent_sync_msg={'orbbec': None},
        recent_publish_time={'orbbec': None},
    )
    cfg = yolo.IntakeConfig(
        camera='orbbec',
        depth=yolo.StreamSpec('/depth'),
    )
    intake = yolo._CompatibleCameraIntake(owner, cfg)
    color_msg = {'kind': 'color'}
    depth_msg = {'kind': 'depth'}
    camera_info = {'kind': 'camera_info'}

    intake._store(color_msg=color_msg, depth_msg=depth_msg)
    intake._camera_info_callback(camera_info)

    assert owner.recent_sync_msg['orbbec'] == (color_msg, depth_msg)
    assert owner.recent_publish_time['orbbec'] is bundle.recv_time
    assert owner.camera_intrinsic['orbbec'] is camera_info


def test_freshness_uses_one_call_time_and_fixed_point_one_second_polls(
    monkeypatch,
):
    served = SimpleNamespace(
        recv_time=_Time(10_100_000_000),
        color_msg={'frame': 'color'},
        depth_msg={'frame': 'depth'},
    )
    bundles = iter([
        None,
        SimpleNamespace(recv_time=_Time(9_000_000_000)),
        served,
        served,
    ])

    class Intake:
        def latest(self):
            return next(bundles)

    class Clock:
        calls = 0

        def now(self):
            self.calls += 1
            return _Time(10_000_000_000)

    clock = Clock()
    sleeps = []
    monkeypatch.setattr(yolo.time, 'sleep', sleeps.append)
    node = SimpleNamespace(
        _camera_intakes={'orbbec': Intake()},
        sync_wait_time_limit=5,
        img_sync_thres=0.2,
        get_clock=lambda: clock,
        get_logger=lambda: _Logger(),
    )

    result = yolo.YOLOSegmentationNode._wait_for_recent_frame(
        node, 'orbbec'
    )

    assert result == (served.color_msg, served.depth_msg)
    assert result[0] is not served.color_msg
    assert clock.calls == 1
    assert sleeps == [0.1, 0.1]


def test_acquire_depth_delegates_ffs_without_binding_it_to_camera_frame():
    depth_msg = object()
    node = SimpleNamespace(
        _native_depth_context=threading.local(),
        get_parameter=lambda name: SimpleNamespace(value=True),
    )

    class Source:
        def acquire(self, align_to_color):
            assert align_to_color is True
            assert node._native_depth_provider() is depth_msg
            return np.ones((2, 2), dtype=np.float32), 'ffs'

    node._depth_source = Source()
    node._native_depth_provider = types.MethodType(
        yolo.YOLOSegmentationNode._native_depth_provider, node
    )

    depth, source = yolo.YOLOSegmentationNode._acquire_depth(
        node, depth_msg
    )

    assert source == 'ffs'
    assert depth.dtype == np.float32
    assert not hasattr(node._native_depth_context, 'depth_msg')


def test_realsense_processing_preserves_bug_compatible_body_axis_math():
    depth = np.array(
        [[0.0, 1.0, 12.0], [0.5, 2.0, 3.0]], dtype=np.float64
    )
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb_msg = object()
    depth_msg = SimpleNamespace(header='depth-header')
    intrinsic = SimpleNamespace(
        k=[4.0, 0.0, 1.0, 0.0, 5.0, 0.5, 0.0, 0.0, 1.0]
    )

    class Bridge:
        def imgmsg_to_cv2(self, msg, encoding):
            assert msg is rgb_msg
            assert encoding == 'bgr8'
            return rgb

    node = SimpleNamespace(
        bridge=Bridge(),
        _last_depth_source='native',
        _acquire_depth=lambda _msg: (depth.copy(), 'ffs'),
    )
    rgb_out, points, valid, header = (
        yolo.YOLOSegmentationNode._process_realsense_data(
            node, rgb_msg, depth_msg, intrinsic
        )
    )

    rows = np.repeat(np.arange(2)[:, None], 3, axis=1)
    cols = np.repeat(np.arange(3)[None, :], 2, axis=0)
    expected = np.stack([
        (rows - 1.0) * depth / 4.0,
        (cols - 0.5) * depth / 5.0,
        np.clip(depth, 0.0, 10.0),
    ], axis=-1)
    expected_valid = np.ones_like(depth)
    expected_valid[depth > 10.0] = 0
    expected_valid[depth < 1e-6] = 0

    assert rgb_out is rgb
    np.testing.assert_allclose(points, expected)
    np.testing.assert_array_equal(valid, expected_valid)
    assert header == 'depth-header'
    assert node._last_depth_source == 'ffs'


def test_orbbec_processing_preserves_negative_minimum_valid_band():
    depth_mm = np.array([[0, 2000, 11000]], dtype=np.uint16)
    intrinsic = SimpleNamespace(
        k=[4.0, 0.0, 1.0, 0.0, 5.0, 0.5, 0.0, 0.0, 1.0]
    )

    class Bridge:
        def imgmsg_to_cv2(self, _msg, encoding):
            assert encoding == 'passthrough'
            return depth_mm

    node = SimpleNamespace(
        bridge=Bridge(),
        min_depth=-10.0,
        max_depth=10.0,
    )

    points, valid = yolo.YOLOSegmentationNode._orbbec_depth_to_array(
        node, object(), intrinsic
    )

    assert points.shape == (1, 3, 3)
    np.testing.assert_array_equal(valid, [[True, True, False]])
    np.testing.assert_allclose(points[:, :, 2], [[0.0, 2.0, 11.0]])


def test_tf_gate_and_orbbec_drop_policy_use_transform_helper():
    class TfHelper:
        def __init__(self):
            self.calls = []
            self.result = None

        def try_lookup(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.result

    helper = TfHelper()
    node = SimpleNamespace(
        _tf_helper=helper,
        get_logger=lambda: _Logger(),
        _frame_supports_tf_transform=lambda camera: (
            camera != 'realsense'
        ),
    )

    tf, ok = yolo.YOLOSegmentationNode._lookup_centroid_transform(
        node, 'camera', 'base', 'stamp', camera='realsense'
    )
    assert (tf, ok) == (None, True)
    assert helper.calls == []

    tf, ok = yolo.YOLOSegmentationNode._lookup_centroid_transform(
        node, 'camera', 'base', 'stamp', camera='orbbec'
    )
    assert (tf, ok) == (None, False)
    assert helper.calls == [
        (
            ('base', 'camera'),
            {'stamp': 'stamp', 'timeout_s': 0.1},
        )
    ]


def test_highest_sort_preserves_closest_fallback_when_tf_is_missing():
    class TfHelper:
        def try_lookup(self, *args, **kwargs):
            assert args == ('map', 'camera')
            assert kwargs == {'stamp': 'stamp', 'timeout_s': 0.1}
            return None

    first = SimpleNamespace(
        centroid=SimpleNamespace(x=2.0, y=0.0, z=0.0)
    )
    second = SimpleNamespace(
        centroid=SimpleNamespace(x=1.0, y=0.0, z=0.0)
    )
    node = SimpleNamespace(
        _tf_helper=TfHelper(),
        get_logger=lambda: _Logger(),
    )

    objects, segments = (
        yolo.YOLOSegmentationNode._sort_objects_and_segments(
            node,
            [first, second],
            ['first-mask', 'second-mask'],
            'highest',
            camera='orbbec',
            source_frame='camera',
            header=SimpleNamespace(stamp='stamp'),
            closest_distances=[2.0, 1.0],
        )
    )

    assert objects == [second, first]
    assert segments == ['second-mask', 'first-mask']
