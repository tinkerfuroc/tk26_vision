"""Unit tests for shared non-throwing TF lookup policy."""
from __future__ import annotations

import importlib
import sys
import types


def _tf_module_with_fakes():
    class TransformException(Exception):
        pass

    class Duration:
        def __init__(self, seconds=0.0):
            self.nanoseconds = int(seconds * 1e9)

    time_module = types.ModuleType('rclpy.time')

    class Time:
        def __init__(self, nanoseconds=0):
            self.nanoseconds = nanoseconds

    time_module.Time = Time

    class PointStamped:
        def __init__(self, header=None, point=None):
            self.header = header or types.SimpleNamespace(
                frame_id='', stamp=time_module.Time()
            )
            self.point = point or types.SimpleNamespace(x=0.0, y=0.0, z=0.0)

    class Buffer:
        def __init__(self, cache_time=None):
            self.cache_time = cache_time
            self.transforms = {}

        def set_transform_static(self, transform, authority):
            key = (
                transform.header.frame_id,
                transform.child_frame_id,
            )
            self.transforms[key] = transform

        def can_transform(self, target, source, stamp):
            return (target, source) in self.transforms

        def lookup_transform(self, target, source, stamp, timeout=None):
            try:
                return self.transforms[(target, source)]
            except KeyError as exc:
                raise TransformException('missing transform') from exc

    class TransformListener:
        def __init__(self, buffer, node):
            self.buffer = buffer
            self.node = node

    def do_transform_point(point, transform):
        translation = transform.transform.translation
        return PointStamped(
            header=types.SimpleNamespace(
                frame_id=transform.header.frame_id,
                stamp=point.header.stamp,
            ),
            point=types.SimpleNamespace(
                x=point.point.x + translation.x,
                y=point.point.y + translation.y,
                z=point.point.z + translation.z,
            ),
        )

    duration_module = types.ModuleType('rclpy.duration')
    duration_module.Duration = Duration
    geometry_module = types.ModuleType('geometry_msgs')
    geometry_msg_module = types.ModuleType('geometry_msgs.msg')
    geometry_msg_module.PointStamped = PointStamped
    geometry_module.msg = geometry_msg_module
    tf2_module = types.ModuleType('tf2_ros')
    tf2_module.Buffer = Buffer
    tf2_module.TransformListener = TransformListener
    tf2_module.TransformException = TransformException
    tf2_module.LookupException = TransformException
    tf2_module.ConnectivityException = TransformException
    tf2_module.ExtrapolationException = TransformException
    tf_geometry_module = types.ModuleType('tf2_geometry_msgs')
    tf_geometry_module.do_transform_point = do_transform_point

    rclpy_module = types.ModuleType('rclpy')
    rclpy_module.duration = duration_module
    rclpy_module.time = time_module
    modules = {
        'rclpy': rclpy_module,
        'rclpy.time': time_module,
        'rclpy.duration': duration_module,
        'geometry_msgs': geometry_module,
        'geometry_msgs.msg': geometry_msg_module,
        'tf2_ros': tf2_module,
        'tf2_geometry_msgs': tf_geometry_module,
    }
    missing = object()
    previous = {
        name: sys.modules.get(name, missing)
        for name in modules
    }
    sys.modules.update(modules)
    sys.modules.pop('vision_util.tf_lookup', None)
    try:
        imported = importlib.import_module('vision_util.tf_lookup')
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return imported, modules


tf_lookup, _TF_FAKES = _tf_module_with_fakes()
PointStamped = _TF_FAKES['geometry_msgs.msg'].PointStamped


class TickClock:
    def __init__(self):
        self.nanoseconds = 0

    def now(self):
        self.nanoseconds += 1_000_000
        return _TF_FAKES['rclpy.time'].Time(self.nanoseconds)


class FakeNode:
    def __init__(self):
        self.clock = TickClock()

    def get_clock(self):
        return self.clock


def _transform(target='base', source='camera', x=1.0, y=2.0, z=3.0):
    return types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id=target),
        child_frame_id=source,
        transform=types.SimpleNamespace(
            translation=types.SimpleNamespace(x=x, y=y, z=z)
        ),
    )


def test_static_lookup_success_and_failure():
    helper = tf_lookup.TransformHelper(FakeNode(), cache_time_s=42.0)
    transform = _transform()
    helper.buffer.set_transform_static(transform, 'test')

    assert helper.try_lookup('base', 'camera') is transform
    assert helper.try_lookup('map', 'camera') is None
    assert helper.buffer.cache_time.nanoseconds == 42_000_000_000


def test_wait_lookup_success_and_timeout():
    helper = tf_lookup.TransformHelper(FakeNode())
    transform = _transform()
    helper.buffer.set_transform_static(transform, 'test')

    assert helper.wait_lookup('base', 'camera', deadline_s=0.1) is transform
    assert helper.wait_lookup('map', 'camera', deadline_s=0.0) is None


def test_transform_point_accepts_transform_or_target_frame():
    helper = tf_lookup.TransformHelper(FakeNode())
    transform = _transform()
    helper.buffer.set_transform_static(transform, 'test')
    point = PointStamped(
        header=types.SimpleNamespace(
            frame_id='camera',
            stamp=_TF_FAKES['rclpy.time'].Time(),
        ),
        point=types.SimpleNamespace(x=4.0, y=5.0, z=6.0),
    )

    direct = helper.transform_point(point, transform)
    looked_up = helper.transform_point(point, 'base')
    assert (direct.point.x, direct.point.y, direct.point.z) == (5.0, 7.0, 9.0)
    assert looked_up.header.frame_id == 'base'
    assert helper.transform_point(point, 'missing') is None
