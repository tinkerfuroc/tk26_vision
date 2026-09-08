"""Unit tests for the shared camera intake."""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pytest


def _camera_module_with_ros_fakes():
    """Import camera_intake against a deterministic minimal ROS surface."""
    class FakeTime:
        def __init__(self, nanoseconds=0):
            self.nanoseconds = int(nanoseconds)

        @classmethod
        def from_msg(cls, stamp):
            return cls(stamp.sec * 1_000_000_000 + stamp.nanosec)

    class QoSProfile:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class ReliabilityPolicy:
        BEST_EFFORT = 'best_effort'
        RELIABLE = 'reliable'

    class HistoryPolicy:
        KEEP_LAST = 'keep_last'

    class Image:
        def __init__(self, array=None, encoding='passthrough', stamp_ns=0):
            self.array = array
            self.encoding = encoding
            self.header = types.SimpleNamespace(
                stamp=types.SimpleNamespace(
                    sec=stamp_ns // 1_000_000_000,
                    nanosec=stamp_ns % 1_000_000_000,
                ),
                frame_id='camera',
            )

    class CameraInfo:
        def __init__(self):
            self.width = 640
            self.height = 480
            self.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    class Header:
        def __init__(self):
            self.stamp = types.SimpleNamespace(sec=0, nanosec=0)
            self.frame_id = ''

    class CvBridge:
        def imgmsg_to_cv2(self, msg, desired_encoding='passthrough'):
            if msg.encoding == 'bad':
                raise ValueError('bad image')
            array = np.asarray(msg.array)
            if desired_encoding == 'bgr8' and msg.encoding == 'rgb8':
                return array[..., ::-1].copy()
            return array

    class Subscriber:
        def __init__(self, node, msg_type, topic, **kwargs):
            self.node = node
            self.msg_type = msg_type
            self.topic = topic
            self.kwargs = kwargs

    class ApproximateTimeSynchronizer:
        def __init__(self, subscribers, queue_size, slop):
            self.subscribers = subscribers
            self.queue_size = queue_size
            self.slop = slop
            self.callback = None

        def registerCallback(self, callback):
            self.callback = callback

        def emit(self, *messages):
            self.callback(*messages)

    modules = {
        'cv_bridge': types.ModuleType('cv_bridge'),
        'message_filters': types.ModuleType('message_filters'),
        'rclpy': types.ModuleType('rclpy'),
        'rclpy.qos': types.ModuleType('rclpy.qos'),
        'rclpy.time': types.ModuleType('rclpy.time'),
        'sensor_msgs': types.ModuleType('sensor_msgs'),
        'sensor_msgs.msg': types.ModuleType('sensor_msgs.msg'),
        'std_msgs': types.ModuleType('std_msgs'),
        'std_msgs.msg': types.ModuleType('std_msgs.msg'),
    }
    modules['cv_bridge'].CvBridge = CvBridge
    modules['message_filters'].Subscriber = Subscriber
    modules['message_filters'].ApproximateTimeSynchronizer = (
        ApproximateTimeSynchronizer
    )
    modules['rclpy.qos'].QoSProfile = QoSProfile
    modules['rclpy.qos'].ReliabilityPolicy = ReliabilityPolicy
    modules['rclpy.qos'].HistoryPolicy = HistoryPolicy
    modules['rclpy.time'].Time = FakeTime
    modules['sensor_msgs.msg'].Image = Image
    modules['sensor_msgs.msg'].CameraInfo = CameraInfo
    modules['std_msgs.msg'].Header = Header
    modules['rclpy'].qos = modules['rclpy.qos']
    modules['rclpy'].time = modules['rclpy.time']
    modules['sensor_msgs'].msg = modules['sensor_msgs.msg']
    modules['std_msgs'].msg = modules['std_msgs.msg']
    missing = object()
    previous = {
        name: sys.modules.get(name, missing)
        for name in modules
    }
    sys.modules.update(modules)
    sys.modules.pop('vision_util.camera_intake', None)
    try:
        imported = importlib.import_module('vision_util.camera_intake')
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return imported, modules


camera_intake, _ROS_FAKES = _camera_module_with_ros_fakes()


class FakeClock:
    def __init__(self, now_ns=0):
        self.now_ns = now_ns

    def now(self):
        return _ROS_FAKES['rclpy.time'].Time(self.now_ns)


class FakeLogger:
    def __init__(self):
        self.warnings = []

    def warn(self, message):
        self.warnings.append(message)


class FakeNode:
    def __init__(self, now_ns=0):
        self.clock = FakeClock(now_ns)
        self.logger = FakeLogger()
        self.subscriptions = []

    def get_clock(self):
        return self.clock

    def get_logger(self):
        return self.logger

    def create_subscription(self, msg_type, topic, callback, **kwargs):
        subscription = types.SimpleNamespace(
            msg_type=msg_type,
            topic=topic,
            callback=callback,
            kwargs=kwargs,
        )
        self.subscriptions.append(subscription)
        return subscription


Image = _ROS_FAKES['sensor_msgs.msg'].Image
CameraInfo = _ROS_FAKES['sensor_msgs.msg'].CameraInfo


def _config(age_source='recv'):
    return camera_intake.IntakeConfig(
        camera='orbbec',
        color=camera_intake.StreamSpec('/color', qos_depth=7),
        depth=camera_intake.StreamSpec('/depth', qos_depth=3),
        camera_info=camera_intake.StreamSpec(
            '/info', best_effort=False, qos_depth=2
        ),
        sync_queue=12,
        sync_slop_s=0.08,
        age_source=age_source,
    )


def _pair(stamp_ns=0, color_encoding='bgr8'):
    color = Image(
        np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8),
        color_encoding,
        stamp_ns,
    )
    depth = Image(
        np.array([[1000, 2000]], dtype=np.uint16),
        '16UC1',
        stamp_ns,
    )
    return color, depth


def _with_intrinsics(intake):
    info = CameraInfo()
    info.width = 2
    info.height = 1
    info.k = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]
    intake._camera_info_callback(info)


def test_sync_pairing_and_per_stream_qos():
    node = FakeNode(now_ns=2_000_000_000)
    intake = camera_intake.CameraIntake(node, _config())
    _with_intrinsics(intake)

    assert intake._sync.queue_size == 12
    assert intake._sync.slop == 0.08
    assert intake._subscriptions[0].kwargs['qos_profile'].depth == 7
    assert intake._subscriptions[1].kwargs['qos_profile'].depth == 3
    info_qos = node.subscriptions[0].kwargs['qos_profile']
    assert info_qos.reliability == 'reliable'

    color, depth = _pair(stamp_ns=1_900_000_000)
    intake._sync.emit(color, depth)
    bundle = intake.latest()
    assert bundle.camera == 'orbbec'
    assert bundle.seq == 1
    assert bundle.color_msg is color
    assert bundle.depth_msg is depth


@pytest.mark.parametrize(
    ('age_source', 'expected'),
    [('recv', True), ('stamp', False)],
)
def test_age_source_recv_and_stamp(age_source, expected):
    node = FakeNode(now_ns=10_000_000_000)
    intake = camera_intake.CameraIntake(node, _config(age_source))
    color, depth = _pair(stamp_ns=8_000_000_000)
    intake._sync.emit(color, depth)
    node.clock.now_ns = 10_500_000_000

    assert (intake.latest(max_age_s=1.0) is not None) is expected


def test_wait_fresh_fail_and_stale_timeout_modes():
    node = FakeNode(now_ns=5_000_000_000)
    intake = camera_intake.CameraIntake(node, _config('stamp'))
    intake._sync.emit(*_pair(stamp_ns=1_000_000_000))

    assert intake.wait_fresh(1.0, 0.0, on_timeout='fail') is None
    stale = intake.wait_fresh(1.0, 0.0, on_timeout='stale')
    assert stale is intake.latest()
    assert node.logger.warnings


def test_latest_new_is_tri_state():
    node = FakeNode()
    intake = camera_intake.CameraIntake(node, _config())
    assert intake.latest_new(0) is None

    intake._sync.emit(*_pair())
    bundle = intake.latest_new(0)
    assert bundle.seq == 1
    assert intake.latest_new(bundle.seq) is camera_intake.NO_NEW_FRAME


def test_decode_failure_drops_bundle_and_restores_previous():
    node = FakeNode()
    intake = camera_intake.CameraIntake(node, _config())
    good_color, depth = _pair()
    intake._sync.emit(good_color, depth)
    good = intake.latest()
    np.testing.assert_array_equal(good.color_bgr(), good_color.array)

    bad_color, bad_depth = _pair()
    bad_color.encoding = 'bad'
    intake._sync.emit(bad_color, bad_depth)
    bad = intake.latest()
    with pytest.raises(ValueError, match='bad image'):
        bad.color_bgr()

    assert intake.latest() is good
    assert 'decode failed' in node.logger.warnings[-1]


def test_rgb8_bgr8_normalization_and_read_only_outputs():
    node = FakeNode()
    intake = camera_intake.CameraIntake(node, _config())
    _with_intrinsics(intake)

    rgb, depth = _pair(color_encoding='rgb8')
    intake._sync.emit(rgb, depth)
    bundle = intake.latest()
    np.testing.assert_array_equal(bundle.color_bgr()[0, 0], [3, 2, 1])
    assert not bundle.color_bgr().flags.writeable
    assert not bundle.depth_m().flags.writeable
    assert bundle.depth_m().dtype == np.float32

    points, valid = bundle.points_xyz()
    assert not points.flags.writeable
    assert not valid.flags.writeable
    assert valid.tolist() == [[True, True]]

    bgr, depth = _pair(color_encoding='bgr8')
    intake._sync.emit(bgr, depth)
    np.testing.assert_array_equal(intake.latest().color_bgr(), bgr.array)


def test_depth_decode_failure_also_restores_previous():
    node = FakeNode()
    intake = camera_intake.CameraIntake(node, _config())
    intake._sync.emit(*_pair())
    good = intake.latest()
    good.depth_m()

    color, bad_depth = _pair()
    bad_depth.encoding = 'bad'
    intake._sync.emit(color, bad_depth)
    with pytest.raises(ValueError):
        intake.latest().depth_m()
    assert intake.latest() is good


def test_service_backend_preserves_bundle_api_and_stamp_freshness(
    monkeypatch,
):
    calls = []
    color, depth = _pair(stamp_ns=9_500_000_000)
    info = CameraInfo()
    info.k = [2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0]

    class Result:
        ok = True
        status = 0
        error_msg = ''
        stamp = types.SimpleNamespace(sec=9, nanosec=400_000_000)
        received_at = types.SimpleNamespace(sec=9, nanosec=600_000_000)
        frame_id = 'provider_optical'
        color_info = info
        depth_info = info

        def __init__(self):
            self.color = color
            self.depth = depth

    class Provider:
        def __init__(self, node, endpoint, **kwargs):
            assert endpoint == '/head_camera_server'

        def snapshot(self, **kwargs):
            calls.append(kwargs)
            return Result()

    import camera_provider

    monkeypatch.setattr(camera_provider, 'CameraProvider', Provider)
    node = FakeNode(now_ns=10_000_000_000)
    cfg = camera_intake.IntakeConfig(
        camera='orbbec',
        color=camera_intake.StreamSpec('/color'),
        depth=camera_intake.StreamSpec('/depth'),
        camera_info=camera_intake.StreamSpec('/info'),
        age_source='stamp',
        backend='service',
        provider_endpoint='/head_camera_server',
    )
    intake = camera_intake.CameraIntake(node, cfg)

    bundle = intake.wait_fresh(max_age_s=1.0, timeout_s=0.7)
    assert bundle.color_msg is color
    assert bundle.depth_msg is depth
    assert bundle.header.stamp.nanosec == 400_000_000
    assert bundle.header.frame_id == 'provider_optical'
    assert bundle.recv_time.nanoseconds == 9_600_000_000
    assert intake.intrinsics()[0] == 2.0
    assert not intake._subscriptions
    assert calls[0]['max_age_s'] == 1.0
    assert calls[0]['wait_timeout_s'] == 0.7
    assert calls[0]['captured_after'].nanoseconds == 9_000_000_000


def test_service_camera_info_only_builds_header_and_recovers_same_stamp(
    monkeypatch,
):
    class Provider:
        def __init__(self, *_args, **_kwargs):
            pass

    import camera_provider

    monkeypatch.setattr(camera_provider, 'CameraProvider', Provider)
    intake = camera_intake.CameraIntake(
        FakeNode(now_ns=10_000_000_000),
        camera_intake.IntakeConfig(
            camera='camera_info_only',
            camera_info=camera_intake.StreamSpec('/info'),
            age_source='stamp',
            backend='service',
            provider_endpoint='/camera_server',
        ),
    )
    invalid = CameraInfo()
    invalid.width = 0
    valid = CameraInfo()
    valid.width = 640
    valid.height = 480
    valid.k = [
        400.0, 0.0, 320.0,
        0.0, 401.0, 240.0,
        0.0, 0.0, 1.0,
    ]

    def result(info):
        return types.SimpleNamespace(
            color=None,
            depth=None,
            stamp=types.SimpleNamespace(sec=9, nanosec=500_000_000),
            received_at=types.SimpleNamespace(
                sec=9, nanosec=600_000_000
            ),
            frame_id='camera_info_frame',
            color_info=info,
            depth_info=invalid,
        )

    first = intake._store_provider_result(result(invalid))
    assert first.header.stamp.nanosec == 500_000_000
    assert first.header.frame_id == 'camera_info_frame'
    assert first.K is None

    second = intake._store_provider_result(result(valid))
    assert second is first
    assert second.K[0] == 400.0
    assert intake.camera_info() is valid
