"""Unit tests for the bounded typed provider clients."""
from __future__ import annotations

import threading
import time

import pytest
from builtin_interfaces.msg import Time
from sensor_msgs.msg import CameraInfo
from tinker_vision_msgs_26.srv import (
    GetCameraPointCloud,
    GetCameraSnapshot,
    GetTransform,
)

from camera_provider import (
    CameraProvider,
    TransformProvider,
    camera_info_is_valid,
    select_camera_info,
)


def _stamp(ns: int) -> Time:
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)


class _Future:
    def __init__(self, response=None):
        self.response = response
        self.callbacks = []
        self.cancelled = False

    def add_done_callback(self, callback):
        self.callbacks.append(callback)
        if self.response is not None:
            callback(self)

    def result(self):
        return self.response

    def cancel(self):
        self.cancelled = True
        for callback in self.callbacks:
            callback(self)

    def complete(self, response):
        self.response = response
        for callback in self.callbacks:
            callback(self)


class _Client:
    def __init__(self, responses=(), available=True):
        self.responses = list(responses)
        self.available = available
        self.requests = []
        self.removed = []
        self.futures = []

    def wait_for_service(self, timeout_sec):
        assert 0.0 < timeout_sec <= 60.0
        return self.available

    def call_async(self, request):
        self.requests.append(request)
        response = self.responses.pop(0) if self.responses else None
        future = _Future(response)
        self.futures.append(future)
        return future

    def remove_pending_request(self, future):
        self.removed.append(future)


class _Node:
    def __init__(self, clients):
        self.clients = iter(clients)
        self.names = []

    def create_client(self, _service, name, callback_group=None):
        self.names.append(name)
        return next(self.clients)


def _snapshot_response(status=0, stamp_ns=2_000_000_000):
    response = GetCameraSnapshot.Response()
    response.status = status
    response.error_msg = 'diagnostic'
    response.stamp = _stamp(stamp_ns)
    response.received_at = _stamp(stamp_ns + 50_000_000)
    response.frame_id = 'camera_optical'
    response.color.header.stamp = _stamp(stamp_ns)
    response.depth.header.stamp = _stamp(stamp_ns)
    response.color_info.k[0] = 500.0
    return response


def _cloud_response(status=0, stamp_ns=2_000_000_000):
    response = GetCameraPointCloud.Response()
    response.status = status
    response.error_msg = 'cloud diagnostic'
    response.stamp = _stamp(stamp_ns)
    response.received_at = _stamp(stamp_ns + 50_000_000)
    response.points.header.stamp = response.stamp
    response.points.header.frame_id = 'camera_optical'
    return response


def _camera_info(*, fx=500.0, fy=500.0, width=640, height=480):
    info = CameraInfo()
    info.width = width
    info.height = height
    info.k[:] = [fx, 0.0, 320.0, 0.0, fy, 240.0, 0.0, 0.0, 1.0]
    return info


def test_camera_info_helpers_handle_ros_numpy_arrays_and_fallback():
    depth = _camera_info()
    color = _camera_info(fx=600.0, fy=600.0)

    assert camera_info_is_valid(depth)
    assert select_camera_info(depth, color) is depth
    assert select_camera_info(depth, color, prefer_depth=False) is color

    depth.k[0] = float('nan')
    assert not camera_info_is_valid(depth)
    assert select_camera_info(depth, color) is color


@pytest.mark.parametrize(
    'info',
    [
        CameraInfo(),
        _camera_info(width=0),
        _camera_info(height=0),
        _camera_info(fx=0.0),
        _camera_info(fy=-1.0),
    ],
)
def test_camera_info_helpers_reject_missing_or_invalid_intrinsics(info):
    assert not camera_info_is_valid(info)


def test_snapshot_preserves_status_stamps_payload_and_request_freshness():
    snapshot_client = _Client([_snapshot_response()])
    provider = CameraProvider(
        _Node([snapshot_client, _Client()]), '/head_camera_server'
    )

    result = provider.snapshot(
        want_color=True,
        want_depth=False,
        want_camera_info=True,
        max_age_s=0.4,
        captured_after=_stamp(1_000_000_000),
        wait_timeout_s=0.7,
    )

    assert result.ok
    assert result.error_msg == 'diagnostic'
    assert result.frame_id == 'camera_optical'
    assert result.received_at.nanosec == 50_000_000
    request = snapshot_client.requests[0]
    assert request.max_age_sec == 0.4
    assert request.captured_after.sec == 1
    assert request.wait_timeout_sec == 0.7
    assert request.want_color and not request.want_depth


def test_unavailable_service_maps_to_no_data_without_sending():
    unavailable = _Client(available=False)
    provider = CameraProvider(_Node([unavailable, _Client()]), '/camera')
    result = provider.snapshot()
    assert result.status == GetCameraSnapshot.Response.STATUS_NO_DATA
    assert 'unavailable' in result.error_msg
    assert not unavailable.requests


@pytest.mark.parametrize(
    'status',
    [
        GetCameraSnapshot.Response.STATUS_STALE,
        GetCameraSnapshot.Response.STATUS_WAIT_TIMEOUT,
        GetCameraSnapshot.Response.STATUS_NO_DATA,
    ],
)
def test_non_ok_snapshot_statuses_are_preserved(status):
    provider = CameraProvider(
        _Node([_Client([_snapshot_response(status)]), _Client()]),
        '/camera',
    )

    result = provider.snapshot()

    assert result.status == status
    assert not result.ok
    assert result.error_msg == 'diagnostic'
    assert result.stamp.sec == 2


def test_response_timeout_removes_and_cancels_pending_request():
    pending_client = _Client()
    provider = CameraProvider(
        _Node([pending_client, _Client()]),
        '/camera',
        response_timeout_s=0.01,
    )
    result = provider.snapshot()
    assert result.status == GetCameraSnapshot.Response.STATUS_NO_DATA
    assert 'timed out' in result.error_msg
    assert pending_client.removed == pending_client.futures
    assert pending_client.futures[0].cancelled


def test_concurrent_calls_complete_independently():
    client = _Client()
    provider = CameraProvider(
        _Node([client, _Client()]),
        '/camera',
        response_timeout_s=0.5,
    )
    results = []

    def call():
        results.append(provider.snapshot())

    threads = [threading.Thread(target=call) for _ in range(2)]
    for thread in threads:
        thread.start()
    while len(client.futures) < 2:
        time.sleep(0.001)
    client.futures[1].complete(_snapshot_response(stamp_ns=2))
    client.futures[0].complete(_snapshot_response(stamp_ns=1))
    for thread in threads:
        thread.join()
    assert sorted(result.stamp.nanosec for result in results) == [1, 2]


def test_bundle_retries_until_snapshot_depth_matches_cloud_stamp():
    provider = CameraProvider.__new__(CameraProvider)
    snapshots = iter([
        _snapshot_response(stamp_ns=10),
        _snapshot_response(stamp_ns=20),
    ])
    clouds = iter([
        _cloud_response(stamp_ns=11),
        _cloud_response(stamp_ns=20),
    ])
    provider.snapshot = lambda **_kwargs: CameraProvider._snapshot_result(
        next(snapshots)
    )
    provider.point_cloud = lambda **_kwargs: CameraProvider._cloud_result(
        next(clouds)
    )

    result = provider.color_cloud_bundle(max_attempts=2)
    assert result.ok
    assert result.snapshot.depth.header.stamp.nanosec == 20
    assert result.cloud.stamp.nanosec == 20


def test_transform_failure_and_success_mapping():
    failure = GetTransform.Response()
    failure.status = GetTransform.Response.STATUS_UNAVAILABLE
    failure.error_msg = 'extrapolation'
    success = GetTransform.Response()
    success.status = GetTransform.Response.STATUS_OK
    success.transform.header.frame_id = 'base_link'
    client = _Client([failure, success])
    provider = TransformProvider(_Node([client]), '/head_camera_server')

    assert not provider.lookup('base_link', 'camera').ok
    result = provider.lookup(
        'base_link', 'camera', lookup_time=_stamp(42), timeout_s=0.3
    )
    assert result.ok
    assert result.transform.header.frame_id == 'base_link'
    assert client.requests[1].lookup_time.nanosec == 42
