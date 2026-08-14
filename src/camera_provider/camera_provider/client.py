"""Bounded and callback-group-safe camera-provider clients.

The clients deliberately do not spin an executor or poll futures. The node's
executor completes the rclpy future in a reentrant callback group and signals
an event. Consumers may either retain the returned :class:`ProviderCall` and
cancel it, or use the bounded convenience methods.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Callable, Generic, Optional, Sequence, TypeVar

from builtin_interfaces.msg import Time as TimeMsg
from rclpy.callback_groups import ReentrantCallbackGroup
from tinker_vision_msgs_26.srv import (
    GetCameraPointCloud,
    GetCameraSnapshot,
    GetTransform,
)


T = TypeVar('T')


def _endpoint(root: str, leaf: str) -> str:
    root = str(root).strip()
    if not root:
        raise ValueError('provider endpoint must be non-empty')
    return root.rstrip('/') + '/' + leaf.lstrip('/')


def _time_msg(value: Any) -> TimeMsg:
    if value is None:
        return TimeMsg()
    if hasattr(value, 'to_msg'):
        return value.to_msg()
    if hasattr(value, 'sec') and hasattr(value, 'nanosec'):
        result = TimeMsg()
        result.sec = int(value.sec)
        result.nanosec = int(value.nanosec)
        return result
    raise TypeError('time value must be a ROS Time or builtin Time message')


def _stamp_ns(value: Any) -> int:
    return int(value.sec) * 1_000_000_000 + int(value.nanosec)


def _positive_timeout(value: float, name: str) -> float:
    result = float(value)
    if not 0.0 < result <= 60.0:
        raise ValueError(f'{name} must be in (0, 60] seconds')
    return result


@dataclass(frozen=True)
class SnapshotResult:
    """One snapshot response, including non-OK provider responses."""

    status: int
    error_msg: str
    stamp: Any = None
    received_at: Any = None
    frame_id: str = ''
    color: Any = None
    depth: Any = None
    color_info: Any = None
    depth_info: Any = None
    transforms: tuple = ()
    transforms_ok: tuple = ()

    @property
    def ok(self) -> bool:
        return self.status == GetCameraSnapshot.Response.STATUS_OK


@dataclass(frozen=True)
class PointCloudResult:
    """One point-cloud response, including its authoritative depth stamp."""

    status: int
    error_msg: str
    stamp: Any = None
    received_at: Any = None
    frame_id: str = ''
    points: Any = None

    @property
    def ok(self) -> bool:
        return self.status == GetCameraPointCloud.Response.STATUS_OK


@dataclass(frozen=True)
class TransformResult:
    """One time-correct transform response."""

    status: int
    error_msg: str
    transform: Any = None

    @property
    def ok(self) -> bool:
        return self.status == GetTransform.Response.STATUS_OK


@dataclass(frozen=True)
class BundleResult:
    """Color/snapshot and cloud responses backed by the same depth stamp."""

    snapshot: SnapshotResult
    cloud: PointCloudResult
    error_msg: str = ''

    @property
    def ok(self) -> bool:
        return self.snapshot.ok and self.cloud.ok and not self.error_msg


class ProviderCall(Generic[T]):
    """A cancellable provider request completed by the consumer executor."""

    def __init__(
        self,
        client,
        future,
        convert: Callable[[Any], T],
        failure: Callable[[str], T],
    ) -> None:
        self._client = client
        self._future = future
        self._convert = convert
        self._failure = failure
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._cancelled = False
        future.add_done_callback(lambda _future: self._event.set())

    def cancel(self) -> None:
        """Cancel locally and remove the request from rclpy's pending map."""
        with self._lock:
            if self._cancelled:
                return
            self._cancelled = True
        try:
            self._client.remove_pending_request(self._future)
        except Exception:
            pass
        try:
            self._future.cancel()
        except Exception:
            pass
        self._event.set()

    def result(self, timeout_s: float) -> T:
        """Wait once for completion and clean up a timed-out request."""
        timeout = max(0.0, float(timeout_s))
        if not self._event.wait(timeout):
            self.cancel()
            return self._failure(
                f'provider response timed out after {timeout:.3f}s'
            )
        with self._lock:
            cancelled = self._cancelled
        if cancelled:
            return self._failure('provider request was cancelled')
        try:
            response = self._future.result()
            if response is None:
                return self._failure('provider returned no response')
            return self._convert(response)
        except Exception as exc:
            return self._failure(f'provider request failed: {exc}')


class _ProviderBase:
    def __init__(
        self,
        node,
        *,
        callback_group=None,
        service_wait_timeout_s: float = 0.5,
        response_timeout_s: float = 3.0,
    ) -> None:
        self._node = node
        self._callback_group = (
            callback_group
            if callback_group is not None
            else ReentrantCallbackGroup()
        )
        self.service_wait_timeout_s = _positive_timeout(
            service_wait_timeout_s, 'service_wait_timeout_s'
        )
        self.response_timeout_s = _positive_timeout(
            response_timeout_s, 'response_timeout_s'
        )

    def _begin(
        self,
        client,
        request,
        convert: Callable[[Any], T],
        failure: Callable[[str], T],
    ) -> ProviderCall[T] | T:
        try:
            available = client.wait_for_service(
                timeout_sec=self.service_wait_timeout_s
            )
        except Exception as exc:
            return failure(f'provider availability check failed: {exc}')
        if not available:
            return failure(
                'provider service unavailable after '
                f'{self.service_wait_timeout_s:.3f}s'
            )
        try:
            return ProviderCall(
                client, client.call_async(request), convert, failure
            )
        except Exception as exc:
            return failure(f'provider request could not be sent: {exc}')

    def _finish(self, pending, timeout_s: Optional[float] = None):
        if isinstance(pending, ProviderCall):
            timeout = (
                self.response_timeout_s
                if timeout_s is None
                else max(self.response_timeout_s, float(timeout_s))
            )
            return pending.result(timeout)
        return pending


class CameraProvider(_ProviderBase):
    """Client for one camera server endpoint."""

    def __init__(
        self,
        node,
        endpoint: str,
        *,
        callback_group=None,
        service_wait_timeout_s: float = 0.5,
        response_timeout_s: float = 3.0,
    ) -> None:
        super().__init__(
            node,
            callback_group=callback_group,
            service_wait_timeout_s=service_wait_timeout_s,
            response_timeout_s=response_timeout_s,
        )
        self.endpoint = str(endpoint).rstrip('/')
        self._snapshot_client = node.create_client(
            GetCameraSnapshot,
            _endpoint(endpoint, 'get_snapshot'),
            callback_group=self._callback_group,
        )
        self._cloud_client = node.create_client(
            GetCameraPointCloud,
            _endpoint(endpoint, 'get_point_cloud'),
            callback_group=self._callback_group,
        )

    @staticmethod
    def _snapshot_failure(message: str) -> SnapshotResult:
        return SnapshotResult(
            GetCameraSnapshot.Response.STATUS_NO_DATA, message
        )

    @staticmethod
    def _cloud_failure(message: str) -> PointCloudResult:
        return PointCloudResult(
            GetCameraPointCloud.Response.STATUS_NO_DATA, message
        )

    @staticmethod
    def _snapshot_result(response) -> SnapshotResult:
        return SnapshotResult(
            status=int(response.status),
            error_msg=str(response.error_msg),
            stamp=response.stamp,
            received_at=getattr(response, 'received_at', TimeMsg()),
            frame_id=str(response.frame_id),
            color=response.color,
            depth=response.depth,
            color_info=response.color_info,
            depth_info=response.depth_info,
            transforms=tuple(response.transforms),
            transforms_ok=tuple(response.transforms_ok),
        )

    @staticmethod
    def _cloud_result(response) -> PointCloudResult:
        points = response.points
        return PointCloudResult(
            status=int(response.status),
            error_msg=str(response.error_msg),
            stamp=response.stamp,
            received_at=getattr(response, 'received_at', TimeMsg()),
            frame_id=str(getattr(points.header, 'frame_id', '')),
            points=points,
        )

    def begin_snapshot(
        self,
        *,
        want_color: bool = True,
        want_depth: bool = True,
        want_camera_info: bool = True,
        target_frames: Sequence[str] = (),
        max_age_s: float = 0.0,
        captured_after=None,
        wait_timeout_s: float = 0.0,
    ):
        request = GetCameraSnapshot.Request()
        request.want_color = bool(want_color)
        request.want_depth = bool(want_depth)
        request.want_camera_info = bool(want_camera_info)
        request.target_frames = [str(frame) for frame in target_frames]
        request.max_age_sec = float(max_age_s)
        request.captured_after = _time_msg(captured_after)
        request.wait_timeout_sec = float(wait_timeout_s)
        return self._begin(
            self._snapshot_client,
            request,
            self._snapshot_result,
            self._snapshot_failure,
        )

    def snapshot(self, **kwargs) -> SnapshotResult:
        wait_timeout = max(
            0.0, float(kwargs.get('wait_timeout_s', 0.0))
        )
        return self._finish(
            self.begin_snapshot(**kwargs),
            (
                wait_timeout + self.service_wait_timeout_s + 0.25
                if wait_timeout > 0.0
                else None
            ),
        )

    def begin_point_cloud(
        self,
        *,
        stride: int = 0,
        include_color: bool = False,
        target_frame: str = '',
        max_age_s: float = 0.0,
        captured_after=None,
        wait_timeout_s: float = 0.0,
    ):
        request = GetCameraPointCloud.Request()
        request.stride = max(0, int(stride))
        request.include_color = bool(include_color)
        request.target_frame = str(target_frame)
        request.max_age_sec = float(max_age_s)
        request.captured_after = _time_msg(captured_after)
        request.wait_timeout_sec = float(wait_timeout_s)
        return self._begin(
            self._cloud_client,
            request,
            self._cloud_result,
            self._cloud_failure,
        )

    def point_cloud(self, **kwargs) -> PointCloudResult:
        wait_timeout = max(
            0.0, float(kwargs.get('wait_timeout_s', 0.0))
        )
        return self._finish(
            self.begin_point_cloud(**kwargs),
            (
                wait_timeout + self.service_wait_timeout_s + 0.25
                if wait_timeout > 0.0
                else None
            ),
        )

    def color_cloud_bundle(
        self,
        *,
        target_frame: str = '',
        stride: int = 0,
        want_camera_info: bool = True,
        max_age_s: float = 0.0,
        captured_after=None,
        wait_timeout_s: float = 0.0,
        max_attempts: int = 3,
    ) -> BundleResult:
        """Acquire color and aligned cloud with identical depth stamps."""
        attempts = max(1, int(max_attempts))
        last_snapshot = self._snapshot_failure('snapshot not attempted')
        last_cloud = self._cloud_failure('point cloud not attempted')
        for _ in range(attempts):
            last_snapshot = self.snapshot(
                want_color=True,
                # Depth is returned so exact pair identity can be checked.
                want_depth=True,
                want_camera_info=want_camera_info,
                max_age_s=max_age_s,
                captured_after=captured_after,
                wait_timeout_s=wait_timeout_s,
            )
            if not last_snapshot.ok:
                return BundleResult(
                    last_snapshot, last_cloud, last_snapshot.error_msg
                )
            last_cloud = self.point_cloud(
                stride=stride,
                include_color=True,
                target_frame=target_frame,
                max_age_s=max_age_s,
                captured_after=captured_after,
                wait_timeout_s=wait_timeout_s,
            )
            if not last_cloud.ok:
                return BundleResult(
                    last_snapshot, last_cloud, last_cloud.error_msg
                )
            depth_stamp = getattr(
                getattr(last_snapshot.depth, 'header', None), 'stamp', None
            )
            if (
                depth_stamp is not None
                and _stamp_ns(depth_stamp) == _stamp_ns(last_cloud.stamp)
            ):
                return BundleResult(last_snapshot, last_cloud)
        return BundleResult(
            last_snapshot,
            last_cloud,
            'camera advanced between snapshot and cloud requests; '
            f'no matching depth stamp after {attempts} attempts',
        )


class TransformProvider(_ProviderBase):
    """Client for time-correct transform lookups."""

    def __init__(
        self,
        node,
        endpoint: str,
        *,
        callback_group=None,
        service_wait_timeout_s: float = 0.5,
        response_timeout_s: float = 3.0,
    ) -> None:
        super().__init__(
            node,
            callback_group=callback_group,
            service_wait_timeout_s=service_wait_timeout_s,
            response_timeout_s=response_timeout_s,
        )
        self.endpoint = str(endpoint).rstrip('/')
        self._client = node.create_client(
            GetTransform,
            _endpoint(endpoint, 'get_transform'),
            callback_group=self._callback_group,
        )

    @staticmethod
    def _failure(message: str) -> TransformResult:
        return TransformResult(
            GetTransform.Response.STATUS_UNAVAILABLE, message
        )

    @staticmethod
    def _convert(response) -> TransformResult:
        return TransformResult(
            int(response.status),
            str(response.error_msg),
            response.transform,
        )

    def begin_lookup(
        self,
        target_frame: str,
        source_frame: str,
        *,
        lookup_time=None,
        timeout_s: float = 0.1,
    ):
        request = GetTransform.Request()
        request.target_frame = str(target_frame)
        request.source_frame = str(source_frame)
        request.lookup_time = _time_msg(lookup_time)
        request.timeout_sec = float(timeout_s)
        return self._begin(
            self._client, request, self._convert, self._failure
        )

    def lookup(self, target_frame: str, source_frame: str, **kwargs):
        lookup_timeout = max(
            0.0, float(kwargs.get('timeout_s', 0.0))
        )
        return self._finish(
            self.begin_lookup(target_frame, source_frame, **kwargs),
            lookup_timeout + self.service_wait_timeout_s + 0.25,
        )


class StampedTransformBuffer:
    """Minimal tf2 Buffer facade pinned to a consumer's capture stamp."""

    def __init__(self, provider: TransformProvider) -> None:
        self.provider = provider
        self.lookup_time = None

    def set_lookup_time(self, lookup_time) -> None:
        self.lookup_time = lookup_time

    def lookup_transform(
        self,
        target_frame=None,
        source_frame=None,
        time=None,
        timeout=None,
        **kwargs,
    ):
        target = target_frame or kwargs.get('target')
        source = source_frame or kwargs.get('source')
        timeout_s = 0.1
        if timeout is not None and hasattr(timeout, 'nanoseconds'):
            timeout_s = max(0.0, timeout.nanoseconds / 1e9)
        result = self.provider.lookup(
            target,
            source,
            lookup_time=self.lookup_time,
            timeout_s=timeout_s,
        )
        if not result.ok:
            raise RuntimeError(result.error_msg)
        return result.transform

    def can_transform(self, target, source, time=None, timeout=None):
        try:
            return (
                self.lookup_transform(target, source, time, timeout)
                is not None
            )
        except Exception:
            return False
