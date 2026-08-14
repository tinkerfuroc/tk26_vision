"""Shared TF lookup and point-transform policy."""
from __future__ import annotations

import time

from rclpy.duration import Duration
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point
import tf2_ros


class _ServiceBuffer:
    """Small tf2 Buffer-compatible facade over TransformProvider."""

    def __init__(self, helper: 'TransformHelper') -> None:
        self._helper = helper

    def lookup_transform(
        self, target, source, stamp, timeout=None
    ):
        timeout_s = 0.1
        if timeout is not None and hasattr(timeout, 'nanoseconds'):
            timeout_s = max(0.0, timeout.nanoseconds / 1e9)
        transform = self._helper.try_lookup(
            target, source, stamp=stamp, timeout_s=timeout_s
        )
        if transform is None:
            exception = getattr(
                tf2_ros, 'LookupException', RuntimeError
            )
            raise exception(
                f'provider has no transform {target} <- {source}'
            )
        return transform

    def can_transform(self, target, source, stamp, timeout=None):
        try:
            return (
                self.lookup_transform(target, source, stamp, timeout)
                is not None
            )
        except Exception:
            return False


class TransformHelper:
    """Own a long-lived TF buffer and provide non-throwing lookup helpers."""

    def __init__(
        self,
        node,
        cache_time_s: float = 180.0,
        *,
        backend: str = 'subscription',
        provider_endpoint: str = '',
        callback_group=None,
        provider_wait_timeout_s: float = 0.5,
        provider_response_timeout_s: float = 3.0,
    ) -> None:
        self._node = node
        self.backend = str(backend)
        if self.backend not in ('subscription', 'service'):
            raise ValueError(
                "backend must be 'subscription' or 'service'"
            )
        self._provider = None
        if self.backend == 'service':
            if not str(provider_endpoint).strip():
                raise ValueError(
                    'provider_endpoint is required for the service backend'
                )
            from camera_provider import TransformProvider

            self._provider = TransformProvider(
                node,
                provider_endpoint,
                callback_group=callback_group,
                service_wait_timeout_s=provider_wait_timeout_s,
                response_timeout_s=provider_response_timeout_s,
            )
            self.buffer = _ServiceBuffer(self)
            self._listener = None
        else:
            self.buffer = tf2_ros.Buffer(
                cache_time=Duration(seconds=float(cache_time_s))
            )
            self._listener = tf2_ros.TransformListener(self.buffer, node)

    @staticmethod
    def _lookup_exceptions():
        return tuple(
            exception
            for exception in (
                getattr(tf2_ros, 'TransformException', None),
                getattr(tf2_ros, 'LookupException', None),
                getattr(tf2_ros, 'ConnectivityException', None),
                getattr(tf2_ros, 'ExtrapolationException', None),
            )
            if isinstance(exception, type)
        )

    def try_lookup(
        self,
        target: str,
        source: str,
        stamp=None,
        timeout_s: float = 0.1,
    ):
        """Return one transform attempt or ``None`` on a TF failure."""
        if stamp is None:
            lookup_stamp = Time()
        elif isinstance(stamp, Time):
            lookup_stamp = stamp
        else:
            lookup_stamp = Time.from_msg(stamp)
        if self.backend == 'service':
            result = self._provider.lookup(
                target,
                source,
                lookup_time=lookup_stamp,
                timeout_s=max(0.0, float(timeout_s)),
            )
            return result.transform if result.ok else None
        try:
            return self.buffer.lookup_transform(
                target,
                source,
                lookup_stamp,
                timeout=Duration(seconds=max(0.0, float(timeout_s))),
            )
        except self._lookup_exceptions():
            return None

    def wait_lookup(
        self,
        target: str,
        source: str,
        deadline_s: float,
        latest: bool = True,
        poll_s: float = 0.02,
        stamp=None,
    ):
        """Poll until a transform is available or the node-clock deadline."""
        lookup_stamp = (
            Time()
            if latest
            else (
                self._node.get_clock().now()
                if stamp is None
                else (
                    stamp
                    if isinstance(stamp, Time)
                    else Time.from_msg(stamp)
                )
            )
        )
        if self.backend == 'service':
            result = self._provider.lookup(
                target,
                source,
                lookup_time=lookup_stamp,
                timeout_s=max(0.0, float(deadline_s)),
            )
            return result.transform if result.ok else None
        deadline_ns = (
            self._node.get_clock().now().nanoseconds
            + int(max(0.0, float(deadline_s)) * 1e9)
        )
        while True:
            try:
                if self.buffer.can_transform(target, source, lookup_stamp):
                    result = self.try_lookup(
                        target,
                        source,
                        stamp=lookup_stamp,
                        timeout_s=0.0,
                    )
                    if result is not None:
                        return result
            except self._lookup_exceptions():
                pass
            if self._node.get_clock().now().nanoseconds >= deadline_ns:
                return None
            time.sleep(max(0.0, float(poll_s)))

    def transform_point(self, pt, transform_or_target):
        """Transform a PointStamped using a transform or target frame name."""
        transform = transform_or_target
        if isinstance(transform_or_target, str):
            if not hasattr(pt, 'header'):
                raise TypeError(
                    'target-frame lookup requires a geometry_msgs PointStamped'
                )
            transform = self.try_lookup(
                transform_or_target,
                pt.header.frame_id,
                stamp=pt.header.stamp,
            )
            if transform is None:
                return None
        try:
            return do_transform_point(pt, transform)
        except self._lookup_exceptions():
            return None
