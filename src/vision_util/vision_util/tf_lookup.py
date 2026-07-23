"""Shared TF lookup and point-transform policy."""
from __future__ import annotations

import time

from rclpy.duration import Duration
from rclpy.time import Time
from tf2_geometry_msgs import do_transform_point
import tf2_ros


class TransformHelper:
    """Own a long-lived TF buffer and provide non-throwing lookup helpers."""

    def __init__(self, node, cache_time_s: float = 180.0) -> None:
        self._node = node
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
        lookup_stamp = Time() if stamp is None else stamp
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
    ):
        """Poll until a transform is available or the node-clock deadline."""
        deadline_ns = (
            self._node.get_clock().now().nanoseconds
            + int(max(0.0, float(deadline_s)) * 1e9)
        )
        while True:
            stamp = Time() if latest else self._node.get_clock().now()
            try:
                if self.buffer.can_transform(target, source, stamp):
                    result = self.try_lookup(
                        target, source, stamp=stamp, timeout_s=0.0
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
