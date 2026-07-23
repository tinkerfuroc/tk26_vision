"""FoundationStereo-preferred depth acquisition with native fallback."""
from __future__ import annotations

import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup


class FfsPreferredDepthSource:
    """Acquire FFS depth first and preserve YOLO's native fallback policy.

    ``native_depth_provider`` is called only when FFS is disabled or fails.
    It may return either a depth Image or a numpy array. Native values are
    always interpreted as millimetres and converted with
    ``astype(float) / 1000.0``, including floating inputs, for behavioral
    compatibility with the current YOLO implementation.
    """

    _PARAM_DEFAULTS = {
        'prefer_ffs': True,
        'ffs_service': '/foundation_stereo/get_depth',
        'ffs_wait_for_service_s': 0.2,
        'ffs_call_timeout_s': 8.0,
        'ffs_fallback_log_period_s': 30.0,
    }

    def __init__(
        self,
        node,
        native_depth_provider: Optional[Callable[[], object]] = None,
        *,
        bridge: Optional[CvBridge] = None,
        service_type=None,
    ) -> None:
        self._node = node
        self._native_depth_provider = native_depth_provider
        self._bridge = bridge if bridge is not None else CvBridge()
        self._service_type = service_type
        self._client = None
        self._callback_group = ReentrantCallbackGroup()
        self._last_fallback_warn = 0.0
        for name, default in self._PARAM_DEFAULTS.items():
            try:
                node.declare_parameter(name, default)
            except Exception:
                pass

    def _parameter(self, name: str):
        return self._node.get_parameter(name).value

    def _type(self):
        if self._service_type is None:
            from tinker_vision_msgs_26.srv import FoundationStereoDepth

            self._service_type = FoundationStereoDepth
        return self._service_type

    def _client_for_current_service(self):
        service_name = str(self._parameter('ffs_service'))
        if self._client is not None and self._client.srv_name == service_name:
            return self._client
        if self._client is not None:
            try:
                self._node.destroy_client(self._client)
            except Exception:
                pass
        self._client = self._node.create_client(
            self._type(),
            service_name,
            callback_group=self._callback_group,
        )
        return self._client

    def _try_ffs(self, align_to_color: bool) -> Optional[np.ndarray]:
        client = self._client_for_current_service()
        if not client.wait_for_service(
            timeout_sec=float(self._parameter('ffs_wait_for_service_s'))
        ):
            return None

        request = self._type().Request()
        request.align_to_color = bool(align_to_color)
        future = client.call_async(request)
        event = threading.Event()
        future.add_done_callback(lambda _future: event.set())
        if not event.wait(
            timeout=float(self._parameter('ffs_call_timeout_s'))
        ):
            try:
                client.remove_pending_request(future)
            except Exception:
                pass
            return None

        try:
            response = future.result()
        except Exception as exc:
            self._node.get_logger().warn(f'FFS call raised: {exc}')
            return None
        if response is None or response.status != 0:
            return None
        try:
            depth = self._bridge.imgmsg_to_cv2(
                response.depth_image, desired_encoding='passthrough'
            )
        except Exception as exc:
            self._node.get_logger().warn(f'FFS depth decode failed: {exc}')
            return None
        return np.asarray(depth).astype(np.float32, copy=False)

    def _warn_fallback(self) -> None:
        period = float(self._parameter('ffs_fallback_log_period_s'))
        now = time.monotonic()
        if now - self._last_fallback_warn >= period:
            self._node.get_logger().warn(
                'FFS depth unavailable; falling back to native realsense depth'
            )
            self._last_fallback_warn = now

    def _native_depth(self) -> np.ndarray:
        if self._native_depth_provider is None:
            raise RuntimeError(
                'native_depth_provider is required for fallback'
            )
        native = self._native_depth_provider()
        if native is None:
            raise RuntimeError('native depth is unavailable')
        if isinstance(native, np.ndarray):
            raw = native
        else:
            raw = self._bridge.imgmsg_to_cv2(
                native, desired_encoding='passthrough'
            )
        return np.asarray(raw).astype(float) / 1000.0

    def acquire(self, align_to_color: bool) -> Tuple[np.ndarray, str]:
        """Return ``(depth_metres, 'ffs'|'native')``."""
        if bool(self._parameter('prefer_ffs')):
            ffs_depth = self._try_ffs(align_to_color)
            if ffs_depth is not None:
                return ffs_depth, 'ffs'
            self._warn_fallback()
        return self._native_depth(), 'native'
