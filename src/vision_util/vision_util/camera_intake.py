"""Shared camera subscription, synchronization, and frame-cache utilities."""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from vision_util.depth_reproject import (
    decode_depth_metres,
    depth_image_to_points,
)


class _NoNewFrame:
    def __repr__(self) -> str:
        return 'NO_NEW_FRAME'


NO_NEW_FRAME = _NoNewFrame()


@dataclass(frozen=True)
class StreamSpec:
    """Configuration for one ROS image or camera-info stream."""

    topic: str
    best_effort: bool = True
    qos_depth: int = 5


@dataclass(frozen=True)
class IntakeConfig:
    """Camera intake stream and synchronization configuration."""

    camera: str
    color: Optional[StreamSpec] = None
    depth: Optional[StreamSpec] = None
    camera_info: Optional[StreamSpec] = None
    sync_queue: int = 10
    sync_slop_s: float = 0.1
    age_source: str = 'recv'
    backend: str = 'subscription'
    provider_endpoint: str = ''
    provider_wait_timeout_s: float = 0.5
    provider_response_timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if self.age_source not in ('recv', 'stamp'):
            raise ValueError("age_source must be 'recv' or 'stamp'")
        if self.backend not in ('subscription', 'service'):
            raise ValueError(
                "backend must be 'subscription' or 'service'"
            )
        if self.backend == 'service' and not self.provider_endpoint.strip():
            raise ValueError(
                'provider_endpoint is required for the service backend'
            )
        if self.backend == 'service' and self.age_source != 'stamp':
            raise ValueError(
                "the service backend requires age_source='stamp'"
            )
        if (
            self.color is None
            and self.depth is None
            and self.camera_info is None
        ):
            raise ValueError('at least one camera stream is required')


def _nanoseconds(value: Any) -> int:
    if hasattr(value, 'nanoseconds'):
        return int(value.nanoseconds)
    if hasattr(value, 'sec') and hasattr(value, 'nanosec'):
        return int(value.sec) * 1_000_000_000 + int(value.nanosec)
    raise TypeError(f'cannot convert {type(value).__name__} to nanoseconds')


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


class FrameBundle:
    """An immutable message bundle with thread-safe lazy numpy decodes."""

    def __init__(
        self,
        *,
        owner: 'CameraIntake',
        camera: str,
        seq: int,
        header,
        recv_time,
        K: Optional[np.ndarray],
        color_msg: Optional[Image],
        depth_msg: Optional[Image],
        previous: Optional['FrameBundle'],
    ) -> None:
        self.camera = camera
        self.seq = seq
        self.header = header
        self.recv_time = recv_time
        self.K = K
        self.color_msg = color_msg
        self.depth_msg = depth_msg
        self._owner = owner
        self._previous = previous
        self._color_bgr: Optional[np.ndarray] = None
        self._depth_m: Optional[np.ndarray] = None
        self._point_cache: Dict[Tuple[Any, Tuple[float, float]], tuple] = {}
        self._invalid = False

    def _decode(
        self,
        attr: str,
        callback: Callable[[], np.ndarray],
    ) -> np.ndarray:
        with self._owner._lock:
            if self._invalid:
                raise RuntimeError(
                    'frame bundle was discarded after a decode failure'
                )
            cached = getattr(self, attr)
            if cached is not None:
                return cached
            try:
                decoded = _readonly(np.asarray(callback()))
            except Exception:
                self._invalid = True
                self._owner._discard_bundle_locked(self)
                raise
            setattr(self, attr, decoded)
            return decoded

    def color_bgr(self) -> np.ndarray:
        """Decode color to a memoized, read-only BGR uint8 array."""
        if self.color_msg is None:
            raise ValueError('this frame bundle has no color stream')
        return self._decode(
            '_color_bgr',
            lambda: self._owner._bridge.imgmsg_to_cv2(
                self.color_msg, desired_encoding='bgr8'
            ),
        )

    def depth_m(self) -> np.ndarray:
        """Decode depth to a memoized, read-only float32 metres array."""
        if self.depth_msg is None:
            raise ValueError('this frame bundle has no depth stream')

        def decode() -> np.ndarray:
            raw = self._owner._bridge.imgmsg_to_cv2(
                self.depth_msg, desired_encoding='passthrough'
            )
            return decode_depth_metres(np.asarray(raw))

        return self._decode('_depth_m', decode)

    def points_xyz(
        self,
        roi=None,
        valid_band: Tuple[float, float] = (1e-6, 10.0),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return memoized read-only optical-frame points and validity mask."""
        if self.K is None:
            raise ValueError(
                'camera intrinsics were unavailable for this frame'
            )
        if roi is None:
            roi_key = None
        elif hasattr(roi, 'x_offset'):
            roi_key = (
                int(roi.x_offset),
                int(roi.y_offset),
                int(roi.width),
                int(roi.height),
            )
        else:
            roi_key = tuple(int(value) for value in roi)
        key = (roi_key, tuple(float(value) for value in valid_band))
        with self._owner._lock:
            if self._invalid:
                raise RuntimeError(
                    'frame bundle was discarded after a decode failure'
                )
            cached = self._point_cache.get(key)
            if cached is not None:
                return cached
            try:
                points, valid = depth_image_to_points(
                    self.depth_m(),
                    self.K,
                    valid_band=valid_band,
                    roi=roi,
                    return_valid_mask=True,
                )
                result = (_readonly(points), _readonly(valid))
            except Exception:
                if not self._invalid:
                    self._invalid = True
                    self._owner._discard_bundle_locked(self)
                raise
            self._point_cache[key] = result
            return result


class CameraIntake:
    """Own subscriptions and expose the newest synchronized camera frame."""

    _STALE_WARN_PERIOD_S = 5.0

    def __init__(
        self,
        node,
        cfg: IntakeConfig,
        callback_group=None,
        *,
        bridge: Optional[CvBridge] = None,
    ) -> None:
        self._node = node
        self.cfg = cfg
        self._bridge = bridge if bridge is not None else CvBridge()
        self._lock = threading.RLock()
        self._latest: Optional[FrameBundle] = None
        self._camera_info = None
        self._K: Optional[np.ndarray] = None
        self._seq = 0
        self._last_stale_warn = 0.0
        self._subscriptions = []
        self._sync = None
        self._provider = None
        self._provider_stamp_ns = None

        if cfg.backend == 'service':
            # Import lazily so subscription-only users and their lightweight
            # unit tests do not need generated provider service modules.
            from camera_provider import CameraProvider

            self._provider = CameraProvider(
                node,
                cfg.provider_endpoint,
                callback_group=callback_group,
                service_wait_timeout_s=cfg.provider_wait_timeout_s,
                response_timeout_s=cfg.provider_response_timeout_s,
            )
        elif cfg.color is not None and cfg.depth is not None:
            color_sub = Subscriber(
                node,
                Image,
                cfg.color.topic,
                qos_profile=self._qos(cfg.color),
                callback_group=callback_group,
            )
            depth_sub = Subscriber(
                node,
                Image,
                cfg.depth.topic,
                qos_profile=self._qos(cfg.depth),
                callback_group=callback_group,
            )
            self._subscriptions.extend((color_sub, depth_sub))
            self._sync = ApproximateTimeSynchronizer(
                [color_sub, depth_sub],
                queue_size=cfg.sync_queue,
                slop=cfg.sync_slop_s,
            )
            self._sync.registerCallback(self._sync_callback)
        elif cfg.color is not None:
            self._subscriptions.append(
                node.create_subscription(
                    Image,
                    cfg.color.topic,
                    self._color_callback,
                    qos_profile=self._qos(cfg.color),
                    callback_group=callback_group,
                )
            )
        elif cfg.depth is not None:
            self._subscriptions.append(
                node.create_subscription(
                    Image,
                    cfg.depth.topic,
                    self._depth_callback,
                    qos_profile=self._qos(cfg.depth),
                    callback_group=callback_group,
                )
            )

        if cfg.backend == 'subscription' and cfg.camera_info is not None:
            self._subscriptions.append(
                node.create_subscription(
                    CameraInfo,
                    cfg.camera_info.topic,
                    self._camera_info_callback,
                    qos_profile=self._qos(cfg.camera_info),
                    callback_group=callback_group,
                )
            )

    @staticmethod
    def _qos(spec: StreamSpec) -> QoSProfile:
        reliability = (
            ReliabilityPolicy.BEST_EFFORT
            if spec.best_effort
            else ReliabilityPolicy.RELIABLE
        )
        return QoSProfile(
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
            depth=spec.qos_depth,
        )

    def _camera_info_callback(self, msg: CameraInfo) -> None:
        k = _readonly(np.asarray(msg.k, dtype=np.float64).reshape(9).copy())
        with self._lock:
            self._camera_info = msg
            self._K = k

    def _sync_callback(self, color_msg: Image, depth_msg: Image) -> None:
        self._store(color_msg=color_msg, depth_msg=depth_msg)

    def _color_callback(self, color_msg: Image) -> None:
        self._store(color_msg=color_msg, depth_msg=None)

    def _depth_callback(self, depth_msg: Image) -> None:
        self._store(color_msg=None, depth_msg=depth_msg)

    def _store(
        self,
        *,
        color_msg: Optional[Image],
        depth_msg: Optional[Image],
    ) -> None:
        recv_time = self._node.get_clock().now()
        header_msg = color_msg if color_msg is not None else depth_msg
        header = getattr(header_msg, 'header', None)
        with self._lock:
            self._seq += 1
            self._latest = FrameBundle(
                owner=self,
                camera=self.cfg.camera,
                seq=self._seq,
                header=header,
                recv_time=recv_time,
                K=self._K,
                color_msg=color_msg,
                depth_msg=depth_msg,
                previous=self._latest,
            )

    def _store_provider_result(self, result) -> Optional[FrameBundle]:
        """Cache a provider payload without using receive time for age."""
        from camera_provider import select_camera_info

        color_msg = result.color if self.cfg.color is not None else None
        depth_msg = result.depth if self.cfg.depth is not None else None
        header_msg = color_msg if color_msg is not None else depth_msg
        payload_header = getattr(header_msg, 'header', None)
        header = (
            copy.copy(payload_header)
            if payload_header is not None
            else Header()
        )
        stamp = getattr(result, 'stamp', None)
        if stamp is None or _nanoseconds(stamp) <= 0:
            stamp = getattr(header, 'stamp', None)
        if stamp is None:
            return None
        stamp_ns = _nanoseconds(stamp)
        if stamp_ns <= 0:
            return None
        header.stamp = stamp
        if getattr(result, 'frame_id', ''):
            header.frame_id = result.frame_id

        info = None
        if self.cfg.camera_info is not None:
            info = select_camera_info(
                result.depth_info,
                result.color_info,
                prefer_depth=self.cfg.depth is not None,
            )
            if info is not None:
                k = _readonly(
                    np.asarray(
                        info.k, dtype=np.float64
                    ).reshape(9).copy()
                )
            else:
                k = None
        else:
            k = self._K

        received_at = getattr(result, 'received_at', None)
        try:
            recv_time = Time.from_msg(received_at)
        except Exception:
            recv_time = self._node.get_clock().now()

        with self._lock:
            if (
                self._provider_stamp_ns == stamp_ns
                and self._latest is not None
            ):
                if info is not None:
                    self._camera_info = info
                    self._K = k
                    self._latest.K = k
                return self._latest
            self._provider_stamp_ns = stamp_ns
            if info is not None:
                self._camera_info = info
                self._K = k
            self._seq += 1
            self._latest = FrameBundle(
                owner=self,
                camera=self.cfg.camera,
                seq=self._seq,
                header=header,
                recv_time=recv_time,
                K=k,
                color_msg=color_msg,
                depth_msg=depth_msg,
                previous=self._latest,
            )
            return self._latest

    def _provider_snapshot(
        self,
        *,
        max_age_s: float = 0.0,
        captured_after=None,
        wait_timeout_s: float = 0.0,
    ):
        result = self._provider.snapshot(
            want_color=self.cfg.color is not None,
            want_depth=self.cfg.depth is not None,
            want_camera_info=self.cfg.camera_info is not None,
            max_age_s=max(0.0, float(max_age_s)),
            captured_after=captured_after,
            wait_timeout_s=max(0.0, float(wait_timeout_s)),
        )
        # STALE and WAIT_TIMEOUT snapshots intentionally carry the newest
        # payload, which is cached for the existing on_timeout='stale' mode.
        if result.status not in (1, 5):
            self._store_provider_result(result)
        return result

    def _discard_bundle_locked(self, bundle: FrameBundle) -> None:
        if self._latest is bundle:
            previous = bundle._previous
            while previous is not None and previous._invalid:
                previous = previous._previous
            self._latest = previous
        try:
            self._node.get_logger().warn(
                f'{self.cfg.camera} frame {bundle.seq} decode failed; '
                'keeping the previous frame'
            )
        except Exception:
            pass

    def _age_s(self, bundle: FrameBundle) -> float:
        now_ns = _nanoseconds(self._node.get_clock().now())
        if self.cfg.age_source == 'recv':
            then_ns = _nanoseconds(bundle.recv_time)
        else:
            if bundle.header is None or not hasattr(bundle.header, 'stamp'):
                return float('inf')
            try:
                then_ns = _nanoseconds(Time.from_msg(bundle.header.stamp))
            except Exception:
                then_ns = _nanoseconds(bundle.header.stamp)
        return (now_ns - then_ns) / 1e9

    def latest(
        self,
        max_age_s: Optional[float] = None,
    ) -> Optional[FrameBundle]:
        """Return the newest bundle, optionally rejecting an over-age frame."""
        if self.cfg.backend == 'service':
            result = self._provider_snapshot(
                max_age_s=0.0 if max_age_s is None else max_age_s
            )
            if not result.ok:
                return None
        with self._lock:
            bundle = self._latest
            if bundle is None:
                return None
            if (
                max_age_s is not None
                and self._age_s(bundle) > float(max_age_s)
            ):
                return None
            return bundle

    def wait_fresh(
        self,
        max_age_s: float,
        timeout_s: float,
        poll_s: float = 0.05,
        on_timeout: str = 'fail',
    ) -> Optional[FrameBundle]:
        """Poll for a fresh bundle, then fail or serve stale on timeout."""
        if on_timeout not in ('fail', 'stale'):
            raise ValueError("on_timeout must be 'fail' or 'stale'")
        if self.cfg.backend == 'service':
            now_ns = _nanoseconds(self._node.get_clock().now())
            after_ns = max(
                1, now_ns - int(max(0.0, float(max_age_s)) * 1e9)
            )
            result = self._provider_snapshot(
                max_age_s=max_age_s,
                captured_after=Time(nanoseconds=after_ns),
                wait_timeout_s=timeout_s,
            )
            if result.ok:
                return self.latest_cached(max_age_s=max_age_s)
            if on_timeout == 'fail':
                return None
            stale = self.latest_cached()
            if stale is not None:
                now = time.monotonic()
                if now - self._last_stale_warn >= self._STALE_WARN_PERIOD_S:
                    self._node.get_logger().warn(
                        f'{self.cfg.camera} frame is stale; proceeding anyway'
                    )
                    self._last_stale_warn = now
            return stale
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            bundle = self.latest(max_age_s=max_age_s)
            if bundle is not None:
                return bundle
            if time.monotonic() >= deadline:
                if on_timeout == 'fail':
                    return None
                stale = self.latest()
                if stale is not None:
                    now = time.monotonic()
                    if (
                        now - self._last_stale_warn
                        >= self._STALE_WARN_PERIOD_S
                    ):
                        self._node.get_logger().warn(
                            f'{self.cfg.camera} frame is stale; '
                            'proceeding anyway'
                        )
                        self._last_stale_warn = now
                return stale
            time.sleep(max(0.0, float(poll_s)))

    def latest_new(self, last_seq) -> Any:
        """Return a new bundle, ``NO_NEW_FRAME``, or ``None`` when empty."""
        if self.cfg.backend == 'service':
            result = self._provider_snapshot()
            if not result.ok:
                return None
        with self._lock:
            bundle = self._latest
            if bundle is None:
                return None
            if last_seq is not None and bundle.seq <= int(last_seq):
                return NO_NEW_FRAME
            return bundle

    def intrinsics(self) -> Optional[np.ndarray]:
        """Return the latest read-only 9-element intrinsic matrix."""
        with self._lock:
            return self._K

    def camera_info(self):
        """Return the latest CameraInfo message, if one is available."""
        with self._lock:
            return self._camera_info

    def latest_cached(
        self,
        max_age_s: Optional[float] = None,
    ) -> Optional[FrameBundle]:
        """Return the local cache without issuing a provider request."""
        with self._lock:
            bundle = self._latest
            if bundle is None:
                return None
            if (
                max_age_s is not None
                and self._age_s(bundle) > float(max_age_s)
            ):
                return None
            return bundle

    @staticmethod
    def with_backend_params(
        node,
        cfg: IntakeConfig,
        *,
        default_backend: str = 'service',
        default_endpoint: str,
    ) -> IntakeConfig:
        """Apply conventional rollout parameters to an intake config."""

        def value(name: str, default):
            try:
                node.declare_parameter(name, default)
            except Exception:
                pass
            try:
                return node.get_parameter(name).value
            except Exception:
                return default

        backend = str(value('camera_backend', default_backend))
        if backend not in ('subscription', 'service'):
            backend = default_backend
        endpoint_name = f'{cfg.camera}_provider_endpoint'
        endpoint = str(value(endpoint_name, default_endpoint))

        def number(name: str, default: float) -> float:
            try:
                return float(value(name, default))
            except (TypeError, ValueError):
                return float(default)

        return replace(
            cfg,
            backend=backend,
            provider_endpoint=endpoint,
            age_source='stamp' if backend == 'service' else cfg.age_source,
            provider_wait_timeout_s=number(
                'camera_provider_wait_timeout_s',
                cfg.provider_wait_timeout_s,
            ),
            provider_response_timeout_s=number(
                'camera_provider_response_timeout_s',
                cfg.provider_response_timeout_s,
            ),
        )

    @staticmethod
    def declare_params(node, camera: str, defaults) -> IntakeConfig:
        """Declare/read a conventional per-camera intake parameter set."""
        if isinstance(defaults, IntakeConfig):
            default_cfg = defaults
        else:
            values = dict(defaults)

            def stream(name):
                value = values.get(name)
                if value is None or isinstance(value, StreamSpec):
                    return value
                if isinstance(value, str):
                    return StreamSpec(value)
                return StreamSpec(**value)

            default_cfg = IntakeConfig(
                camera=camera,
                color=stream('color'),
                depth=stream('depth'),
                camera_info=stream('camera_info'),
                sync_queue=int(values.get('sync_queue', 10)),
                sync_slop_s=float(values.get('sync_slop_s', 0.1)),
                age_source=str(values.get('age_source', 'recv')),
                backend=str(values.get('backend', 'subscription')),
                provider_endpoint=str(values.get('provider_endpoint', '')),
                provider_wait_timeout_s=float(
                    values.get('provider_wait_timeout_s', 0.5)
                ),
                provider_response_timeout_s=float(
                    values.get('provider_response_timeout_s', 10.0)
                ),
            )

        def parameter(name: str, default):
            full_name = f'{camera}_{name}'
            try:
                node.declare_parameter(full_name, default)
            except Exception:
                pass
            return node.get_parameter(full_name).value

        def configured_stream(name: str, spec: Optional[StreamSpec]):
            if spec is None:
                return None
            topic = str(parameter(f'{name}_topic', spec.topic))
            if not topic:
                return None
            return StreamSpec(
                topic=topic,
                best_effort=bool(
                    parameter(f'{name}_best_effort', spec.best_effort)
                ),
                qos_depth=int(parameter(f'{name}_qos_depth', spec.qos_depth)),
            )

        return IntakeConfig(
            camera=camera,
            color=configured_stream('color', default_cfg.color),
            depth=configured_stream('depth', default_cfg.depth),
            camera_info=configured_stream(
                'camera_info', default_cfg.camera_info
            ),
            sync_queue=int(parameter('sync_queue', default_cfg.sync_queue)),
            sync_slop_s=float(
                parameter('sync_slop_s', default_cfg.sync_slop_s)
            ),
            age_source=str(parameter('age_source', default_cfg.age_source)),
            backend=str(parameter('backend', default_cfg.backend)),
            provider_endpoint=str(
                parameter(
                    'provider_endpoint', default_cfg.provider_endpoint
                )
            ),
            provider_wait_timeout_s=float(
                parameter(
                    'provider_wait_timeout_s',
                    default_cfg.provider_wait_timeout_s,
                )
            ),
            provider_response_timeout_s=float(
                parameter(
                    'provider_response_timeout_s',
                    default_cfg.provider_response_timeout_s,
                )
            ),
        )


def configure_camera_backend(
    node,
    cfg: IntakeConfig,
    *,
    default_backend: str = 'service',
    default_endpoint: str,
) -> IntakeConfig:
    """Apply the standard camera-provider rollout parameters."""
    return CameraIntake.with_backend_params(
        node,
        cfg,
        default_backend=default_backend,
        default_endpoint=default_endpoint,
    )
