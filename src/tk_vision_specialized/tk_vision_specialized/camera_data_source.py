"""ROS-aware camera composable for object_match_all.

Owns:
  - color + depth + camera_info subscribers for realsense and orbbec
  - ApproximateTimeSynchronizer per camera
  - TF buffer + listener
  - VisionLogger
  - depth-to-3D + centroid + TF helpers

Logic is lifted from object_detection_new.YOLOSegmentationNode but the
class is plain (no Node subclass)."""

from __future__ import annotations

import copy
import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy.duration
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from std_msgs.msg import Header
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
from tf2_ros import (
    ConnectivityException,
    ExtrapolationException,
    LookupException,
)
from tf2_geometry_msgs import do_transform_point

from vision_util.vision_logging import VisionLogger
from vision_util.tf_lookup import TransformHelper
from vision_util.depth_reproject import decode_depth_metres


@dataclass
class CameraTopics:
    realsense_image: str
    realsense_depth: str
    realsense_camera_info: str
    orbbec_image: str
    orbbec_depth: str
    orbbec_camera_info: str


class CameraDataSource:
    """Camera-sync + intrinsics + depth-to-3D + TF + logging, composable.

    Construct from a Node so we can attach subscribers/TF listener to the
    same lifecycle. All public methods are thread-safe in the sense that
    they take their own locks; the underlying ros2 callback group should
    still be MutuallyExclusive on the service to serialise calls."""

    def __init__(
        self, ros_node, *, topics: CameraTopics, params, logger=None,
    ):
        self._node = ros_node
        self._log = logger or ros_node.get_logger()
        self._params = params

        self.bridge = CvBridge()
        try:
            self.backend = str(
                ros_node.get_parameter('camera_backend').value
            )
        except Exception:
            self.backend = 'service'
        if self.backend not in ('subscription', 'service'):
            raise ValueError(
                "camera_backend must be 'subscription' or 'service'"
            )

        if self.backend == 'service':
            from camera_provider import CameraProvider

            wait_timeout = float(
                ros_node.get_parameter(
                    'camera_provider_wait_timeout_s'
                ).value
            )
            response_timeout = float(
                ros_node.get_parameter(
                    'camera_provider_response_timeout_s'
                ).value
            )
            self._providers = {
                camera: CameraProvider(
                    ros_node,
                    ros_node.get_parameter(
                        f'{camera}_provider_endpoint'
                    ).value,
                    service_wait_timeout_s=wait_timeout,
                    response_timeout_s=response_timeout,
                )
                for camera in ('realsense', 'orbbec')
            }
            transform_endpoint = ros_node.get_parameter(
                'transform_provider_endpoint'
            ).value
            self._transform_helper = TransformHelper(
                ros_node,
                backend='service',
                provider_endpoint=transform_endpoint,
                provider_wait_timeout_s=wait_timeout,
                provider_response_timeout_s=response_timeout,
            )
        else:
            self._providers = {}
            self._transform_helper = TransformHelper(
                ros_node, cache_time_s=60.0
            )
        self.tf_buffer = self._transform_helper.buffer
        self.tf_listener = self._transform_helper._listener

        self.lock_msg = threading.Lock()
        self.lock_info = threading.Lock()
        self.camera_intrinsic: dict[str, dict | None] = {
            'realsense': None, 'orbbec': None,
        }
        self.recent_sync_msg: dict[str, tuple | None] = {
            'realsense': None, 'orbbec': None,
        }
        self.recent_publish_time: dict[str, object] = {
            'realsense': None, 'orbbec': None,
        }

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        if self.backend == 'subscription':
            self._subscribe(topics, qos)

        # VisionLogger takes (node, enabled, base_folder). We read the
        # corresponding ROS params (declared by the owning node, e.g.
        # ObjectMatchAllServer in Task 10) and pass them in directly.
        try:
            log_enabled = bool(
                ros_node.get_parameter('vision_logging_enabled').value
            )
        except Exception:    # noqa: BLE001 — param may not be declared
            log_enabled = True
        try:
            log_folder = str(
                ros_node.get_parameter('vision_log_folder').value
            )
        except Exception:    # noqa: BLE001
            log_folder = 'vision_log'
        self.vision_logger = VisionLogger(
            ros_node,
            enabled=log_enabled,
            base_folder=log_folder,
        )

    def _subscribe(self, topics, qos):
        """Create the rollback subscription backend."""
        # Realsense (color Image + aligned depth Image + CameraInfo)
        self._rs_color = Subscriber(
            self._node, Image, topics.realsense_image, qos_profile=qos,
        )
        self._rs_depth = Subscriber(
            self._node, Image, topics.realsense_depth, qos_profile=qos,
        )
        self._rs_info = self._node.create_subscription(
            CameraInfo, topics.realsense_camera_info,
            lambda msg: self._set_intrinsic('realsense', msg),
            qos_profile=qos,
        )
        self._rs_sync = ApproximateTimeSynchronizer(
            [self._rs_color, self._rs_depth], queue_size=5, slop=0.05,
        )
        self._rs_sync.registerCallback(
            lambda c, d: self._on_sync('realsense', (c, d)),
        )

        # Orbbec (color Image + PointCloud2 depth + CameraInfo)
        self._ob_color = Subscriber(
            self._node, Image, topics.orbbec_image, qos_profile=qos,
        )
        self._ob_depth = Subscriber(
            self._node, PointCloud2, topics.orbbec_depth, qos_profile=qos,
        )
        self._ob_info = self._node.create_subscription(
            CameraInfo, topics.orbbec_camera_info,
            lambda msg: self._set_intrinsic('orbbec', msg),
            qos_profile=qos,
        )
        self._ob_sync = ApproximateTimeSynchronizer(
            [self._ob_color, self._ob_depth], queue_size=5, slop=0.05,
        )
        self._ob_sync.registerCallback(
            lambda c, d: self._on_sync('orbbec', (c, d)),
        )

    # ---------------- subscriber callbacks ----------------
    def _set_intrinsic(self, camera: str, msg: CameraInfo) -> None:
        with self.lock_info:
            self.camera_intrinsic[camera] = {
                'fx': msg.k[0], 'fy': msg.k[4],
                'cx': msg.k[2], 'cy': msg.k[5],
                'width': msg.width, 'height': msg.height,
                'frame_id': msg.header.frame_id,
            }

    def _on_sync(self, camera: str, msg_pair: tuple) -> None:
        with self.lock_msg:
            self.recent_sync_msg[camera] = msg_pair
            self.recent_publish_time[camera] = (
                self._node.get_clock().now()
            )

    # ---------------- public API ----------------
    def snapshot(self, camera: str):
        """Wait briefly for a recent (color, depth) pair and return it
        alongside processed arrays. Returns None on timeout."""
        if self.backend == 'service':
            return self._provider_snapshot(camera)
        sync_thres_s = float(self._params.img_sync_thres_s)
        deadline = (
            self._node.get_clock().now()
            + rclpy.duration.Duration(
                seconds=float(self._params.sync_wait_time_s),
            )
        )
        while self._node.get_clock().now() < deadline:
            with self.lock_msg:
                pair = self.recent_sync_msg.get(camera)
                rt = self.recent_publish_time.get(camera)
            if pair is not None and rt is not None:
                age = (
                    self._node.get_clock().now() - rt
                ).nanoseconds / 1e9
                if age <= sync_thres_s:
                    with self.lock_info:
                        intrinsic = copy.deepcopy(
                            self.camera_intrinsic.get(camera),
                        )
                    if intrinsic is None:
                        return None
                    color_msg, depth_msg = pair
                    if camera == 'realsense':
                        return self._process_realsense(
                            color_msg, depth_msg, intrinsic,
                        )
                    return self._process_orbbec(
                        color_msg, depth_msg, intrinsic,
                    )
            time.sleep(0.05)
        return None

    def _provider_snapshot(self, camera: str):
        provider = self._providers.get(camera)
        if provider is None:
            return None
        if camera == 'orbbec':
            bundle = provider.color_cloud_bundle(
                want_camera_info=True,
                max_age_s=float(self._params.img_sync_thres_s),
                wait_timeout_s=float(self._params.sync_wait_time_s),
            )
            if not bundle.ok:
                self._log.warning(
                    f'orbbec provider bundle failed: {bundle.error_msg}'
                )
                return None
            response = bundle.snapshot
        else:
            response = provider.snapshot(
                want_color=True,
                want_depth=True,
                want_camera_info=True,
                max_age_s=float(self._params.img_sync_thres_s),
                wait_timeout_s=float(self._params.sync_wait_time_s),
            )
            if not response.ok:
                self._log.warning(
                    f'realsense provider snapshot failed: '
                    f'{response.error_msg}'
                )
                return None
        from camera_provider import select_camera_info

        info = select_camera_info(
            response.depth_info, response.color_info
        )
        if info is None:
            return None
        intrinsic = {
            'fx': info.k[0], 'fy': info.k[4],
            'cx': info.k[2], 'cy': info.k[5],
            'width': info.width, 'height': info.height,
            'frame_id': info.header.frame_id,
        }
        # The provider cloud is requested above to guarantee the custom
        # consumer's color/cloud pair identity. The snapshot depth remains
        # the pixel-aligned representation used by the mask-centroid logic.
        return self._process_realsense(
            response.color, response.depth, intrinsic,
        )

    def _process_realsense(self, color_msg, depth_msg, intrinsic):
        rgb_bgr = self.bridge.imgmsg_to_cv2(
            color_msg, desired_encoding='bgr8',
        )
        depth_mm = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough',
        )
        depth_m = decode_depth_metres(np.asarray(depth_mm))
        h, w = depth_m.shape[:2]
        fx, fy = intrinsic['fx'], intrinsic['fy']
        cx, cy = intrinsic['cx'], intrinsic['cy']
        u = np.arange(w, dtype=np.float32)[None, :].repeat(h, axis=0)
        v = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
        z = depth_m
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        points_xyz = np.stack([x, y, z], axis=-1)
        valid = (
            (z > self._params.min_depth_m)
            & (z < self._params.max_depth_m)
        )
        header = Header(
            stamp=color_msg.header.stamp,
            frame_id=color_msg.header.frame_id,
        )
        return rgb_bgr, points_xyz, valid, header, color_msg

    def _process_orbbec(self, color_msg, points_msg, intrinsic):
        # Orbbec depth arrives as PointCloud2 already aligned to color.
        # Reproject into the image grid using the (cx, fx, cy, fy)
        # intrinsic.
        rgb_bgr = self.bridge.imgmsg_to_cv2(
            color_msg, desired_encoding='bgr8',
        )
        h, w = rgb_bgr.shape[:2]
        from sensor_msgs_py import point_cloud2 as pc2
        pts = np.asarray(
            list(pc2.read_points(
                points_msg,
                field_names=('x', 'y', 'z'),
                skip_nans=False,
            )),
            dtype=np.float32,
        )
        if pts.size == 0 or pts.shape[0] != h * w:
            return None
        pts = pts.reshape(h, w, 3)
        valid = (
            np.isfinite(pts[:, :, 2])
            & (pts[:, :, 2] > self._params.min_depth_m)
            & (pts[:, :, 2] < self._params.max_depth_m)
        )
        header = Header(
            stamp=color_msg.header.stamp,
            frame_id=color_msg.header.frame_id,
        )
        return rgb_bgr, pts, valid, header, color_msg

    def centroid_for(
        self,
        points_xyz: np.ndarray,
        mask: np.ndarray,
        valid_mask: np.ndarray,
        bbox,
        camera: str,
    ):
        x1, y1, x2, y2 = bbox
        h, w = points_xyz.shape[:2]
        x1 = max(0, min(int(x1), w - 1))
        x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return None
        sub_pts = points_xyz[y1:y2, x1:x2]
        sub_mask = mask[y1:y2, x1:x2] & valid_mask[y1:y2, x1:x2]
        if not np.any(sub_mask):
            return None
        sel = sub_pts[sub_mask]
        if sel.shape[0] < int(self._params.min_valid_centroid_pixels):
            return None
        med = np.median(sel, axis=0)
        if not np.all(np.isfinite(med)):
            return None
        p = geometry_msgs.msg.Point()
        p.x, p.y, p.z = float(med[0]), float(med[1]), float(med[2])
        return p

    def frame_supports_tf_transform(self, camera: str) -> bool:
        # Both realsense + orbbec frames are published on /tf; the gate
        # exists for hypothetical synthetic/non-TF cameras.
        return True

    def transform_point(
        self, point, target_frame: str, source_frame: str, stamp,
    ):
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, source_frame, stamp,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except (
            LookupException, ConnectivityException, ExtrapolationException,
        ) as exc:
            self._log.warning(
                f'TF {source_frame}->{target_frame} failed: {exc}'
            )
            return None
        try:
            ps = PointStamped()
            ps.header = Header(stamp=stamp, frame_id=source_frame)
            ps.point = point
            return do_transform_point(ps, tf).point
        except Exception as exc:    # noqa: BLE001
            self._log.warning(f'do_transform_point failed: {exc}')
            return None

    def write(
        self, rgb_img, detections, *, request_ctx, branch, timings,
    ):
        try:
            self.vision_logger.write(
                rgb_img, detections,
                request_ctx=request_ctx,
                branch=branch,
                timings=timings,
            )
        except Exception as exc:    # noqa: BLE001
            self._log.warning(f'vision_logger.write failed: {exc}')
