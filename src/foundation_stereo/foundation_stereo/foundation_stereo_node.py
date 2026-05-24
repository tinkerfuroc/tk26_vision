"""FoundationStereo ROS2 node — service + action + optional streaming worker.

Spec: docs/superpowers/specs/2026-05-24-foundation-stereo-design.md.
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

from tinker_vision_msgs_26.srv import FoundationStereoDepth as FSSrv

from foundation_stereo.color_align import reproject_ir_to_color
from foundation_stereo import stereo_runner as _sr
from foundation_stereo.stereo_runner import StereoRunner


# Per-camera-profile defaults. Picked by the `camera_profile` ROS param.
_PROFILES = {
    "d435": dict(
        left_topic="/camera/xarm_camera/infra1/image_rect_raw",
        right_topic="/camera/xarm_camera/infra2/image_rect_raw",
        left_info_topic="/camera/xarm_camera/infra1/camera_info",
        color_info_topic="/camera/xarm_camera/color/camera_info",
        extrinsics_topic="/camera/xarm_camera/extrinsics/depth_to_color",
        baseline_m=0.050,
    ),
    "d405": dict(
        left_topic="/camera/camera/infra1/image_rect_raw",
        right_topic="/camera/camera/infra2/image_rect_raw",
        left_info_topic="/camera/camera/infra1/camera_info",
        color_info_topic="/camera/camera/color/camera_info",
        extrinsics_topic="/camera/camera/extrinsics/depth_to_color",
        baseline_m=0.018,
    ),
}


def _info_to_K(info: CameraInfo) -> np.ndarray:
    """Convert a sensor_msgs/CameraInfo into a (3, 3) intrinsics matrix.
    Prefers the rectified-projection K-block of P over plain K when both
    are populated — matches how realsense2_camera publishes infra1's
    rect intrinsics through P."""
    P = np.asarray(info.p, dtype=np.float32).reshape(3, 4)
    if np.any(P[:3, :3] != 0):
        return P[:3, :3].copy()
    return np.asarray(info.k, dtype=np.float32).reshape(3, 3).copy()


class FoundationStereoNode(Node):

    def __init__(self):
        super().__init__("foundation_stereo")
        self._declare_parameters()
        self._bridge = CvBridge()

        self._runner = StereoRunner(weights_root=self._p("weights_root"))

        # Latest synced stereo triple (left, right, info), under a lock.
        self._latest_lock = threading.Lock()
        self._latest: Optional[Tuple[Image, Image, CameraInfo]] = None

        # Latched-style holders for color CameraInfo + IR1→Color extrinsics.
        self._color_info: Optional[CameraInfo] = None
        self._extrinsics: Optional[Tuple[np.ndarray, np.ndarray]] = None

        self._setup_subscribers()
        self._setup_service()

        self.get_logger().info(
            f"foundation_stereo ready: profile={self._p('camera_profile')}, "
            f"default_model={self._p('default_model_kind')}, "
            f"weights_root={self._p('weights_root')}, "
            f"stream_enabled={self._p('stream_enabled')}, "
            f"trt_variants={list(_sr.TRT_VARIANTS.keys())}"
        )

    # ---------- parameters ----------

    def _declare_parameters(self) -> None:
        self.declare_parameter("weights_root",
                               "/home/tinker/projects/vision_tests/dualrRGB-foundationStereo")
        self.declare_parameter("camera_profile", "d435")
        self.declare_parameter("default_model_kind", "fast_trt")
        self.declare_parameter("default_trt_variant", "output_two_stage")
        self.declare_parameter("default_scale", 0.5)
        self.declare_parameter("default_iters", 0)
        self.declare_parameter("default_z_far", 10.0)

        # Topic params with profile-derived defaults applied at runtime
        # (declared empty so the profile fills them in).
        self.declare_parameter("left_topic", "")
        self.declare_parameter("right_topic", "")
        self.declare_parameter("left_info_topic", "")
        self.declare_parameter("color_info_topic", "")
        self.declare_parameter("extrinsics_topic", "")
        self.declare_parameter("baseline_m", 0.0)

        self.declare_parameter("sync_slop_sec", 0.05)
        self.declare_parameter("sync_queue_size", 5)
        self.declare_parameter("measure_forward_ms", True)

        # Streaming-mode params — declared even when stream_enabled=false
        # so the launch file can preset them uniformly.
        self.declare_parameter("stream_enabled", False)
        self.declare_parameter("stream_align_to_color", True)
        self.declare_parameter("stream_depth_topic", "")
        self.declare_parameter("stream_info_topic", "")
        self.declare_parameter("stream_dtype", "16UC1_mm")
        self.declare_parameter("output_frame_id", "")
        self.declare_parameter("stream_publish_vis", False)
        self.declare_parameter("stream_max_fps", 0.0)
        self.declare_parameter("extrinsics_warmup_timeout_sec", 5.0)
        self.declare_parameter("stream_measure_forward_ms", False)

        self.declare_parameter("vision_logging_enabled", False)
        self.declare_parameter("vision_log_folder", "vision_log")

    def _p(self, name: str):
        return self.get_parameter(name).value

    def _topic_for(self, key: str) -> str:
        explicit = self._p(key)
        if explicit:
            return explicit
        profile = self._p("camera_profile")
        if profile not in _PROFILES:
            raise ValueError(f"unknown camera_profile: {profile!r}")
        return _PROFILES[profile][key]

    def _baseline(self) -> float:
        explicit = float(self._p("baseline_m"))
        if explicit > 0:
            return explicit
        profile = self._p("camera_profile")
        return float(_PROFILES[profile]["baseline_m"])

    # ---------- subscribers ----------

    def _setup_subscribers(self) -> None:
        sub_left = Subscriber(self, Image, self._topic_for("left_topic"),
                              qos_profile=qos_profile_sensor_data)
        sub_right = Subscriber(self, Image, self._topic_for("right_topic"),
                               qos_profile=qos_profile_sensor_data)
        sub_info = Subscriber(self, CameraInfo, self._topic_for("left_info_topic"),
                              qos_profile=qos_profile_sensor_data)
        self._sync = ApproximateTimeSynchronizer(
            [sub_left, sub_right, sub_info],
            queue_size=int(self._p("sync_queue_size")),
            slop=float(self._p("sync_slop_sec")),
        )
        self._sync.registerCallback(self._on_synced)

        # Color CameraInfo (one-shot latest cache).
        self.create_subscription(
            CameraInfo, self._topic_for("color_info_topic"),
            self._on_color_info, qos_profile_sensor_data,
        )
        # IR1->Color extrinsics (latched; small dance to avoid hard dep at import time).
        try:
            from realsense2_camera_msgs.msg import Extrinsics  # type: ignore
            self.create_subscription(
                Extrinsics, self._topic_for("extrinsics_topic"),
                self._on_extrinsics, qos_profile_sensor_data,
            )
        except ImportError:
            self.get_logger().warn(
                "realsense2_camera_msgs not available; color alignment disabled."
            )

    def _on_synced(self, left: Image, right: Image, info: CameraInfo) -> None:
        with self._latest_lock:
            self._latest = (left, right, info)

    def _on_color_info(self, info: CameraInfo) -> None:
        self._color_info = info

    def _on_extrinsics(self, msg) -> None:
        # realsense2_camera_msgs/Extrinsics: rotation (row-major 3x3),
        # translation (3,). They sit in the librealsense *optical* CS;
        # ROS optical CS is identical (x right, y down, z forward), so
        # we can use them directly.
        R = np.asarray(msg.rotation, dtype=np.float32).reshape(3, 3)
        T = np.asarray(msg.translation, dtype=np.float32).reshape(3)
        self._extrinsics = (R, T)

    # ---------- service ----------
    # Note: req.want_pointcloud is intentionally ignored in this task —
    # Task 9 / 10 will handle pointcloud generation if needed.

    def _setup_service(self) -> None:
        self.create_service(FSSrv, "~/get_depth", self._on_get_depth)

    def _on_get_depth(self, req: FSSrv.Request, resp: FSSrv.Response) -> FSSrv.Response:
        # TODO(task-9): extract _run_inference helper to share with action handler
        wall_t0 = time.time()

        with self._latest_lock:
            cached = self._latest
        if cached is None:
            resp.status = 1
            resp.error_msg = "no synced stereo frame"
            return resp

        left_msg, right_msg, info_msg = cached
        try:
            left = self._bridge.imgmsg_to_cv2(left_msg, desired_encoding="rgb8")
            right = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding="rgb8")
        except Exception as exc:
            resp.status = 3
            resp.error_msg = f"cv_bridge: {exc}"
            return resp

        K_ir = _info_to_K(info_msg)
        baseline = self._baseline()

        kind = req.model_kind or self._p("default_model_kind")
        trt_variant = req.trt_variant or self._p("default_trt_variant")
        scale = float(req.scale) if req.scale > 0 else float(self._p("default_scale"))
        iters = int(req.iters) if req.iters > 0 else int(self._p("default_iters"))
        z_far = float(req.z_far) if req.z_far > 0 else float(self._p("default_z_far"))
        measure_fwd = bool(self._p("measure_forward_ms"))

        try:
            result = self._runner.infer(
                left_rgb=left, right_rgb=right, K=K_ir, baseline=baseline,
                kind=kind, scale=scale,
                valid_iters=(iters or None), z_far=z_far,
                trt_variant=trt_variant,
                live=False,
                measure_forward_ms=measure_fwd,
                want_debug_jpeg=bool(req.want_debug_jpeg),
            )
        except FileNotFoundError as exc:
            resp.status = 2
            resp.error_msg = str(exc)
            return resp
        except Exception as exc:  # noqa: BLE001
            resp.status = 3
            resp.error_msg = f"{type(exc).__name__}: {exc}"
            return resp

        depth = result.depth  # float32 m at the scaled grid

        # Optionally align into color frame.
        if req.align_to_color:
            if self._color_info is None or self._extrinsics is None:
                resp.status = 3
                resp.error_msg = "extrinsics not available"
                return resp
            K_color = _info_to_K(self._color_info)
            K_ir_scaled = K_ir.copy()
            K_ir_scaled[:2] *= result.scale_used  # cx, cy, fx, fy scale with resize; K[2,2] stays 1
            R, T = self._extrinsics
            depth = reproject_ir_to_color(
                depth, K_ir_scaled, K_color, R, T,
                out_hw=(self._color_info.height, self._color_info.width),
            )
            out_info = self._color_info
        else:
            out_info = info_msg

        # 32FC1 m for srv/action (the streaming worker handles 16UC1 conversion).
        depth_msg = self._bridge.cv2_to_imgmsg(depth.astype(np.float32),
                                               encoding="32FC1")
        depth_msg.header = out_info.header
        resp.depth_image = depth_msg
        resp.camera_info = out_info

        if req.want_debug_jpeg and result.vis_jpg:
            cmp = CompressedImage()
            cmp.header = depth_msg.header
            cmp.format = "jpeg"
            cmp.data = list(result.vis_jpg)
            resp.debug_jpeg = cmp

        resp.status = 0
        resp.error_msg = ""
        resp.forward_ms = float(result.forward_ms)
        resp.load_s = float(result.load_s)
        resp.end_to_end_s = float(time.time() - wall_t0)
        resp.model_used = self._runner.current_model or kind
        resp.trt_variant_used = self._runner.current_trt_variant or ""
        return resp


def main(args=None):
    rclpy.init(args=args)
    node = FoundationStereoNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
