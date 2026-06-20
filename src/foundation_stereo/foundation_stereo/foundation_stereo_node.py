"""FoundationStereo ROS2 node — service + action + optional streaming worker.

Spec: docs/superpowers/specs/2026-05-24-foundation-stereo-design.md.
"""

from __future__ import annotations

import copy
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from std_msgs.msg import Header

from tinker_vision_msgs_26.action import FoundationStereoDepth as FSAction
from tinker_vision_msgs_26.srv import FoundationStereoDepth as FSSrv

from foundation_stereo.color_align_rs2 import RealsenseAligner
from foundation_stereo import stereo_runner as _sr
from foundation_stereo.stereo_runner import StereoRunner


# realsense2_camera publishes the depth_to_color Extrinsics topic latched
# (RELIABLE + TRANSIENT_LOCAL). A VOLATILE subscriber never sees the
# already-published message, so use a matching latched-style QoS.
_LATCHED_QOS = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


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


def _depth_to_pointcloud2(depth_m: np.ndarray, K: np.ndarray, header) -> PointCloud2:
    """Deproject a (H, W) float32 depth grid into a sensor_msgs/PointCloud2.

    Points with depth==0 are skipped. K is the 3x3 intrinsics for the same grid.
    The output cloud is in the same optical frame as `header.frame_id`.
    """
    H, W = depth_m.shape
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    vv, uu = np.indices((H, W), dtype=np.float32)
    Z = depth_m
    valid = Z > 0
    if not np.any(valid):
        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = 0
        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 0
        msg.is_dense = True
        msg.data = b""
        return msg

    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=-1)[valid]  # (N, 3) float32

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(pts.shape[0])
    msg.fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = pts.astype(np.float32).tobytes()
    return msg


def _depth_to_msg(depth_m: np.ndarray, dtype: str, bridge: CvBridge,
                  header) -> Image:
    if dtype == "16UC1_mm":
        mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        msg = bridge.cv2_to_imgmsg(mm, encoding="16UC1")
    else:  # 32FC1_m
        msg = bridge.cv2_to_imgmsg(depth_m.astype(np.float32), encoding="32FC1")
    msg.header = header
    return msg


def _resolve_stream_topics(node, align: bool) -> Tuple[str, str]:
    depth_topic = node._p("stream_depth_topic")
    info_topic = node._p("stream_info_topic")
    if depth_topic and info_topic:
        return depth_topic, info_topic
    if align:
        return ("~/aligned_depth_to_color/image_rect_raw",
                "~/aligned_depth_to_color/camera_info")
    return "~/depth/image_rect_raw", "~/depth/camera_info"


def _mem_report(node, tag):
    """Intra-process GPU memory breakdown (gated by CUMOTION_MEM_PROFILE env).
    live_tensors = torch.cuda.memory_allocated; torch_pool = memory_reserved;
    ctx+libs+graphs = NVML_total - torch_pool (CUDA context, cuBLAS/cuDNN,
    TensorRT engine, Warp) — the part torch's counters and external tools miss.
    No-op unless CUMOTION_MEM_PROFILE is set."""
    import os
    if not os.environ.get('CUMOTION_MEM_PROFILE'):
        return
    import subprocess
    try:
        import torch
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() / 2**20
        resv = torch.cuda.memory_reserved() / 2**20
        total = float('nan')
        pid = os.getpid()
        out = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5).stdout
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                total = float(parts[1])
                break
        node.get_logger().info(
            f'[MEMPROF {tag}] live_tensors={alloc:.0f}MB '
            f'torch_pool={resv:.0f}MB(slack={resv - alloc:.0f}) '
            f'nvsmi_total={total:.0f}MB ctx+libs+graphs={total - resv:.0f}MB')
    except Exception as e:
        node.get_logger().warn(f'[MEMPROF {tag}] failed: {e}')


class FoundationStereoNode(Node):

    def __init__(self):
        super().__init__("foundation_stereo")
        _mem_report(self, '00_init_start')
        self._declare_parameters()
        self._bridge = CvBridge()

        self._runner = StereoRunner(weights_root=self._p("weights_root"))

        # Latest synced stereo triple (left, right, info), under a lock.
        self._latest_lock = threading.Lock()
        self._latest: Optional[Tuple[Image, Image, CameraInfo]] = None

        # Latched-style holders for color CameraInfo + IR1→Color extrinsics.
        self._color_info: Optional[CameraInfo] = None
        self._extrinsics: Optional[Tuple[np.ndarray, np.ndarray]] = None

        # Cached IR1→color aligner (rs.align via software_device). Re-built
        # only when intrinsics/extrinsics/output shape change.
        self._aligner: Optional[RealsenseAligner] = None
        self._aligner_key: Optional[tuple] = None

        # Set BEFORE _setup_stream() so the stream worker thread (which may
        # start inside _setup_stream) can always reference it. Cleared while a
        # background warmup is in flight; set once the model is loaded + warm
        # (or immediately when warmup_on_launch is false, since the runner
        # lazy-loads the engine on the first real infer() under its own lock).
        self._model_ready = threading.Event()

        self._setup_subscribers()
        self._setup_service()
        self._setup_action()
        self._setup_stream()

        self._log_dir = None
        if self._p("vision_logging_enabled"):
            from foundation_stereo._logging import resolve_session_dir
            self._log_dir = resolve_session_dir(self._p("vision_log_folder"))
            self.get_logger().info(f"vision_log session dir: {self._log_dir}")

        if bool(self._p("warmup_on_launch")):
            # Run the cold TRT engine load (~2-5 s) OFF the synchronous init
            # path in a background daemon thread so __init__ returns and
            # main() reaches rclpy.spin() in milliseconds. With the executor
            # live, the latched extrinsics + color_info subscription
            # callbacks fire right away and the stream loop's align-to-color
            # readiness gate passes well inside its window — instead of the
            # gate's clock burning down during a blocking warmup. The worker
            # sets self._model_ready when done; the stream loop waits on it
            # before its first inference (see _stream_loop), and the runner's
            # own lock serializes engine access so warm + use can't race.
            self._warmup_thread = threading.Thread(
                target=self._warmup_model_threaded,
                name="fs-warmup", daemon=True,
            )
            self._warmup_thread.start()
        else:
            # No launch-time warmup: the runner lazy-loads on first infer()
            # under its own lock, so the model is "ready" to be used now.
            self._model_ready.set()

        self.get_logger().info(
            f"foundation_stereo ready: profile={self._p('camera_profile')}, "
            f"trt_variant={self._p('default_trt_variant')}, "
            f"weights_root={self._p('weights_root')}, "
            f"stream_enabled={self._p('stream_enabled')}, "
            f"trt_variants={list(_sr.TRT_VARIANTS.keys())}"
        )

        self.create_timer(5.0, lambda: _mem_report(self, 'periodic'))

    # ---------- parameters ----------

    def _declare_parameters(self) -> None:
        self.declare_parameter("weights_root",
                               "/home/tinker/projects/vision_tests/dualrRGB-foundationStereo")
        self.declare_parameter("camera_profile", "d435")
        # `default_model_kind` is kept for backwards-compat but ignored;
        # the node serves only `fast_trt`. Any other value in a request
        # is rejected. (See _run_inference.)
        self.declare_parameter("default_model_kind", "fast_trt")
        self.declare_parameter("default_trt_variant", "output_two_stage")
        # Warm the default TRT engine (load + one live forward) at node
        # startup so the first real request lands at warm latency
        # (~30 ms) instead of cold (~2-5 s for TRT engine load + CUDA
        # init). Disable with -p warmup_on_launch:=false for fast dev
        # iteration when you don't care about first-call latency.
        self.declare_parameter("warmup_on_launch", True)
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
        self.declare_parameter("stream_max_fps", 15.0)
        self.declare_parameter("extrinsics_warmup_timeout_sec", 15.0)
        self.declare_parameter("stream_measure_forward_ms", False)
        # QoS reliability for the streaming depth + camera_info publishers.
        # 'reliable' (default) is a drop-in for realsense aligned_depth_to_color
        # and shows up in default-QoS RViz; 'best_effort' is the lower-overhead
        # sensor-stream profile (use it if a consumer subscribes BEST_EFFORT or
        # the link drops frames under load).
        self.declare_parameter("stream_qos_reliability", "reliable")

        self.declare_parameter("vision_logging_enabled", False)
        self.declare_parameter("vision_log_folder", "vision_log")

    def _p(self, name: str):
        return self.get_parameter(name).value

    def _stream_qos(self) -> QoSProfile:
        """QoS for the streaming depth + camera_info publishers.

        Reliability is selectable via `stream_qos_reliability`; durability is
        VOLATILE (the info is republished every frame, so no latching needed)
        and history KEEP_LAST/5 to match realsense's sensor streams.
        """
        val = str(self._p("stream_qos_reliability") or "reliable").lower()
        if val not in ("reliable", "best_effort"):
            self.get_logger().warn(
                f"stream_qos_reliability='{val}' invalid; using 'reliable'"
            )
            val = "reliable"
        reliability = (ReliabilityPolicy.RELIABLE if val == "reliable"
                       else ReliabilityPolicy.BEST_EFFORT)
        return QoSProfile(
            depth=5,
            history=HistoryPolicy.KEEP_LAST,
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
        )

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
        # realsense2_camera publishes this once at startup with TRANSIENT_LOCAL,
        # so the subscription must match that durability or it never fires.
        try:
            from realsense2_camera_msgs.msg import Extrinsics  # type: ignore
            self.create_subscription(
                Extrinsics, self._topic_for("extrinsics_topic"),
                self._on_extrinsics, _LATCHED_QOS,
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

    def _warmup_model_threaded(self) -> None:
        """Background-thread entry point for launch-time warmup.

        Runs the (slow, cold) TRT engine load + first forward off the
        synchronous __init__ path so main() reaches rclpy.spin()
        immediately. Sets self._model_ready in a finally block so the
        stream loop's model-ready gate is released whether the warmup
        forward succeeds or fails — a genuine failure surfaces again on
        the first real infer() (same as before), but we must never leave
        the stream worker blocked forever. Engine access is serialized by
        the runner's internal lock, so this thread and the stream worker
        can't race on the TRT engine.
        """
        try:
            self._warmup_model()
            _mem_report(self, '01_after_model_load')
        finally:
            self._model_ready.set()

    def _warmup_model(self) -> None:
        """Load the default TRT engine + run one live forward so the
        first real request lands at warm latency.

        Hot path inside `infer()` does: TRT engine load (~2-5 s for the
        first variant), lazy buffer allocation on first execute (~50-
        100 ms), CUDA kernel JIT for the resize ops (~5-20 ms). Doing
        all that here means the user's first service call is ~30 ms
        instead of ~5 s.

        We only warm the *default* variant. Per-request overrides will
        still pay cold cost on first use of a non-default variant —
        that's a one-time hit and intentional. Non-TRT model_kinds are
        rejected at the service layer, so we don't bother with them.
        """
        variant = str(self._p("default_trt_variant")) or "output_two_stage"
        if variant not in _sr.TRT_VARIANTS:
            self.get_logger().warn(
                f"warmup skipped: default_trt_variant={variant!r} not "
                f"in available {list(_sr.TRT_VARIANTS.keys())}")
            return

        # Build dummy IR pair at the camera profile's native resolution.
        # `_run` inside the runner will downsample to the engine's input
        # shape; what matters is that the call exercises the full
        # forward path (resize → engine forward → resize back).
        H, W = 480, 848  # native IR/color size for both d435 and d405
        dummy_l = np.zeros((H, W, 3), dtype=np.uint8)
        dummy_r = np.zeros((H, W, 3), dtype=np.uint8)
        # Build a plausible K at IR resolution (used only for depth-math
        # we'll skip via live=True).
        K = np.array([[423.0, 0.0, W / 2.0],
                      [0.0,   423.0, H / 2.0],
                      [0.0,   0.0,   1.0]], dtype=np.float32)
        baseline = float(self._baseline()) or 0.05

        t0 = time.time()
        try:
            self._runner.infer(
                left_rgb=dummy_l, right_rgb=dummy_r,
                K=K, baseline=baseline,
                kind="fast_trt", scale=float(self._p("default_scale")),
                valid_iters=None, z_far=1.0,
                trt_variant=variant,
                live=True,                # skip depth math + pointcloud
                measure_forward_ms=False,
                want_debug_jpeg=False,
            )
            self.get_logger().info(
                f"warmup ok: variant={variant} loaded + first-forward "
                f"in {time.time() - t0:.2f}s")
        except Exception as exc:  # noqa: BLE001
            # Don't crash — the node may still be useful for other purposes
            # (e.g. just streaming after extrinsics arrive). Real requests
            # will surface the same error.
            self.get_logger().error(
                f"warmup FAILED (variant={variant}): {exc}. The first "
                f"real request will pay cold-load cost instead.")

    def _get_aligner(self, *, K_ir: np.ndarray, ir_info: CameraInfo,
                     ir_hw: Tuple[int, int]) -> RealsenseAligner:
        """Build (or reuse a cached) RealsenseAligner for IR1→color
        alignment. The aligner is rebuilt only when K_color, K_ir, R, T,
        or the IR/color output shape change."""
        K_color = _info_to_K(self._color_info)
        R, T = self._extrinsics
        out_hw = (self._color_info.height, self._color_info.width)
        D_color = tuple(self._color_info.d) if self._color_info.d else None
        D_ir = tuple(ir_info.d) if ir_info.d else None
        key = (
            K_color.tobytes(), K_ir.tobytes(),
            R.tobytes(), T.tobytes(),
            tuple(ir_hw), tuple(out_hw),
            D_color, D_ir,
        )
        if self._aligner is None or self._aligner_key != key:
            self._aligner = RealsenseAligner(
                K_ir=K_ir, K_color=K_color,
                R_ir_to_color=R, T_ir_to_color=T,
                ir_hw=ir_hw, color_hw=out_hw,
                D_color=list(D_color) if D_color else None,
                D_ir=list(D_ir) if D_ir else None,
            )
            self._aligner_key = key
            self.get_logger().info(
                f"RealsenseAligner (re)built  ir={ir_hw} color={out_hw}  "
                f"|T_d2c|={float(np.linalg.norm(T))*1000:.2f}mm  "
                f"D_color_nonzero={bool(D_color) and any(D_color)}")
        return self._aligner

    # ---------- shared inference core ----------

    def _run_inference(
        self,
        *,
        model_kind: str,
        trt_variant: str,
        scale: float,
        iters: int,
        z_far: float,
        want_pointcloud: bool,
        want_debug_jpeg: bool,
        align_to_color: bool,
        on_stage=None,
    ) -> dict:
        """Single inference + optional color alignment.

        Returns a dict the caller copies into srv response / action result
        fields. `on_stage(stage_str)` is invoked at each major phase
        boundary when provided (for action feedback)."""
        wall_t0 = time.time()
        out: dict = {
            "status": 0,
            "error_msg": "",
            "depth_image": None,
            "camera_info": None,
            "pointcloud": None,
            "debug_jpeg": None,
            "forward_ms": 0.0,
            "load_s": 0.0,
            "end_to_end_s": 0.0,
            "model_used": "",
            "trt_variant_used": "",
        }

        def _stage(name: str) -> None:
            if on_stage is not None:
                try:
                    on_stage(name)
                except Exception:  # noqa: BLE001 — feedback must not break inference
                    self.get_logger().warn(f"on_stage callback raised at {name!r}")

        # TRT-only: reject any non-fast_trt model kind. Empty string
        # means "use the default" (which is also fast_trt).
        if model_kind and model_kind != "fast_trt":
            out["status"] = 4
            out["error_msg"] = (
                f"model_kind={model_kind!r} not supported; this node "
                "only serves 'fast_trt' (TRT-engine inference). Leave "
                "model_kind empty or set it to 'fast_trt'."
            )
            out["end_to_end_s"] = float(time.time() - wall_t0)
            return out
        model_kind = "fast_trt"

        with self._latest_lock:
            cached = self._latest
        if cached is None:
            out["status"] = 1
            out["error_msg"] = "no synced stereo frame"
            out["end_to_end_s"] = float(time.time() - wall_t0)
            return out

        left_msg, right_msg, info_msg = cached
        try:
            left = self._bridge.imgmsg_to_cv2(left_msg, desired_encoding="rgb8")
            right = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding="rgb8")
        except Exception as exc:  # noqa: BLE001
            out["status"] = 3
            out["error_msg"] = f"cv_bridge: {exc}"
            out["end_to_end_s"] = float(time.time() - wall_t0)
            return out

        K_ir = _info_to_K(info_msg)
        baseline = self._baseline()
        measure_fwd = bool(self._p("measure_forward_ms"))

        _stage("running_forward")
        try:
            result = self._runner.infer(
                left_rgb=left, right_rgb=right, K=K_ir, baseline=baseline,
                kind=model_kind, scale=scale,
                valid_iters=(iters or None), z_far=z_far,
                trt_variant=trt_variant,
                live=False,
                measure_forward_ms=measure_fwd,
                want_debug_jpeg=want_debug_jpeg,
            )
        except FileNotFoundError as exc:
            out["status"] = 2
            out["error_msg"] = str(exc)
            out["end_to_end_s"] = float(time.time() - wall_t0)
            return out
        except Exception as exc:  # noqa: BLE001
            out["status"] = 3
            out["error_msg"] = f"{type(exc).__name__}: {exc}"
            out["end_to_end_s"] = float(time.time() - wall_t0)
            return out

        depth = result.depth  # float32 m at the scaled grid

        # Optionally align into color frame.
        # Always preserve the synced stereo frame's timestamp; frame_id
        # depends on whether we aligned (color) or stayed in IR (info_msg).
        if align_to_color:
            _stage("aligning_to_color")
            if self._color_info is None or self._extrinsics is None:
                out["status"] = 3
                out["error_msg"] = "extrinsics not available"
                out["end_to_end_s"] = float(time.time() - wall_t0)
                return out
            K_color = _info_to_K(self._color_info)
            K_ir_scaled = K_ir.copy()
            K_ir_scaled[:2] *= result.scale_used  # cx, cy, fx, fy scale with resize; K[2,2] stays 1
            aligner = self._get_aligner(
                K_ir=K_ir_scaled, ir_info=info_msg, ir_hw=depth.shape)
            depth = aligner.align(np.ascontiguousarray(depth, dtype=np.float32))
            out_info = self._color_info
            K_for_cloud = K_color
        else:
            out_info = info_msg
            K_ir_scaled = K_ir.copy()
            K_ir_scaled[:2] *= result.scale_used
            K_for_cloud = K_ir_scaled

        depth_header = Header(
            stamp=info_msg.header.stamp,
            frame_id=out_info.header.frame_id,
        )

        # 32FC1 m for srv/action (the streaming worker handles 16UC1 conversion).
        depth_msg = self._bridge.cv2_to_imgmsg(depth.astype(np.float32),
                                               encoding="32FC1")
        depth_msg.header = depth_header
        out["depth_image"] = depth_msg
        # camera_info copy still mirrors out_info (height/width/intrinsics),
        # but its header gets the synced stamp too.
        info_for_resp = copy.copy(out_info)
        info_for_resp.header = depth_header
        out["camera_info"] = info_for_resp

        if want_pointcloud:
            out["pointcloud"] = _depth_to_pointcloud2(depth, K_for_cloud, depth_header)

        if want_debug_jpeg and result.vis_jpg:
            _stage("encoding_debug")
            cmp = CompressedImage()
            cmp.header = depth_header
            cmp.format = "jpeg"
            cmp.data = list(result.vis_jpg)
            out["debug_jpeg"] = cmp

        out["status"] = 0
        out["error_msg"] = ""
        out["forward_ms"] = float(result.forward_ms)
        out["load_s"] = float(result.load_s)
        out["end_to_end_s"] = float(time.time() - wall_t0)
        out["model_used"] = self._runner.current_model or model_kind
        out["trt_variant_used"] = self._runner.current_trt_variant or ""

        if self._log_dir is not None:
            self._log_call(left, depth, depth_header)

        return out

    def _log_call(self, left_rgb, depth, header):
        """Dump the input left image + output depth to vision_log/."""
        try:
            ts = time.strftime("%Y%m%d_%H%M%S")
            ms = int(time.time() * 1000) % 1000
            stem = f"foundation_stereo_node_get_depth_{ts}_{ms:03d}"
            cv2.imwrite(f"{self._log_dir}/{stem}_orig.jpg",
                        cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
            mm = np.clip(depth * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(f"{self._log_dir}/{stem}_depth.png", mm)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"vision_log write failed: {exc}",
                                   throttle_duration_sec=10.0)

    # ---------- service ----------

    def _setup_service(self) -> None:
        self.create_service(FSSrv, "~/get_depth", self._on_get_depth)

    def _on_get_depth(self, req: FSSrv.Request, resp: FSSrv.Response) -> FSSrv.Response:
        result = self._run_inference(
            model_kind=(req.model_kind or "fast_trt"),
            trt_variant=(req.trt_variant or self._p("default_trt_variant")),
            scale=float(req.scale) if req.scale > 0 else float(self._p("default_scale")),
            iters=int(req.iters) if req.iters > 0 else int(self._p("default_iters")),
            z_far=float(req.z_far) if req.z_far > 0 else float(self._p("default_z_far")),
            want_pointcloud=bool(req.want_pointcloud),
            want_debug_jpeg=bool(req.want_debug_jpeg),
            align_to_color=bool(req.align_to_color),
        )

        resp.status = result["status"]
        resp.error_msg = result["error_msg"]
        if result["depth_image"] is not None:
            resp.depth_image = result["depth_image"]
        if result["camera_info"] is not None:
            resp.camera_info = result["camera_info"]
        if result["pointcloud"] is not None:
            resp.pointcloud = result["pointcloud"]
        if result["debug_jpeg"] is not None:
            resp.debug_jpeg = result["debug_jpeg"]
        resp.forward_ms = result["forward_ms"]
        resp.load_s = result["load_s"]
        resp.end_to_end_s = result["end_to_end_s"]
        resp.model_used = result["model_used"]
        resp.trt_variant_used = result["trt_variant_used"]
        return resp

    # ---------- action ----------

    def _setup_action(self) -> None:
        self._action = ActionServer(
            self,
            FSAction,
            "~/infer_depth",
            execute_callback=self._on_infer_depth,
            goal_callback=lambda goal: GoalResponse.ACCEPT,
            cancel_callback=lambda goal: CancelResponse.ACCEPT,
        )

    def _on_infer_depth(self, goal_handle):
        req = goal_handle.request
        resp = FSAction.Result()
        feedback = FSAction.Feedback()
        action_t0 = time.time()

        def fb(stage: str) -> None:
            feedback.current_stage = stage
            feedback.elapsed_s = float(time.time() - action_t0)
            goal_handle.publish_feedback(feedback)

        # Cancel-check before doing any work.
        if goal_handle.is_cancel_requested:
            goal_handle.canceled()
            resp.status = 3
            resp.error_msg = "cancelled before inference"
            return resp

        result = self._run_inference(
            model_kind=(req.model_kind or "fast_trt"),
            trt_variant=(req.trt_variant or self._p("default_trt_variant")),
            scale=float(req.scale) if req.scale > 0 else float(self._p("default_scale")),
            iters=int(req.iters) if req.iters > 0 else int(self._p("default_iters")),
            z_far=float(req.z_far) if req.z_far > 0 else float(self._p("default_z_far")),
            want_pointcloud=bool(req.want_pointcloud),
            want_debug_jpeg=bool(req.want_debug_jpeg),
            align_to_color=bool(req.align_to_color),
            on_stage=fb,
        )

        # Copy fields onto the action Result.
        resp.status = result["status"]
        resp.error_msg = result["error_msg"]
        if result["depth_image"] is not None:
            resp.depth_image = result["depth_image"]
        if result["camera_info"] is not None:
            resp.camera_info = result["camera_info"]
        if result["pointcloud"] is not None:
            resp.pointcloud = result["pointcloud"]
        if result["debug_jpeg"] is not None:
            resp.debug_jpeg = result["debug_jpeg"]
        resp.forward_ms = result["forward_ms"]
        resp.load_s = result["load_s"]
        resp.end_to_end_s = result["end_to_end_s"]
        resp.model_used = result["model_used"]
        resp.trt_variant_used = result["trt_variant_used"]

        goal_handle.succeed()
        return resp

    # ---------- streaming worker ----------

    def _setup_stream(self) -> None:
        if not self._p("stream_enabled"):
            return
        align = bool(self._p("stream_align_to_color"))

        # IMPORTANT: do NOT wait for extrinsics here — __init__ runs before
        # rclpy.spin(), so subscription callbacks can't fire and the loop
        # would always time out. The worker thread (below) does the warmup
        # wait once the executor is alive.

        depth_topic, info_topic = _resolve_stream_topics(self, align)
        stream_qos = self._stream_qos()
        self.get_logger().info(
            f"stream depth/info QoS reliability="
            f"{stream_qos.reliability.name.lower()}"
        )
        self._stream_depth_pub = self.create_publisher(
            Image, depth_topic, stream_qos
        )
        self._stream_info_pub = self.create_publisher(
            CameraInfo, info_topic, stream_qos
        )
        self._stream_vis_pub = (
            self.create_publisher(CompressedImage, "~/debug/disparity/compressed",
                                  qos_profile_sensor_data)
            if self._p("stream_publish_vis") else None
        )

        self._stream_stop = threading.Event()
        self._stream_thread = threading.Thread(
            target=self._stream_loop, name="fs-stream", daemon=True,
        )
        self._stream_thread.start()
        self.get_logger().info(
            f"streaming publisher created: depth={depth_topic}, "
            f"info={info_topic}, dtype={self._p('stream_dtype')}, align={align}"
        )

    def _stream_loop(self) -> None:
        align = bool(self._p("stream_align_to_color"))
        dtype = str(self._p("stream_dtype"))
        out_frame = str(self._p("output_frame_id"))
        max_fps = float(self._p("stream_max_fps"))
        min_period = (1.0 / max_fps) if max_fps > 0 else 0.0
        measure_fwd = bool(self._p("stream_measure_forward_ms"))

        # Extrinsics warm-up runs here so rclpy.spin()'s executor is
        # already running in the main thread and the latched extrinsics +
        # color_info callbacks can fire.
        if align:
            warmup = float(self._p("extrinsics_warmup_timeout_sec"))
            deadline = time.time() + warmup
            while time.time() < deadline and not self._stream_stop.is_set():
                if self._extrinsics is not None and self._color_info is not None:
                    break
                time.sleep(0.1)
            if self._extrinsics is None or self._color_info is None:
                self.get_logger().error(
                    "stream_align_to_color=true but extrinsics or "
                    f"color_info not received within {warmup} s; "
                    "publisher not emitting."
                )
                return

        # Inputs are ready; now block until the model is warmed before the
        # first inference. With warmup_on_launch=true this waits on the
        # background warmup thread (the cold TRT load runs off the init path,
        # so the executor stayed live and the inputs gate above already
        # passed); with warmup_on_launch=false the Event is pre-set, so this
        # returns immediately and the runner lazy-loads under its own lock.
        # This guarantees no inference output is emitted before the engine is
        # actually loaded. Polled so _stream_stop can break a long warmup.
        while not self._model_ready.is_set():
            if self._stream_stop.is_set():
                return
            if self._model_ready.wait(timeout=0.1):
                break
        self.get_logger().info("stream worker: model ready, emitting depth")

        last_seq = None
        last_emit = 0.0

        while not self._stream_stop.is_set():
            with self._latest_lock:
                cached = self._latest
            if cached is None:
                time.sleep(0.01)
                continue

            left_msg, right_msg, info_msg = cached
            seq = (left_msg.header.stamp.sec, left_msg.header.stamp.nanosec)
            if seq == last_seq:
                time.sleep(0.001)
                continue
            if min_period > 0 and (time.time() - last_emit) < min_period:
                time.sleep(0.001)
                continue
            last_seq = seq

            try:
                left = self._bridge.imgmsg_to_cv2(left_msg, desired_encoding="rgb8")
                right = self._bridge.imgmsg_to_cv2(right_msg, desired_encoding="rgb8")
            except Exception as exc:
                self.get_logger().warn(f"cv_bridge: {exc}", throttle_duration_sec=5.0)
                continue

            K_ir = _info_to_K(info_msg)
            try:
                result = self._runner.infer(
                    left_rgb=left, right_rgb=right, K=K_ir,
                    baseline=self._baseline(),
                    kind="fast_trt",
                    scale=float(self._p("default_scale")),
                    valid_iters=(int(self._p("default_iters")) or None),
                    z_far=float(self._p("default_z_far")),
                    trt_variant=self._p("default_trt_variant"),
                    live=False,
                    measure_forward_ms=measure_fwd,
                    want_debug_jpeg=bool(self._stream_vis_pub),
                )
            except FileNotFoundError as exc:
                self.get_logger().error(
                    f"streaming worker exiting — weights missing: {exc}"
                )
                return
            except Exception as exc:  # noqa: BLE001
                self.get_logger().exception("stream inference failed")
                time.sleep(0.05)
                continue

            depth = result.depth
            if align:
                color_info = self._color_info
                extrinsics = self._extrinsics
                if color_info is None or extrinsics is None:
                    # Shouldn't happen — warmup gate already passed — but defensive.
                    continue
                K_ir_scaled = K_ir.copy()
                K_ir_scaled[:2] *= result.scale_used  # cx, cy, fx, fy scale with resize; K[2,2] stays 1
                aligner = self._get_aligner(
                    K_ir=K_ir_scaled, ir_info=info_msg, ir_hw=depth.shape)
                depth = aligner.align(np.ascontiguousarray(depth, dtype=np.float32))
                out_info = color_info
            else:
                out_info = info_msg

            header = Header(
                stamp=info_msg.header.stamp,
                frame_id=(out_frame or out_info.header.frame_id),
            )

            depth_msg = _depth_to_msg(depth, dtype, self._bridge, header)
            info_out = copy.copy(out_info)
            info_out.header = header

            self._stream_depth_pub.publish(depth_msg)
            self._stream_info_pub.publish(info_out)

            if self._stream_vis_pub is not None and result.vis_jpg:
                cmp = CompressedImage()
                cmp.header = header
                cmp.format = "jpeg"
                cmp.data = list(result.vis_jpg)
                self._stream_vis_pub.publish(cmp)

            last_emit = time.time()

    def destroy_node(self):
        if getattr(self, "_stream_stop", None) is not None:
            self._stream_stop.set()
        if getattr(self, "_stream_thread", None) is not None:
            self._stream_thread.join(timeout=2.0)
        super().destroy_node()


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
