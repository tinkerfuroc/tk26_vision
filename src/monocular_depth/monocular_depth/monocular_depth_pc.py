"""Monocular-depth-fused depth-image action server.

Runs Depth Anything 3 on the latest RGB frame, scale-shift-aligns the
relative monocular depth against the live sensor depth on overlapping
valid pixels, fuses the two (holes-only by default), and returns the
fused 32FC1 depth image (metres) at the source RGB resolution.

When the goal sets ``debug_publish=True`` the colored PointCloud2 is
also computed and published on ``debug_pc_topic`` (default
``~/debug_points``, SensorDataQoS) — but the cloud is **not** part of
the action result. ``stride`` subsamples the debug cloud only;
``depth_image`` is always at the source resolution so it stays
pixel-aligned to the source RGB.

Subscribes to both RealSense (`/camera/xarm_camera/color/image_raw` +
`/camera/xarm_camera/aligned_depth_to_color/image_raw`) and Orbbec
(`/camera/color/image_raw` + `/camera/depth/image_raw`); the goal's
``camera`` field selects which one.

Action: ``tinker_vision_msgs_26/action/MonocularDepthPC``.

Runs under the dedicated `.venv-da3` venv — `depth_anything_3` pins
`numpy<2` so this node lives in its own package, isolated from the
shared `.venv-vision-main`.
"""

import copy
import threading

import numpy as np
import rclpy
import rclpy.executors
import torch
from cv_bridge import CvBridge
from depth_anything_3.api import DepthAnything3
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from tinker_vision_msgs_26.action import MonocularDepthPC

from vision_util._pc_utils import (
    build_xy_table_cuda,
    make_pc2_xyzrgb,
    pack_rgb_u8_to_float32_cuda,
)
from vision_util.camera_intake import (
    CameraIntake,
    IntakeConfig,
    StreamSpec,
    configure_camera_backend,
)


_REALSENSE = 'realsense'
_ORBBEC = 'orbbec'

_TOPICS = {
    _REALSENSE: {
        'color': '/camera/xarm_camera/color/image_raw',
        'depth': '/camera/xarm_camera/aligned_depth_to_color/image_raw',
        'info': '/camera/xarm_camera/aligned_depth_to_color/camera_info',
    },
    _ORBBEC: {
        'color': '/camera/color/image_raw',
        'depth': '/camera/depth/image_raw',
        'info': '/camera/color/camera_info',
    },
}

_FILL_HOLES_ONLY = 'holes_only'
_FILL_FULL_OVERRIDE = 'full_override'


class MonocularDepthPCService(Node):
    def __init__(self):
        super().__init__('monocular_depth_pc')

        if not torch.cuda.is_available():
            raise RuntimeError(
                'monocular_depth_pc requires CUDA — no GPU device available'
            )
        self.device = torch.device('cuda')
        self.get_logger().info(
            f'CUDA OK: {torch.cuda.get_device_name(0)} '
            f'(torch {torch.__version__}, numpy {np.__version__})'
        )

        self.declare_parameter('da3_model', 'depth-anything/DA3-SMALL')
        self.declare_parameter('fill_mode', _FILL_HOLES_ONLY)
        self.declare_parameter('align_min_overlap_pixels', 2000)
        self.declare_parameter('align_trim_frac', 0.05)
        self.declare_parameter('output_frame_id', '')
        self.declare_parameter('debug_pc_topic', '~/debug_points')
        self.declare_parameter('vision_logging_enabled', False)
        self.declare_parameter('camera_backend', 'service')
        self.declare_parameter(
            'realsense_provider_endpoint', '/wrist_camera_server')
        self.declare_parameter(
            'orbbec_provider_endpoint', '/head_camera_server')
        self.declare_parameter('camera_provider_wait_timeout_s', 0.5)
        self.declare_parameter('camera_provider_response_timeout_s', 5.0)

        self.da3_model_id = (
            self.get_parameter('da3_model').get_parameter_value().string_value
        )
        self.fill_mode = (
            self.get_parameter('fill_mode').get_parameter_value().string_value
        )
        self.align_min_overlap_pixels = int(
            self.get_parameter('align_min_overlap_pixels')
            .get_parameter_value().integer_value
        )
        self.align_trim_frac = float(
            self.get_parameter('align_trim_frac')
            .get_parameter_value().double_value
        )
        self._output_frame_override = (
            self.get_parameter('output_frame_id')
            .get_parameter_value().string_value
        )
        self.vision_logging_enabled = (
            self.get_parameter('vision_logging_enabled')
            .get_parameter_value().bool_value
        )

        self.bridge = CvBridge()

        self._load_da3()

        self.lock_intr = threading.Lock()
        self._intr_key = {_REALSENSE: None, _ORBBEC: None}
        self._xy_table_cuda = {_REALSENSE: None, _ORBBEC: None}

        self._subscribe_cameras()

        self.action_cb_group = MutuallyExclusiveCallbackGroup()
        self._action_server = ActionServer(
            self,
            MonocularDepthPC,
            'monocular_depth_pc',
            self._execute_callback,
            callback_group=self.action_cb_group,
        )

        debug_topic = (
            self.get_parameter('debug_pc_topic')
            .get_parameter_value().string_value
        )
        # Reliable QoS (depth=5) so RViz default subscriber matches without
        # extra config. Debug PC is one cloud per goal — drop is fine but
        # publishing reliably keeps every consumer happy.
        self._debug_pc_pub = self.create_publisher(
            PointCloud2, debug_topic, 5,
        )

        self.get_logger().info(
            f'monocular_depth_pc action server initialized '
            f'(model={self.da3_model_id}, fill_mode={self.fill_mode}, '
            f'debug_pc_topic={debug_topic}).'
        )

    def _load_da3(self):
        self.get_logger().info(f'Loading DA3 model: {self.da3_model_id}')
        self.model = DepthAnything3.from_pretrained(self.da3_model_id)
        self.model = self.model.to(device=self.device).eval()
        self.get_logger().info('DA3 model ready.')

    def _subscribe_cameras(self):
        self._intakes = {}
        for cam, topics in _TOPICS.items():
            cb_sync = MutuallyExclusiveCallbackGroup()
            cfg = configure_camera_backend(
                self,
                IntakeConfig(
                    camera=cam,
                    color=StreamSpec(
                        topics['color'], best_effort=True, qos_depth=5),
                    depth=StreamSpec(
                        topics['depth'], best_effort=True, qos_depth=5),
                    camera_info=StreamSpec(
                        topics['info'], best_effort=False, qos_depth=10),
                    sync_queue=3,
                    sync_slop_s=0.05,
                    age_source='stamp',
                ),
                default_endpoint=(
                    '/wrist_camera_server'
                    if cam == _REALSENSE
                    else '/head_camera_server'
                ),
            )
            self._intakes[cam] = CameraIntake(
                self,
                cfg,
                callback_group=cb_sync,
                bridge=self.bridge,
            )

    def _maybe_update_xy_table(self, cam: str, info: CameraInfo):
        h, w = int(info.height), int(info.width)
        if h == 0 or w == 0:
            return
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        key = (h, w, fx, fy, cx, cy)
        with self.lock_intr:
            if key == self._intr_key[cam]:
                return
            self._xy_table_cuda[cam] = build_xy_table_cuda(
                h, w, fx, fy, cx, cy, self.device,
            )
            self._intr_key[cam] = key
            self.get_logger().info(
                f'[{cam}] Intrinsics cached: {w}x{h} fx={fx:.2f} fy={fy:.2f} '
                f'cx={cx:.2f} cy={cy:.2f}'
            )

    async def _execute_callback(self, goal_handle):
        request = goal_handle.request
        feedback = MonocularDepthPC.Feedback()
        result = MonocularDepthPC.Result()

        cam = request.camera
        stride = max(1, int(request.stride))

        try:
            self._publish_stage(goal_handle, feedback, 'snapshot')
            color_msg, depth_msg, _info = self._snapshot(cam)

            self._publish_stage(goal_handle, feedback, 'da3_inference')
            inv_p = self._run_da3(color_msg)

            self._publish_stage(goal_handle, feedback, 'alignment')
            Z_s = self._depth_to_metres_cuda(depth_msg)
            self._validate_shapes(cam, Z_s, inv_p)
            a, b = self._fit_affine_inverse(inv_p, Z_s)

            self._publish_stage(goal_handle, feedback, 'hole_fill')
            Z_filled = self._apply_fill(Z_s, inv_p, a, b, self.fill_mode)

            self._publish_stage(goal_handle, feedback, 'publish_depth')
            result.depth_image = self._depth_to_image_msg(Z_filled, depth_msg)

            if request.debug_publish:
                self._publish_stage(goal_handle, feedback, 'debug_publish_pc')
                cloud = self._deproject(
                    Z_filled, color_msg, depth_msg, cam, stride,
                )
                self._debug_pc_pub.publish(cloud)
        except RuntimeError as e:
            self.get_logger().error(f'monocular_depth_pc aborted: {e}')
            goal_handle.abort()
            return result

        goal_handle.succeed()
        return result

    @staticmethod
    def _publish_stage(goal_handle, feedback, stage):
        feedback.current_stage = stage
        goal_handle.publish_feedback(feedback)

    def _snapshot(self, cam: str):
        if cam not in _TOPICS:
            raise RuntimeError(f"Unknown camera '{cam}'")
        bundle = self._intakes[cam].wait_fresh(
            max_age_s=0.5,
            timeout_s=1.5,
            on_timeout='fail',
        )
        info = self._intakes[cam].camera_info()
        if bundle is None or info is None:
            raise RuntimeError(f"No fresh provider frame for '{cam}'")
        color_msg = bundle.color_msg
        depth_msg = bundle.depth_msg
        self._maybe_update_xy_table(cam, info)
        return color_msg, depth_msg, info

    @torch.inference_mode()
    def _run_da3(self, color_msg: Image) -> torch.Tensor:
        rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        h, w = rgb.shape[:2]
        # DA3.inference accepts a list of np.ndarray or paths.
        prediction = self.model.inference([rgb])
        depth_arr = prediction.depth[0]      # (h_proc, w_proc) metric, on cuda or cpu
        if isinstance(depth_arr, np.ndarray):
            depth_t = torch.from_numpy(np.ascontiguousarray(depth_arr)).to(
                self.device, non_blocking=True,
            )
        else:
            depth_t = depth_arr.to(self.device, non_blocking=True).float()
        if depth_t.dim() == 2:
            depth_t = depth_t.unsqueeze(0).unsqueeze(0)         # (1, 1, h, w)
        elif depth_t.dim() == 3:
            depth_t = depth_t.unsqueeze(1)                      # (N, 1, h, w)
        depth_resized = torch.nn.functional.interpolate(
            depth_t, size=(h, w), mode='bilinear', align_corners=False,
        ).squeeze(1).squeeze(0)                                  # (H, W)
        # DA3 outputs metric depth — convert to inverse-depth-like for the
        # affine fit (handles bias robustly across scenes).
        valid = depth_resized > 1e-6
        inv = torch.zeros_like(depth_resized)
        inv[valid] = 1.0 / depth_resized[valid]
        return inv

    def _depth_to_metres_cuda(self, depth_msg: Image) -> torch.Tensor:
        arr = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough',
        )
        if arr.dtype == np.uint16:
            arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
            t = torch.from_numpy(arr_f32).to(self.device, non_blocking=True)
            return t * 0.001
        if arr.dtype == np.float32:
            return torch.from_numpy(np.ascontiguousarray(arr)).to(
                self.device, non_blocking=True,
            )
        raise RuntimeError(
            f'Unsupported depth dtype {arr.dtype} (encoding '
            f'{depth_msg.encoding!r}); expected uint16 mm or float32 m.'
        )

    def _validate_shapes(self, cam: str, Z_s: torch.Tensor, inv_p: torch.Tensor):
        with self.lock_intr:
            key = self._intr_key[cam]
        if key is None:
            raise RuntimeError(f"No intrinsics cached for '{cam}'")
        h_intr, w_intr = key[0], key[1]
        if Z_s.shape != (h_intr, w_intr):
            raise RuntimeError(
                f'Depth shape {tuple(Z_s.shape)} != CameraInfo {h_intr}x{w_intr}'
            )
        if inv_p.shape != (h_intr, w_intr):
            raise RuntimeError(
                f'DA3 output shape {tuple(inv_p.shape)} != color {h_intr}x{w_intr}'
            )

    def _fit_affine_inverse(self, inv_p: torch.Tensor, Z_s: torch.Tensor):
        valid = (
            (Z_s > 0.1) & (Z_s < 10.0)
            & torch.isfinite(inv_p) & (inv_p > 0)
        )
        n = int(valid.sum().item())
        if n < self.align_min_overlap_pixels:
            raise RuntimeError(
                f'Sensor/DA3 overlap {n} px below threshold '
                f'{self.align_min_overlap_pixels}'
            )
        x = inv_p[valid].detach().cpu().numpy().astype(np.float64)
        inv_s_valid = (
            (1.0 / Z_s[valid]).detach().cpu().numpy().astype(np.float64)
        )
        a, b = np.polyfit(x, inv_s_valid, 1)
        r = np.abs(inv_s_valid - (a * x + b))
        keep_thresh = np.quantile(r, 1.0 - self.align_trim_frac)
        keep = r <= keep_thresh
        if int(keep.sum()) >= 2:
            a, b = np.polyfit(x[keep], inv_s_valid[keep], 1)
        return float(a), float(b)

    def _apply_fill(
        self,
        Z_s: torch.Tensor,
        inv_p: torch.Tensor,
        a: float,
        b: float,
        mode: str,
    ) -> torch.Tensor:
        inv_filled = a * inv_p + b
        inv_filled = torch.clamp(inv_filled, min=1e-3)
        Z_pred = 1.0 / inv_filled
        valid_pred = (Z_pred > 0.05) & (Z_pred < 20.0)
        Z_pred = torch.where(valid_pred, Z_pred, torch.zeros_like(Z_pred))
        if mode == _FILL_HOLES_ONLY:
            return torch.where(Z_s > 0, Z_s, Z_pred)
        if mode == _FILL_FULL_OVERRIDE:
            return Z_pred
        raise RuntimeError(f"Unknown fill_mode '{mode}'")

    def _deproject(
        self,
        Z: torch.Tensor,
        color_msg: Image,
        depth_msg: Image,
        cam: str,
        stride: int,
    ):
        with self.lock_intr:
            xy_table = self._xy_table_cuda[cam]

        rgb = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        rgb_t = torch.from_numpy(np.ascontiguousarray(rgb)).to(
            self.device, non_blocking=True,
        )
        if rgb_t.shape[:2] != Z.shape:
            raise RuntimeError(
                f'Color shape {tuple(rgb_t.shape[:2])} != depth {tuple(Z.shape)}'
            )

        if stride > 1:
            Z = Z[::stride, ::stride].contiguous()
            xy = xy_table[::stride, ::stride].contiguous()
            rgb_t = rgb_t[::stride, ::stride].contiguous()
        else:
            xy = xy_table

        x = xy[..., 0] * Z
        y = xy[..., 1] * Z
        mask = Z > 0.0
        xyz = torch.stack([x, y, Z], dim=-1)[mask]
        rgb_u32 = pack_rgb_u8_to_float32_cuda(rgb_t)[mask]

        xyz_np = xyz.cpu().numpy()
        rgb_packed_f32 = (
            rgb_u32.cpu().numpy().astype(np.uint32).view(np.float32)
        )
        return make_pc2_xyzrgb(self._make_header(depth_msg), xyz_np, rgb_packed_f32)

    def _depth_to_image_msg(self, Z: torch.Tensor, depth_msg: Image) -> Image:
        depth_np = np.ascontiguousarray(
            Z.detach().cpu().numpy(), dtype=np.float32,
        )
        msg = self.bridge.cv2_to_imgmsg(depth_np, encoding='32FC1')
        msg.header = self._make_header(depth_msg)
        return msg

    def _make_header(self, depth_msg: Image):
        hdr = copy.deepcopy(depth_msg.header)
        if self._output_frame_override:
            hdr.frame_id = self._output_frame_override
        return hdr


def main():
    rclpy.init()
    node = MonocularDepthPCService()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
