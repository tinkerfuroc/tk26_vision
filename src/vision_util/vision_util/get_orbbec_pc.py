"""CUDA-deprojected Orbbec point-cloud service.

Background: the vendored Orbbec depth engine ``libdepthengine.so.2.0`` is
incompatible with NVIDIA driver 575.x — see
``src/tk26_vision/orbbec_diagnosis.md``. The current iGPU bind-mount
workaround keeps depth flowing but caps ``/camera/depth_registered/points``
at ~5 Hz because the SDK's colored-PC step does a 1280x720 CPU xy-table
reprojection (``ob_camera_node.cpp:1718``) in series with the iGPU GL
command stream.

This node moves the PC reprojection off the iGPU/CPU contention path
entirely: it subscribes to ``/camera/depth/image_raw`` plus the color
intrinsics, deprojects on NVIDIA CUDA via PyTorch, and serves the result
over ``/get_orbbec_pc``. Depth is assumed to be registered to color
(``depth_registration:=true``), so the color CameraInfo carries the right
intrinsics.

Service: ``tinker_vision_msgs_26/srv/GetOrbbecPC``.
"""

import copy
import threading

import numpy as np
import rclpy
import rclpy.executors
import torch
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tinker_vision_msgs_26.srv import GetOrbbecPC

from vision_util._pc_utils import (
    build_xy_table_cuda,
    make_pc2_xyz,
    make_pc2_xyzrgb,
    pack_rgb_u8_to_float32_cuda,
)


class GetOrbbecPCService(Node):
    def __init__(self):
        super().__init__('get_orbbec_pc_service')

        if not torch.cuda.is_available():
            raise RuntimeError(
                'get_orbbec_pc requires CUDA — no GPU device available'
            )
        self.device = torch.device('cuda')
        self.get_logger().info(
            f'CUDA OK: {torch.cuda.get_device_name(0)} '
            f'(torch {torch.__version__})'
        )

        self.declare_parameter('output_frame_id', '')
        self._output_frame_override = (
            self.get_parameter('output_frame_id')
            .get_parameter_value().string_value
        )

        self.bridge = CvBridge()

        cb_sync = MutuallyExclusiveCallbackGroup()
        depth_sub = Subscriber(
            self, Image, '/camera/depth/image_raw', callback_group=cb_sync,
        )
        color_sub = Subscriber(
            self, Image, '/camera/color/image_raw', callback_group=cb_sync,
        )
        self._sync = ApproximateTimeSynchronizer(
            [depth_sub, color_sub], queue_size=3, slop=0.05,
        )
        self._sync.registerCallback(self._frame_callback)

        self._info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self._info_callback,
            qos_profile=10,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self._lock_frame = threading.Lock()
        self._latest_depth = None
        self._latest_color = None

        self._lock_intr = threading.Lock()
        self._intr_key = None              # (h, w, fx, fy, cx, cy)
        self._xy_table_cuda = None         # (H, W, 2) float32 on cuda

        self.srv = self.create_service(
            GetOrbbecPC,
            'get_orbbec_pc',
            self._service_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info('get_orbbec_pc service initialized.')

    def _frame_callback(self, depth_msg, color_msg):
        with self._lock_frame:
            self._latest_depth = depth_msg
            self._latest_color = color_msg

    def _info_callback(self, info: CameraInfo):
        h, w = int(info.height), int(info.width)
        if h == 0 or w == 0:
            return
        fx, fy = float(info.k[0]), float(info.k[4])
        cx, cy = float(info.k[2]), float(info.k[5])
        key = (h, w, fx, fy, cx, cy)
        with self._lock_intr:
            if key == self._intr_key:
                return
            self._xy_table_cuda = build_xy_table_cuda(
                h, w, fx, fy, cx, cy, self.device,
            )
            self._intr_key = key
            self.get_logger().info(
                f'Intrinsics cached: {w}x{h} fx={fx:.2f} fy={fy:.2f} '
                f'cx={cx:.2f} cy={cy:.2f}'
            )

    def _depth_to_metres_cuda(self, depth_msg: Image) -> torch.Tensor:
        """16UC1/mono16 mm or 32FC1 m → float32 metres on CUDA.

        torch.from_numpy does not accept uint16, so we cast through float32
        in numpy first.
        """
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

    def _color_to_rgb_cuda(self, color_msg: Image) -> torch.Tensor:
        arr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='rgb8')
        return torch.from_numpy(np.ascontiguousarray(arr)).to(
            self.device, non_blocking=True,
        )

    def _service_callback(
        self,
        request: GetOrbbecPC.Request,
        response: GetOrbbecPC.Response,
    ):
        with self._lock_frame:
            depth_msg = self._latest_depth
            color_msg = self._latest_color
        with self._lock_intr:
            xy_table = self._xy_table_cuda
            intr_key = self._intr_key

        if depth_msg is None or xy_table is None:
            response.status = 1
            response.error_msg = 'No Orbbec depth/intrinsics yet.'
            return response
        if request.include_color and color_msg is None:
            response.status = 1
            response.error_msg = (
                'No synced color frame for include_color=true.'
            )
            return response

        try:
            depth_m = self._depth_to_metres_cuda(depth_msg)  # (H, W)
        except Exception as e:
            response.status = 1
            response.error_msg = f'cv_bridge depth conversion failed: {e}'
            return response

        # Validate depth shape matches cached intrinsics; reject mismatches
        # rather than silently producing junk geometry.
        h_intr, w_intr = intr_key[0], intr_key[1]
        if depth_m.shape != (h_intr, w_intr):
            response.status = 1
            response.error_msg = (
                f'Depth shape {tuple(depth_m.shape)} does not match '
                f'CameraInfo {h_intr}x{w_intr}.'
            )
            return response

        s = max(1, int(request.stride))
        if s > 1:
            depth_m = depth_m[::s, ::s].contiguous()
            xy = xy_table[::s, ::s].contiguous()
        else:
            xy = xy_table

        # Deproject and mask Z>0.
        z = depth_m
        x = xy[..., 0] * z
        y = xy[..., 1] * z
        mask = z > 0.0
        xyz = torch.stack([x, y, z], dim=-1)[mask]   # (N, 3)

        if request.include_color:
            try:
                rgb = self._color_to_rgb_cuda(color_msg)  # (Hc, Wc, 3) uint8
            except Exception as e:
                response.status = 1
                response.error_msg = f'cv_bridge color conversion failed: {e}'
                return response
            if rgb.shape[:2] != (h_intr, w_intr):
                response.status = 1
                response.error_msg = (
                    f'Color shape {tuple(rgb.shape[:2])} does not match '
                    f'CameraInfo {h_intr}x{w_intr}.'
                )
                return response
            if s > 1:
                rgb = rgb[::s, ::s].contiguous()
            rgb_u32 = pack_rgb_u8_to_float32_cuda(rgb)[mask]
            xyz_np = xyz.cpu().numpy()
            rgb_packed_f32 = (
                rgb_u32.cpu().numpy().astype(np.uint32).view(np.float32)
            )
            cloud = make_pc2_xyzrgb(
                self._make_header(depth_msg), xyz_np, rgb_packed_f32,
            )
        else:
            cloud = make_pc2_xyz(self._make_header(depth_msg), xyz.cpu().numpy())

        response.status = 0
        response.error_msg = ''
        response.points = cloud
        return response

    def _make_header(self, depth_msg: Image):
        hdr = copy.deepcopy(depth_msg.header)
        if self._output_frame_override:
            hdr.frame_id = self._output_frame_override
        return hdr


def main():
    rclpy.init()
    node = GetOrbbecPCService()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
