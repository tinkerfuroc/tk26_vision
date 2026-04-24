"""LLM-backed empty-seat recommendation with bounding box + 3D centroid.

Sibling to `feature_recognition`'s `/seat_recommend_service` — asks
Gemini 2.5 Flash (via OpenRouter) for both a recommendation sentence
AND a 2D bbox of the recommended empty seat, then projects the bbox
centre to 3D by unprojecting the synchronized depth image at that pixel
(mirrors `vision_track.person_track_node._depth_image_to_points`).
Optionally TF-transforms the centroid to a caller-chosen frame.

Kept separate from `feature_recognition` so the old string-only service
stays wire-compatible for BT callers that expect
`SeatRecommendation.srv`.
"""

import copy
import threading
import time

import numpy as np
import rclpy
import rclpy.executors
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_geometry_msgs import do_transform_point
from tinker_vision_msgs_26.msg import BoundingBox
from tinker_vision_msgs_26.srv import SeatRecommendBbox
from vision_util.vision_logging import VisionLogger

from ._env import load_env, require_api_key
from ._seat_vlm import VlmSeatError, request_seat


class SeatRecommendBboxService(Node):
    def __init__(self):
        super().__init__(f'seat_recommend_bbox_service_{int(time.time())}')

        self.camera_types = ['orbbec']

        self.declare_parameter('log_prompts', True)
        self.declare_parameter('llm_model', 'google/gemini-2.5-flash')
        self.declare_parameter('vlm_timeout_s', 20.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 10.0)

        self.log_prompts = self.get_parameter('log_prompts').get_parameter_value().bool_value
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        self.vlm_timeout_s = (
            self.get_parameter('vlm_timeout_s').get_parameter_value().double_value
        )
        self.vlm_max_retries = (
            self.get_parameter('vlm_max_retries').get_parameter_value().integer_value
        )
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = (
            self.get_parameter('camera_info_topic').get_parameter_value().string_value
        )
        self.min_depth_m = (
            self.get_parameter('min_depth_m').get_parameter_value().double_value
        )
        self.max_depth_m = (
            self.get_parameter('max_depth_m').get_parameter_value().double_value
        )
        self._vision_logger = VisionLogger(
            self,
            self.get_parameter('vision_logging_enabled')
            .get_parameter_value()
            .bool_value,
            self.get_parameter('vision_log_folder')
            .get_parameter_value()
            .string_value,
        )

        # Fail-fast on missing key — matches feature_recognition pattern so
        # the T1 negative test (no .env) surfaces at node init.
        require_api_key()

        self.server_cb_group = MutuallyExclusiveCallbackGroup()
        self.camera_cb_group = MutuallyExclusiveCallbackGroup()

        self.bridge = CvBridge()

        self.lock_img = threading.Lock()
        self.recent_sync = {'orbbec': None}  # (color_msg, depth_msg)
        self.lock_info = threading.Lock()
        self.camera_intrinsic = {'orbbec': None}

        color_sub = Subscriber(
            self, Image, image_topic, callback_group=self.camera_cb_group,
        )
        depth_sub = Subscriber(
            self, Image, depth_topic, callback_group=self.camera_cb_group,
        )
        self._sync = ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=3, slop=0.1,
        )
        self._sync.registerCallback(self.sync_orbbec_callback)
        self._color_sub = color_sub  # keep alive
        self._depth_sub = depth_sub

        self.camera_info_sub_orbbec = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_orbbec_callback,
            qos_profile=10,
            callback_group=self.camera_cb_group,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.seat_srv = self.create_service(
            SeatRecommendBbox,
            'seat_recommend_bbox_service',
            self.seat_recommend_bbox_callback,
            callback_group=self.server_cb_group,
        )

        self.get_logger().info(
            f'Seat-recommend-bbox service initialized '
            f'(model={self.llm_model}, image={image_topic}, depth={depth_topic}).'
        )

    def camera_info_orbbec_callback(self, info):
        with self.lock_info:
            self.camera_intrinsic['orbbec'] = info

    def sync_orbbec_callback(self, color_msg, depth_msg):
        with self.lock_img:
            self.recent_sync['orbbec'] = (color_msg, depth_msg)

    def _fail(self, response, msg: str, *, log: bool = True):
        if log:
            self.get_logger().warn(msg)
        response.status = 1
        response.error_msg = msg
        return response

    def _sample_depth_at(self, depth_arr_m: np.ndarray, u: int, v: int):
        """Return depth (metres) at pixel (u, v) or None.

        Walks a 5x5 neighbourhood when the centre pixel has no valid depth
        (Orbbec depth holes are common at object edges).
        """
        h, w = depth_arr_m.shape
        if w == 0 or h == 0:
            return None
        u = max(0, min(int(u), w - 1))
        v = max(0, min(int(v), h - 1))

        offsets = [(0, 0)]
        for r in range(1, 3):
            for du in range(-r, r + 1):
                for dv in range(-r, r + 1):
                    if abs(du) == r or abs(dv) == r:
                        offsets.append((du, dv))

        for du, dv in offsets:
            uu = u + du
            vv = v + dv
            if 0 <= uu < w and 0 <= vv < h:
                z = float(depth_arr_m[vv, uu])
                if np.isfinite(z) and self.min_depth_m < z < self.max_depth_m:
                    return uu, vv, z
        return None

    async def seat_recommend_bbox_callback(
        self,
        request: SeatRecommendBbox.Request,
        response: SeatRecommendBbox.Response,
    ):
        start_time = time.time_ns()

        # 1. Latest synced frame + intrinsics.
        if not any(cam in request.camera for cam in self.camera_types):
            return self._fail(response, f'Unsupported camera: {request.camera}.')

        with self.lock_img:
            synced = copy.copy(self.recent_sync['orbbec'])
        if synced is None:
            return self._fail(response, f'No camera data for {request.camera}.')
        img_msg, depth_msg = synced

        with self.lock_info:
            intrinsic = self.camera_intrinsic['orbbec']
        if intrinsic is None:
            return self._fail(response, 'No camera_info received yet.')

        try:
            color_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except Exception as exc:  # noqa: BLE001
            return self._fail(response, f'cv_bridge conversion failed: {exc}')

        try:
            # Orbbec Femto Bolt default: 16UC1 depth in millimeters.
            depth_arr_m = (
                np.frombuffer(depth_msg.data, dtype=np.uint16)
                .reshape(depth_msg.height, depth_msg.width)
                .astype(np.float32)
                * 0.001
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(response, f'depth image decode failed: {exc}')

        # 2. Gemini call.
        try:
            rec_text, bbox_xyxy, vlm_elapsed = request_seat(
                color_img,
                request.names,
                request.features,
                model=self.llm_model,
                timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries,
                logger=self.get_logger(),
            )
        except VlmSeatError as exc:
            return self._fail(response, f'VLM unavailable: {exc}')

        if self.log_prompts:
            self.get_logger().info(
                f'VLM seat recommendation: {rec_text!r} (elapsed {vlm_elapsed:.2f}s, '
                f'bbox={bbox_xyxy})'
            )

        response.recommendation = rec_text

        request_ctx = {
            'service': 'seat_recommend_bbox',
            'camera': request.camera,
            'names': list(request.names),
            'features': list(request.features),
            'target_frame': request.target_frame,
            'recommendation': rec_text,
        }
        log_timings = {'vlm': vlm_elapsed}
        log_extras: dict = {}

        def _write_log(detections, branch='seat_recommend_bbox'):
            if self._vision_logger.enabled:
                self._vision_logger.write(
                    color_img, detections,
                    request_ctx=request_ctx,
                    branch=branch,
                    extras=dict(log_extras) or None,
                    timings=dict(log_timings),
                )

        def _fail_with_log(msg, detections):
            _write_log(detections)
            return self._fail(response, msg)

        if bbox_xyxy is None:
            log_extras['event'] = 'no_empty_seat'
            _write_log(None)
            return self._fail(response, 'No empty seat detected by VLM.')

        response.bbox = BoundingBox(
            xmin=int(bbox_xyxy[0]),
            ymin=int(bbox_xyxy[1]),
            xmax=int(bbox_xyxy[2]),
            ymax=int(bbox_xyxy[3]),
        )
        cx = (bbox_xyxy[0] + bbox_xyxy[2]) // 2
        cy = (bbox_xyxy[1] + bbox_xyxy[3]) // 2
        log_det = {
            'bbox': bbox_xyxy,
            'cls_name': 'empty_seat',
            'centroid': (cx, cy),
        }

        # 3. Unproject bbox centre from depth.
        sampled = self._sample_depth_at(depth_arr_m, cx, cy)
        if sampled is None:
            log_extras['event'] = 'no_depth_at_centre'
            log_extras['depth_frame'] = depth_msg.header.frame_id
            return _fail_with_log(
                f'No valid depth near bbox centre ({cx},{cy}).', [log_det],
            )
        uu, vv, z = sampled

        fx, fy = float(intrinsic.k[0]), float(intrinsic.k[4])
        px, py = float(intrinsic.k[2]), float(intrinsic.k[5])
        x = (uu - px) * z / fx
        y = (vv - py) * z / fy

        # Depth is `depth_registration:=true`-aligned at launch, so it
        # carries the color optical frame. Use the depth header so stamps
        # reflect the measurement time.
        centroid_header = depth_msg.header
        centroid_point = Point(x=float(x), y=float(y), z=float(z))

        # 4. Optional TF to target_frame.
        if request.target_frame and request.target_frame != centroid_header.frame_id:
            src = PointStamped(header=centroid_header, point=centroid_point)
            try:
                transform = self.tf_buffer.lookup_transform(
                    request.target_frame,
                    centroid_header.frame_id,
                    centroid_header.stamp,
                    rclpy.duration.Duration(seconds=1.0),
                )
                transformed = do_transform_point(src, transform)
                centroid_header = transformed.header
                centroid_point = transformed.point
            except (TransformException, Exception) as exc:  # noqa: BLE001
                log_extras['event'] = 'tf_failed'
                log_extras['centroid_3d_camera'] = [float(x), float(y), float(z)]
                log_extras['depth_frame'] = depth_msg.header.frame_id
                return _fail_with_log(
                    f'TF {depth_msg.header.frame_id} -> {request.target_frame} failed: {exc}',
                    [log_det],
                )

        response.centroid = PointStamped(header=centroid_header, point=centroid_point)
        response.status = 0
        response.error_msg = ''

        total_elapsed = (time.time_ns() - start_time) / 1e9
        log_timings['total'] = total_elapsed
        log_extras['centroid_3d'] = [
            float(centroid_point.x),
            float(centroid_point.y),
            float(centroid_point.z),
        ]
        log_extras['centroid_frame'] = centroid_header.frame_id
        log_extras['depth_frame'] = depth_msg.header.frame_id
        log_extras['depth_pixel'] = [int(uu), int(vv)]
        _write_log([log_det])

        self.get_logger().info(
            f'Seat recommended. Total time: {total_elapsed * 1e3:.2f} ms'
        )
        return response


def main():
    load_env()
    rclpy.init()
    node = SeatRecommendBboxService()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
