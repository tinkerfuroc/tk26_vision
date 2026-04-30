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
        # Pro is the default — Flash's pointing accuracy on cluttered
        # multi-seat scenes was unusable in the 2026-04-30 logs (point
        # landed on people / coffee tables / armrests, never on the
        # named cushion). Pro is documented by Google's spatial-
        # understanding cookbook as the higher-precision tier; the ~1 s
        # extra latency is acceptable for an HRI intake step. Override
        # with `-p llm_model:=google/gemini-2.5-flash` for cheap regression
        # checks where accuracy isn't being measured.
        self.declare_parameter('llm_model', 'google/gemini-2.5-pro')
        # 35 s (was 20) — Pro + thinking adds 3–8 s vs. Flash; 20 s was
        # already tight on cluttered scenes and tripped the timeout when
        # thinking is enforced.
        self.declare_parameter('vlm_timeout_s', 35.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 10.0)
        # Half-size in pixels of the bbox synthesized around the VLM pointing
        # pixel for the response's `bbox` field (used for overlay and pan-tilt
        # aiming; depth is sampled at the point itself, not the bbox centre).
        self.declare_parameter('point_bbox_halfsize_px', 40)
        # Snap-to-horizontal-surface: after the VLM returns a pixel, we fit a
        # plane to a local depth patch and require the surface normal to point
        # approximately along world-up. Backrests / walls / backpack fabric
        # fail this test, so the snap search spirals outward until it hits a
        # cushion-like surface (or gives up and fails clean).
        self.declare_parameter('snap_enabled', True)
        self.declare_parameter('snap_patch_half_px', 8)       # 17x17 plane fit
        # 200 px (was 80) — VLM pointing error is regularly >80 px on cluttered
        # scenes, so the spiral has to reach the next cushion over, not just
        # denoise within the current surface.
        self.declare_parameter('snap_search_radius_px', 200)
        # 0.85 (was 0.6) — 0.6 admitted ~53° tilts, accepting armrests, slanted
        # laptop screens, and the side of a person's lap as "horizontal". 0.85
        # (~32°) requires a near-level surface like an actual seat cushion.
        # Side-effect: more `point_not_on_horizontal_surface` failures (correct
        # behavior; the BT retries on status=1 instead of seating on a wall).
        self.declare_parameter('snap_min_horizontality', 0.85)  # |n_y|: 1=level, 0=vertical

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
        self.point_bbox_halfsize_px = int(
            self.get_parameter('point_bbox_halfsize_px')
            .get_parameter_value()
            .integer_value
        )
        self.snap_enabled = bool(
            self.get_parameter('snap_enabled').get_parameter_value().bool_value
        )
        self.snap_patch_half_px = int(
            self.get_parameter('snap_patch_half_px')
            .get_parameter_value()
            .integer_value
        )
        self.snap_search_radius_px = int(
            self.get_parameter('snap_search_radius_px')
            .get_parameter_value()
            .integer_value
        )
        self.snap_min_horizontality = float(
            self.get_parameter('snap_min_horizontality')
            .get_parameter_value()
            .double_value
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

        # 60 s cache: VLM seat calls take 10-25 s and the TF lookup uses the
        # depth image's stamp (not "now"), so default 10 s buffer falls behind
        # when the VLM stalls or retries. Sized to absorb vlm_max_retries *
        # vlm_timeout_s (3 * 20 = 60 s default) without falling off the back.
        self.tf_buffer = Buffer(
            cache_time=rclpy.duration.Duration(seconds=120.0)
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.seat_srv = self.create_service(
            SeatRecommendBbox,
            'seat_recommend_bbox_service',
            self.seat_recommend_bbox_callback,
            callback_group=self.server_cb_group,
        )

        self.get_logger().info(
            f'Seat-recommend-bbox service initialized '
            f'(model={self.llm_model}, image={image_topic}, depth={depth_topic}, '
            f'snap={"on" if self.snap_enabled else "off"} '
            f'r={self.snap_search_radius_px}px min|ny|={self.snap_min_horizontality:.2f}).'
        )

    def camera_info_orbbec_callback(self, info):
        self.lock_info.acquire()
        self.camera_intrinsic['orbbec'] = info
        self.lock_info.release()

    def sync_orbbec_callback(self, color_msg, depth_msg):
        self.lock_img.acquire()
        self.recent_sync['orbbec'] = (color_msg, depth_msg)
        self.lock_img.release()

    def _fail(self, response, msg: str, *, log: bool = True):
        if log:
            self.get_logger().warn(msg)
        response.status = 1
        response.error_msg = msg
        return response

    def _local_normal(
        self,
        depth_arr_m: np.ndarray,
        K: tuple[float, float, float, float],
        u: int,
        v: int,
    ) -> np.ndarray | None:
        """Fit a plane to the local depth patch and return its unit normal.

        Works in the camera's optical frame (X right, Y down, Z forward).
        Returns None if the patch has fewer valid depth samples than needed
        to fit a plane (sparse depth / holes).
        """
        h, w = depth_arr_m.shape
        hp = self.snap_patch_half_px
        u0 = max(0, u - hp)
        v0 = max(0, v - hp)
        u1 = min(w, u + hp + 1)
        v1 = min(h, v + hp + 1)
        patch = depth_arr_m[v0:v1, u0:u1]
        valid = (
            np.isfinite(patch)
            & (patch > self.min_depth_m)
            & (patch < self.max_depth_m)
        )
        if valid.sum() < max(25, patch.size // 3):
            return None
        vs, us = np.mgrid[v0:v1, u0:u1]
        fx, fy, cx, cy = K
        z = patch[valid].astype(np.float64)
        x = (us[valid].astype(np.float64) - cx) * z / fx
        y = (vs[valid].astype(np.float64) - cy) * z / fy
        pts = np.stack([x, y, z], axis=1)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        normal = vh[-1]
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            return None
        return normal / norm

    def _snap_to_horizontal(
        self,
        depth_arr_m: np.ndarray,
        K: tuple[float, float, float, float],
        u: int,
        v: int,
    ):
        """Spiral-search for the nearest horizontal (world-up) surface.

        Returns (u', v', |n_y|, normal, ok). `ok` is True iff the best
        candidate within ``snap_search_radius_px`` has |n_y| >=
        ``snap_min_horizontality``. Returns None only if no pixel in the
        search region had enough valid depth to fit any plane at all.
        """
        h, w = depth_arr_m.shape

        def _score_at(uu: int, vv: int):
            n = self._local_normal(depth_arr_m, K, uu, vv)
            if n is None:
                return None
            return abs(float(n[1])), n

        best_uv = None
        best_score = -1.0
        best_normal = None

        hit = _score_at(u, v)
        if hit is not None:
            best_score, best_normal = hit
            best_uv = (u, v)
            if best_score >= self.snap_min_horizontality:
                return best_uv[0], best_uv[1], best_score, best_normal, True

        step = 4
        for r in range(step, self.snap_search_radius_px + 1, step):
            # One pixel per ~`step` arc-length keeps coverage uniform across rings.
            num = max(12, int(2.0 * np.pi * r / step))
            for k in range(num):
                theta = 2.0 * np.pi * k / num
                uu = u + int(round(r * np.cos(theta)))
                vv = v + int(round(r * np.sin(theta)))
                if not (0 <= uu < w and 0 <= vv < h):
                    continue
                hit = _score_at(uu, vv)
                if hit is None:
                    continue
                score, normal = hit
                if score > best_score:
                    best_uv = (uu, vv)
                    best_score = score
                    best_normal = normal
                    if score >= self.snap_min_horizontality:
                        return best_uv[0], best_uv[1], best_score, best_normal, True

        if best_uv is None:
            return None
        return best_uv[0], best_uv[1], best_score, best_normal, False

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

    def seat_recommend_bbox_callback(
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

        self.get_logger().info(
            f'Received seat recommendation request for camera {request.camera} '
            f'(model={self.llm_model}, names={request.names}, features={request.features}, target_frame={request.target_frame}).'
        )
        # if transform needed, record TF at this point for use after Gemini fishes processing
        transform = None
        try:
            transform = self.tf_buffer.lookup_transform(
                request.target_frame,
                depth_msg.header.frame_id,
                depth_msg.header.stamp,
                rclpy.duration.Duration(seconds=1.0),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'TF lookup failed for frame {request.target_frame}: {exc}'
            )
            response.status = 1
            response.error_msg = f'TF lookup failed for frame {request.target_frame}: {exc}'
            return response

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

        self.get_logger().info(
            'Preparing for VLM call'
            f'Camera data ready (color {color_img.shape[1]}x{color_img.shape[0]}, depth {depth_arr_m.shape[1]}x{depth_arr_m.shape[0]}). '
            f'Elapsed {(time.time_ns() - start_time) / 1e9:.2f}s.'
        )
        # 2. Gemini call — returns a pointing pixel + short label.
        try:
            label, point_xy, visible_seats, vlm_elapsed = request_seat(
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
                f'VLM seat point={point_xy}, label={label!r}, '
                f'visible_seats={visible_seats} '
                f'(elapsed {vlm_elapsed:.2f}s)'
            )

        # Populate the recommendation field with the short label so
        # callers reading the srv still get a human-readable identifier.
        response.recommendation = label

        request_ctx = {
            'service': 'seat_recommend_bbox',
            'camera': request.camera,
            'names': list(request.names),
            'features': list(request.features),
            'target_frame': request.target_frame,
            'label': label,
            'visible_seats': visible_seats,
        }
        log_timings = {'vlm': vlm_elapsed}
        log_extras: dict = {}

        self.get_logger().info(
            f'VLM returned label={label!r}, point={point_xy} '
            f'(elapsed {vlm_elapsed:.2f}s). '
            f'Preparing response (snap={"on" if self.snap_enabled else "off"}).'
        )

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

        if point_xy is None:
            log_extras['event'] = 'no_empty_seat'
            _write_log(None)
            return self._fail(response, 'No empty seat detected by VLM.')

        vlm_px = (int(point_xy[0]), int(point_xy[1]))
        log_extras['vlm_point'] = [vlm_px[0], vlm_px[1]]

        fx = float(intrinsic.k[0])
        fy = float(intrinsic.k[4])
        px = float(intrinsic.k[2])
        py = float(intrinsic.k[5])
        K = (fx, fy, px, py)

        # Snap the VLM point to the nearest horizontal (world-up) surface.
        # Backrests, walls, and backpack fabric fail the |n_y| test, so the
        # spiral search walks outward until it hits a cushion-like surface.
        # On a hard miss we fail clean — returning a 3D centroid on a wall
        # is worse than telling the caller to retry.
        if self.snap_enabled:
            snap_res = self._snap_to_horizontal(depth_arr_m, K, vlm_px[0], vlm_px[1])
            if snap_res is None:
                cx, cy = vlm_px
                log_extras['snap'] = {'status': 'skipped_sparse_depth'}
            else:
                su, sv, score, normal, ok = snap_res
                log_extras['snap'] = {
                    'status': 'ok' if ok else 'best_below_threshold',
                    'horizontality': round(score, 3),
                    'normal_camera': [
                        round(float(normal[0]), 3),
                        round(float(normal[1]), 3),
                        round(float(normal[2]), 3),
                    ],
                    'moved_px': int(abs(su - vlm_px[0]) + abs(sv - vlm_px[1])),
                }
                if ok:
                    cx, cy = int(su), int(sv)
                else:
                    log_extras['event'] = 'point_not_on_horizontal_surface'
                    bad_det = {
                        'bbox': (
                            max(0, vlm_px[0] - 20), max(0, vlm_px[1] - 20),
                            vlm_px[0] + 20, vlm_px[1] + 20,
                        ),
                        'cls_name': f'{label or "vlm_point"} (|n_y|={score:.2f})',
                        'centroid': vlm_px,
                    }
                    return _fail_with_log(
                        f'VLM point ({vlm_px[0]},{vlm_px[1]}) not on a '
                        f'horizontal surface (best |n_y|={score:.2f}).',
                        [bad_det],
                    )
        else:
            cx, cy = vlm_px

        # Synthesize a small bbox around the (possibly snapped) point for the
        # response's bbox field (used by callers for overlay and pan-tilt aiming).
        h_img, w_img = color_img.shape[:2]
        r = max(1, int(self.point_bbox_halfsize_px))
        bbox_xyxy = (
            max(0, cx - r),
            max(0, cy - r),
            min(w_img - 1, cx + r),
            min(h_img - 1, cy + r),
        )
        response.bbox = BoundingBox(
            xmin=int(bbox_xyxy[0]),
            ymin=int(bbox_xyxy[1]),
            xmax=int(bbox_xyxy[2]),
            ymax=int(bbox_xyxy[3]),
        )
        log_det = {
            'bbox': bbox_xyxy,
            'cls_name': label or 'empty_seat',
            'centroid': (cx, cy),
        }
        # When snap actually moved the point, log the VLM raw pixel as a
        # second detection so the overlay shows both markers side-by-side.
        extra_dets = []
        if (cx, cy) != vlm_px:
            extra_dets.append({
                'bbox': (
                    max(0, vlm_px[0] - 12), max(0, vlm_px[1] - 12),
                    vlm_px[0] + 12, vlm_px[1] + 12,
                ),
                'cls_name': 'vlm_raw',
                'centroid': vlm_px,
            })

        # 3. Unproject the (snapped) pixel from depth.
        sampled = self._sample_depth_at(depth_arr_m, cx, cy)
        if sampled is None:
            log_extras['event'] = 'no_depth_at_point'
            log_extras['depth_frame'] = depth_msg.header.frame_id
            return _fail_with_log(
                f'No valid depth near point ({cx},{cy}).',
                [log_det] + extra_dets,
            )
        uu, vv, z = sampled

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
            if transform is None:
                response.status = 2
                response.error_msg = f'TF lookup failed for frame {request.target_frame}.'
                response.centroid = src
                return response
            try:
                transformed = do_transform_point(src, transform)
                centroid_header = transformed.header
                centroid_point = transformed.point
            except (TransformException, Exception) as exc:  # noqa: BLE001
                log_extras['event'] = 'tf_failed'
                log_extras['centroid_3d_camera'] = [float(x), float(y), float(z)]
                log_extras['depth_frame'] = depth_msg.header.frame_id
                response.status = 3
                response.error_msg = f'TF {depth_msg.header.frame_id} -> {request.target_frame} failed: {exc}'
                response.centroid = PointStamped(header=centroid_header, point=centroid_point)
                return response
                return _fail_with_log(
                    f'TF {depth_msg.header.frame_id} -> {request.target_frame} failed: {exc}',
                    [log_det] + extra_dets,
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
        _write_log([log_det] + extra_dets)

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
