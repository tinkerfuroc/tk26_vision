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
import os
import threading
import time

import cv2
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
from tinker_vision_msgs_26.srv import ObjectDetectionGeneralist, SeatRecommendBbox
from vision_util.vision_logging import VisionLogger

from ._env import load_env, require_api_key
from ._seat_vlm import VlmSeatError, request_seat, request_seat_choice


class SeatRecommendBboxService(Node):
    def __init__(self):
        super().__init__(f'seat_recommend_bbox_service_{int(time.time())}')

        self.camera_types = ['orbbec']

        self.declare_parameter('log_prompts', True)
        # Set to 'google/gemini-2.5-pro' for harder multi-seat scenes; Flash
        # is cheaper / faster and works for most cases.
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
        self.declare_parameter('snap_search_radius_px', 80)
        self.declare_parameter('snap_min_horizontality', 0.6)  # |n_y|: 1=level, 0=vertical
        # Set-of-Mark pipeline (depth-proposed candidates + YOLO person-mask
        # filter + VLM picks a number). Replaces the VLM-pointing path when
        # enabled; flip som_enabled to false to restore the prior flow.
        self.declare_parameter('som_enabled', True)
        self.declare_parameter('som_min_area_px', 500)
        self.declare_parameter('som_max_area_frac', 0.30)
        self.declare_parameter('som_depth_range_m', 0.50)
        # Floor on component depth range — a flat wall facing the camera has
        # near-zero dZ/du and dZ/dv, so a noisy cross-product normal can score
        # high |n_y| by chance. Requiring a little real depth spread across
        # the component's pixels knocks those wall patches out.
        self.declare_parameter('som_min_depth_range_m', 0.03)
        # Tighter than the class-level max_depth_m so far walls don't enter
        # the horizontality map in the first place.
        self.declare_parameter('som_max_depth_m', 4.0)
        self.declare_parameter('som_max_candidates', 10)
        self.declare_parameter('som_person_cover_frac', 0.60)
        self.declare_parameter('som_detection_timeout_s', 1.0)
        self.declare_parameter('detection_service', 'object_detection_generalist')

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
        self.som_enabled = bool(
            self.get_parameter('som_enabled').get_parameter_value().bool_value
        )
        self.som_min_area_px = int(
            self.get_parameter('som_min_area_px').get_parameter_value().integer_value
        )
        self.som_max_area_frac = float(
            self.get_parameter('som_max_area_frac').get_parameter_value().double_value
        )
        self.som_depth_range_m = float(
            self.get_parameter('som_depth_range_m').get_parameter_value().double_value
        )
        self.som_min_depth_range_m = float(
            self.get_parameter('som_min_depth_range_m')
            .get_parameter_value()
            .double_value
        )
        self.som_max_depth_m = float(
            self.get_parameter('som_max_depth_m').get_parameter_value().double_value
        )
        self.som_max_candidates = int(
            self.get_parameter('som_max_candidates').get_parameter_value().integer_value
        )
        self.som_person_cover_frac = float(
            self.get_parameter('som_person_cover_frac')
            .get_parameter_value()
            .double_value
        )
        self.som_detection_timeout_s = float(
            self.get_parameter('som_detection_timeout_s')
            .get_parameter_value()
            .double_value
        )
        detection_service = (
            self.get_parameter('detection_service').get_parameter_value().string_value
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
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

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

        # Async client for person masks (best-effort; node keeps working if down).
        self.detection_cli = self.create_client(
            ObjectDetectionGeneralist,
            detection_service,
            callback_group=self.client_cb_group,
        )
        self._detection_service_name = detection_service

        self.seat_srv = self.create_service(
            SeatRecommendBbox,
            'seat_recommend_bbox_service',
            self.seat_recommend_bbox_callback,
            callback_group=self.server_cb_group,
        )

        self.get_logger().info(
            f'Seat-recommend-bbox service initialized '
            f'(model={self.llm_model}, image={image_topic}, depth={depth_topic}, '
            f'som={"on" if self.som_enabled else "off"} '
            f'detection={detection_service}, '
            f'snap={"on" if self.snap_enabled else "off"} '
            f'r={self.snap_search_radius_px}px min|ny|={self.snap_min_horizontality:.2f}).'
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

    def _horizontality_map(
        self,
        depth_arr_m: np.ndarray,
        K: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Per-pixel world-up alignment score |n_y|/||n|| of the depth surface.

        Computes a vectorised analytical normal by forward-differencing an
        unprojected 3D point grid. 1.0 = perfectly horizontal (world-up
        surface normal), 0.0 = vertical. NaN where the local patch can't
        form a valid cross product (hole / edge).
        """
        fx, fy, cx, cy = K
        h, w = depth_arr_m.shape
        Z = depth_arr_m.astype(np.float32, copy=False)
        bad = ~(np.isfinite(Z) & (Z > self.min_depth_m) & (Z < self.max_depth_m))
        Z = np.where(bad, np.nan, Z)
        us = np.arange(w, dtype=np.float32)[np.newaxis, :]
        vs = np.arange(h, dtype=np.float32)[:, np.newaxis]
        X = (us - cx) * Z / fx
        Y = (vs - cy) * Z / fy
        # np.gradient returns [d/dv, d/du] for 2D input indexed [v, u].
        dX_dv, dX_du = np.gradient(X)
        dY_dv, dY_du = np.gradient(Y)
        dZ_dv, dZ_du = np.gradient(Z)
        nx = dY_du * dZ_dv - dZ_du * dY_dv
        ny = dZ_du * dX_dv - dX_du * dZ_dv
        nz = dX_du * dY_dv - dY_du * dX_dv
        mag = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9
        return np.abs(ny) / mag

    def _propose_candidates(
        self,
        depth_arr_m: np.ndarray,
        K: tuple[float, float, float, float],
    ):
        """Enumerate horizontal patches from depth.

        Returns ``(candidates, labels)`` where ``candidates`` is a list of
        dicts (sorted by area descending, capped at ``som_max_candidates``)
        and ``labels`` is the full-image int32 label map from
        ``connectedComponentsWithStats`` (used by person-mask exclusion).
        """
        h_map = self._horizontality_map(depth_arr_m, K)
        effective_max_depth = min(self.max_depth_m, self.som_max_depth_m)
        mask = (h_map > self.snap_min_horizontality) & np.isfinite(h_map)
        mask = mask & (depth_arr_m > self.min_depth_m) & (depth_arr_m < effective_max_depth)
        mask_u8 = mask.astype(np.uint8)
        # Dilate so depth holes between adjacent pixels (common on cushion
        # edges / seams on Orbbec) do not split one cushion into two labels.
        mask_u8 = cv2.dilate(mask_u8, np.ones((5, 5), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8)

        h, w = depth_arr_m.shape
        img_area = float(h * w)
        max_area = int(self.som_max_area_frac * img_area)

        cands: list[dict] = []
        for i in range(1, n):  # label 0 is background
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.som_min_area_px or area > max_area:
                continue
            x0 = int(stats[i, cv2.CC_STAT_LEFT])
            y0 = int(stats[i, cv2.CC_STAT_TOP])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            short = min(cw, ch)
            long_ = max(cw, ch)
            if short < 1 or long_ / short > 8:
                continue
            patch_labels = labels[y0:y0 + ch, x0:x0 + cw]
            patch_depth_all = depth_arr_m[y0:y0 + ch, x0:x0 + cw]
            patch_sel = patch_labels == i
            patch_depth = patch_depth_all[patch_sel]
            patch_depth = patch_depth[
                np.isfinite(patch_depth)
                & (patch_depth > self.min_depth_m)
                & (patch_depth < self.max_depth_m)
            ]
            if patch_depth.size == 0:
                continue
            d_range = float(patch_depth.max() - patch_depth.min())
            if d_range > self.som_depth_range_m:
                continue
            # Walls facing the camera have near-zero depth variation across
            # the component — their "horizontality" is noise. Require enough
            # real depth spread to separate them from genuine cushions.
            if d_range < self.som_min_depth_range_m:
                continue
            cx_c = int(round(float(centroids[i, 0])))
            cy_c = int(round(float(centroids[i, 1])))
            # If the weighted centroid falls on a non-component pixel (happens
            # with C- / L-shaped patches), move it onto the nearest pixel that
            # is part of this component.
            if not (
                0 <= cx_c < w and 0 <= cy_c < h and labels[cy_c, cx_c] == i
            ):
                ys, xs = np.where(patch_sel)
                if ys.size:
                    mid = ys.size // 2
                    order = np.argsort(ys + xs)  # deterministic tie-break
                    pick = order[mid]
                    cx_c = int(x0 + xs[pick])
                    cy_c = int(y0 + ys[pick])
            cands.append({
                'label_id': int(i),
                'centroid_px': (cx_c, cy_c),
                'bbox_xyxy': (x0, y0, x0 + cw - 1, y0 + ch - 1),
                'area_px': area,
                'depth_range_m': d_range,
                'median_depth_m': float(np.median(patch_depth)),
            })
        cands.sort(key=lambda c: -c['area_px'])
        cands = cands[: self.som_max_candidates]
        return cands, labels

    def _apply_person_mask(
        self,
        candidates: list[dict],
        labels: np.ndarray,
        person_mask_msgs,
    ) -> tuple[list[dict], int]:
        """Drop candidates >som_person_cover_frac covered by any person mask.

        Returns ``(kept, n_dropped)``. If no usable masks are supplied, the
        input list is returned verbatim with ``n_dropped=0``.
        """
        if not person_mask_msgs:
            return candidates, 0
        combined = None
        for mask_msg in person_mask_msgs:
            try:
                m = self.bridge.imgmsg_to_cv2(mask_msg, '8UC1')
            except Exception:  # noqa: BLE001
                continue
            if m.shape != labels.shape:
                continue
            m_bool = m > 0
            combined = m_bool if combined is None else (combined | m_bool)
        if combined is None:
            return candidates, 0
        kept: list[dict] = []
        dropped = 0
        for cand in candidates:
            patch = labels == cand['label_id']
            area = int(patch.sum())
            if area == 0:
                kept.append(cand)
                continue
            overlap = int((patch & combined).sum())
            if overlap / area >= self.som_person_cover_frac:
                dropped += 1
                continue
            kept.append(cand)
        return kept, dropped

    def _render_som(
        self,
        color_img_bgr: np.ndarray,
        candidates: list[dict],
    ) -> np.ndarray:
        """Overlay numbered yellow/black circles at each candidate centroid."""
        out = color_img_bgr.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, cand in enumerate(candidates, start=1):
            cx, cy = cand['centroid_px']
            cv2.circle(out, (cx, cy), 26, (0, 220, 255), 3)
            cv2.circle(out, (cx, cy), 22, (0, 0, 0), -1)
            text = str(i)
            (tw, th), _ = cv2.getTextSize(text, font, 0.9, 2)
            cv2.putText(
                out, text, (cx - tw // 2, cy + th // 2), font, 0.9,
                (0, 220, 255), 2, cv2.LINE_AA,
            )
        return out

    async def _fetch_person_masks(self, camera: str):
        """Best-effort call to the generalist for person masks.

        Returns ``None`` when the service isn't reachable or the call
        errors (caller should treat as "unavailable" — no filtering).
        Returns a possibly-empty list of ``sensor_msgs/Image`` when the
        call succeeded (empty ⇒ no persons detected, also fine).
        """
        if not self.detection_cli.wait_for_service(timeout_sec=0.5):
            return None
        req = ObjectDetectionGeneralist.Request()
        req.camera = camera
        req.prompt = 'person'
        req.target_frame = ''
        req.sort_closest = False
        req.sort_highest = False
        req.return_rgb_image = False
        req.return_depth_image = False
        req.return_segments = True
        req.force_vlm_sam = False
        req.use_vlm_sam_fallback = False
        try:
            future = self.detection_cli.call_async(req)
            await future
            res = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'person-mask call failed: {exc}')
            return None
        if res is None:
            return None
        # detection_source == 'none' means the generalist didn't actually run
        # a detector — e.g. it couldn't sync color+depth within its threshold.
        # Treat as "unavailable" rather than silently pretending there were
        # no people in the scene.
        if getattr(res, 'detection_source', '') == 'none':
            return None
        masks: list = []
        for i, obj in enumerate(res.objects):
            if obj.cls == 'person' and i < len(res.segments):
                masks.append(res.segments[i])
        return masks

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

        # 2. Intrinsic K — shared by both paths.
        fx = float(intrinsic.k[0])
        fy = float(intrinsic.k[4])
        px = float(intrinsic.k[2])
        py = float(intrinsic.k[5])
        K = (fx, fy, px, py)

        request_ctx: dict = {
            'service': 'seat_recommend_bbox',
            'camera': request.camera,
            'names': list(request.names),
            'features': list(request.features),
            'target_frame': request.target_frame,
        }
        log_timings: dict[str, float] = {}
        log_extras: dict = {}
        som_annotated_img: np.ndarray | None = None

        def _write_log(detections, branch='seat_recommend_bbox'):
            if not self._vision_logger.enabled:
                return
            ts = self._vision_logger.write(
                color_img, detections,
                request_ctx=request_ctx,
                branch=branch,
                extras=dict(log_extras) or None,
                timings=dict(log_timings),
            )
            if ts and som_annotated_img is not None:
                run_dir = self._vision_logger.run_dir
                if run_dir:
                    try:
                        cv2.imwrite(
                            os.path.join(run_dir, f'som_{ts}.jpg'),
                            som_annotated_img,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.get_logger().warn(f'som overlay write failed: {exc}')

        def _fail_with_log(msg, detections):
            _write_log(detections)
            return self._fail(response, msg)

        extra_dets: list[dict] = []

        if self.som_enabled:
            # === SoM path ===
            t_prop = time.perf_counter()
            candidates, labels = self._propose_candidates(depth_arr_m, K)
            log_timings['propose'] = time.perf_counter() - t_prop
            n_raw = len(candidates)
            som_log: dict = {
                'n_candidates_raw': n_raw,
                'person_mask_status': 'skipped',
                'person_dropped': 0,
            }
            log_extras['som'] = som_log
            if n_raw == 0:
                log_extras['event'] = 'som_no_candidates'
                return _fail_with_log(
                    'SoM produced no horizontal candidate seats from depth.', [],
                )

            # Stage 3: best-effort person-mask exclusion.
            t_det = time.perf_counter()
            person_masks = await self._fetch_person_masks(request.camera)
            log_timings['person_mask'] = time.perf_counter() - t_det
            if person_masks is None:
                som_log['person_mask_status'] = 'unavailable'
            else:
                som_log['person_mask_status'] = 'ok'
                candidates, n_dropped = self._apply_person_mask(
                    candidates, labels, person_masks,
                )
                som_log['person_dropped'] = int(n_dropped)
            som_log['n_candidates'] = len(candidates)
            if not candidates:
                log_extras['event'] = 'som_all_occupied_by_person'
                return _fail_with_log(
                    'All horizontal candidates covered by person masks.', [],
                )

            # Stage 4: SoM annotation that the VLM will see.
            som_annotated_img = self._render_som(color_img, candidates)

            # Stage 5: VLM picks a number.
            try:
                choice, vlm_elapsed = request_seat_choice(
                    som_annotated_img,
                    len(candidates),
                    request.names,
                    request.features,
                    model=self.llm_model,
                    timeout_s=self.vlm_timeout_s,
                    max_retries=self.vlm_max_retries,
                    logger=self.get_logger(),
                )
            except VlmSeatError as exc:
                return _fail_with_log(f'VLM unavailable: {exc}', [])
            log_timings['vlm'] = vlm_elapsed
            som_log['choice'] = int(choice)

            if self.log_prompts:
                self.get_logger().info(
                    f'SoM candidates={len(candidates)} (raw {n_raw}, '
                    f'person_drop {som_log["person_dropped"]}), '
                    f'VLM choice={choice} (elapsed {vlm_elapsed:.2f}s)'
                )

            if choice < 1 or choice > len(candidates):
                log_extras['event'] = 'som_vlm_none'
                # Dump the candidates we showed the VLM so we can inspect why.
                _write_log(
                    [
                        {
                            'bbox': c['bbox_xyxy'],
                            'cls_name': f'cand_{i}',
                            'centroid': c['centroid_px'],
                        }
                        for i, c in enumerate(candidates, start=1)
                    ],
                )
                return self._fail(response, 'VLM returned no valid seat choice (0).')

            picked = candidates[choice - 1]
            cx, cy = picked['centroid_px']
            bbox_xyxy = picked['bbox_xyxy']
            label = f'candidate_{choice}_of_{len(candidates)}'
            response.recommendation = label
            response.bbox = BoundingBox(
                xmin=int(bbox_xyxy[0]),
                ymin=int(bbox_xyxy[1]),
                xmax=int(bbox_xyxy[2]),
                ymax=int(bbox_xyxy[3]),
            )
            log_det = {
                'bbox': bbox_xyxy,
                'cls_name': label,
                'centroid': (cx, cy),
            }
            # Non-chosen candidates land in the overlay too so the dumped
            # artifacts show what the VLM picked from.
            for i, c in enumerate(candidates, start=1):
                if i == choice:
                    continue
                extra_dets.append({
                    'bbox': c['bbox_xyxy'],
                    'cls_name': f'cand_{i}',
                    'centroid': c['centroid_px'],
                })
        else:
            # === Legacy B path: VLM pointing pixel + snap-to-horizontal ===
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
            log_timings['vlm'] = vlm_elapsed
            request_ctx['label'] = label
            request_ctx['visible_seats'] = visible_seats
            response.recommendation = label

            if self.log_prompts:
                self.get_logger().info(
                    f'VLM seat point={point_xy}, label={label!r}, '
                    f'visible_seats={visible_seats} '
                    f'(elapsed {vlm_elapsed:.2f}s)'
                )

            if point_xy is None:
                log_extras['event'] = 'no_empty_seat'
                _write_log(None)
                return self._fail(response, 'No empty seat detected by VLM.')

            vlm_px = (int(point_xy[0]), int(point_xy[1]))
            log_extras['vlm_point'] = [vlm_px[0], vlm_px[1]]

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
