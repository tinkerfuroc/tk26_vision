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

from ._env import load_env, require_api_key, require_dashscope_api_key
from ._seat_bbox_vlm import request_seat_bbox_chain, VlmSeatBboxError
from ._seat_fewshot import load_fewshots
from ._seat_vlm import VlmSeatError, request_seat_chain


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
        self.declare_parameter('vlm_timeout_s', 25.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('min_depth_m', 0.1)
        self.declare_parameter('max_depth_m', 10.0)
        # Half-size in pixels of the neighbourhood walked by `_sample_depth_at`
        # (5 → 11x11). Bump higher when Orbbec depth holes are large near the
        # VLM target (transparent / specular / edge pixels).
        self.declare_parameter('sample_depth_halfsize_px', 5)
        # Last-resort depth (metres) when every fallback tier fails. The
        # caller asked for an always-valid centroid, so we surface a
        # plausible seat distance rather than a service failure.
        self.declare_parameter('fallback_depth_m', 1.5)
        # Half-size in pixels of the bbox synthesized around the VLM pointing
        # pixel for the response's `bbox` field (used for overlay and pan-tilt
        # aiming; depth is sampled at the point itself, not the bbox centre).
        self.declare_parameter('point_bbox_halfsize_px', 40)
        # Snap-to-horizontal-surface: after the VLM returns a pixel, we fit a
        # plane to a local depth patch and require the surface normal to point
        # approximately along world-up. Backrests / walls / backpack fabric
        # fail this test, so the snap search spirals outward until it hits a
        # cushion-like surface (or gives up and fails clean).
        # Default OFF as of 2026-06-02: bbox_select localizes the cushion via the
        # chosen box, so the box centre is already on the seat — snap-to-horizontal
        # adds latency and can wander. Re-enable with -p snap_enabled:=true for the
        # legacy point path or noisy depth.
        self.declare_parameter('snap_enabled', False)
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
        # Few-shot in-context examples (kimi_api/fewshot/<slug>/answer.json,
        # produced by `seat_fewshot_annotator`). Off by default so existing
        # deployments stay bit-identical until examples are curated.
        self.declare_parameter('fewshot_enabled', False)
        self.declare_parameter('max_fewshots', 3)
        # --- VLM strategy / provider (2026-06-02: switch to Qwen bbox+select) ---
        # 'bbox_select' (default) = one structured call returns a cushion box +
        # occupancy per seat + the chosen empty seat (benchmark winner, S1).
        # 'point' = legacy Gemini pointing via _seat_vlm.request_seat (rollback).
        self.declare_parameter('vlm_strategy', 'bbox_select')
        # Primary provider for bbox_select, then fallback. 'qwen' = DashScope
        # qwen3-vl-plus (benchmark best); 'gemini' = OpenRouter gemini-2.5-pro.
        self.declare_parameter('vlm_provider', 'qwen')
        self.declare_parameter('vlm_fallback_provider', 'gemini')  # '' to disable
        self.declare_parameter('bbox_model_qwen', 'qwen3-vl-plus')
        self.declare_parameter('bbox_model_gemini', 'google/gemini-2.5-pro')

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
        self.sample_depth_halfsize_px = max(1, int(
            self.get_parameter('sample_depth_halfsize_px')
            .get_parameter_value()
            .integer_value
        ))
        self.fallback_depth_m = float(
            self.get_parameter('fallback_depth_m')
            .get_parameter_value()
            .double_value
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
        self.fewshot_enabled = bool(
            self.get_parameter('fewshot_enabled').get_parameter_value().bool_value
        )
        self.max_fewshots = int(
            self.get_parameter('max_fewshots').get_parameter_value().integer_value
        )
        self.vlm_strategy = (
            self.get_parameter('vlm_strategy').get_parameter_value().string_value
        )
        self.vlm_provider = (
            self.get_parameter('vlm_provider').get_parameter_value().string_value
        )
        self.vlm_fallback_provider = (
            self.get_parameter('vlm_fallback_provider').get_parameter_value().string_value
        )
        self.bbox_model_qwen = (
            self.get_parameter('bbox_model_qwen').get_parameter_value().string_value
        )
        self.bbox_model_gemini = (
            self.get_parameter('bbox_model_gemini').get_parameter_value().string_value
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

        # Fail-fast on the API key(s) the configured strategy/providers need —
        # matches feature_recognition pattern so the T1 negative test (no .env)
        # surfaces at node init. bbox_select builds an ordered provider chain
        # (primary required, fallback dropped-with-warning if its key is absent);
        # the legacy point path keeps its historical Gemini primary
        # (self.llm_model, needing only OPENROUTER_API_KEY) with an optional
        # Qwen fallback — it must NOT inherit the bbox_select chain, whose
        # default is qwen-primary with the bbox-tuned models.
        if self.vlm_strategy == 'bbox_select':
            self._provider_models = self._resolve_provider_chain()
        else:
            require_api_key()
            self._provider_models = self._resolve_point_provider_chain()

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

        # VLM seat calls take 10-25 s and the TF lookup uses the depth
        # image's stamp (not "now"), so the default 10 s buffer falls behind
        # when the VLM stalls or retries. Sized to absorb the provider
        # chain's worst case — 2 providers x vlm_max_retries x vlm_timeout_s
        # + backoff (2 x 3 x 25 + 3 ≈ 155 s at defaults) — without the
        # successful-fallback path falling off the back.
        self.tf_buffer = Buffer(
            cache_time=rclpy.duration.Duration(seconds=180.0)
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

    def _model_for(self, provider: str) -> str:
        return self.bbox_model_qwen if provider == 'qwen' else self.bbox_model_gemini

    def _has_provider_key(self, provider: str) -> bool:
        try:
            (require_dashscope_api_key if provider == 'qwen' else require_api_key)()
            return True
        except RuntimeError:
            return False

    def _resolve_provider_chain(self) -> list:
        """Ordered (provider, model) chain for bbox_select. Primary key is
        required (raises at init if missing); a fallback whose key is absent is
        dropped with a warning."""
        primary = self.vlm_provider
        if not self._has_provider_key(primary):
            # Re-call to raise the descriptive RuntimeError for the missing key.
            (require_dashscope_api_key if primary == 'qwen' else require_api_key)()
        chain = [(primary, self._model_for(primary))]
        fb = self.vlm_fallback_provider
        if fb and fb != primary:
            if fb not in ('qwen', 'gemini'):
                # _has_provider_key maps any non-'qwen' string to the Gemini
                # key check, so without this guard a typo'd fallback would be
                # appended as a chain entry that errors at request time.
                self.get_logger().warn(
                    f'Unknown fallback provider {fb!r}; fallback disabled.'
                )
            elif self._has_provider_key(fb):
                chain.append((fb, self._model_for(fb)))
            else:
                self.get_logger().warn(
                    f'Fallback provider {fb!r} key missing; fallback disabled.'
                )
        self.get_logger().info(
            f'bbox+select provider chain: {[p for p, _ in chain]}'
        )
        return chain

    def _resolve_point_provider_chain(self) -> list:
        """Ordered (provider, model) chain for the legacy 'point' strategy:
        Gemini primary on the historical self.llm_model (rollback semantics —
        needs only OPENROUTER_API_KEY, honors -p llm_model overrides), plus
        the Qwen fallback (bbox_model_qwen) when its key is present.

        `vlm_fallback_provider`'s default ('gemini') is oriented to the
        bbox_select chain, whose primary is qwen; here the primary IS gemini,
        so any non-empty fallback setting resolves to qwen — only the empty
        string disables the fallback."""
        chain = [('gemini', self.llm_model)]
        if self.vlm_fallback_provider:
            if self._has_provider_key('qwen'):
                chain.append(('qwen', self.bbox_model_qwen))
            else:
                self.get_logger().warn(
                    'Point-path qwen fallback key missing; fallback disabled.'
                )
        self.get_logger().info(
            f'point provider chain: {[p for p, _ in chain]}'
        )
        return chain

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
        """Return (u, v, depth_m) at pixel (u, v) or None.

        Walks a (2*hp+1)x(2*hp+1) neighbourhood (hp = ``sample_depth_halfsize_px``)
        ring-by-ring outward. Orbbec depth holes near edges / dark / specular
        surfaces are common, so the radius is configurable.
        """
        h, w = depth_arr_m.shape
        if w == 0 or h == 0:
            return None
        u = max(0, min(int(u), w - 1))
        v = max(0, min(int(v), h - 1))

        hp = self.sample_depth_halfsize_px
        offsets = [(0, 0)]
        for r in range(1, hp + 1):
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

    def _region_median_depth(
        self,
        depth_arr_m: np.ndarray,
        u0: int, v0: int, u1: int, v1: int,
    ):
        """Median valid depth in [u0,u1)x[v0,v1) or None."""
        h, w = depth_arr_m.shape
        u0 = max(0, int(u0)); v0 = max(0, int(v0))
        u1 = min(w, int(u1)); v1 = min(h, int(v1))
        if u1 <= u0 or v1 <= v0:
            return None
        patch = depth_arr_m[v0:v1, u0:u1]
        valid = (
            np.isfinite(patch)
            & (patch > self.min_depth_m)
            & (patch < self.max_depth_m)
        )
        if not valid.any():
            return None
        return float(np.median(patch[valid]))

    def _resolve_depth_robust(
        self,
        depth_arr_m: np.ndarray,
        cx: int, cy: int,
        bbox_xyxy: tuple[int, int, int, int],
    ):
        """Always return (uu, vv, z, tier).

        Tier order:
          0 'point'        : valid pixel in (2*hp+1)^2 neighbourhood at (cx,cy).
          1 'bbox_median'  : median depth across synthesized response bbox.
          2 'roi_median'   : median in expanded ROI (8x bbox half-size).
          3 'image_median' : median over whole depth image.
          4 'fallback'     : constant ``fallback_depth_m`` — last resort.
        Pixel (uu, vv) is the original (cx, cy) for tiers >= 1; only the
        depth value differs. The arm gets a well-defined ray every time.
        """
        h, w = depth_arr_m.shape
        cx = max(0, min(int(cx), max(0, w - 1)))
        cy = max(0, min(int(cy), max(0, h - 1)))

        sampled = self._sample_depth_at(depth_arr_m, cx, cy)
        if sampled is not None:
            uu, vv, z = sampled
            return uu, vv, z, 'point'

        x0, y0, x1, y1 = bbox_xyxy
        z = self._region_median_depth(depth_arr_m, x0, y0, x1 + 1, y1 + 1)
        if z is not None:
            return cx, cy, z, 'bbox_median'

        r = max(self.point_bbox_halfsize_px * 8, 200)
        z = self._region_median_depth(
            depth_arr_m, cx - r, cy - r, cx + r + 1, cy + r + 1,
        )
        if z is not None:
            return cx, cy, z, 'roi_median'

        z = self._region_median_depth(depth_arr_m, 0, 0, w, h)
        if z is not None:
            return cx, cy, z, 'image_median'

        return cx, cy, float(self.fallback_depth_m), 'fallback'

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

        known_seats = list(request.known_seats)
        self.get_logger().info(
            f'Received seat recommendation request for camera {request.camera} '
            f'(model={self.llm_model}, names={request.names}, features={request.features}, '
            f'target_frame={request.target_frame}, known_seats={known_seats}).'
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
        # 2. VLM call. bbox_select (default) returns a cushion box + chosen seat
        # across a Qwen->Gemini provider chain; point is the legacy Gemini path.
        # `box_px` is the chosen cushion box in pixels (None for the point path
        # or a "none" result); `point_xy` is the legacy pointing pixel.
        box_px = None
        provider_used = ''
        fewshots = None
        if self.vlm_strategy == 'bbox_select':
            try:
                sel = request_seat_bbox_chain(
                    color_img,
                    request.names,
                    request.features,
                    provider_models=self._provider_models,
                    known_seats=known_seats or None,
                    timeout_s=self.vlm_timeout_s,
                    max_retries=self.vlm_max_retries,
                    logger=self.get_logger(),
                )
            except VlmSeatBboxError as exc:
                return self._fail(response, f'VLM bbox+select unavailable: {exc}')
            label = sel.label
            visible_seats = sel.seats
            vlm_elapsed = sel.elapsed_s
            provider_used = sel.provider
            box_px = tuple(sel.box_xyxy) if sel.box_xyxy else None
            point_xy = None
            overridden_from = sel.overridden_from
        else:
            if self.fewshot_enabled:
                fewshots = load_fewshots(self.max_fewshots, logger=self.get_logger())
                self.get_logger().info(
                    f'Few-shot enabled: applying {len(fewshots)} example(s) '
                    f'(max_fewshots={self.max_fewshots}).'
                )
            try:
                label, point_xy, visible_seats, vlm_elapsed, provider_used = (
                    request_seat_chain(
                        color_img,
                        request.names,
                        request.features,
                        provider_models=self._provider_models,
                        timeout_s=self.vlm_timeout_s,
                        max_retries=self.vlm_max_retries,
                        logger=self.get_logger(),
                        fewshots=fewshots,
                        known_seats=known_seats,
                    )
                )
            except VlmSeatError as exc:
                return self._fail(response, f'VLM unavailable: {exc}')
            overridden_from = None

        if overridden_from:
            self.get_logger().info(
                f'Suitability re-rank: overrode VLM choice {overridden_from!r} '
                f'-> {label!r} (stool/bench or narrower seat than a better '
                f'unoccupied option).'
            )

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
            'known_seats': list(known_seats),
            'label': label,
            'visible_seats': visible_seats,
            'fewshot_enabled': bool(self.fewshot_enabled),
            'n_fewshots': int(len(fewshots)) if fewshots is not None else 0,
            'vlm_strategy': self.vlm_strategy,
            'vlm_provider': provider_used,
            'overridden_from': overridden_from,
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

        # No empty seat: point path yields point_xy is None; bbox_select yields a
        # "none" choice (box_px is None) with no error.
        if point_xy is None and box_px is None:
            log_extras['event'] = 'no_empty_seat'
            _write_log(None)
            return self._fail(response, 'No empty seat detected by VLM.')

        # When a catalog is supplied, the VLM must pick exactly one of the
        # listed labels (the prompt tells it as much). Reject anything else
        # before downstream snap/depth work commits to a hallucinated seat.
        if known_seats and label not in known_seats:
            log_extras['event'] = 'out_of_catalog_label'
            return _fail_with_log(
                f'VLM returned out-of-catalog label {label!r}; '
                f'catalog={list(known_seats)}.',
                None,
            )

        # Working pixel: chosen box centre for bbox_select, else the VLM point.
        if box_px is not None:
            bx0, by0, bx1, by1 = box_px
            vlm_px = ((bx0 + bx1) // 2, (by0 + by1) // 2)
            log_extras['vlm_box'] = [int(bx0), int(by0), int(bx1), int(by1)]
        else:
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

        # bbox field: use the VLM's actual cushion box (bbox_select); else
        # synthesize a small box around the (possibly snapped) point.
        h_img, w_img = color_img.shape[:2]
        if box_px is not None:
            bbox_xyxy = (
                max(0, min(int(box_px[0]), w_img - 1)),
                max(0, min(int(box_px[1]), h_img - 1)),
                max(0, min(int(box_px[2]), w_img - 1)),
                max(0, min(int(box_px[3]), h_img - 1)),
            )
        else:
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

        # 3. Unproject the (snapped) pixel from depth. Always succeed —
        # caller (BT) needs a 3D point to drive the arm even when Orbbec
        # depth has a hole at the seat. `_resolve_depth_robust` walks
        # progressively wider regions then falls back to a constant.
        uu, vv, z, depth_tier = self._resolve_depth_robust(
            depth_arr_m, cx, cy, bbox_xyxy,
        )
        log_extras['depth_tier'] = depth_tier
        log_extras['depth_frame'] = depth_msg.header.frame_id
        if depth_tier != 'point':
            self.get_logger().warning(
                f'Depth fallback tier={depth_tier} at ({cx},{cy}); z={z:.3f} m.'
            )

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
