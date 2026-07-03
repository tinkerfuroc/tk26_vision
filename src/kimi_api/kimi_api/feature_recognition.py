"""LLM-backed person feature extraction and seat recommendation.

Two services:
- `feature_extraction_service`: calls `object_detection_generalist`, picks
  the person most likely addressing the robot (see `select_best_person_idx`:
  size-gated, largest-first; centermost/closest only breaks near-size
  ties), crops to a tight bbox, runs a vision
  LLM on the crop for a structured text description, and returns BOTH the
  description and the crop as `sensor_msgs/Image` for downstream image-vs-image
  matching.
- `seat_recommend_service`: takes a list of named/feature-described people,
  sends the current frame, asks the model where the new guest should sit.
"""

import math
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from message_filters import Subscriber
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tinker_vision_msgs_26.srv import (
    FeatureExtraction,
    ObjectDetectionGeneralist as ObjectDetection,
    SeatRecommendation,
)
from vision_util.vision_logging import VisionLogger

from ._env import (
    default_flash_model,
    load_env,
    require_api_key,
    resolve_qwen_target,
)
from ._feature_vlm import FeatureVlmError, request_feature_description_chain
from ._image_utils import bbox_from_mask, encode_to_data_url


# Minimum person bbox height, as a fraction of frame height, to be considered
# a candidate at all — 2026-07-01 incident: a 17x23 px detection (~3% of a
# 720 px-tall frame) of a person visible through a distant doorway won the
# old pure pixel-center-distance selection over the actual foreground person,
# because the foreground person's mask centroid was pulled off-center by
# being cut off at the bottom of frame (very close to the camera). A
# background blob that small is never the intended subject, so it is gated
# out before scoring.
MIN_PERSON_HEIGHT_FRAC = 0.15

# How close two candidates' normalized image-center offsets must be (as a
# fraction of the frame half-diagonal) to be treated as "roughly equidistant
# from the optical center" and have depth break the tie. Kept small so depth
# only refines near-ties instead of overriding a clear centering difference —
# see the 2026-07-01 replay note on DEPTH_TIE_EPS below for why depth cannot
# be a primary, additively-weighted term here.
DEPTH_TIE_EPS = 0.08

# Candidates whose bbox height is at least this fraction of the tallest
# survivor's height count as "comparably sized"; only they compete on
# centering/depth. Anyone shorter is background — 2026-07-02 incident
# (vision_log/20260702_070449, 09:17:58): at the arena door a crowd member
# ~3 m behind the barrier was 21.7% of frame height (past the 15% absolute
# gate) and almost exactly on the optical center, while the guest filling
# the frame was bottom-cut and therefore far off-center (offset 0.27 vs
# 0.03 — outside DEPTH_TIE_EPS, so depth never got a say). Absolute size
# gating cannot separate a close crowd from a close guest; relative size
# can: the guest was 3x the crowd member's height.
HEIGHT_TIE_FRAC = 0.75

# Feature-description prompt, cut to five slots on 2026-07-02 (spec:
# docs/superpowers/specs/2026-07-02-feature-5slot-cut-design.md). The
# output sentence is spoken verbatim by the Receptionist introduction and
# is matched against candidate crops by feature_matching — keep it a
# natural sentence and keep the slots in sync with
# feature_matching.MATCH_EVIDENCE.
FEATURE_SYS_PROMPT = (
    'You will be asked to extract features of one single designated person in an image.'
    ' Describe EXACTLY these five attributes and nothing else: (1) gender — decide male'
    ' or female and commit to the more likely one, never "gender-neutral" or "unknown",'
    ' (2) approximate age in years (give in words, such as "twenty", not numeric'
    ' numerals), (3) hair color (you may qualify it with hair length, such as "short'
    ' black hair"), (4) whether the person is wearing glasses or not, and (5) the most'
    ' prominent upper-body garment as color plus garment type (such as "red shirt"; if a'
    ' jacket or coat covers the shirt, describe the jacket). Output in the format of'
    ' "[gender pronoun] is [gender], approximately [age in words] years old, has'
    ' [hair color] hair, is [wearing glasses | not wearing glasses], and is wearing a'
    ' [color] [garment]" (use "are" instead of "is" where the pronoun requires it).'
    ' Do not mention lower-body clothing, shoes, accessories, or any other information.'
)


def select_best_person_idx(
    bboxes,
    depths_m,
    frame_w: int,
    frame_h: int,
    *,
    min_height_frac: float = MIN_PERSON_HEIGHT_FRAC,
    depth_tie_eps: float = DEPTH_TIE_EPS,
    height_tie_frac: float = HEIGHT_TIE_FRAC,
) -> int:
    """Pick the index of the person most likely addressing the robot.

    Three stages: (1) drop candidates whose bbox height is below
    ``min_height_frac`` of the frame height — tiny background clutter can
    never win; (2) keep only candidates whose bbox height is at least
    ``height_tie_frac`` of the tallest survivor's — apparent size is the
    primary signal, so a dominant (largest) person always beats clearly
    smaller ones regardless of centering; (3) among the comparably-sized
    rest, rank by image-center offset (normalized by the frame's
    half-diagonal); when two or more are within ``depth_tie_eps`` of the
    smallest offset (i.e. "roughly equidistant from the optical center"),
    the closest valid depth among that tied group wins instead.

    Depth is deliberately only a tie-breaker, not an additive score term.
    Replaying this against real `object_detection_generalist` logs
    (2026-07-01) showed most small/distant detections report a sentinel
    ``centroid.z == 0.0`` when depth lookup fails (not ``None`` — the
    upstream depth pipeline can produce a "valid" all-zero point for a
    depth-hole region). An additive ``offset + weight * depth`` score treats
    that unmeasured 0.0 as "closer than anything real", so a genuinely close
    and clearly-centered subject can lose to a small background detection
    with a bogus zero reading. Restricting depth to break only near-ties
    (and ignoring non-positive readings as invalid rather than "very
    close") keeps a clear centering advantage decisive regardless of
    upstream depth data quality.

    ``bboxes``: sequence of (x1, y1, x2, y2) pixel boxes.
    ``depths_m``: sequence of forward-distance-in-metres or ``None``,
    aligned index-for-index with ``bboxes``. Non-positive values are
    treated as an invalid/missing reading, not as "0 m away".

    Returns -1 if no candidate survives the size gate.
    """
    cx_frame, cy_frame = frame_w / 2.0, frame_h / 2.0
    half_diag = math.hypot(cx_frame, cy_frame) or 1.0
    min_height_px = min_height_frac * frame_h

    survivors = []  # (index, height_px, offset_norm, depth_or_None)
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        height = y2 - y1
        if height < min_height_px:
            continue
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        offset_norm = math.hypot(cx - cx_frame, cy - cy_frame) / half_diag
        depth = depths_m[i] if i < len(depths_m) else None
        if depth is not None and depth <= 0.0:
            depth = None
        survivors.append((i, height, offset_norm, depth))

    if not survivors:
        return -1

    max_height = max(s[1] for s in survivors)
    comparable = [s for s in survivors if s[1] >= height_tie_frac * max_height]

    best_offset = min(s[2] for s in comparable)
    tied = [s for s in comparable if s[2] - best_offset <= depth_tie_eps]

    def _tie_key(s):
        _, _, offset_norm, depth = s
        return (0, depth) if depth is not None else (1, offset_norm)

    return min(tied, key=_tie_key)[0]


class FeatureService(Node):
    def __init__(self):
        super().__init__(f'feature_service_{int(time.time())}')

        self.camera_types = ['orbbec']

        self.declare_parameter('log_prompts', True)
        self.declare_parameter('llm_model', default_flash_model())
        self.declare_parameter('detection_service', 'object_detection_generalist')
        self.declare_parameter('vlm_timeout_s', 20.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('vlm_fallback_provider', 'qwen')  # '' to disable
        self.declare_parameter('feature_model_qwen', '')
        self.declare_parameter('qwen_api_backend', 'dashscope')
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self.log_prompts = self.get_parameter('log_prompts').get_parameter_value().bool_value
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        detection_service = self.get_parameter('detection_service').get_parameter_value().string_value
        self.vlm_timeout_s = self.get_parameter('vlm_timeout_s').get_parameter_value().double_value
        self.vlm_max_retries = self.get_parameter('vlm_max_retries').get_parameter_value().integer_value
        self.vlm_fallback_provider = (
            self.get_parameter('vlm_fallback_provider').get_parameter_value().string_value
        )
        self.feature_model_qwen = (
            self.get_parameter('feature_model_qwen').get_parameter_value().string_value
        )
        self.qwen_api_backend = self.get_parameter('qwen_api_backend').value

        self._vision_logger = VisionLogger(
            self,
            self.get_parameter('vision_logging_enabled').get_parameter_value().bool_value,
            self.get_parameter('vision_log_folder').get_parameter_value().string_value,
        )

        self.server_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        # seat_recommend still uses the live frame from the camera topic.
        self.image_subs = []
        image_sub = Subscriber(self, Image, '/camera/color/image_raw')
        image_sub.registerCallback(self.img_orbbec_callback)
        self.image_subs.append(image_sub)

        self.recent_msg = {'orbbec': None}

        self.camera_info_sub_orbbec = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_orbbec_callback,
            qos_profile=10,
        )
        self.camera_intrinsic = {'orbbec': None}

        self.bridge = CvBridge()

        require_api_key()  # fail fast at init if the primary Gemini key is missing
        self._feature_provider_chain = self._resolve_feature_provider_chain()

        self.detection_cli = self.create_client(
            ObjectDetection, detection_service, callback_group=self.client_cb_group,
        )

        self.extraction_srv = self.create_service(
            FeatureExtraction,
            'feature_extraction_service',
            self.feature_extraction_srv_callback,
            callback_group=self.server_cb_group,
        )
        self.seat_recommend_srv = self.create_service(
            SeatRecommendation, 'seat_recommend_service', self.seat_recommend_srv_callback,
        )
        self.get_logger().info(
            f'Feature services initialized (model={self.llm_model}, '
            f'detection_service={detection_service}).'
        )

    def _resolve_feature_provider_chain(self) -> list:
        """Ordered (provider, model) chain for feature extraction: Gemini
        (self.llm_model, already required at init) then, if configured, a
        Qwen fallback that is dropped with a warning when its key is
        missing rather than failing node startup."""
        chain = [('gemini', self.llm_model)]
        fb = self.vlm_fallback_provider
        if fb and fb != 'gemini':
            if fb != 'qwen':
                self.get_logger().warn(f'Unknown fallback provider {fb!r}; ignoring.')
            else:
                try:
                    _, _, resolved_model = resolve_qwen_target(
                        self.qwen_api_backend, self.feature_model_qwen)
                    chain.append(('qwen', resolved_model))
                except RuntimeError:
                    self.get_logger().warn(
                        f'Fallback provider {fb!r} key missing; fallback disabled.'
                    )
        self.get_logger().info(
            f'feature_extraction provider chain: {[p for p, _ in chain]}'
        )
        return chain

    def camera_info_orbbec_callback(self, info):
        self.camera_intrinsic['orbbec'] = info

    def img_orbbec_callback(self, color_msg):
        self.recent_msg['orbbec'] = color_msg

    async def feature_extraction_srv_callback(
        self,
        request: FeatureExtraction.Request,
        response: FeatureExtraction.Response,
    ):
        self.get_logger().info('Feature extraction request received.')
        start_time = time.time_ns()

        if not self.detection_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Detection service unavailable.')
            response.status = 1
            response.error_msg = 'Detection service unavailable.'
            response.feature = ''
            response.comparison_image = Image()
            return response

        detection_req = ObjectDetection.Request()
        detection_req.camera = request.camera
        detection_req.prompt = 'person'
        detection_req.return_rgb_image = True
        detection_req.return_segments = True
        detection_req.use_vlm_sam_fallback = True
        detection_req.target_frame = ''

        detection_future = self.detection_cli.call_async(detection_req)
        await detection_future
        detection_res = detection_future.result()

        if detection_res is None or detection_res.status != 0:
            err = detection_res.error_msg if detection_res is not None else 'no response'
            self.get_logger().warn(f'Detection service failed: {err}')
            response.status = 1
            response.error_msg = f'Detection failed: {err}.'
            response.feature = ''
            response.comparison_image = Image()
            return response

        t_det_ms = (time.time_ns() - start_time) / 1e6
        self.get_logger().info(
            f'Detection finished. Time spent: {t_det_ms:.2f} ms'
        )

        color_img = self.bridge.imgmsg_to_cv2(detection_res.rgb_image, 'bgr8')
        h, w = color_img.shape[:2]

        candidate_indices = []  # index into detection_res.objects/segments
        candidate_bboxes = []   # (x1, y1, x2, y2)
        candidate_depths = []   # forward distance in metres
        n_persons_detected = 0
        for i, obj in enumerate(detection_res.objects):
            if obj.cls != 'person':
                continue
            n_persons_detected += 1
            seg = self.bridge.imgmsg_to_cv2(detection_res.segments[i], '8UC1')
            y1, x1, y2, x2 = bbox_from_mask(seg)
            candidate_indices.append(i)
            candidate_bboxes.append((x1, y1, x2, y2))
            # Forward distance: orbbec optical → z; realsense post-axis-swap → x.
            candidate_depths.append(
                obj.centroid.z if 'orbbec' in request.camera else obj.centroid.x
            )

        sel = select_best_person_idx(candidate_bboxes, candidate_depths, w, h)
        if sel >= 0:
            best_idx = candidate_indices[sel]
            x1, y1, x2, y2 = candidate_bboxes[sel]
            best_bbox = (y1, x1, y2, x2)
        else:
            best_idx = -1
            best_bbox = None

        detections = []
        crop = None
        if best_idx >= 0:
            y1, x1, y2, x2 = best_bbox
            crop = color_img[y1:y2, x1:x2]
            detections = [{
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'cls_name': 'Selected',
                'detection_idx': int(best_idx),
            }]

        def _emit_vision_log(feature_text, status, error_msg, t_vlm_ms, provider_used=''):
            request_ctx = {
                'camera': request.camera,
                'n_persons_detected': int(n_persons_detected),
                'best_idx': int(best_idx),
            }
            extras = {
                'feature': str(feature_text or ''),
                'vlm_status': int(status),
                'vlm_error_msg': str(error_msg),
                'vlm_provider_used': str(provider_used or ''),
                'crop_size': (
                    [int(crop.shape[1]), int(crop.shape[0])]
                    if crop is not None else None
                ),
                'crop_path': None,
            }
            timings = {'det': t_det_ms / 1000.0}
            if t_vlm_ms is not None:
                timings['vlm'] = t_vlm_ms / 1000.0
            ts = self._vision_logger.write(
                rgb_img=color_img,
                detections=detections,
                request_ctx=request_ctx,
                branch='feature_extraction',
                extras=extras,
                timings=timings,
            )
            if ts is None or self._vision_logger.run_dir is None or crop is None:
                return
            out_path = self._vision_logger.aux_path(
                ts, 'crop', 'jpg', branch='feature_extraction',
            )
            try:
                cv2.imwrite(out_path, crop)
                self.get_logger().info(
                    f'feature_extraction: dumped crop to {out_path}'
                )
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(
                    f'feature_extraction: crop dump failed: {exc}'
                )

        if best_idx < 0:
            self.get_logger().warn('No person detected.')
            response.status = 1
            response.error_msg = 'No person detected.'
            response.feature = ''
            response.comparison_image = Image()
            _emit_vision_log('', 1, 'No person detected.', None)
            return response

        comparison_image = self.bridge.cv2_to_imgmsg(crop, 'bgr8')
        comparison_image.header = detection_res.header
        response.comparison_image = comparison_image

        self.get_logger().info(
            f'Selected person idx={best_idx}, bbox={best_bbox}. '
            f'Crop: {crop.shape}. Time: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )

        crop_url = encode_to_data_url(crop)

        sys_prompt = FEATURE_SYS_PROMPT

        t_vlm_start = time.time_ns()
        try:
            vlm_result = request_feature_description_chain(
                crop_url, sys_prompt,
                'extract the features of the person shown in the image.',
                provider_models=self._feature_provider_chain,
                qwen_api_backend=self.qwen_api_backend,
                timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries,
                logger=self.get_logger(),
            )
        except FeatureVlmError as exc:
            t_vlm_ms = (time.time_ns() - t_vlm_start) / 1e6
            self.get_logger().error(f'VLM call failed on every provider: {exc}')
            response.feature = ''
            response.status = 1
            response.error_msg = f'VLM call failed on every provider: {exc}.'
            _emit_vision_log('', 1, str(exc), t_vlm_ms, '')
            return response

        t_vlm_ms = (time.time_ns() - t_vlm_start) / 1e6
        self.get_logger().info(
            f'LLM finished. Total time: {(time.time_ns() - start_time) / 1e9:.3f} s'
        )
        response.feature = vlm_result.text
        response.status = 0
        response.error_msg = ''
        _emit_vision_log(response.feature, 0, '', t_vlm_ms, vlm_result.provider)
        return response

    def seat_recommend_srv_callback(
        self,
        request: SeatRecommendation.Request,
        response: SeatRecommendation.Response,
    ):
        color_img = None
        for cam in self.camera_types:
            if cam in request.camera:
                rec_msg = self.recent_msg[cam]
                if rec_msg is not None:
                    color_img = self.bridge.imgmsg_to_cv2(rec_msg, 'bgr8')
                    break

        if color_img is None:
            self.get_logger().warn('No camera data.')
            response.status = 1
            response.error_msg = f'No camera data for {request.camera}.'
            response.recommendation = ''
            return response

        color_image_url = encode_to_data_url(color_img)

        sys_prompt_recommend = (
            'You will be asked to recommend a seat for a new guest. give your answer in the'
            ' format of "Please sit at ...[a description of what furniture they should sit on'
            ' and where they should sit relative (to the right hand or left hand of people in'
            ' the picture)". Do not offer explanations. Be as accurate as possible. You can use'
            ' 1-2 sentences'
        )
        text_prompt = 'Recommend a seat for a new guest.'
        for name, feature in zip(request.names, request.features):
            text_prompt += ' The person matching description: ' + feature + ' is called ' + name + '.'

        def _emit_vision_log(rec_text, status, error_msg, t_vlm_ms, provider_used=''):
            request_ctx = {
                'camera': request.camera,
                'n_named_people': len(request.names),
            }
            extras = {
                'names': list(request.names),
                'features_text': list(request.features),
                'recommendation': str(rec_text or ''),
                'vlm_status': int(status),
                'vlm_error_msg': str(error_msg),
                'vlm_provider_used': str(provider_used or ''),
            }
            timings = {}
            if t_vlm_ms is not None:
                timings['vlm'] = t_vlm_ms / 1000.0
            self._vision_logger.write(
                rgb_img=color_img,
                detections=[],
                request_ctx=request_ctx,
                branch='seat_recommend',
                extras=extras,
                timings=timings,
            )

        start_time = time.time_ns()
        try:
            vlm_result = request_feature_description_chain(
                color_image_url, sys_prompt_recommend, text_prompt,
                provider_models=self._feature_provider_chain,
                qwen_api_backend=self.qwen_api_backend,
                timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries,
                logger=self.get_logger(),
            )
        except FeatureVlmError as exc:
            t_vlm_ms = (time.time_ns() - start_time) / 1e6
            self.get_logger().error(f'Seat recommend VLM call failed on every provider: {exc}')
            response.status = 1
            response.error_msg = f'VLM call failed on every provider: {exc}.'
            response.recommendation = ''
            _emit_vision_log('', 1, str(exc), t_vlm_ms, '')
            return response

        t_vlm_ms = (time.time_ns() - start_time) / 1e6
        self.get_logger().info(
            f'Finished, time = {(time.time_ns() - start_time) / 1e9:.3f} s'
        )
        if self.log_prompts:
            self.get_logger().info('seat recommendation prompt: ' + text_prompt)
        response.status = 0
        response.error_msg = ''
        response.recommendation = vlm_result.text
        _emit_vision_log(response.recommendation, 0, '', t_vlm_ms, vlm_result.provider)
        return response


def main():
    load_env()
    rclpy.init()
    feature_service = FeatureService()
    rclpy.spin(feature_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
