"""LLM-backed person re-identification.

Calls `object_detection_generalist` (the generalist YOLO/VLM service) to
crop each detected person in the current scene, then asks a vision LLM to
match each caller-supplied feature description (and reference image, when
provided) to one of the candidate crops.

Two prompt modes:
  * Image+text — when the caller sends a reference image per feature, the
    VLM matches references to candidates using the photo with text as a
    tiebreaker.
  * Text-only (legacy) — when the caller sends fewer reference images than
    features (including zero), comparison_images is dropped entirely and
    the VLM matches descriptions to candidate crops by text alone, mirroring
    the tk23 behavior.

In both modes the VLM is required to match every entry to some candidate
(no -1). Results that fail strict validation are patched in-place
(coercion / clamp / cyclic fallback for unrecoverable cells); only
structurally unsalvageable output (non-list, empty list) triggers a VLM
retry, and only after exhausting all retries is status=1 returned.
"""

import os
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PointStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from tf2_geometry_msgs import do_transform_point
from tf2_ros import (
    Buffer,
    ConnectivityException,
    ExtrapolationException,
    LookupException,
    TransformListener,
)
from tinker_vision_msgs_26.srv import FeatureMatching
from tinker_vision_msgs_26.srv import ObjectDetectionGeneralist as ObjectDetection
from vision_util.vision_logging import VisionLogger

from ._env import (
    default_model,
    load_env,
    require_api_key,
    require_dashscope_api_key,
)
from ._image_utils import bbox_from_mask, encode_to_data_url
from ._match_vlm import MatchVlmError, request_match_indices_chain

# Match-evidence attribute list — keep in sync with the five slots
# feature_recognition.FEATURE_SYS_PROMPT asks for (spec:
# docs/superpowers/specs/2026-07-02-feature-5slot-cut-design.md).
MATCH_EVIDENCE = (
    'hair color and length, gender, apparent age, glasses, and '
    'upper-body clothing color and type'
)


def build_matching_sys_prompt(n_cand: int, n_feats: int, text_only: bool) -> str:
    """Sys-prompt for the match call; pure so tests can pin its content.

    ``text_only`` selects the legacy descriptions-only wording; otherwise
    the reference-image wording is produced. Both keep the JSON-list
    output contract and the forced-match rule unchanged.
    """
    if text_only:
        return (
            f'You will be shown {n_cand} CANDIDATE crops of people and {n_feats} '
            f'textual DESCRIPTIONS. For each description, output the candidate index '
            f'(0..{n_cand - 1}) whose person best matches that description. '
            f'Output ONLY a JSON list of length {n_feats}, e.g. "[0, 3, 1]". '
            'EVERY description MUST be matched to a candidate. If you are uncertain, '
            f'pick the candidate whose visible features ({MATCH_EVIDENCE}) '
            'are CLOSEST to the description. NEVER use -1 or any negative number. '
            'Multiple descriptions MAY map to the same candidate. '
            'Do not include explanations.'
        )
    return (
        f'You will be shown {n_feats} REFERENCE images of specific people, then '
        f'{n_cand} CANDIDATE crops taken from a wider scene. For each reference '
        f'(0..{n_feats - 1}), output the candidate index whose person is the SAME '
        f'individual as the reference. Use {MATCH_EVIDENCE} as evidence. '
        'The user may also provide a textual description per reference; treat it '
        'as a tiebreaker hint only. '
        f'Output ONLY a JSON list of length {n_feats}, e.g. "[0, 2, 1]". '
        'EVERY reference MUST be matched to a candidate. If you are uncertain, '
        f'pick the candidate whose features ({MATCH_EVIDENCE}, and the '
        'description) are CLOSEST to the reference. NEVER use -1 or any '
        'negative number. Do not include explanations.'
    )


class FeatureMatchingService(Node):
    def __init__(self):
        super().__init__(f'feature_matching_service_{int(time.time())}')

        self.max_person_per_image = 5

        self.server_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        self.declare_parameter('log_prompts', True)
        self.declare_parameter('llm_model', default_model())
        self.declare_parameter('detection_service', 'object_detection_generalist')
        self.declare_parameter('vlm_timeout_s', 20.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('vlm_fallback_provider', 'qwen')  # '' to disable
        self.declare_parameter('match_model_qwen', 'qwen3-vl-plus')
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
        self.match_model_qwen = (
            self.get_parameter('match_model_qwen').get_parameter_value().string_value
        )

        self._vision_logger = VisionLogger(
            self,
            self.get_parameter('vision_logging_enabled').get_parameter_value().bool_value,
            self.get_parameter('vision_log_folder').get_parameter_value().string_value,
        )

        self.bridge = CvBridge()

        # TF backstop: detection nodes are now expected to express
        # centroids in `request.target_frame`, but if a code path skips
        # that (or a future detection backend forgets), this node
        # transforms each centroid here so the BT always receives points
        # in its requested frame. The lookup uses the pre-VLM detection
        # stamp, so the cache must outlast the provider chain's worst case
        # — 2 providers x vlm_max_retries x vlm_timeout_s + backoff
        # (2 x 3 x 20 + 3 ≈ 125 s at defaults); 180 s keeps the successful-
        # Qwen-fallback path from falling off the back of the buffer.
        self.tf_buffer = Buffer(
            cache_time=rclpy.duration.Duration(seconds=180.0)
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        require_api_key()  # fail fast at init if the primary Gemini key is missing
        self._match_provider_chain = self._resolve_match_provider_chain()

        self.detection_cli = self.create_client(
            ObjectDetection, detection_service, callback_group=self.client_cb_group,
        )

        self.matching_srv = self.create_service(
            FeatureMatching,
            'feature_matching_service',
            self.feature_matching_srv_callback,
            callback_group=self.server_cb_group,
        )

        self.get_logger().info(
            f'Feature matching service initialized (model={self.llm_model}, '
            f'detection_service={detection_service}).'
        )

    def _resolve_match_provider_chain(self) -> list:
        """Ordered (provider, model) chain for feature matching: Gemini
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
                    require_dashscope_api_key()
                    chain.append(('qwen', self.match_model_qwen))
                except RuntimeError:
                    self.get_logger().warn(
                        f'Fallback provider {fb!r} key missing; fallback disabled.'
                    )
        self.get_logger().info(
            f'feature_matching provider chain: {[p for p, _ in chain]}'
        )
        return chain

    def _stamped_in_target_frame(
        self,
        point: Point,
        det_header,
        target_frame: str,
    ) -> PointStamped:
        """Wrap a centroid Point as PointStamped in target_frame.

        No-op when target_frame is empty or already matches detection's
        header (the detection node has done the transform). On TF lookup
        failure, log a warn and return the un-transformed point with the
        detection-frame header so the message is at least self-consistent
        — downstream consumers can still detect the frame mismatch and
        abort if they care.
        """
        src_frame = det_header.frame_id
        if not target_frame or target_frame == src_frame:
            return PointStamped(header=det_header, point=point)
        try:
            tf = self.tf_buffer.lookup_transform(
                target_frame, src_frame, det_header.stamp,
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
        except (LookupException, ConnectivityException,
                ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF {src_frame} -> {target_frame} failed: {e}; '
                'returning detection-frame point'
            )
            return PointStamped(header=det_header, point=point)
        ps = PointStamped(header=det_header, point=point)
        ps_out = do_transform_point(ps, tf)
        ps_out.header.frame_id = target_frame
        return ps_out

    async def feature_matching_srv_callback(
        self,
        request: FeatureMatching.Request,
        response: FeatureMatching.Response,
    ):
        n_refs = len(request.comparison_images)
        n_feats = len(request.features)

        if n_feats == 0:
            self.get_logger().warn('No features provided.')
            response.status = 1
            response.error_msg = 'No features provided.'
            response.centroids = []
            return response
        if n_feats >= 26:
            self.get_logger().warn('Too many features provided; consider reducing to <26 to fit within VLM context window.')
            response.status = 1
            response.error_msg = 'Too many features; reduce to <26.'
            response.centroids = []
            return response

        # Two prompt modes: text-only legacy when references are short
        # (the user-requested fallback — discard partial refs entirely),
        # image+text when every feature has a paired reference image.
        text_only_mode = (n_refs < n_feats)
        if text_only_mode:
            self.get_logger().info(
                f'Reference images ({n_refs}) < features ({n_feats}); '
                f'using legacy text-only matching.'
            )
            n_refs = 0
        elif n_refs > n_feats:
            self.get_logger().warn(
                f'More reference images ({n_refs}) than features ({n_feats}).'
            )
            response.status = 1
            response.error_msg = (
                f'More reference images ({n_refs}) than features ({n_feats}).'
            )
            response.centroids = []
            return response

        self.get_logger().info(
            f'Request received with {n_feats} features '
            f'({"text-only" if text_only_mode else f"{n_refs} references"}).'
        )

        start_time = time.time_ns()

        self.get_logger().info('Calling detection service...')
        if not self.detection_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Detection service unavailable.')
            response.status = 1
            response.error_msg = 'Detection service unavailable.'
            response.centroids = []
            return response

        self.get_logger().info('Detection service available, sending request...')
        detection_req = ObjectDetection.Request()
        detection_req.camera = request.camera
        detection_req.prompt = 'person'
        detection_req.return_rgb_image = True
        detection_req.return_segments = True
        detection_req.use_vlm_sam_fallback = True
        detection_req.target_frame = request.target_frame

        detection_future = self.detection_cli.call_async(detection_req)
        await detection_future
        detection_res = detection_future.result()

        self.get_logger().info('Detection service responded, processing results...')
        if detection_res is None or detection_res.status != 0:
            err = detection_res.error_msg if detection_res is not None else 'no response'
            self.get_logger().warn(f'Detection service failed: {err}')
            response.status = 1
            response.error_msg = f'Detection failed: {err}.'
            response.centroids = []
            return response

        self.get_logger().info(
            f'Detection finished. Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )

        color_img = self.bridge.imgmsg_to_cv2(detection_res.rgb_image, 'bgr8')

        cropped_person_imgs = []
        n_person_total = 0
        n_person_far = 0
        for i, obj in enumerate(detection_res.objects):
            if obj.cls != 'person':
                continue
            n_person_total += 1
            depth = obj.centroid.z if 'orbbec' in request.camera else obj.centroid.x
            if request.max_distance >= 0.01 and depth >= request.max_distance:
                n_person_far += 1
                self.get_logger().info(
                    f'Person {i} skipped: depth={depth:.2f}m '
                    f'>= max_distance={request.max_distance:.2f}m'
                )
                continue
            seg = self.bridge.imgmsg_to_cv2(detection_res.segments[i], '8UC1')
            bbox = bbox_from_mask(seg)
            cropped_person_imgs.append(
                (i, color_img[bbox[0]:bbox[2], bbox[1]:bbox[3]], obj.centroid)
            )
            self.get_logger().info(f'Person {i} detected: {bbox}, depth = {depth:.2f}m')

        if len(cropped_person_imgs) == 0:
            if n_person_total > 0 and n_person_far == n_person_total:
                msg = (
                    f'All {n_person_total} detected persons exceeded '
                    f'max_distance={request.max_distance:.2f}m.'
                )
            elif n_person_total > 0:
                msg = f'{n_person_total} persons detected but none usable (segmentation/depth missing).'
            else:
                msg = 'No person detected.'
            self.get_logger().warn(msg)
            response.status = 1
            response.error_msg = msg
            response.centroids = []
            return response

        if len(cropped_person_imgs) > self.max_person_per_image:
            cropped_person_imgs = cropped_person_imgs[: self.max_person_per_image]

        candidate_urls = [encode_to_data_url(img) for _, img, _ in cropped_person_imgs]

        self.get_logger().info(
            f'Persons cropped and encoded. Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )
        reference_urls = []
        reference_imgs = []
        if not text_only_mode:
            for ref_msg in request.comparison_images:
                ref_img = self.bridge.imgmsg_to_cv2(ref_msg, 'bgr8')
                reference_urls.append(encode_to_data_url(ref_img))
                reference_imgs.append(ref_img)

        self.get_logger().info(
            f'Persons cropped + references encoded. Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )

        n_cand = len(cropped_person_imgs)
        sys_prompt = build_matching_sys_prompt(n_cand, n_feats, text_only_mode)

        user_content = []
        for i, ref_url in enumerate(reference_urls):
            user_content.append({'type': 'text', 'text': f'Reference {i}:'})
            user_content.append({'type': 'image_url', 'image_url': {'url': ref_url}})

        for j, cand_url in enumerate(candidate_urls):
            user_content.append({'type': 'text', 'text': f'Candidate {j}:'})
            user_content.append({'type': 'image_url', 'image_url': {'url': cand_url}})

        if text_only_mode:
            text_tail = 'Descriptions:\n'
            for i, feat in enumerate(request.features):
                text_tail += f'- Description {i}: {feat or "(none)"}\n'
            text_tail += (
                f'Now output the JSON list of length {n_feats} mapping each description '
                'to the matching candidate index.'
            )
        else:
            text_tail = 'Textual hints per reference:\n'
            for i, feat in enumerate(request.features):
                text_tail += f'- Reference {i}: {feat or "(none)"}\n'
            text_tail += (
                f'Now output the JSON list of length {n_feats} mapping each reference '
                'to the matching candidate index.'
            )
        user_content.append({'type': 'text', 'text': text_tail})

        if self.log_prompts:
            self.get_logger().info(f'text_tail: {text_tail}')

        self.get_logger().info('Sending request to VLM...')
        result = None
        provider_used = ''
        last_error = ''
        try:
            match_res = request_match_indices_chain(
                sys_prompt, user_content,
                n_feats=n_feats, n_cand=n_cand,
                provider_models=self._match_provider_chain,
                timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries,
                logger=self.get_logger(),
            )
            result = match_res.indices
            provider_used = match_res.provider
            self.get_logger().info(
                f'VLM accepted (provider={provider_used}). '
                f'Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
            )
        except MatchVlmError as exc:
            last_error = str(exc)
            self.get_logger().warn(f'VLM match failed on every provider: {exc}')

        # Build a detection record per candidate crop so the vision_log overlay
        # carries the labeled candidate index ('Cand j') the VLM sees in its
        # prompt. Filled with the matched reference index after VLM result is
        # parsed below, so a reviewer can read 'Cand 2 <- Ref 0' off the bbox.
        detections = []
        for j, (det_idx, _crop, centroid) in enumerate(cropped_person_imgs):
            y1, x1, y2, x2 = bbox_from_mask(
                self.bridge.imgmsg_to_cv2(detection_res.segments[det_idx], '8UC1')
            )
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'cls_name': f'Cand {j}',
                'candidate_idx': j,
                'detection_idx': det_idx,
                'centroid_3d': [
                    float(centroid.x), float(centroid.y), float(centroid.z),
                ],
                'matched_ref_idx': None,
            })

        def _emit_vision_log(parsed_result, vlm_status, vlm_error_msg, provider_used=''):
            extras = {
                'reference_paths': [],
                'features_text': list(request.features),
                'text_only_mode': bool(text_only_mode),
                'matches': [
                    {
                        'ref': ref_i,
                        'cand': (
                            int(parsed_result[ref_i])
                            if isinstance(parsed_result, list)
                            and ref_i < len(parsed_result)
                            and isinstance(parsed_result[ref_i], int)
                            else None
                        ),
                    }
                    for ref_i in range(n_feats)
                ],
                'vlm_status': int(vlm_status),
                'vlm_error_msg': str(vlm_error_msg),
                'vlm_provider_used': str(provider_used or ''),
            }
            request_ctx = {
                'camera': request.camera,
                'max_distance': float(request.max_distance),
                'target_frame': request.target_frame,
                'n_features': n_feats,
                'n_references': n_refs,
                'n_candidates': n_cand,
            }
            ts = self._vision_logger.write(
                rgb_img=color_img,
                detections=detections,
                request_ctx=request_ctx,
                branch='feature_matching',
                extras=extras,
            )
            if ts is None or self._vision_logger.run_dir is None:
                return
            for ref_i, ref_img in enumerate(reference_imgs):
                if ref_img is None:
                    extras['reference_paths'].append(None)
                    continue
                out_path = self._vision_logger.aux_path(
                    ts, f'ref{ref_i}', 'jpg', branch='feature_matching',
                )
                try:
                    cv2.imwrite(out_path, ref_img)
                    extras['reference_paths'].append(os.path.basename(out_path))
                except Exception as exc:  # noqa: BLE001
                    self.get_logger().warn(
                        f'feature_matching: reference image dump failed: {exc}'
                    )
                    extras['reference_paths'].append(None)

        if result is None:
            # last_error is the MatchVlmError text, which carries the
            # per-provider attempt breakdown — no attempt count here, the
            # chain makes up to retries x providers attempts.
            self.get_logger().warn(
                f'VLM match failed on every provider; returning status=1: '
                f'{last_error}'
            )
            response.status = 1
            response.error_msg = (
                f'VLM match failed on every provider: {last_error}.'
            )
            response.centroids = []
            _emit_vision_log(None, response.status, response.error_msg)
            return response

        # patch_result (in _match_vlm.py) guarantees every value is in
        # [0, n_cand) — build centroids directly without per-element
        # re-validation.
        response.error_msg = ''
        response.centroids = [
            self._stamped_in_target_frame(
                cropped_person_imgs[cand_id][2],
                detection_res.header,
                request.target_frame,
            )
            for cand_id in result
        ]
        response.status = 0

        # Annotate overlay bbox labels with 'Cand j <- Ref i' for log review.
        for ref_i, cand_id in enumerate(result):
            if 0 <= cand_id < len(detections):
                detections[cand_id]['matched_ref_idx'] = ref_i
                detections[cand_id]['cls_name'] = (
                    f'Cand {cand_id} <- Ref {ref_i}'
                )

        self.get_logger().info(
            f'Result processed.   Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )
        _emit_vision_log(result, response.status, response.error_msg, provider_used)
        return response


def main():
    load_env()
    rclpy.init()
    feature_matching_service = FeatureMatchingService()
    rclpy.spin(feature_matching_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
