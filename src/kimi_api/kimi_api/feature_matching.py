"""LLM-backed person re-identification by image-vs-image matching.

Calls `object_detection_generalist` (the generalist YOLO/VLM service) to
crop each detected person in the current scene, then asks a vision LLM to
match each caller-supplied REFERENCE image (the comparison image captured
during feature extraction) to one of the candidate crops. The text feature
description is supplied as a tiebreaker hint only.
"""

import ast
import time

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from openai import OpenAI
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from tinker_vision_msgs_26.srv import FeatureMatching
from tinker_vision_msgs_26.srv import ObjectDetectionGeneralist as ObjectDetection

from ._env import base_url, default_model, load_env, require_api_key
from ._image_utils import bbox_from_mask, encode_to_data_url


class FeatureMatchingService(Node):
    def __init__(self):
        super().__init__(f'feature_matching_service_{int(time.time())}')

        self.max_person_per_image = 5

        self.server_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        self.declare_parameter('log_prompts', True)
        self.declare_parameter('llm_model', default_model())
        self.declare_parameter('detection_service', 'object_detection_generalist')
        self.log_prompts = self.get_parameter('log_prompts').get_parameter_value().bool_value
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        detection_service = self.get_parameter('detection_service').get_parameter_value().string_value

        self.bridge = CvBridge()

        self.client = OpenAI(api_key=require_api_key(), base_url=base_url())

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

    async def feature_matching_srv_callback(
        self,
        request: FeatureMatching.Request,
        response: FeatureMatching.Response,
    ):
        n_refs = len(request.comparison_images)
        assert 0 < n_refs <= 26, 'Too few (or too many) references to match.'
        assert len(request.features) == n_refs, (
            f'features ({len(request.features)}) and comparison_images ({n_refs}) length mismatch.'
        )
        self.get_logger().info(f'Request received with {n_refs} references.')

        start_time = time.time_ns()

        if not self.detection_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Detection service unavailable.')
            response.status = 1
            response.error_msg = 'Detection service unavailable.'
            response.centroids = []
            return response

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
        for i, obj in enumerate(detection_res.objects):
            if obj.cls == 'person':
                depth = obj.centroid.z if 'orbbec' in request.camera else obj.centroid.x
                if request.max_distance < 0.01 or depth < request.max_distance:
                    seg = self.bridge.imgmsg_to_cv2(detection_res.segments[i], '8UC1')
                    bbox = bbox_from_mask(seg)
                    cropped_person_imgs.append(
                        (i, color_img[bbox[0]:bbox[2], bbox[1]:bbox[3]], obj.centroid)
                    )
                    self.get_logger().info(f'Person {i} detected: {bbox}, depth = {depth}')

        if len(cropped_person_imgs) == 0:
            self.get_logger().warn('No person detected.')
            response.status = 1
            response.error_msg = 'No person detected.'
            response.centroids = []
            return response

        if len(cropped_person_imgs) > self.max_person_per_image:
            cropped_person_imgs = cropped_person_imgs[: self.max_person_per_image]

        candidate_urls = [encode_to_data_url(img) for _, img, _ in cropped_person_imgs]

        reference_urls = []
        for ref_msg in request.comparison_images:
            if len(ref_msg.data) == 0:
                reference_urls.append(None)
            else:
                ref_img = self.bridge.imgmsg_to_cv2(ref_msg, 'bgr8')
                reference_urls.append(encode_to_data_url(ref_img))

        self.get_logger().info(
            f'Persons cropped + references encoded. Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )

        n_cand = len(cropped_person_imgs)
        sys_prompt = (
            f'You will be shown {n_refs} REFERENCE images of specific people, then '
            f'{n_cand} CANDIDATE crops taken from a wider scene. For each reference '
            f'(0..{n_refs - 1}), output the candidate index whose person is the SAME '
            'individual as the reference. Use clothing, hair color/length, body shape, '
            'and posture as evidence. The user may also provide a textual description '
            'per reference; treat it as a tiebreaker hint only. '
            f'Output ONLY a JSON list of length {n_refs}, e.g. "[0, 2, 1]". '
            'Use -1 for a reference with no plausible match in the candidates. '
            'Do not include explanations.'
        )

        user_content = []
        for i, ref_url in enumerate(reference_urls):
            if ref_url is None:
                user_content.append(
                    {'type': 'text', 'text': f'Reference {i} (text-only, see hints below):'}
                )
            else:
                user_content.append({'type': 'text', 'text': f'Reference {i}:'})
                user_content.append({'type': 'image_url', 'image_url': {'url': ref_url}})

        for j, cand_url in enumerate(candidate_urls):
            user_content.append({'type': 'text', 'text': f'Candidate {j}:'})
            user_content.append({'type': 'image_url', 'image_url': {'url': cand_url}})

        text_tail = 'Textual hints per reference:\n'
        for i, feat in enumerate(request.features):
            text_tail += f'- Reference {i}: {feat or "(none)"}\n'
        text_tail += (
            f'Now output the JSON list of length {n_refs} mapping each reference '
            'to the matching candidate index.'
        )
        user_content.append({'type': 'text', 'text': text_tail})

        if self.log_prompts:
            self.get_logger().info(f'text_tail: {text_tail}')

        result = None
        for it in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {'role': 'system', 'content': sys_prompt},
                        {'role': 'user', 'content': user_content},
                    ],
                )
            except Exception as e:
                self.get_logger().warn(f'API call failed: {e}')
                completion = None

            self.get_logger().info(
                f'LLM finished.      Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
            )

            try:
                raw = completion.choices[0].message.content
                self.get_logger().info(f'LLM response: {raw}')
                result = ast.literal_eval(raw)
                self.get_logger().info(f'Parse response succeed ({it + 1}/3).')
                break
            except Exception:
                self.get_logger().info(f'Parse response failed ({it + 1}/3).')
                continue

        if result is None:
            self.get_logger().warn('Failed to parse response. Falling back...')
            result = [i % n_cand for i in range(n_refs)]

        response.error_msg = ''
        response.centroids = []

        if not isinstance(result, list):
            response.error_msg = f'Not a list: {result}.'
        elif len(result) != n_refs:
            response.error_msg = f'Invalid length: {result}.'

        if len(response.error_msg) == 0:
            for i in range(n_refs):
                cand_id = result[i]

                if not isinstance(cand_id, int):
                    response.error_msg = f'result[{i}] contains non-int values: {result}.'
                    break

                if cand_id < -1 or cand_id >= n_cand:
                    response.error_msg = f'result[{i}] contains invalid Candidate ID: {result}.'
                    break

                if cand_id == -1:
                    response.error_msg = f'result[{i}] unmatched: {result}.'
                    break

                response.centroids.append(
                    PointStamped(header=detection_res.header, point=cropped_person_imgs[cand_id][2])
                )

        if len(response.error_msg) > 0:
            response.status = 1
            response.centroids = []
        else:
            response.status = 0

        self.get_logger().info(
            f'Result processed.   Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )
        return response


def main():
    load_env()
    rclpy.init()
    feature_matching_service = FeatureMatchingService()
    rclpy.spin(feature_matching_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
