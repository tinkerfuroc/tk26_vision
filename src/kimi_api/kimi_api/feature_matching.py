"""LLM-backed person re-identification by description matching.

Ports tk23 `kimi_api/feature_matching.py`. Calls `object_detection` (the
generalist YOLO service), crops each detected person, then asks an OpenRouter
vision model to assign each caller-supplied description to one of the cropped
images. Returns a PointStamped per matched description.

Changes from tk23:
- API key/base URL/model from environment.
- `detection_service` ROS param lets the operator retarget between
  `object_detection` (default-YOLO generalist) and `object_detection_yolo`
  (custom-trained variant).
- Temporary JPEGs written via `tempfile.NamedTemporaryFile`.
"""

import ast
import base64
import os
import tempfile
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from openai import OpenAI
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from tinker_vision_msgs.srv import FeatureMatching
from tinker_vision_msgs_26.srv import ObjectDetection

from ._env import base_url, default_model, load_env, require_api_key


def bbox_from_mask(mask):
    nonzero = np.nonzero(mask)
    x1, y1, x2, y2 = np.min(nonzero[0]), np.min(nonzero[1]), np.max(nonzero[0]), np.max(nonzero[1])
    return x1, y1, x2, y2


def _encode_to_data_url(img) -> str:
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cv2.imwrite(tmp_path, img)
        with open(tmp_path, 'rb') as f:
            data = f.read()
    finally:
        os.unlink(tmp_path)
    return f'data:image/jpg;base64,{base64.b64encode(data).decode("utf-8")}'


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
        assert 0 < len(request.features) <= 26, 'Too few (or too many) features to match.'
        self.get_logger().info('Request received.')

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

        if detection_res.status != 0:
            self.get_logger().warn('Detection service failed.')
            response.status = 1
            response.error_msg = (
                f'Detection failed (status {detection_res.status}): {detection_res.error_msg}.'
            )
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

        person_img_urls = []
        for _, img, _ in cropped_person_imgs:
            person_img_urls.append(_encode_to_data_url(img))

        self.get_logger().info(
            f'Person cropped.     Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )

        sys_prompt = (
            f"You will be given {len(person_img_urls)} person's images and "
            f"{len(request.features)} descriptions. "
            'For each description, determine which image matches the described person best. '
            "Each description will contain the person's gender, approximate age, facial features, "
            'hair color, and clothing. '
            f'Output a list of length {len(request.features)}, in the format of '
            '"[image ID matching description 0, image ID matching description 1, ...]".'
            f'Descriptions are numbered from 0 to {len(request.features) - 1}, and images are '
            f'numbered from 0 to {len(person_img_urls) - 1}.'
            'For each description, there will be exactly one image that matches it best.'
            'Output the final list ONLY. Do not include explanations or other information.'
            'Example output (3 descriptions and 4 images): [0, 3, 1].'
            'Example output (5 descriptions and 3 images): [0, 2, 0, 1, 1].'
        )

        text_prompt = 'Match the following descriptions to the images above:\n'
        for i, feat in enumerate(request.features):
            text_prompt += f'- Description {i}: {feat}\n'
        if self.log_prompts:
            self.get_logger().info(f'text_prompt: {text_prompt}')

        image_contents = []
        for i, img_url in enumerate(person_img_urls):
            image_contents.append({'type': 'text', 'text': f'Image {i}:'})
            image_contents.append({'type': 'image_url', 'image_url': {'url': img_url}})

        result = None
        for it in range(3):
            try:
                completion = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {'role': 'system', 'content': sys_prompt},
                        {
                            'role': 'user',
                            'content': image_contents + [{'type': 'text', 'text': text_prompt}],
                        },
                    ],
                )
            except Exception as e:
                self.get_logger().warn(f'API call failed: {e}')
                completion = None

            self.get_logger().info(
                f'GPT finished.      Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
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
            result = [i % len(cropped_person_imgs) for i in range(len(request.features))]

        response.error_msg = ''
        response.centroids = []

        if not isinstance(result, list):
            response.error_msg = f'Not a list: {result}.'
        elif len(result) != len(request.features):
            response.error_msg = f'Invalid length: {result}.'

        if len(response.error_msg) == 0:
            for i in range(len(request.features)):
                img_id = result[i]

                if not isinstance(img_id, int):
                    response.error_msg = f'result[{i}] contains non-int values: {result}.'
                    break

                if img_id < -1 or img_id >= len(cropped_person_imgs):
                    response.error_msg = f'result[{i}] contains invalid Image ID: {result}.'
                    break

                if img_id == -1:
                    response.error_msg = f'result[{i}] unmatched: {result}.'
                    break

                response.centroids.append(
                    PointStamped(header=detection_res.header, point=cropped_person_imgs[img_id][2])
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
