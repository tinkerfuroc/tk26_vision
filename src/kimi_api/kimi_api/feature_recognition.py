"""LLM-backed person feature extraction and seat recommendation.

Ports tk23 `kimi_api/feature_recognition.py`. Two services:
- `feature_extraction_service`: sends the latest Orbbec RGB frame to an
  OpenRouter vision model and asks for a structured person description.
- `seat_recommend_service`: takes a list of named/feature-described people,
  sends the current frame, asks the model where the new guest should sit.

Changes from tk23:
- API key/base URL/model from environment (via `.env.example`) instead of
  hardcoded literals.
- RealSense branches dropped (tk23 hardcoded camera_types=['orbbec'] so they
  were dead code).
- Temporary JPEG is written with `tempfile.NamedTemporaryFile` instead of
  CWD-relative `feature_extraction.jpg`.
"""

import base64
import os
import tempfile
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from message_filters import Subscriber
from openai import OpenAI
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from tinker_vision_msgs.srv import FeatureExtraction, SeatRecommendation

from ._env import base_url, default_model, load_env, require_api_key


class FeatureService(Node):
    def __init__(self):
        super().__init__(f'feature_service_{int(time.time())}')

        self.camera_types = ['orbbec']

        self.declare_parameter('log_prompts', True)
        self.declare_parameter('llm_model', default_model())
        self.log_prompts = self.get_parameter('log_prompts').get_parameter_value().bool_value
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value

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

        self.client = OpenAI(api_key=require_api_key(), base_url=base_url())

        self.extraction_srv = self.create_service(
            FeatureExtraction, 'feature_extraction_service', self.feature_extraction_srv_callback,
        )
        self.seat_recommend_srv = self.create_service(
            SeatRecommendation, 'seat_recommend_service', self.seat_recommend_srv_callback,
        )
        self.get_logger().info(f'Feature services initialized (model={self.llm_model}).')

    def camera_info_orbbec_callback(self, info):
        self.camera_intrinsic['orbbec'] = info

    def img_orbbec_callback(self, color_msg):
        self.recent_msg['orbbec'] = color_msg

    def _encode_image(self, img) -> str:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            cv2.imwrite(tmp_path, img)
            with open(tmp_path, 'rb') as f:
                data = f.read()
        finally:
            os.unlink(tmp_path)
        return f'data:image/jpg;base64,{base64.b64encode(data).decode("utf-8")}'

    def feature_extraction_srv_callback(
        self,
        request: FeatureExtraction.Request,
        response: FeatureExtraction.Response,
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
            response.feature = ''
            return response

        color_image_url = self._encode_image(color_img)

        sys_prompt = (
            'You will be asked to extract features of one single designated person in an image,'
            ' including their gender, approximate age in years, facial features (hair length,'
            ' with or without glasses), hair color, and atleast two pieces of clothing (the more'
            ' the better, but no more than five). Output in the format of "[gender pronoun] is'
            ' [gender], [gender pronoun] are approximately [approximate age in years (give in'
            ' words, such as "twenty", not numeric numerals)] years-old, [gender pronoun] has'
            ' [hair color] hair and [facial features]. [gender pronoun] is wearing [clothing]",'
            ' do not include other information'
        )

        start_time = time.time_ns()
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image_url', 'image_url': {'url': color_image_url}},
                            {
                                'type': 'text',
                                'text': 'extract the features of the single person in the center of the image.',
                            },
                        ],
                    },
                ],
            )

            self.get_logger().info(
                f'Finished, time = {(time.time_ns() - start_time) / 1e9:.3f} s'
            )
            features = completion.choices[0].message.content
            response.feature = features
            response.status = 0
            response.error_msg = ''
        except Exception as e:
            self.get_logger().error(f'Error in LLM call: {e}')
            response.feature = ''
            response.status = 1
            response.error_msg = 'API call failed.'
            self.get_logger().warn('API call failed.')

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

        color_image_url = self._encode_image(color_img)

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

        start_time = time.time_ns()
        completion = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {'role': 'system', 'content': sys_prompt_recommend},
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': color_image_url}},
                        {'type': 'text', 'text': text_prompt},
                    ],
                },
            ],
        )

        self.get_logger().info(
            f'Finished, time = {(time.time_ns() - start_time) / 1e9:.3f} s'
        )
        if self.log_prompts:
            self.get_logger().info('seat recommendation prompt: ' + text_prompt)
        response.status = 0
        response.error_msg = ''
        response.recommendation = completion.choices[0].message.content
        return response


def main():
    load_env()
    rclpy.init()
    feature_service = FeatureService()
    rclpy.spin(feature_service)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
