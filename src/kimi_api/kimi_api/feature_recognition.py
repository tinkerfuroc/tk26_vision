"""LLM-backed person feature extraction and seat recommendation.

Two services:
- `feature_extraction_service`: calls `object_detection_generalist`, picks
  the centermost detected person, crops to a tight bbox, runs a vision LLM
  on the crop for a structured text description, and returns BOTH the
  description and the crop as `sensor_msgs/Image` for downstream image-vs-image
  matching.
- `seat_recommend_service`: takes a list of named/feature-described people,
  sends the current frame, asks the model where the new guest should sit.
"""

import time

import cv2
import rclpy
from cv_bridge import CvBridge
from message_filters import Subscriber
from openai import OpenAI
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tinker_vision_msgs_26.srv import (
    FeatureExtraction,
    ObjectDetectionGeneralist as ObjectDetection,
    SeatRecommendation,
)

from ._env import base_url, default_model, load_env, require_api_key
from ._image_utils import bbox_from_mask, encode_to_data_url


class FeatureService(Node):
    def __init__(self):
        super().__init__(f'feature_service_{int(time.time())}')

        self.camera_types = ['orbbec']

        self.declare_parameter('log_prompts', True)
        self.declare_parameter('llm_model', default_model())
        self.declare_parameter('detection_service', 'object_detection_generalist')
        self.log_prompts = self.get_parameter('log_prompts').get_parameter_value().bool_value
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        detection_service = self.get_parameter('detection_service').get_parameter_value().string_value

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

        self.client = OpenAI(api_key=require_api_key(), base_url=base_url())

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

        self.get_logger().info(
            f'Detection finished. Time spent: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )

        color_img = self.bridge.imgmsg_to_cv2(detection_res.rgb_image, 'bgr8')
        h, w = color_img.shape[:2]
        cx_frame, cy_frame = w / 2.0, h / 2.0

        best_idx = -1
        best_dist = None
        best_bbox = None
        for i, obj in enumerate(detection_res.objects):
            if obj.cls != 'person':
                continue
            seg = self.bridge.imgmsg_to_cv2(detection_res.segments[i], '8UC1')
            x1, y1, x2, y2 = bbox_from_mask(seg)
            cx = (y1 + y2) / 2.0
            cy = (x1 + x2) / 2.0
            d = (cx - cx_frame) ** 2 + (cy - cy_frame) ** 2
            if best_dist is None or d < best_dist:
                best_dist = d
                best_idx = i
                best_bbox = (x1, y1, x2, y2)

        if best_idx < 0:
            self.get_logger().warn('No person detected.')
            response.status = 1
            response.error_msg = 'No person detected.'
            response.feature = ''
            response.comparison_image = Image()
            return response

        x1, y1, x2, y2 = best_bbox
        crop = color_img[x1:x2, y1:y2]
        comparison_image = self.bridge.cv2_to_imgmsg(crop, 'bgr8')
        comparison_image.header = detection_res.header
        response.comparison_image = comparison_image

        self.get_logger().info(
            f'Centermost person idx={best_idx}, bbox={best_bbox}. '
            f'Crop: {crop.shape}. Time: {(time.time_ns() - start_time) / 1e6:.2f} ms'
        )

        crop_url = encode_to_data_url(crop)

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

        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'image_url', 'image_url': {'url': crop_url}},
                            {
                                'type': 'text',
                                'text': 'extract the features of the person shown in the image.',
                            },
                        ],
                    },
                ],
            )

            self.get_logger().info(
                f'LLM finished. Total time: {(time.time_ns() - start_time) / 1e9:.3f} s'
            )
            response.feature = completion.choices[0].message.content
            response.status = 0
            response.error_msg = ''
        except Exception as e:
            self.get_logger().error(f'Error in LLM call: {e}')
            response.feature = ''
            response.status = 1
            response.error_msg = 'API call failed.'

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
