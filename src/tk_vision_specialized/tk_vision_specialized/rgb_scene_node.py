"""On-demand VLM nodes using only the Orbbec RGB stream, never RGB-D sync."""
import copy
import math
import threading
import time

import cv2
from cv_bridge import CvBridge
from dotenv import load_dotenv
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, RegionOfInterest
from tinker_vision_msgs_26.msg import SceneObservation
from tinker_vision_msgs_26.srv import DetectRgbScene
from vision_util.vision_logging import VisionLogger
from vision_util.vlm_models import vision_flash_model, vision_qwen_model

from .scene_logging import log_scene
from .scene_vlm import infer_scene


class RgbSceneNode(Node):
    """Each service captures a stable image reference; intake runs independently."""

    def __init__(self, name, task):
        load_dotenv()
        super().__init__(name)
        self.task = task
        defaults = {
            'rgb_topic': '/camera/color/image_raw',
            'max_image_age_s': 2.0,
            'vlm_provider': 'qwen', 'vlm_fallback_provider': 'gemini',
            'vlm_model_qwen': vision_qwen_model(),
            'vlm_model_gemini': vision_flash_model(),
            'vlm_timeout_s': 30.0, 'vlm_attempts': 2,
            'vision_log_folder': 'vision_log',
        }
        self.config = {k: self.declare_parameter(k, v).value for k, v in defaults.items()}
        for key in ('max_image_age_s', 'vlm_timeout_s'):
            if not math.isfinite(float(self.config[key])) or self.config[key] <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if not 1 <= self.config['vlm_attempts'] <= 5:
            raise ValueError('vlm_attempts must be in 1..5')
        self.providers = []
        for key in ('vlm_provider', 'vlm_fallback_provider'):
            provider = self.config[key]
            if provider == '' and key == 'vlm_fallback_provider':
                continue
            if provider not in ('qwen', 'gemini'):
                raise ValueError(f'{key} must be qwen or gemini')
            item = (provider, self.config['vlm_model_' + provider])
            if item not in self.providers:
                self.providers.append(item)
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame = None
        self._received_at = 0.0
        # Logging is mandatory for these nodes, as part of the service contract.
        self.artifacts = VisionLogger(self, True, self.config['vision_log_folder'])
        self._intake_group = MutuallyExclusiveCallbackGroup()
        self._service_group = MutuallyExclusiveCallbackGroup()
        self.subscription = self.create_subscription(
            Image, self.config['rgb_topic'], self._on_image, qos_profile_sensor_data,
            callback_group=self._intake_group)
        self.service = self.create_service(
            DetectRgbScene, '~/detect', self._detect,
            callback_group=self._service_group)

    def _on_image(self, message):
        with self._lock:
            self._frame = message
            self._received_at = time.monotonic()

    def _snapshot(self):
        with self._lock:
            message, received = self._frame, self._received_at
        if message is None or time.monotonic() - received > self.config['max_image_age_s']:
            raise ValueError('No fresh Orbbec RGB image')
        stamp_ns = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
        age = (self.get_clock().now().nanoseconds - stamp_ns) / 1e9
        if stamp_ns <= 0 or age < -0.5 or age > self.config['max_image_age_s']:
            raise ValueError('RGB capture stamp is missing, stale or in the future')
        # The callback never mutates cached messages. The service owns a copy.
        return copy.deepcopy(message)

    def _detect(self, request, response):
        image = None
        observations = []
        model_info = {}
        capture = None
        response.status = DetectRgbScene.Response.STATUS_NO_IMAGE
        try:
            message = self._snapshot()
            response.header = copy.deepcopy(message.header)
            response.rgb_image = message
            capture = {'frame_id': message.header.frame_id,
                       'stamp': {'sec': message.header.stamp.sec,
                                 'nanosec': message.header.stamp.nanosec}}
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8').copy()
            response.status = DetectRgbScene.Response.STATUS_VLM_ERROR
            observations, model_info = infer_scene(
                image, self.task, self.providers,
                timeout_s=self.config['vlm_timeout_s'], attempts=self.config['vlm_attempts'])
            response.status = DetectRgbScene.Response.STATUS_OK
        except Exception as exc:
            response.error_msg = str(exc)

        response.detected = bool(observations)
        overlay = image.copy() if image is not None else None
        for item in observations:
            x1, y1, x2, y2 = item.bbox
            row = SceneObservation()
            row.labels = list(item.labels)
            row.description = item.description
            row.bbox = RegionOfInterest(x_offset=x1, y_offset=y1,
                                        width=x2-x1, height=y2-y1, do_rectify=False)
            response.observations.append(row)
            cv2.rectangle(overlay, (x1, y1), (x2-1, y2-1), (0, 255, 0), 2)
            cv2.putText(overlay, ','.join(item.labels), (x1, max(15, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if overlay is not None:
            response.annotated_image = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            response.annotated_image.header = copy.deepcopy(response.header)
        try:
            response.log_json = log_scene(
                self.artifacts, image, observations, task=self.task,
                capture_header=capture, status=response.status,
                error=response.error_msg, model_info=model_info,
                annotated_image=overlay)
        except Exception as exc:
            response.status = DetectRgbScene.Response.STATUS_LOG_ERROR
            response.error_msg = (response.error_msg + '; ' if response.error_msg else '') + str(exc)
            self.get_logger().error('Scene logging failed: ' + str(exc))
        return response


def run_node(name, task, args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=2)
    try:
        node = RgbSceneNode(name, task)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
