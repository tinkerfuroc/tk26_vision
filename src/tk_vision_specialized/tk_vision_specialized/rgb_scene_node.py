"""VLM scene detection with optional capture-matched behaviour localization."""
import copy
from collections import deque
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
from sensor_msgs.msg import Image, RegionOfInterest, CameraInfo
from geometry_msgs.msg import PointStamped
from tinker_vision_msgs_26.msg import SceneObservation
from tinker_vision_msgs_26.srv import DetectRgbScene
from vision_util.vision_logging import VisionLogger
from vision_util.vlm_models import vision_flash_model, vision_qwen_model

from .scene_logging import log_scene
from .scene_vlm import infer_scene
from .scene_geometry import estimate_position, matched_depth_index


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
        if task == 'behaviour':
            defaults.update(depth_topic='/camera/depth/image_raw',
                            camera_info_topic='/camera/color/camera_info',
                            target_frame='map', sync_slop_s=0.1,
                            min_depth_m=0.2, max_depth_m=8.0,
                            min_depth_pixels=25, tf_timeout_s=0.5)
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
        if task == 'behaviour':
            from vision_util.tf_lookup import TransformHelper
            for key in ('sync_slop_s', 'min_depth_m', 'max_depth_m', 'tf_timeout_s'):
                if not math.isfinite(self.config[key]) or self.config[key] <= 0:
                    raise ValueError(f'{key} must be finite and positive')
            if self.config['min_depth_m'] >= self.config['max_depth_m'] or self.config['min_depth_pixels'] < 1:
                raise ValueError('Invalid depth range or minimum pixel count')
            if not self.config['target_frame'].strip():
                raise ValueError('target_frame must not be empty')
            self._depth_frames = deque(maxlen=20)
            self._camera_info = None
            self.transform_helper = TransformHelper(self)
            self.depth_subscription = self.create_subscription(
                Image, self.config['depth_topic'], self._on_depth, qos_profile_sensor_data,
                callback_group=self._intake_group)
            self.info_subscription = self.create_subscription(
                CameraInfo, self.config['camera_info_topic'], self._on_info,
                qos_profile_sensor_data, callback_group=self._intake_group)
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

    def _on_depth(self, message):
        with self._lock:
            self._depth_frames.append((message, time.monotonic()))

    def _on_info(self, message):
        with self._lock:
            self._camera_info = message

    @staticmethod
    def _stamp_ns(message):
        return message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec

    def _localization_snapshot(self, rgb, target):
        """Freeze registered depth, calibration and capture-time TF BEFORE VLM."""
        from vision_util.depth_reproject import decode_depth_metres
        with self._lock:
            frames = list(self._depth_frames)
            info = copy.deepcopy(self._camera_info)
        if not frames or info is None:
            raise ValueError('Missing registered depth or RGB CameraInfo')
        index = matched_depth_index(
            [self._stamp_ns(pair[0]) for pair in frames], [pair[1] for pair in frames],
            self._stamp_ns(rgb), time.monotonic(), self.config['max_image_age_s'],
            self.config['sync_slop_s'])
        depth = copy.deepcopy(frames[index][0])
        if not rgb.header.frame_id or info.header.frame_id != rgb.header.frame_id:
            raise ValueError('RGB CameraInfo frame does not match RGB optical frame')
        if (depth.width, depth.height) != (rgb.width, rgb.height) or (info.width, info.height) != (rgb.width, rgb.height):
            raise ValueError('Depth and CameraInfo must be registered at RGB resolution')
        if depth.encoding not in ('16UC1', 'mono16', '32FC1'):
            raise ValueError('Unsupported depth encoding; expected uint16 mm or float32 m')
        data = decode_depth_metres(self.bridge.imgmsg_to_cv2(depth, desired_encoding='passthrough')).copy()
        transform = None
        if target != rgb.header.frame_id:
            transform = self.transform_helper.try_lookup(
                target, rgb.header.frame_id, stamp=rgb.header.stamp,
                timeout_s=self.config['tf_timeout_s'])
            if transform is None:
                raise ValueError(f'No capture-time TF: {rgb.header.frame_id} -> {target}')
        return data, list(info.k), transform, copy.deepcopy(depth.header)

    def _locate(self, row, item, header, target, snapshot, error):
        row.position_error = error
        if snapshot is None:
            return
        try:
            depth, k, transform, _ = snapshot
            xyz = estimate_position(depth, k, item.bbox,
                                    min_depth=self.config['min_depth_m'],
                                    max_depth=self.config['max_depth_m'],
                                    min_pixels=self.config['min_depth_pixels'])
            point = PointStamped()
            point.header = copy.deepcopy(header)
            point.point.x, point.point.y, point.point.z = map(float, xyz)
            if transform is not None:
                point = self.transform_helper.transform_point(point, transform)
                if point is None:
                    raise ValueError('Point transform failed')
            if not all(math.isfinite(v) for v in (point.point.x, point.point.y, point.point.z)):
                raise ValueError('Non-finite transformed position')
            point.header.frame_id = target
            point.header.stamp = copy.deepcopy(header.stamp)
            row.position = point
            row.position_valid = True
            row.position_error = ''
        except Exception as exc:
            row.position_error = str(exc)

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
        snapshot = None
        location_error = 'Localization is not enabled for litter detection'
        target = (request.target_frame.strip() or self.config.get('target_frame', ''))
        response.status = DetectRgbScene.Response.STATUS_NO_IMAGE
        response.vlm_elapsed_s = -1.0
        try:
            message = self._snapshot()
            response.header = copy.deepcopy(message.header)
            response.rgb_image = message
            capture = {'frame_id': message.header.frame_id,
                       'stamp': {'sec': message.header.stamp.sec,
                                 'nanosec': message.header.stamp.nanosec}}
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8').copy()
            if self.task == 'behaviour':
                try:
                    snapshot = self._localization_snapshot(message, target)
                    location_error = ''
                except Exception as exc:
                    location_error = str(exc)
            response.status = DetectRgbScene.Response.STATUS_VLM_ERROR
            started = time.monotonic()
            try:
                observations, model_info = infer_scene(
                    image, self.task, self.providers,
                    timeout_s=self.config['vlm_timeout_s'], attempts=self.config['vlm_attempts'])
            finally:
                response.vlm_elapsed_s = time.monotonic() - started
            response.vlm_provider = model_info.get('provider', '')
            response.vlm_model = model_info.get('model', '')
            response.status = DetectRgbScene.Response.STATUS_OK
        except Exception as exc:
            response.error_msg = str(exc)

        response.detected = bool(observations)
        model_info['vlm_elapsed_s'] = response.vlm_elapsed_s
        overlay = image.copy() if image is not None else None
        for item in observations:
            x1, y1, x2, y2 = item.bbox
            row = SceneObservation()
            row.labels = list(item.labels)
            row.description = item.description
            row.bbox = RegionOfInterest(x_offset=x1, y_offset=y1,
                                        width=x2-x1, height=y2-y1, do_rectify=False)
            self._locate(row, item, response.header, target, snapshot, location_error)
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
                annotated_image=overlay,
                localization={
                    'target_frame': target if self.task == 'behaviour' else '',
                    'error': location_error,
                    'method': 'central_bbox_median_depth' if self.task == 'behaviour' else 'disabled',
                    'depth_header': ({'frame_id': snapshot[3].frame_id,
                                      'stamp': {'sec': snapshot[3].stamp.sec,
                                                'nanosec': snapshot[3].stamp.nanosec}}
                                     if snapshot else None),
                    'positions': [dict(valid=r.position_valid, error=r.position_error,
                                       frame_id=r.position.header.frame_id,
                                       stamp={'sec': r.position.header.stamp.sec,
                                              'nanosec': r.position.header.stamp.nanosec},
                                       xyz=([r.position.point.x, r.position.point.y, r.position.point.z]
                                            if r.position_valid else None))
                                  for r in response.observations]})
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
