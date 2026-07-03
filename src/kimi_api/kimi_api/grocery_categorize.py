"""LLM-backed grocery categorization onto a multi-layer shelf.

Ports tk23 `kimi_api/grocery_categorize.py`. Calls `object_detection` to spot
shelf items, clusters them by height (scipy kmeans), asks an OpenRouter vision
model which shelf layer a new item belongs on, then picks an empty spot in
that layer and returns a PointStamped.

Changes from tk23:
- API key/base URL/model from environment.
- `detection_service` ROS param (default `object_detection`) for retargeting.
- JPEG encoding shared with feature_recognition / feature_matching via
  `kimi_api._image_utils.encode_to_data_url` (in-memory `cv2.imencode`).
- Dead Chinese-prompt block removed.
"""

import threading
import time

import geometry_msgs.msg
import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform)
from cv_bridge import CvBridge
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from scipy.cluster.vq import kmeans2
from sensor_msgs.msg import PointCloud2
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tinker_vision_msgs_26.action import Categorize
from tinker_vision_msgs_26.srv import ObjectDetectionGeneralist as ObjectDetection

from ._categorize_vlm import ShelfVlmError, request_shelf_layer_chain
from ._env import default_model, load_env, require_api_key, resolve_qwen_target
from ._image_utils import encode_to_data_url

USE_SHELF_HEIGHT = False
PROJECT_ON_LINE = False


def get_bounding_box(mask):
    nonzero = np.nonzero(mask)
    x1, y1, x2, y2 = np.min(nonzero[0]), np.min(nonzero[1]), np.max(nonzero[0]), np.max(nonzero[1])
    return x1, y1, x2, y2


class GroceryCategorizeAction(Node):
    def __init__(self):
        super().__init__('grocery_categorize')

        self.shelf_height = [0.5, 1.0, 1.5]  # bottom to top
        self.raise_height = 0.25
        self.push_in_distance = 0.05

        self.declare_parameter('llm_model', default_model())
        self.declare_parameter('detection_service', 'object_detection_generalist')
        # 60 s, not the 20 s the other kimi_api nodes use: the default model
        # here is gemini-2.5-pro on a two-image prompt, and the seat-bench
        # measurements for that model (n=144) put the median at 15.5 s and
        # p90 at 28.4 s — a 20 s cap would cancel roughly a third of calls
        # that the pre-fallback code (no per-call timeout) completed.
        self.declare_parameter('vlm_timeout_s', 60.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('vlm_fallback_provider', 'qwen')  # '' to disable
        self.declare_parameter('categorize_model_qwen', '')
        self.declare_parameter('qwen_api_backend', 'dashscope')
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        detection_service = self.get_parameter('detection_service').get_parameter_value().string_value
        self.vlm_timeout_s = self.get_parameter('vlm_timeout_s').get_parameter_value().double_value
        self.vlm_max_retries = (
            self.get_parameter('vlm_max_retries').get_parameter_value().integer_value
        )
        self.vlm_fallback_provider = (
            self.get_parameter('vlm_fallback_provider').get_parameter_value().string_value
        )
        self.categorize_model_qwen = (
            self.get_parameter('categorize_model_qwen').get_parameter_value().string_value
        )
        self.qwen_api_backend = self.get_parameter('qwen_api_backend').value

        self.server_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        self.action_server = ActionServer(
            self,
            Categorize,
            'grocery_categorize',
            self.grocery_categorize_callback,
            callback_group=self.server_cb_group,
        )

        require_api_key()  # fail fast at init if the primary Gemini key is missing
        self._categorize_provider_chain = self._resolve_categorize_provider_chain()

        self.orbec_pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/points',
            self.orbec_pc_callback,
            qos_profile=1,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.detection_cli = self.create_client(
            ObjectDetection, detection_service, callback_group=self.client_cb_group,
        )

        self.bridge = CvBridge()

        self.debug_pub = self.create_publisher(geometry_msgs.msg.PointStamped, 'categorize_debug1', 10)
        self.shelf_l_pub = self.create_publisher(geometry_msgs.msg.PointStamped, 'categorize_shelf_left', 10)
        self.shelf_r_pub = self.create_publisher(geometry_msgs.msg.PointStamped, 'categorize_shelf_right', 10)
        self.env_pc = None
        self.env_pc_lock = threading.Lock()
        self.last_time = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(
            f'GroceryCategorize initialized (model={self.llm_model}, '
            f'detection_service={detection_service}).'
        )

    def _resolve_categorize_provider_chain(self) -> list:
        """Ordered (provider, model) chain for shelf-layer categorization:
        Gemini (self.llm_model, already required at init) then, if
        configured, a Qwen fallback that is dropped with a warning when
        its key is missing rather than failing node startup."""
        chain = [('gemini', self.llm_model)]
        fb = self.vlm_fallback_provider
        if fb and fb != 'gemini':
            if fb != 'qwen':
                self.get_logger().warn(f'Unknown fallback provider {fb!r}; ignoring.')
            else:
                try:
                    _, _, resolved_model = resolve_qwen_target(
                        self.qwen_api_backend, self.categorize_model_qwen)
                    chain.append(('qwen', resolved_model))
                except RuntimeError:
                    self.get_logger().warn(
                        f'Fallback provider {fb!r} key missing; fallback disabled.'
                    )
        self.get_logger().info(
            f'grocery_categorize provider chain: {[p for p, _ in chain]}'
        )
        return chain

    async def orbec_pc_callback(self, msg):
        if self.last_time is None or self.last_time + 0.5 < time.time():
            with self.env_pc_lock:
                self.env_pc = msg
            self.last_time = time.time()

    async def grocery_categorize_callback(self, goal_handle):
        self.get_logger().info('Executing goal...')
        self.pt_shelf_left = goal_handle.request.pt_shelf_left
        self.pt_shelf_right = goal_handle.request.pt_shelf_right

        result = Categorize.Result()

        feedback_msg = Categorize.Feedback()
        feedback_msg.status = 0
        feedback_msg.message = 'Segmenting objects from shelf..'
        goal_handle.publish_feedback(feedback_msg)

        if self.pt_shelf_left is None or self.pt_shelf_right is None:
            self.get_logger().error('Shelf points not provided')
            result.status = 3
            result.error_msg = 'Shelf points not provided.'
            return result
        if self.pt_shelf_left.header.frame_id != 'map' or self.pt_shelf_right.header.frame_id != 'map':
            self.get_logger().error('Shelf point frame is not map.')
            result.status = 3
            result.error_msg = 'Shelf point frame is not map.'
            return result

        # 1. Get image from orbbec and call object detection, get shelf + items segmentation
        if not self.detection_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Detection service unavailable.')
            result.status = 1
            result.error_msg = 'Detection service unavailable.'
            return result
        detection_req = ObjectDetection.Request()
        detection_req.camera = 'orbbec'
        detection_req.prompt = goal_handle.request.prompt + ' . shelf'
        detection_req.return_rgb_image = True
        detection_req.return_segments = True
        detection_req.use_vlm_sam_fallback = True
        detection_req.target_frame = goal_handle.request.target_frame

        detection_future = self.detection_cli.call_async(detection_req)
        await detection_future
        detection_res = detection_future.result()

        if detection_res.status != 0:
            self.get_logger().warn('Detection service failed.')
            result.status = 1
            result.error_msg = (
                f'Detection failed (status {detection_res.status}): {detection_res.error_msg}.'
            )
            return result

        result.rgb_image = detection_res.rgb_image
        result.depth_image = detection_res.depth_image
        rgb_image = self.bridge.imgmsg_to_cv2(detection_res.rgb_image, 'bgr8')
        cluster_objects = []
        shelf_object = None
        for i, obj in enumerate(detection_res.objects):
            self.get_logger().info(
                f'Object {i}: {obj.cls}, ({obj.centroid.x}, {obj.centroid.y}, {obj.centroid.z})'
            )
            if obj.cls != 'shelf':
                cluster_objects.append(obj)
            else:
                if shelf_object is not None:
                    self.get_logger().warn('Multiple shelf objects detected, using the first one.')
                    continue
                shelf_object = obj

        # 2. Cluster items into n_layers using scipy kmeans
        if not USE_SHELF_HEIGHT:
            feedback_msg.status = 0
            feedback_msg.message = 'Clustering shelf items...'
            goal_handle.publish_feedback(feedback_msg)
            z_coords = np.array([obj.centroid.z for obj in cluster_objects]).reshape(-1, 1)
            k = goal_handle.request.n_layers
            centroids, labels = kmeans2(z_coords, k, minit='points')

            if goal_handle.request.target_frame == 'base_link':
                sorted_indices = np.argsort(centroids.flatten())
            else:
                sorted_indices = np.argsort(centroids.flatten())[::-1]
            new_labels = np.zeros_like(labels)
            for new_label, old_label in enumerate(sorted_indices):
                new_labels[labels == old_label] = new_label

            clusters = [[] for _ in range(k)]
            for point, label in zip(cluster_objects, new_labels):
                clusters[label].append(point)
        else:
            feedback_msg.status = 0
            feedback_msg.message = 'Categorizing shelf items according to height..'
            shelf_objects = []
            for i in range(len(self.shelf_height) - 1):
                shelf_objects.append([])
                for obj in cluster_objects:
                    if obj.centroid.z >= self.shelf_height[i] and obj.centroid.z < self.shelf_height[i + 1]:
                        shelf_objects[i].append(obj)

        # 3. Crop the grasped object from the table image
        feedback_msg.status = 0
        feedback_msg.message = 'Extracting item from table image...'
        goal_handle.publish_feedback(feedback_msg)

        img_table = self.bridge.imgmsg_to_cv2(goal_handle.request.img_table, 'bgr8')
        obj_segment = self.bridge.imgmsg_to_cv2(goal_handle.request.segment_object, '8UC1')
        obj_bbox = get_bounding_box(obj_segment)
        obj_segment = img_table[obj_bbox[0]:obj_bbox[2], obj_bbox[1]:obj_bbox[3]]

        # 4. Ask the LLM which layer
        feedback_msg.status = 0
        feedback_msg.message = 'Determining layer to put on...'
        goal_handle.publish_feedback(feedback_msg)

        obj_seg_url = encode_to_data_url(obj_segment)
        shelf_img_url = encode_to_data_url(rgb_image)

        sys_prompt = (
            f'You will be given a picture of a shelf with {goal_handle.request.n_layers} main'
            ' visible layers. Items on each layer of the shelf is already grouped according to'
            ' three categories: food, drink, and utilities. You will then be given a picture of'
            ' an object. Please determine which layer the object should be placed on the shelf.'
            'Given your output in a json format as follows: \n'
            '{\n'
            '   "object_description": [description of the new object to be grouped],\n'
            '  "shelf_description": [description of items on each main visible layer and their'
            ' attributes, in detail, with the bottom layer being layer 0],\n'
            '  "reason": [reason for placing the object in the desired layer in one or two'
            ' sentences],\n'
            '  "layer": [integer number of the layer the new object should be placed on, with the'
            ' bottom layer being layer 0]\n'
            '}'
            '\nYour descriptions should be as detailed as possible.'
        )

        self.get_logger().info(f'API prompt: {sys_prompt}')

        try:
            shelf_res = request_shelf_layer_chain(
                sys_prompt, shelf_img_url, obj_seg_url,
                provider_models=self._categorize_provider_chain,
                qwen_api_backend=self.qwen_api_backend,
                timeout_s=self.vlm_timeout_s,
                max_retries=self.vlm_max_retries,
                logger=self.get_logger(),
            )
        except ShelfVlmError as exc:
            self.get_logger().error(f'Shelf-layer VLM call failed on every provider: {exc}')
            result.status = 4
            result.error_msg = f'VLM call failed on every provider: {exc}.'
            return result

        response = shelf_res.response
        self.get_logger().info(f'API response (provider={shelf_res.provider}): {response}')
        layer = response['layer']
        result.place_reason = str(response['reason'])

        self.get_logger().info(f'Layer to put on: {layer}.')
        self.get_logger().info(f"Description of items on each layer: {response['shelf_description']}.")

        # 5. Find empty space in that layer
        feedback_msg.status = 0
        feedback_msg.message = f'Determing the position to put in layer {layer}...'
        goal_handle.publish_feedback(feedback_msg)

        if not USE_SHELF_HEIGHT:
            layer_objects = clusters[int(layer)]
            layer_objects = sorted(layer_objects, key=lambda obj: obj.centroid.y)

            max_dis, pt = -1, None
            assert len(layer_objects) > 0, f'Layer has only {len(layer_objects)} objects'

            for i in range(len(layer_objects) - 1):
                dis = (
                    (layer_objects[i + 1].centroid.x - layer_objects[i].centroid.x) ** 2
                    + (layer_objects[i + 1].centroid.y - layer_objects[i].centroid.y) ** 2
                    + (layer_objects[i + 1].centroid.z - layer_objects[i].centroid.z) ** 2
                )
                if dis > max_dis:
                    max_dis = dis
                    pt = layer_objects[i].centroid
                    pt.x += (layer_objects[i + 1].centroid.x - layer_objects[i].centroid.x) / 2
                    pt.y += (layer_objects[i + 1].centroid.y - layer_objects[i].centroid.y) / 2
                    pt.z += (layer_objects[i + 1].centroid.z - layer_objects[i].centroid.z) / 2

            if max_dis < 0:
                pt = layer_objects[0].centroid
                if shelf_object is None or shelf_object.centroid.y > pt.y:
                    pt.y += 0.2
                else:
                    pt.y -= 0.2
            pt.z += 0.1
        else:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame='base_link',
                    source_frame=self.pt_shelf_left.header.frame_id,
                    time=rclpy.time.Time(),
                )
            except Exception:
                self.get_logger().warn(
                    f'Failed to lookup transform from {self.pt_shelf_left.header.frame_id} to base_link.'
                )
                result.status = 2
                result.error_msg = (
                    f'Failed to lookup transform from {self.pt_shelf_left.header.frame_id} to base_link.'
                )
                goal_handle.abort()
                return result

            self.pt_shelf_left = tf2_geometry_msgs.do_transform_point(self.pt_shelf_left, transform)
            self.pt_shelf_right = tf2_geometry_msgs.do_transform_point(self.pt_shelf_right, transform)

            layer_objects = shelf_objects[int(layer)]
            max_dis, pt = -1, None
            layer_centroids = [obj.centroid for obj in layer_objects]
            layer_centroids.append(self.pt_shelf_left.point)
            layer_centroids.append(self.pt_shelf_right.point)

            if PROJECT_ON_LINE:
                for i in range(len(layer_centroids)):
                    vec = np.array([
                        layer_centroids[i].x - self.pt_shelf_left.point.x,
                        layer_centroids[i].y - self.pt_shelf_left.point.y,
                    ])
                    vec2 = np.array([
                        self.pt_shelf_right.point.x - self.pt_shelf_left.point.x,
                        self.pt_shelf_right.point.y - self.pt_shelf_left.point.y,
                    ])
                    vec2 = vec2 / np.linalg.norm(vec2)
                    dot = np.dot(vec, vec2)
                    layer_centroids[i].x = self.pt_shelf_left.point.x + dot * vec2[0]
                    layer_centroids[i].y = self.pt_shelf_left.point.y + dot * vec2[1]

            layer_centroids = sorted(layer_centroids, key=lambda obj: obj.y)

            for i in range(len(layer_centroids) - 1):
                dis = (
                    (layer_centroids[i + 1].x - layer_centroids[i].x) ** 2
                    + (layer_centroids[i + 1].y - layer_centroids[i].y) ** 2
                )
                if dis > max_dis:
                    max_dis = dis
                    pt = layer_centroids[i]
                    pt.x += (layer_centroids[i + 1].x - layer_centroids[i].x) / 2
                    pt.y += (layer_centroids[i + 1].y - layer_centroids[i].y) / 2

            vec = np.array([pt.x - self.pt_shelf_left.point.x, pt.y - self.pt_shelf_left.point.y])
            vec2 = np.array([
                self.pt_shelf_right.point.x - self.pt_shelf_left.point.x,
                self.pt_shelf_right.point.y - self.pt_shelf_left.point.y,
            ])
            vec2 = vec2 / np.linalg.norm(vec2)
            dot = np.dot(vec, vec2)
            pt.x = self.pt_shelf_left.point.x + dot * vec2[0]
            pt.y = self.pt_shelf_left.point.y + dot * vec2[1]
            pt.z = self.shelf_height[int(layer)] + self.raise_height
            pt.x += -vec2[1] * self.push_in_distance
            pt.y += vec2[0] * self.push_in_distance

        header = detection_res.header
        point_stamped = geometry_msgs.msg.PointStamped(header=header, point=pt)

        feedback_msg.status = 0
        feedback_msg.message = 'Done!'
        goal_handle.publish_feedback(feedback_msg)

        result.status = 0
        result.place_point = point_stamped
        result.shelf_layer = layer
        result.error_msg = ''

        self.debug_pub.publish(result.place_point)
        self.shelf_l_pub.publish(self.pt_shelf_left)
        self.shelf_r_pub.publish(self.pt_shelf_right)

        goal_handle.succeed()

        with self.env_pc_lock:
            result.env_points = self.env_pc

        return result


def main():
    load_env()
    rclpy.init()
    action = GroceryCategorizeAction()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(action)
    executor.spin()

    action.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
