"""LLM-backed grocery categorization onto a multi-layer shelf.

Ports tk23 `kimi_api/grocery_categorize.py`. Calls `object_detection` to spot
shelf items, clusters them by height (scipy kmeans), asks an OpenRouter vision
model which shelf layer a new item belongs on, then picks an empty spot in
that layer and returns a PointStamped.

Changes from tk23:
- API key/base URL/model from environment.
- `detection_service` ROS param (default `object_detection`) for retargeting.
- Temporary JPEG encoded via `tempfile.NamedTemporaryFile`.
- Dead Chinese-prompt block removed.
"""

import base64
import json
import os
import tempfile
import threading
import time

import cv2
import geometry_msgs.msg
import numpy as np
import rclpy
import tf2_geometry_msgs  # noqa: F401  (registers PointStamped transform)
from cv_bridge import CvBridge
from openai import OpenAI
from rclpy.action import ActionServer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from scipy.cluster.vq import kmeans2
from sensor_msgs.msg import PointCloud2
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tinker_vision_msgs.action import Categorize
from tinker_vision_msgs_26.srv import ObjectDetection

from ._env import base_url, default_model, load_env, require_api_key

USE_SHELF_HEIGHT = False
PROJECT_ON_LINE = False


def get_bounding_box(mask):
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


class GroceryCategorizeAction(Node):
    def __init__(self):
        super().__init__('grocery_categorize')

        self.shelf_height = [0.5, 1.0, 1.5]  # bottom to top
        self.raise_height = 0.25
        self.push_in_distance = 0.05

        self.declare_parameter('llm_model', default_model())
        self.declare_parameter('detection_service', 'object_detection_generalist')
        self.llm_model = self.get_parameter('llm_model').get_parameter_value().string_value
        detection_service = self.get_parameter('detection_service').get_parameter_value().string_value

        self.server_cb_group = MutuallyExclusiveCallbackGroup()
        self.client_cb_group = MutuallyExclusiveCallbackGroup()

        self.action_server = ActionServer(
            self,
            Categorize,
            'grocery_categorize',
            self.grocery_categorize_callback,
            callback_group=self.server_cb_group,
        )

        self.client = OpenAI(api_key=require_api_key(), base_url=base_url())

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

        obj_seg_url = _encode_to_data_url(obj_segment)
        shelf_img_url = _encode_to_data_url(rgb_image)

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
            ' sentences]\n'
            '  "layer": [integer number of the layer the new object should be placed on, with the'
            ' bottom layer being layer 0]\n'
            '}'
            '\nYour descriptions should be as detailed as possible.'
        )

        self.get_logger().info(f'API prompt: {sys_prompt}')

        completion = None
        try:
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {'role': 'system', 'content': sys_prompt},
                    {
                        'role': 'user',
                        'content': [
                            {'type': 'text', 'text': 'picture of shelf'},
                            {'type': 'image_url', 'image_url': {'url': shelf_img_url}},
                            {'type': 'text', 'text': 'picture of new object.'},
                            {'type': 'image_url', 'image_url': {'url': obj_seg_url}},
                        ],
                    },
                ],
            )
        except Exception as e:
            self.get_logger().warn(f'API call failed: {e}')

        response = None
        try:
            response = json.loads(completion.choices[0].message.content)
        except Exception as e:
            self.get_logger().error(f'Failed to parse response: {e}.')
            self.get_logger().info(f'API response: {completion}.')
            result.status = 4
            result.error_msg = f'Failed to parse response: {completion}.'
            return result

        if 'layer' not in response:
            result.status = 4
            result.error_msg = "Response missing 'layer' field."
            return result
        if 'shelf_description' not in response:
            result.status = 4
            result.error_msg = "Response missing 'shelf_description' field."
            return result
        self.get_logger().info(f'API response: {response}')
        layer = response['layer']
        result.place_reason = response['reason']

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
