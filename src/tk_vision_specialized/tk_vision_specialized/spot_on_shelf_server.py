#!/usr/bin/env python3
"""
SpotOnShelf Action Server.

This node provides an action server that detects objects on a shelf
and categorizes them by vertical layer and horizontal position.
"""

import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
import numpy as np
import tf2_ros
import tf2_geometry_msgs

from tinker_vision_msgs_26.action import SpotOnShelf
from tinker_vision_msgs.srv import ObjectDetection


class SpotOnShelfServer(Node):
    """Action server for spotting objects on shelf and categorizing position."""

    def __init__(self):
        super().__init__('spot_on_shelf_server')

        # Parameters
        self.declare_parameter('detection_service', 'object_detection_yolo')
        self.declare_parameter('target_frame', 'base_link')
        self.declare_parameter('default_camera', 'orbbec')

        self.detection_service_name = self.get_parameter(
            'detection_service').value
        self.default_target_frame = self.get_parameter('target_frame').value
        self.default_camera = self.get_parameter('default_camera').value

        # Create service client for object detection with reentrant callback group
        self.callback_group = ReentrantCallbackGroup()
        self.detection_cli = self.create_client(
            ObjectDetection,
            self.detection_service_name,
            callback_group=self.callback_group
        )

        # TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Create action server with reentrant callback group
        self._action_server = ActionServer(
            self,
            SpotOnShelf,
            'spot_on_shelf',
            self.execute_callback,
            callback_group=self.callback_group
        )

        self.get_logger().info('SpotOnShelf action server initialized')

    async def execute_callback(self, goal_handle):
        """Execute the SpotOnShelf action."""
        self.get_logger().info('Executing SpotOnShelf action...')

        request = goal_handle.request
        result = SpotOnShelf.Result()
        feedback = SpotOnShelf.Feedback()

        # Validate request
        if len(request.shelf_heights) == 0:
            result.status = -1
            result.error_msg = 'shelf_heights cannot be empty'
            goal_handle.abort()
            return result

        if len(request.item_ids) == 0:
            result.status = -1
            result.error_msg = 'item_ids cannot be empty'
            goal_handle.abort()
            return result

        # Transform shelf endpoints to target frame
        try:
            shelf_left, shelf_right = await self._transform_shelf_endpoints(
                request.shelf_left,
                request.shelf_right
            )
        except Exception as e:
            result.status = -2
            result.error_msg = f'Failed to transform shelf endpoints: {e}'
            self.get_logger().error(result.error_msg)
            goal_handle.abort()
            return result

        # Detect all requested objects
        feedback.status = 0
        feedback.message = 'Detecting objects...'
        goal_handle.publish_feedback(feedback)

        # Determine which camera to use (can be specified in request or use default)
        camera = self.default_camera

        all_objects = []
        for item_id in request.item_ids:
            objects = await self._detect_objects(item_id, camera=camera)
            if objects:
                all_objects.extend(objects)
                self.get_logger().info(
                    f'Found {len(objects)} instance(s) of "{item_id}" using {camera}')
            else:
                self.get_logger().warn(f'No objects found for "{item_id}"')

        if len(all_objects) == 0:
            result.status = 1
            result.error_msg = 'No objects detected'
            goal_handle.succeed()
            return result

        # Categorize objects by vertical layer
        feedback.status = 1
        feedback.message = 'Categorizing by vertical layers...'
        goal_handle.publish_feedback(feedback)

        layer_assignments = self._assign_to_layers(
            all_objects,
            request.shelf_heights
        )

        # Categorize objects by horizontal position
        feedback.status = 2
        feedback.message = 'Categorizing by horizontal position...'
        goal_handle.publish_feedback(feedback)

        horizontal_assignments = self._assign_horizontal_positions(
            all_objects,
            shelf_left,
            shelf_right
        )

        # Build result
        result.status = 0
        result.error_msg = ''
        result.item_height_grids = layer_assignments
        result.item_horizontal_grids = horizontal_assignments

        feedback.status = 3
        feedback.message = 'Done!'
        goal_handle.publish_feedback(feedback)

        self.get_logger().info(
            f'Categorized {len(all_objects)} objects: '
            f'layers={layer_assignments}, horizontal={horizontal_assignments}'
        )

        goal_handle.succeed()
        return result

    async def _transform_shelf_endpoints(self, shelf_left, shelf_right):
        """Transform shelf endpoint poses to base_link frame."""
        target_frame = 'base_link'

        # Wait for transform to be available
        await self._wait_for_transform(
            target_frame,
            shelf_left.header.frame_id
        )

        # Get transform
        transform = self.tf_buffer.lookup_transform(
            target_frame,
            shelf_left.header.frame_id,
            rclpy.time.Time()
        )

        # Transform both points
        left_point_stamped = tf2_geometry_msgs.do_transform_pose(
            shelf_left, transform)
        right_point_stamped = tf2_geometry_msgs.do_transform_pose(
            shelf_right, transform)

        return left_point_stamped, right_point_stamped

    async def _wait_for_transform(
            self, target_frame, source_frame,
            timeout_sec=5.0):
        """Wait for transform to become available."""
        rate = self.create_rate(10)
        start_time = self.get_clock().now()

        while rclpy.ok():
            if self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                rclpy.time.Time()
            ):
                return True

            elapsed = (self.get_clock().now() - start_time).nanoseconds / 1e9
            if elapsed > timeout_sec:
                raise RuntimeError(
                    f'Transform from {source_frame} to {target_frame} '
                    f'not available after {timeout_sec}s'
                )

            await rate.sleep()

        return False

    async def _detect_objects(self, prompt, camera=None):
        """Call object detection service for a specific item."""
        # Use default camera if not specified
        if camera is None:
            camera = self.default_camera

        # Wait for service
        if not self.detection_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                f'Detection service {self.detection_service_name} '
                'not available'
            )
            return []

        # Create request
        request = ObjectDetection.Request()
        request.flags = []
        request.prompt = prompt
        request.camera = camera
        request.target_frame = self.default_target_frame

        # Call service
        try:
            future = self.detection_cli.call_async(request)
            await future
            response = future.result()

            if response.status != 0:
                self.get_logger().warn(
                    f'Detection failed for "{prompt}" on {camera}: '
                    f'status={response.status}'
                )
                return []

            return response.objects

        except Exception as e:
            self.get_logger().error(
                f'Exception calling detection service: {e}'
            )
            return []

    def _assign_to_layers(self, objects, shelf_heights):
        """
        Assign objects to vertical layers based on their z-coordinates.

        Parameters
        ----------
        objects : list
            List of detected Object messages
        shelf_heights : list of float
            Heights of shelf layers (boundaries)

        Returns
        -------
        list of int
            Layer index for each object

        """
        layer_assignments = []

        for obj in objects:
            z = obj.centroid.z
            layer = self._find_layer(z, shelf_heights)
            layer_assignments.append(layer)

        return layer_assignments

    def _find_layer(self, z, shelf_heights):
        """
        Find which layer a z-coordinate belongs to.

        Parameters
        ----------
        z : float
            Z-coordinate (height) of object
        shelf_heights : list of float
            Sorted list of layer boundaries

        Returns
        -------
        int
            Layer index (0-based from bottom)

        """
        # Sort heights to ensure correct ordering
        heights = sorted(shelf_heights)

        # Find the layer
        for i in range(len(heights) - 1):
            if heights[i] <= z < heights[i + 1]:
                return i

        # If above all boundaries, assign to top layer
        if z >= heights[-1]:
            return len(heights) - 2

        # If below all boundaries, assign to bottom layer
        return 0

    def _assign_horizontal_positions(self, objects, shelf_left, shelf_right):
        """
        Assign objects to horizontal positions (left=0, middle=1, right=2).

        Projects object centroids onto the line between shelf endpoints
        and divides into thirds.

        Parameters
        ----------
        objects : list
            List of detected Object messages
        shelf_left : PoseStamped
            Left endpoint of shelf
        shelf_right : PoseStamped
            Right endpoint of shelf

        Returns
        -------
        list of int
            Horizontal position for each object (0=left, 1=middle, 2=right)

        """
        horizontal_assignments = []

        # Get shelf line vector
        left_pt = shelf_left.pose.position
        right_pt = shelf_right.pose.position

        shelf_vec = np.array([
            right_pt.x - left_pt.x,
            right_pt.y - left_pt.y
        ])
        shelf_length = np.linalg.norm(shelf_vec)
        shelf_vec_normalized = shelf_vec / shelf_length

        for obj in objects:
            # Project object centroid onto shelf line
            obj_vec = np.array([
                obj.centroid.x - left_pt.x,
                obj.centroid.y - left_pt.y
            ])

            # Distance along shelf line
            distance_along = np.dot(obj_vec, shelf_vec_normalized)

            # Normalize to [0, 1]
            normalized_pos = distance_along / shelf_length

            # Assign to horizontal position
            if normalized_pos < 1.0 / 3.0:
                position = 0  # Left
            elif normalized_pos < 2.0 / 3.0:
                position = 1  # Middle
            else:
                position = 2  # Right

            horizontal_assignments.append(position)

        return horizontal_assignments


def main(args=None):
    """Run the SpotOnShelf action server."""
    rclpy.init(args=args)

    server = SpotOnShelfServer()

    # Use MultiThreadedExecutor for concurrent callback processing
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
