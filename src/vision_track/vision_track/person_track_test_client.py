#!/usr/bin/env python3
"""
Person Tracking Action Client with Visualization

This script is a test client for the TrackPerson action server.
It sends a tracking goal and visualizes the results including:
- RGB image with bounding box overlay
- Tracked person's 3D position
- Segmentation mask (optional)

Author: TinkerFuroc
Date: 2025
"""

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle, GoalStatus

import cv2
import numpy as np
import threading
import time
from collections import deque

# ROS2 messages
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped

# Action definition
from tinker_vision_msgs_26.action import TrackPerson

# CV Bridge
from cv_bridge import CvBridge


class PersonTrackTestClient(Node):
    """
    Test client for the TrackPerson action server with visualization.
    """

    def __init__(self):
        super().__init__('person_track_test_client')
        
        # Action client
        self.action_client = ActionClient(self, TrackPerson, 'track_person')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Visualization state
        self.latest_rgb = None
        self.latest_segment = None
        self.latest_position: PointStamped = None
        self.target_track_id = -1
        self.target_lost = False
        self.lock = threading.Lock()
        
        # Position history for trajectory visualization
        self.position_history = deque(maxlen=100)
        
        # Visualization window
        self.window_name = 'Person Tracking Test'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        # Goal handle
        self.goal_handle: ClientGoalHandle = None
        self.tracking_active = False
        
        # Stats
        self.feedback_count = 0
        self.last_feedback_time = time.time()
        self.fps = 0.0
        
        # Debug mode flag
        self.debug_mode = False
        
        self.get_logger().info('Person Track Test Client initialized')

    def send_goal(self, return_rgb=True, return_depth=False, return_segment=True, debug=False):
        """Send a tracking goal to the action server."""
        self.get_logger().info('Waiting for action server...')
        
        if not self.action_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error('Action server not available!')
            return False
        
        self.get_logger().info('Action server available, sending goal...')
        
        # Create goal
        goal = TrackPerson.Goal()
        goal.return_rgb_img = return_rgb
        goal.return_depth_img = return_depth
        goal.return_segment = return_segment
        goal.debug = debug  # Enable debug visualization
        
        # Send goal
        self.tracking_active = True
        send_goal_future = self.action_client.send_goal_async(
            goal,
            feedback_callback=self._feedback_callback
        )
        send_goal_future.add_done_callback(self._goal_response_callback)
        
        return True

    def _goal_response_callback(self, future):
        """Handle goal response."""
        self.goal_handle = future.result()
        
        if not self.goal_handle.accepted:
            self.get_logger().warn('Goal rejected by server')
            self.tracking_active = False
            return
        
        self.get_logger().info('Goal accepted!')
        
        # Get result asynchronously
        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._result_callback)

    def _result_callback(self, future):
        """Handle tracking result."""
        result = future.result()
        status = result.status
        
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f'Tracking completed: {result.result.message}')
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().warn(f'Tracking aborted: {result.result.message}')
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info(f'Tracking canceled: {result.result.message}')
        else:
            self.get_logger().info(f'Tracking ended with status: {status}')
        
        self.tracking_active = False

    def _feedback_callback(self, feedback_msg):
        """Handle tracking feedback."""
        feedback = feedback_msg.feedback
        
        # Update FPS calculation
        self.feedback_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_feedback_time
        if elapsed >= 1.0:
            self.fps = self.feedback_count / elapsed
            self.feedback_count = 0
            self.last_feedback_time = current_time
        
        with self.lock:
            self.target_lost = feedback.target_lost
            self.target_track_id = feedback.target_track_id
            
            # Update position
            if feedback.target_position is not None:
                self.latest_position = feedback.target_position
                
                # Add to history if not lost and position is valid
                if not feedback.target_lost:
                    point = feedback.target_position.point
                    if point.z > 0:  # Valid depth
                        self.position_history.append((point.x, point.y, point.z))
            
            # Update RGB image
            if feedback.rgb_img is not None and feedback.rgb_img.data:
                try:
                    self.latest_rgb = self.bridge.imgmsg_to_cv2(feedback.rgb_img, "bgr8")
                except Exception as e:
                    self.get_logger().warn(f'Failed to convert RGB: {e}')
            
            # Update segmentation mask
            if feedback.segment_img is not None and feedback.segment_img.data:
                try:
                    self.latest_segment = self.bridge.imgmsg_to_cv2(
                        feedback.segment_img, "mono8"
                    )
                except Exception as e:
                    self.get_logger().warn(f'Failed to convert segment: {e}')

    def cancel_tracking(self):
        """Cancel the current tracking goal."""
        if self.goal_handle is not None and self.tracking_active:
            self.get_logger().info('Canceling tracking...')
            cancel_future = self.goal_handle.cancel_goal_async()
            cancel_future.add_done_callback(self._cancel_callback)

    def _cancel_callback(self, future):
        """Handle cancel response."""
        cancel_response = future.result()
        if len(cancel_response.goals_canceling) > 0:
            self.get_logger().info('Goal successfully canceled')
        else:
            self.get_logger().warn('Goal cancel failed')

    def visualize(self):
        """Create and display visualization."""
        with self.lock:
            rgb = self.latest_rgb
            segment = self.latest_segment
            position = self.latest_position
            target_lost = self.target_lost
            track_id = self.target_track_id
            history = list(self.position_history)
        
        if rgb is None:
            self._show_placeholder()
            return
        
        vis = self._apply_segment_mask(rgb, segment)
        h, w = vis.shape[:2]

        self._draw_info_panel(vis, target_lost, track_id, position)
        self._draw_position_view(vis, position, history, w, h)
        self._draw_crosshair(vis, w, h)

        cv2.imshow(self.window_name, vis)

    def _show_placeholder(self):
        placeholder = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(
            placeholder,
            "Waiting for tracking data...",
            (400, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        cv2.imshow(self.window_name, placeholder)

    def _apply_segment_mask(self, rgb, segment):
        vis = rgb.copy()
        if segment is not None and segment.shape[:2] == rgb.shape[:2]:
            mask_overlay = np.zeros_like(vis)
            mask_overlay[:, :, 1] = segment
            vis = cv2.addWeighted(vis, 0.7, mask_overlay, 0.3, 0)
        return vis

    def _draw_info_panel(self, vis, target_lost, track_id, position):
        status_color = (0, 0, 255) if target_lost else (0, 255, 0)
        status_text = "TARGET LOST" if target_lost else "TRACKING"
        cv2.rectangle(vis, (10, 10), (400, 180), (0, 0, 0), -1)
        cv2.rectangle(vis, (10, 10), (400, 180), (255, 255, 255), 2)
        cv2.putText(vis, f"Status: {status_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(vis, f"Track ID: {track_id}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis, f"FPS: {self.fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if position is not None:
            point = position.point
            pos_text = f"Position: ({point.x:.2f}, {point.y:.2f}, {point.z:.2f})"
            cv2.putText(vis, pos_text, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            distance = np.sqrt(point.x**2 + point.y**2 + point.z**2)
            cv2.putText(vis, f"Distance: {distance:.2f}m", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def _draw_crosshair(self, vis, w, h):
        center_x, center_y = w // 2, h // 2
        cv2.line(vis, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 255), 1)
        cv2.line(vis, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 255), 1)

    def _draw_position_view(self, vis, position, history, img_w, img_h):
        """Draw a bird's eye view of the target position."""
        # Position view dimensions and location
        view_w, view_h = 200, 200
        view_x, view_y = img_w - view_w - 20, img_h - view_h - 20
        
        # Draw background
        cv2.rectangle(vis, (view_x, view_y), (view_x + view_w, view_y + view_h), (50, 50, 50), -1)
        cv2.rectangle(vis, (view_x, view_y), (view_x + view_w, view_y + view_h), (255, 255, 255), 2)
        
        # Draw title
        cv2.putText(
            vis,
            "Bird's Eye View",
            (view_x + 40, view_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Draw grid
        grid_center_x = view_x + view_w // 2
        grid_center_y = view_y + view_h - 20  # Camera at bottom
        
        # Draw axis
        cv2.line(vis, (grid_center_x, view_y + 30), (grid_center_x, view_y + view_h - 10), (100, 100, 100), 1)
        cv2.line(vis, (view_x + 10, grid_center_y), (view_x + view_w - 10, grid_center_y), (100, 100, 100), 1)
        
        # Draw camera position
        cv2.circle(vis, (grid_center_x, grid_center_y), 5, (0, 255, 0), -1)
        
        # Scale: 1 meter = 20 pixels
        scale = 20.0
        
        # Draw trajectory history
        if len(history) > 1:
            for i in range(1, len(history)):
                x1, y1, z1 = history[i - 1]
                x2, y2, z2 = history[i]
                
                # Convert to view coordinates (x -> horizontal, z -> vertical/forward)
                px1 = int(grid_center_x + x1 * scale)
                py1 = int(grid_center_y - z1 * scale)
                px2 = int(grid_center_x + x2 * scale)
                py2 = int(grid_center_y - z2 * scale)
                
                # Clip to view bounds
                px1 = np.clip(px1, view_x + 5, view_x + view_w - 5)
                py1 = np.clip(py1, view_y + 25, view_y + view_h - 5)
                px2 = np.clip(px2, view_x + 5, view_x + view_w - 5)
                py2 = np.clip(py2, view_y + 25, view_y + view_h - 5)
                
                # Draw trajectory line (fading color based on age)
                alpha = int(255 * (i / len(history)))
                color = (alpha // 2, alpha, alpha)
                cv2.line(vis, (px1, py1), (px2, py2), color, 1)
        
        # Draw current position
        if position is not None:
            point = position.point
            if point.z > 0:
                px = int(grid_center_x + point.x * scale)
                py = int(grid_center_y - point.z * scale)
                
                # Clip to view bounds
                px = np.clip(px, view_x + 5, view_x + view_w - 5)
                py = np.clip(py, view_y + 25, view_y + view_h - 5)
                
                # Draw target position
                cv2.circle(vis, (px, py), 8, (0, 0, 255), -1)
                cv2.circle(vis, (px, py), 8, (255, 255, 255), 2)
        
        # Draw scale reference
        cv2.putText(
            vis,
            "1m",
            (view_x + view_w - 35, view_y + view_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )
        cv2.line(
            vis,
            (view_x + view_w - 40, view_y + view_h - 15),
            (view_x + view_w - 40 + int(scale), view_y + view_h - 15),
            (255, 255, 255),
            2
        )

    def run(self):
        """Main run loop with visualization."""
        # Send goal
        if not self.send_goal(return_rgb=True, return_depth=False, return_segment=True, debug=self.debug_mode):
            return
        
        self.get_logger().info('Press "q" to quit, "c" to cancel tracking, "d" to toggle debug mode')
        
        try:
            while rclpy.ok():
                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.01)
                
                # Update visualization
                self.visualize()
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.get_logger().info('Quit requested')
                    self.cancel_tracking()
                    break
                elif key == ord('c'):
                    self.cancel_tracking()
                elif key == ord('r'):
                    # Restart tracking
                    if not self.tracking_active:
                        self.position_history.clear()
                        self.send_goal(return_rgb=True, return_depth=False, return_segment=True, debug=self.debug_mode)
                elif key == ord('d'):
                    # Toggle debug mode
                    self.debug_mode = not self.debug_mode
                    self.get_logger().info(f'Debug mode: {self.debug_mode}')
                
                # Exit if tracking ended
                if not self.tracking_active and self.goal_handle is not None:
                    # Wait a bit to show final state
                    time.sleep(2.0)
                    break
        
        except KeyboardInterrupt:
            self.get_logger().info('Interrupted')
            self.cancel_tracking()
        
        finally:
            cv2.destroyAllWindows()


def main(args=None):
    import argparse
    parser = argparse.ArgumentParser(description='Person Tracking Test Client')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug visualization')
    parsed_args, remaining = parser.parse_known_args()
    
    rclpy.init(args=remaining)
    
    client = PersonTrackTestClient()
    client.debug_mode = parsed_args.debug
    
    try:
        client.run()
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
