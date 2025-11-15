#!/usr/bin/env python3
"""
Example client for testing object detection with sorting and visualization.

This demonstrates how to call the object detection service with different
sorting modes and request segmentation masks.
"""

import rclpy
from rclpy.node import Node
from tinker_vision_msgs.srv import ObjectDetection


class DetectionTestClient(Node):
    def __init__(self):
        super().__init__('detection_test_client')
        self.client = self.create_client(ObjectDetection, 'object_detection_yolo')
        
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')
    
    def send_request(self, camera='orbbec', target_class='bottle', 
                     sort_mode=None, request_segments=False):
        """
        Send detection request.
        
        Parameters
        ----------
        camera : str
            'realsense' or 'orbbec'
        target_class : str
            Object class to detect (e.g., 'bottle', 'cup', 'person')
        sort_mode : str, optional
            'closest', 'highest', or None for default
        request_segments : bool
            Whether to request segmentation masks
        """
        request = ObjectDetection.Request()
        request.camera = camera
        request.prompt = target_class
        
        # Build flags
        flags = []
        if sort_mode == 'closest':
            flags.append('sort_closest')
        elif sort_mode == 'highest':
            flags.append('sort_highest')
        elif sort_mode == 'none':
            flags.append('sort_none')
        
        if request_segments:
            flags.append('request_segments')
        
        request.flags = ' '.join(flags)
        
        self.get_logger().info(f'Sending request: camera={camera}, '
                              f'class={target_class}, flags="{request.flags}"')
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        
        if response.status == 0:
            self.get_logger().info(f'Detected {len(response.objects)} objects:')
            for i, obj in enumerate(response.objects):
                self.get_logger().info(
                    f'  #{i+1}: {obj.cls} (conf={obj.conf:.2f}, '
                    f'x={obj.centroid.x:.2f}m, y={obj.centroid.y:.2f}m, '
                    f'z={obj.centroid.z:.2f}m)'
                )
            
            if request_segments:
                self.get_logger().info(f'Received {len(response.segments)} segments')
                # Verify alignment
                if len(response.segments) == len(response.objects):
                    self.get_logger().info('✓ Objects and segments are aligned')
                else:
                    self.get_logger().warn('✗ Misalignment detected!')
        else:
            self.get_logger().warn('Detection failed or no objects found')
        
        return response


def main(args=None):
    rclpy.init(args=args)
    client = DetectionTestClient()
    
    # Example 1: Find closest bottle
    print("\n=== Example 1: Find closest bottle ===")
    client.send_request(
        camera='orbbec',
        target_class='bottle',
        sort_mode='closest',
        request_segments=True
    )
    
    # Example 2: Find highest cup
    print("\n=== Example 2: Find highest cup ===")
    client.send_request(
        camera='orbbec',
        target_class='cup',
        sort_mode='highest',
        request_segments=True
    )
    
    # Example 3: Find all persons without sorting
    print("\n=== Example 3: Find all persons (no sorting) ===")
    client.send_request(
        camera='orbbec',
        target_class='person',
        sort_mode='none',
        request_segments=False
    )
    
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
