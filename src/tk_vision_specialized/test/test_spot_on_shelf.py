#!/usr/bin/env python3
"""Test cases for SpotOnShelf action server."""

import unittest
from geometry_msgs.msg import Point, PoseStamped
from tinker_vision_msgs.msg import Object


class TestSpotOnShelfUtils(unittest.TestCase):
    """Test utility functions for SpotOnShelf server."""

    def test_find_layer_bottom(self):
        """Test layer assignment for bottom layer."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        # Heights: [0.0, 0.5, 1.0, 1.5]
        # Layers: 0=[0.0, 0.5), 1=[0.5, 1.0), 2=[1.0, 1.5)
        shelf_heights = [0.0, 0.5, 1.0, 1.5]

        # Object at z=0.3 should be in layer 0
        layer = server._find_layer(0.3, shelf_heights)
        self.assertEqual(layer, 0)

        server.destroy_node()
        rclpy.shutdown()

    def test_find_layer_middle(self):
        """Test layer assignment for middle layer."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        shelf_heights = [0.0, 0.5, 1.0, 1.5]

        # Object at z=0.7 should be in layer 1
        layer = server._find_layer(0.7, shelf_heights)
        self.assertEqual(layer, 1)

        server.destroy_node()
        rclpy.shutdown()

    def test_find_layer_top(self):
        """Test layer assignment for top layer."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        shelf_heights = [0.0, 0.5, 1.0, 1.5]

        # Object at z=1.2 should be in layer 2
        layer = server._find_layer(1.2, shelf_heights)
        self.assertEqual(layer, 2)

        server.destroy_node()
        rclpy.shutdown()

    def test_find_layer_boundary(self):
        """Test layer assignment at boundaries."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        shelf_heights = [0.0, 0.5, 1.0, 1.5]

        # Object at exact boundary should be in lower layer
        layer = server._find_layer(0.5, shelf_heights)
        self.assertEqual(layer, 1)

        layer = server._find_layer(1.0, shelf_heights)
        self.assertEqual(layer, 2)

        server.destroy_node()
        rclpy.shutdown()

    def test_horizontal_position_left(self):
        """Test horizontal position assignment for left position."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        # Shelf from (0, 0) to (3, 0)
        shelf_left = PoseStamped()
        shelf_left.pose.position = Point(x=0.0, y=0.0, z=0.0)

        shelf_right = PoseStamped()
        shelf_right.pose.position = Point(x=3.0, y=0.0, z=0.0)

        # Object at (0.5, 0, 0) should be on the left (position 0)
        obj = Object()
        obj.centroid = Point(x=0.5, y=0.0, z=0.0)

        positions = server._assign_horizontal_positions(
            [obj], shelf_left, shelf_right
        )
        self.assertEqual(positions[0], 0)

        server.destroy_node()
        rclpy.shutdown()

    def test_horizontal_position_middle(self):
        """Test horizontal position assignment for middle position."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        # Shelf from (0, 0) to (3, 0)
        shelf_left = PoseStamped()
        shelf_left.pose.position = Point(x=0.0, y=0.0, z=0.0)

        shelf_right = PoseStamped()
        shelf_right.pose.position = Point(x=3.0, y=0.0, z=0.0)

        # Object at (1.5, 0, 0) should be in the middle (position 1)
        obj = Object()
        obj.centroid = Point(x=1.5, y=0.0, z=0.0)

        positions = server._assign_horizontal_positions(
            [obj], shelf_left, shelf_right
        )
        self.assertEqual(positions[0], 1)

        server.destroy_node()
        rclpy.shutdown()

    def test_horizontal_position_right(self):
        """Test horizontal position assignment for right position."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        # Shelf from (0, 0) to (3, 0)
        shelf_left = PoseStamped()
        shelf_left.pose.position = Point(x=0.0, y=0.0, z=0.0)

        shelf_right = PoseStamped()
        shelf_right.pose.position = Point(x=3.0, y=0.0, z=0.0)

        # Object at (2.5, 0, 0) should be on the right (position 2)
        obj = Object()
        obj.centroid = Point(x=2.5, y=0.0, z=0.0)

        positions = server._assign_horizontal_positions(
            [obj], shelf_left, shelf_right
        )
        self.assertEqual(positions[0], 2)

        server.destroy_node()
        rclpy.shutdown()

    def test_horizontal_position_angled_shelf(self):
        """Test horizontal position with angled shelf."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        # Shelf from (0, 0) to (3, 3) - 45 degree angle
        shelf_left = PoseStamped()
        shelf_left.pose.position = Point(x=0.0, y=0.0, z=0.0)

        shelf_right = PoseStamped()
        shelf_right.pose.position = Point(x=3.0, y=3.0, z=0.0)

        # Object near the middle
        obj = Object()
        obj.centroid = Point(x=1.5, y=1.5, z=0.0)

        positions = server._assign_horizontal_positions(
            [obj], shelf_left, shelf_right
        )
        self.assertEqual(positions[0], 1)

        server.destroy_node()
        rclpy.shutdown()

    def test_assign_multiple_objects(self):
        """Test assigning multiple objects to layers."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        # Create objects at different heights
        objects = []
        for z in [0.2, 0.6, 1.1]:
            obj = Object()
            obj.centroid = Point(x=0.0, y=0.0, z=z)
            objects.append(obj)

        shelf_heights = [0.0, 0.5, 1.0, 1.5]
        layers = server._assign_to_layers(objects, shelf_heights)

        self.assertEqual(layers, [0, 1, 2])

        server.destroy_node()
        rclpy.shutdown()

    def test_assign_multiple_horizontal_positions(self):
        """Test assigning multiple objects to horizontal positions."""
        from tk_vision_specialized.spot_on_shelf_server import (
            SpotOnShelfServer
        )
        import rclpy
        rclpy.init()

        server = SpotOnShelfServer()

        # Shelf from (0, 0) to (6, 0)
        shelf_left = PoseStamped()
        shelf_left.pose.position = Point(x=0.0, y=0.0, z=0.0)

        shelf_right = PoseStamped()
        shelf_right.pose.position = Point(x=6.0, y=0.0, z=0.0)

        # Create objects at different positions: left, middle, right
        objects = []
        for x in [1.0, 3.0, 5.0]:
            obj = Object()
            obj.centroid = Point(x=x, y=0.0, z=0.0)
            objects.append(obj)

        positions = server._assign_horizontal_positions(
            objects, shelf_left, shelf_right
        )

        self.assertEqual(positions, [0, 1, 2])

        server.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
