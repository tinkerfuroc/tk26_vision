"""Standalone camera-server launch.

The head server is intended for the vision driver bringup. The wrist server
is normally owned by the manipulation bringup that owns the RealSense. The
legacy bridge remains opt-in until consumer cutover.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


WRIST_PARAMS = {
    'color_topic': '/camera/xarm_camera/color/image_raw',
    'depth_topic': '/camera/xarm_camera/aligned_depth_to_color/image_raw',
    'color_info_topic': '/camera/xarm_camera/color/camera_info',
    'depth_info_topic': '/camera/xarm_camera/aligned_depth_to_color/camera_info',
}

HEAD_PARAMS = {
    'color_topic': '/camera/color/image_raw',
    'depth_topic': '/camera/depth/image_raw',
    'color_info_topic': '/camera/color/camera_info',
    # Registered Orbbec depth is on the color pixel grid and uses the color
    # intrinsics/frame, matching vision_util/get_orbbec_pc.py.
    'depth_info_topic': '/camera/color/camera_info',
}


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'enable_wrist', default_value='false',
            description='Launch the wrist RealSense camera server.',
        ),
        DeclareLaunchArgument(
            'enable_head', default_value='true',
            description='Launch the head Orbbec camera server.',
        ),
        DeclareLaunchArgument(
            'enable_legacy_services', default_value='false',
            description=(
                'Launch the legacy service bridge. Keep false while Python '
                'utility nodes own the legacy names.'
            ),
        ),
        Node(
            package='camera_server', executable='camera_server_node',
            name='wrist_camera_server', output='screen',
            parameters=[WRIST_PARAMS],
            condition=IfCondition(LaunchConfiguration('enable_wrist')),
        ),
        Node(
            package='camera_server', executable='camera_server_node',
            name='head_camera_server', output='screen',
            parameters=[HEAD_PARAMS],
            condition=IfCondition(LaunchConfiguration('enable_head')),
        ),
        Node(
            package='camera_server', executable='camera_compat_bridge',
            name='camera_compat_bridge', output='screen',
            condition=IfCondition(LaunchConfiguration('enable_legacy_services')),
        ),
    ])
