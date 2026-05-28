from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
import os


def generate_launch_description():
    rs_launch = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py',
    )

    camera_namespace = LaunchConfiguration('camera_namespace')
    camera_name = LaunchConfiguration('camera_name')
    depth_min_m = LaunchConfiguration('depth_min_m')
    depth_max_m = LaunchConfiguration('depth_max_m')

    depth_topic = PythonExpression([
        "'/' + '", camera_namespace, "' + '/' + '", camera_name,
        "' + '/depth/image_rect_raw'",
    ])

    return LaunchDescription([
        DeclareLaunchArgument('camera_namespace', default_value='camera'),
        DeclareLaunchArgument('camera_name', default_value='camera'),
        DeclareLaunchArgument('depth_min_m', default_value='0.3',
                              description='distance rendered as yellow (m)'),
        DeclareLaunchArgument('depth_max_m', default_value='3.0',
                              description='distance rendered as red (m)'),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(rs_launch),
            launch_arguments={
                'camera_namespace': camera_namespace,
                'camera_name': camera_name,
            }.items(),
        ),

        Node(
            package='vision_util',
            executable='depth_colorizer',
            name='depth_colorizer',
            output='screen',
            parameters=[{
                'input_topic': depth_topic,
                'output_topic': 'depth_colorized',
                'depth_min_m': depth_min_m,
                'depth_max_m': depth_max_m,
            }],
        ),
    ])
