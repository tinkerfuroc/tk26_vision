import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# Camera frames arrive at 30 Hz only if the SHM FastDDS profile is set on the
# SUBSCRIBER too (this node subscribes the color topic for MJPEG). Set it here.
_FASTRTPS = '/home/tinker/tk25_ws/src/tk26_vision/config/fastdds_shm.xml'


def generate_launch_description():
    bind = LaunchConfiguration('bind')
    port = LaunchConfiguration('port')
    camera_topic = LaunchConfiguration('camera_topic')
    return LaunchDescription([
        DeclareLaunchArgument('bind', default_value='0.0.0.0'),
        DeclareLaunchArgument('port', default_value='8768'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/color/image_raw'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE',
                               os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE', _FASTRTPS)),
        Node(
            package='restaurant_nav_test_web',
            executable='restaurant_nav_test_web',
            output='screen',
            parameters=[{
                'bind': ParameterValue(bind, value_type=str),
                'port': ParameterValue(port, value_type=int),
                'camera_topic': ParameterValue(camera_topic, value_type=str),
            }],
        ),
    ])
