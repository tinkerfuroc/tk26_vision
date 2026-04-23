from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import FindExecutable
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration('config')
    device = LaunchConfiguration('device')
    urdf_path = PathJoinSubstitution(
        [FindPackageShare('pan_tilt'), 'urdf', 'pan_tilt.urdf.xacro'],
    )
    robot_description = Command([FindExecutable(name='xacro'), ' ', urdf_path])

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'config',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('pan_tilt'), 'config', 'pan_tilt.yaml'],
                ),
            ),
            DeclareLaunchArgument('device', default_value='/dev/ttyUSB0'),
            Node(
                package='pan_tilt',
                executable='controller',
                output='screen',
                parameters=[config, {'device': device}],
            ),
            Node(
                package='pan_tilt',
                executable='state_publisher',
                output='screen',
                parameters=[config],
            ),
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                output='screen',
                parameters=[{'robot_description': robot_description}],
            ),
        ],
    )
