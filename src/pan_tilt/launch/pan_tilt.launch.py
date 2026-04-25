from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import FindExecutable
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration('config')
    device = LaunchConfiguration('device')
    launch_rsp = LaunchConfiguration('launch_robot_state_publisher')
    urdf_path = PathJoinSubstitution(
        [FindPackageShare('tinker_urdf'), 'src', 'pan_tilt_standalone.urdf.xacro'],
    )
    robot_description = Command([FindExecutable(name='xacro'), ' ', urdf_path])

    # pan_tilt publishes its URDF + joint-state feed on private topics so that
    # running this launch alongside the main robot bringup (grasp_bringup etc.)
    # does not collide with the xArm's /robot_description latched topic or
    # the shared /joint_states aggregator. /tf and /tf_static stay global so
    # downstream consumers see one merged TF tree.
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}],
        remappings=[
            ('/robot_description', '/pan_tilt/robot_description'),
            ('/joint_states', '/pan_tilt/joint_states'),
            ('robot_description', '/pan_tilt/robot_description'),
            ('joint_states', '/pan_tilt/joint_states'),
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'config',
                default_value=PathJoinSubstitution(
                    [FindPackageShare('pan_tilt'), 'config', 'pan_tilt.yaml'],
                ),
            ),
            DeclareLaunchArgument('device', default_value='/dev/ttyUSB0'),
            DeclareLaunchArgument(
                'launch_robot_state_publisher',
                default_value='true',
                description=(
                    'Set to false when another launch already owns '
                    '/robot_description (e.g. mobile_bringup grasp_bringup, '
                    'which publishes the merged mobile_manipulator URDF).'
                ),
            ),
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
            rsp_node,
        ],
    )
