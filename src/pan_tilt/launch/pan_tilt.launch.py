from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = LaunchConfiguration('config')
    device = LaunchConfiguration('device')
    launch_rsp = LaunchConfiguration('launch_robot_state_publisher')

    # Publish the URDF via tinker_robot_config's robot_description.launch.py
    # wrapper, which renders pan_tilt_standalone.urdf.xacro. The pan/tilt
    # joint offsets that used to be threaded through here as a
    # 'pan_tilt.urdf_overrides' mapping are now read directly by
    # pan_tilt_state_publisher from the per-robot profile (see
    # pan_tilt_state_publisher._load_per_robot_offsets) — that deprecated
    # overrides_key plumbing has been dropped. Private topics are preserved
    # (belt-and-suspenders) so this launch can run alongside grasp_bringup's
    # xArm RSP without colliding with /robot_description or /joint_states;
    # the operational disable is still launch_robot_state_publisher:=false.
    rsp_include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('tinker_robot_config'),
            '/launch/robot_description.launch.py',
        ]),
        launch_arguments={
            'xacro_path': PathJoinSubstitution([
                FindPackageShare('tinker_urdf'), 'src',
                'pan_tilt_standalone.urdf.xacro',
            ]),
            'remappings': (
                '/robot_description=/pan_tilt/robot_description;'
                '/joint_states=/pan_tilt/joint_states;'
                'robot_description=/pan_tilt/robot_description;'
                'joint_states=/pan_tilt/joint_states'
            ),
        }.items(),
        condition=IfCondition(launch_rsp),
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
            rsp_include,
        ],
    )
