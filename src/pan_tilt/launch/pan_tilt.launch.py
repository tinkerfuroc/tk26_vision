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
    # wrapper, which renders pan_tilt_standalone.urdf.xacro. The deprecated
    # overrides_key='pan_tilt.urdf_overrides' mapping (URDF MOUNT GEOMETRY —
    # attach_xyz/attach_rpy/camera_mount_xyz/camera_mount_rpy — never the
    # joint offsets) is dropped: since tk25_basic db1524a, pan_tilt.urdf.xacro
    # sources per-robot mount geometry itself via a ROBOT_NAME-guarded
    # <xacro:include> of robots/$ROBOT_NAME/pan_tilt/pan_tilt_overrides.xacro
    # at xacro-parse time, independent of launch args — per-robot geometry is
    # preserved whenever ROBOT_NAME is set. The joint offsets are a separate
    # concern: always ROS params, now read at runtime from the per-robot
    # profile by pan_tilt_state_publisher._load_per_robot_offsets (with the
    # package-yaml params as warned fallback). Private topics are preserved
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
