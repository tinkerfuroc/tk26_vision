"""Vision driver bringup — the hardware / driver layer of the tk26 vision stack.

Brings up, in one command:
  - the pan-tilt head (controller + state_publisher + URDF RSP),
  - the Orbbec Femto Bolt (the tk26-tuned overrides),
  - the RealSense (aligned depth + the tk26 QoS overrides), and
  - FoundationStereo with its streaming depth publisher ENABLED.

RealSense is included because FoundationStereo's streaming consumes the
RealSense IR pair — without it, ``stream_enabled`` produces nothing.

Sets ``FASTRTPS_DEFAULT_PROFILES_FILE`` to the FastDDS SHM profile for the whole
launch so the cameras (and any subscriber that inherits the env) negotiate
shared-memory transport and hit ~30 Hz instead of the ~3 Hz the vendored
launches drop to over UDP. See src/tk26_vision/CAMERA_BRINGUP.md for the
full root-cause writeup.

Usage::

    ros2 launch vision_bringup vision_driver.launch.py
    ros2 launch vision_bringup vision_driver.launch.py enable_realsense:=false
    # alongside grasp_bringup (which already owns /robot_description):
    ros2 launch vision_bringup vision_driver.launch.py launch_robot_state_publisher:=false

Each subsystem is gated by an ``enable_*`` argument (all default ``true``).
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('vision_bringup')
    fastdds_profile = PathJoinSubstitution([pkg_share, 'config', 'fastdds_shm.xml'])
    realsense_qos = PathJoinSubstitution([pkg_share, 'config', 'realsense_qos.yaml'])

    args = [
        DeclareLaunchArgument('enable_pan_tilt', default_value='true'),
        DeclareLaunchArgument('enable_orbbec', default_value='true'),
        DeclareLaunchArgument('enable_realsense', default_value='true'),
        DeclareLaunchArgument('enable_ffs', default_value='true'),
        DeclareLaunchArgument(
            'device', default_value='/dev/ttyUSB0',
            description='pan-tilt servo serial device',
        ),
        DeclareLaunchArgument(
            'launch_robot_state_publisher', default_value='true',
            description=(
                'Set false when another launch already owns /robot_description '
                '(e.g. mobile_bringup grasp_bringup).'
            ),
        ),
        DeclareLaunchArgument(
            'camera_profile', default_value='d435',
            description='FoundationStereo camera profile.',
        ),
    ]

    # Whole-launch env: every child process (cameras + FFS) inherits the SHM
    # profile. Downstream subscribers must set the same env to negotiate SHM.
    set_dds = SetEnvironmentVariable(
        'FASTRTPS_DEFAULT_PROFILES_FILE', fastdds_profile,
    )

    pan_tilt = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('pan_tilt'), '/launch/pan_tilt.launch.py',
        ]),
        launch_arguments={
            'device': LaunchConfiguration('device'),
            'launch_robot_state_publisher':
                LaunchConfiguration('launch_robot_state_publisher'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_pan_tilt')),
    )

    orbbec = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('orbbec_camera'), '/launch/femto_bolt.launch.py',
        ]),
        launch_arguments={
            'depth_registration': 'true',
            'enable_colored_point_cloud': 'true',
            'enable_ir': 'false',
            'enable_frame_sync': 'false',
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_orbbec')),
    )

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('realsense2_camera'), '/launch/rs_launch.py',
        ]),
        launch_arguments={
            'camera_name': 'xarm_camera',
            'align_depth.enable': 'true',
            'config_file': realsense_qos,
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_realsense')),
    )

    ffs = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('foundation_stereo'),
            '/launch/foundation_stereo.launch.py',
        ]),
        launch_arguments={
            'stream_enabled': 'true',
            'stream_align_to_color': 'true',
            'camera_profile': LaunchConfiguration('camera_profile'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_ffs')),
    )

    return LaunchDescription(args + [set_dds, pan_tilt, orbbec, realsense, ffs])
