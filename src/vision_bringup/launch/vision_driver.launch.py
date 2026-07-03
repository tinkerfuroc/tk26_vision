"""Vision driver bringup — the sensor / hardware layer of the tk26 vision stack.

Brings up, in one command:
  - the pan-tilt head (controller + state_publisher + URDF RSP),
  - the Orbbec Femto Bolt (the tk26-tuned overrides), and
  - FoundationStereo with its streaming depth publisher ENABLED.

RealSense is deliberately NOT here — the manipulation launch (grasp_bringup /
arm_bringup_cumotion) owns the ``xarm_camera`` RealSense and is the only place
that enables its IR pair (``enable_infra1/2``). Launching it here too would
double-bind the same serial/node. FoundationStereo consumes that
manipulation-owned IR pair, so its streamed depth is non-empty only when the
manipulation stack is already up (see the cross-launch contract in
``docs/vision-bringup-design.md``).

FastDDS transport — the whole launch (pan-tilt + Orbbec + FFS) runs under
``FASTRTPS_DEFAULT_PROFILES_FILE`` = the SHM profile (``fastdds_shm.xml``):
  * the Orbbec *publisher* must offer SHM so the perception subscribers in
    vision_bringup negotiate shared memory and sustain ~30 Hz instead of the
    ~3 Hz the vendored launch drops to over UDP (src/tk26_vision/CAMERA_BRINGUP.md);
  * FoundationStereo ALSO needs it — the RealSense IR pair it subscribes to
    (~0.82 MB combined) exceeds the *default* ~512 KB FastDDS SHM segment, so a
    frame drops and FFS time-sync collapses; the profile's larger (20 MB)
    segment fixes that. (An earlier "SHM corrupts cuMotion collision voxels"
    concern was experimentally refuted — the 20 MB segment is data-safe; the
    real IR-pair reliability fix lives on the camera *owner*, the manipulation
    launch, which must publish under the same profile. See
    ``docs/vision-bringup-design.md``.)

Usage::

    ros2 launch vision_bringup vision_driver.launch.py
    ros2 launch vision_bringup vision_driver.launch.py enable_ffs:=false
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

    args = [
        DeclareLaunchArgument('enable_pan_tilt', default_value='true'),
        DeclareLaunchArgument('enable_orbbec', default_value='true'),
        DeclareLaunchArgument(
            'enable_ffs', default_value='true',
            description=(
                'FoundationStereo streaming depth. Needs the manipulation-owned '
                'RealSense IR pair to produce output; set false for vision-only '
                'bench runs without the arm stack.'
            ),
        ),
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
        DeclareLaunchArgument(
            'ffs_stream_enabled', default_value='true',
            description='Publish the FFS streaming depth topic.',
        ),
        DeclareLaunchArgument(
            'ffs_stream_align_to_color', default_value='false',
            description=(
                'false = non-aligned /foundation_stereo/depth/* (the form the '
                'cuMotion nvblox collision path consumes). Do not enable unless '
                'a consumer specifically needs aligned-to-color depth.'
            ),
        ),
        DeclareLaunchArgument(
            'color_width', default_value='1280',
            description=(
                'Orbbec color stream width. Task launch scripts override this '
                '(e.g. HRI raises it for face/feature enrollment quality); '
                'default matches the vendored femto_bolt.launch.py default. '
                'Depth stream resolution is untouched -- SW alignment handles '
                'any color/depth size mismatch.'
            ),
        ),
        DeclareLaunchArgument(
            'color_height', default_value='720',
            description='Orbbec color stream height; see color_width.',
        ),
    ]

    # SHM profile for the whole launch: the Orbbec publisher offers SHM to the
    # perception subscribers, and FFS gets the larger SHM segment its RealSense
    # IR pair needs. Perception subscribers in vision_bringup set the same env.
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
            'color_width': LaunchConfiguration('color_width'),
            'color_height': LaunchConfiguration('color_height'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_orbbec')),
    )

    ffs = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('foundation_stereo'),
            '/launch/foundation_stereo.launch.py',
        ]),
        launch_arguments={
            'stream_enabled': LaunchConfiguration('ffs_stream_enabled'),
            'stream_align_to_color':
                LaunchConfiguration('ffs_stream_align_to_color'),
            'camera_profile': LaunchConfiguration('camera_profile'),
        }.items(),
        condition=IfCondition(LaunchConfiguration('enable_ffs')),
    )

    return LaunchDescription(args + [set_dds, pan_tilt, orbbec, ffs])
