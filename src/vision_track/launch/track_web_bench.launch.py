"""Bench bringup for the track_web dashboard.

Starts the person tracker (with the track_web debug telemetry ENABLED), the
waving server, and the dashboard in one command:

    ros2 launch vision_track track_web_bench.launch.py
    ros2 launch vision_track track_web_bench.launch.py port:=9000 with_waving:=false

Cameras are deliberately NOT included — start them first per
src/tk26_vision/CAMERA_BRINGUP.md (the vendored launches alone drop to ~3 Hz).
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    bind = LaunchConfiguration('bind')
    port = LaunchConfiguration('port')
    with_waving = LaunchConfiguration('with_waving')
    perf_logging = LaunchConfiguration('perf_logging')

    # Resolved at LAUNCH time (substitutions), keeping generate_launch_description
    # pure and unit-testable without a sourced workspace.
    tracker_params = PathJoinSubstitution(
        [FindPackageShare('vision_track'), 'config', 'default.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'bind', default_value='0.0.0.0',
            description='dashboard bind address (0.0.0.0 = operator laptop on LAN)'),
        DeclareLaunchArgument(
            'port', default_value='8766', description='dashboard HTTP port'),
        DeclareLaunchArgument(
            'with_waving', default_value='true',
            description='also start waving_person_server (the 👋 button backend)'),
        DeclareLaunchArgument(
            'perf_logging', default_value='false',
            description='log per-frame [perf] track/post/loop timings on the tracker'),
        # Tracker: production config first, bench telemetry overrides after —
        # the three debug flags are default-OFF in code/yaml; the bench is the
        # one place they are deliberately ON.
        Node(
            package='vision_track', executable='person_track_server',
            output='screen',
            parameters=[
                tracker_params,
                {'debug_state_enabled': True,
                 'gallery_keep_crops': True,
                 'debug_image_enabled': True,
                 'perf_logging_enabled': ParameterValue(
                     perf_logging, value_type=bool)},
            ]),
        Node(
            package='tk_vision_specialized', executable='waving_person_server',
            output='screen', condition=IfCondition(with_waving)),
        Node(
            package='vision_track', executable='track_web', output='screen',
            parameters=[{
                'bind': ParameterValue(bind, value_type=str),
                'port': ParameterValue(port, value_type=int),
            }]),
    ])
