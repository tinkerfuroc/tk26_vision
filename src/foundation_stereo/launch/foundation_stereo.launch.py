"""Launch the foundation_stereo node with the canonical config yaml.

Override individual params via `ros2 launch foundation_stereo
foundation_stereo.launch.py stream_enabled:=true camera_profile:=d405`.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    args = []
    for name, default in [
        ("camera_profile", "d435"),
        ("stream_enabled", "false"),
        ("stream_align_to_color", "true"),
        ("stream_qos_reliability", "reliable"),
        ("default_model_kind", "fast_trt"),
        ("default_trt_variant", "output_two_stage"),
    ]:
        args.append(DeclareLaunchArgument(name, default_value=default))

    pkg_share = FindPackageShare("foundation_stereo")
    config_path = [pkg_share, "/config/foundation_stereo.yaml"]

    node = Node(
        package="foundation_stereo",
        executable="foundation_stereo_node",
        name="foundation_stereo",
        output="screen",
        parameters=[
            config_path,
            {
                "camera_profile": LaunchConfiguration("camera_profile"),
                "stream_enabled": LaunchConfiguration("stream_enabled"),
                "stream_align_to_color": LaunchConfiguration("stream_align_to_color"),
                "stream_qos_reliability": LaunchConfiguration("stream_qos_reliability"),
                "default_model_kind": LaunchConfiguration("default_model_kind"),
                "default_trt_variant": LaunchConfiguration("default_trt_variant"),
            },
        ],
    )

    return LaunchDescription(args + [node])
