"""Launch the waving-person service together with rqt_image_view on /detect_waving_debug_image.

Usage:
    ros2 launch tk_vision_specialized detect_waving.launch.py

Common overrides:
    ros2 launch tk_vision_specialized detect_waving.launch.py launch_rqt:=false
    ros2 launch tk_vision_specialized detect_waving.launch.py model_path:=yolo11l-seg.pt
    ros2 launch tk_vision_specialized detect_waving.launch.py min_person_conf:=0.5

The node's `show_window` param is forced to `false` so we don't get two
rqt windows — the launch file owns the viewer. To suppress the viewer
entirely (headless robot, ssh-without-X, CI), pass `launch_rqt:=false`
and the node will run alone, still publishing on /detect_waving_debug_image for any
external subscriber.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    launch_rqt = LaunchConfiguration('launch_rqt')
    model_path = LaunchConfiguration('model_path')
    min_person_conf = LaunchConfiguration('min_person_conf')
    color_topic = LaunchConfiguration('color_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    sync_slop_sec = LaunchConfiguration('sync_slop_sec')
    vision_logging_enabled = LaunchConfiguration('vision_logging_enabled')

    args = [
        DeclareLaunchArgument(
            'launch_rqt', default_value='true',
            description='Spawn rqt_image_view subscribed to /detect_waving_debug_image. '
                        'Set false for headless deployments.'),
        DeclareLaunchArgument(
            'model_path', default_value='yolo11m-seg.pt',
            description='YOLO weights (resolved by vision_util.weights_cache).'),
        DeclareLaunchArgument(
            'min_person_conf', default_value='0.4',
            description='Min YOLO confidence to accept a person bbox.'),
        DeclareLaunchArgument(
            'color_topic', default_value='/camera/color/image_raw'),
        DeclareLaunchArgument(
            'depth_topic', default_value='/camera/depth/image_raw'),
        DeclareLaunchArgument(
            'camera_info_topic', default_value='/camera/color/camera_info'),
        DeclareLaunchArgument(
            'sync_slop_sec', default_value='0.1'),
        DeclareLaunchArgument(
            'vision_logging_enabled', default_value='true'),
    ]

    waving_node = Node(
        package='tk_vision_specialized',
        executable='waving_person_server',
        name='detect_waving_persons_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            # Force show_window off — the launch file owns the rqt viewer.
            # If the operator wants the node-spawned viewer instead, run the
            # node directly with `ros2 run ... -p show_window:=true`.
            'show_window': False,
            'model_path': model_path,
            'min_person_conf': min_person_conf,
            'color_topic': color_topic,
            'depth_topic': depth_topic,
            'camera_info_topic': camera_info_topic,
            'sync_slop_sec': sync_slop_sec,
            'vision_logging_enabled': vision_logging_enabled,
        }],
    )

    # rqt_image_view as raw ExecuteProcess (not Node) — launch_ros's Node
    # action appends `--ros-args ...` to the cmd, which rqt_image_view's
    # argparse treats as unknown flags and the topic-preselect positional
    # arg gets shadowed. ExecuteProcess passes the cmd verbatim.
    # 1 s delay lets the node register the /detect_waving_debug_image publisher before
    # rqt scans the topic list on startup.
    rqt_viewer = TimerAction(
        period=1.0,
        actions=[ExecuteProcess(
            cmd=['ros2', 'run', 'rqt_image_view', 'rqt_image_view', '/detect_waving_debug_image'],
            output='screen',
        )],
        condition=IfCondition(launch_rqt),
    )

    return LaunchDescription(args + [waving_node, rqt_viewer])
