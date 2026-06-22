"""Vision perception bringup — everything except the driver / hardware layer.

Starts the vision nodes that normally run one-per-terminal via ``ros2 run``:
detection, person tracking, the specialized action servers, the kimi_api LLM
services, and the utility services. Each group is gated by an ``enable_*``
argument so whole subsystems can be toggled per task without editing this file.

Start the cameras + pan-tilt FIRST::

    ros2 launch vision_bringup vision_driver.launch.py

Then::

    ros2 launch vision_bringup vision_bringup.launch.py
    ros2 launch vision_bringup vision_bringup.launch.py enable_llm:=false
    ros2 launch vision_bringup vision_bringup.launch.py \
        enable_follow_head:=true enable_monocular_depth:=true

Like the driver launch, this sets ``FASTRTPS_DEFAULT_PROFILES_FILE`` to the
FastDDS SHM profile so the camera subscribers (detection / tracker) negotiate
shared memory and see ~30 Hz — without it they fall back to UDP and ~3 Hz
(src/tk26_vision/CAMERA_BRINGUP.md).

Group defaults
--------------
- ``enable_detection``         (true)  specialist YOLO-seg + generalist
- ``enable_default_detection`` (false) legacy COCO node (off: triple camera sub)
- ``enable_tracker``           (true)  person_track_server
- ``enable_specialized``       (true)  spot_on_shelf + waving + object_match_all
- ``enable_match_extra``       (false) object_match_server + placing_location
- ``enable_llm``               (true)  kimi_api services (need API keys in .env)
- ``enable_utils``             (true)  door_detection + point-cloud relays
- ``enable_follow_head``       (false) drives the servo; needs driver pan-tilt up
- ``enable_monocular_depth``   (false) DA3 action server (separate .venv-da3)

The kimi_api LLM nodes crash at init if ``OPENROUTER_API_KEY`` (and, for the
Qwen paths, ``DASHSCOPE_API_KEY``) are absent — they load ``.env`` from the
launch CWD upward, so launch from the workspace root. Set ``enable_llm:=false``
to skip them. Mixed venvs are fine: each node runs via its own re-shebang'd
entry-point script (FFS->.venv-fs, monocular_depth->.venv-da3, rest->.venv-vision-main).
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _node(package, executable, **kwargs):
    return Node(package=package, executable=executable, output='screen', **kwargs)


def _group(arg, *nodes):
    return GroupAction(
        condition=IfCondition(LaunchConfiguration(arg)),
        actions=list(nodes),
    )


def generate_launch_description():
    pkg_share = FindPackageShare('vision_bringup')
    fastdds_profile = PathJoinSubstitution([pkg_share, 'config', 'fastdds_shm.xml'])
    pan_tilt_cfg = PathJoinSubstitution([
        FindPackageShare('pan_tilt'), 'config', 'pan_tilt.yaml',
    ])

    args = [
        DeclareLaunchArgument('enable_detection', default_value='true'),
        DeclareLaunchArgument('enable_default_detection', default_value='false'),
        DeclareLaunchArgument('enable_tracker', default_value='true'),
        DeclareLaunchArgument('enable_specialized', default_value='true'),
        DeclareLaunchArgument('enable_match_extra', default_value='false'),
        DeclareLaunchArgument('enable_llm', default_value='true'),
        DeclareLaunchArgument('enable_utils', default_value='true'),
        DeclareLaunchArgument('enable_follow_head', default_value='false'),
        DeclareLaunchArgument('enable_monocular_depth', default_value='false'),
    ]

    set_dds = SetEnvironmentVariable(
        'FASTRTPS_DEFAULT_PROFILES_FILE', fastdds_profile,
    )

    groups = [
        _group(
            'enable_detection',
            _node('object_detection_new', 'yolo_seg_node'),
            _node('object_detection_generalist', 'generalist_node'),
        ),
        _group(
            'enable_default_detection',
            _node('object_detection_new', 'yolo_seg_default_node'),
        ),
        _group(
            'enable_tracker',
            _node('vision_track', 'person_track_server'),
        ),
        _group(
            'enable_specialized',
            _node('tk_vision_specialized', 'spot_on_shelf_server'),
            _node('tk_vision_specialized', 'waving_person_server'),
            _node('tk_vision_specialized', 'object_match_all_server'),
        ),
        _group(
            'enable_match_extra',
            _node('tk_vision_specialized', 'object_match_server'),
            _node('tk_vision_specialized', 'placing_location_server'),
        ),
        _group(
            'enable_llm',
            _node('kimi_api', 'feature_recognition'),
            _node('kimi_api', 'feature_matching'),
            _node('kimi_api', 'grocery_categorize'),
            _node('kimi_api', 'seat_recommend_bbox'),
        ),
        _group(
            'enable_utils',
            _node('vision_util', 'door_detection'),
            _node('vision_util', 'get_point_cloud'),
            _node('vision_util', 'get_orbbec_pc'),
        ),
        _group(
            'enable_follow_head',
            _node('pan_tilt', 'follow_head', parameters=[pan_tilt_cfg]),
        ),
        _group(
            'enable_monocular_depth',
            _node('monocular_depth', 'monocular_depth_pc'),
        ),
    ]

    return LaunchDescription(args + [set_dds] + groups)
