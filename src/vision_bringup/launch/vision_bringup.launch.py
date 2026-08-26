"""Vision perception bringup — the BT-facing vision nodes.

Starts only the vision nodes the behavior_tree actually calls, selected by
auditing the production task trees (HRI+Follow, GPSR, Restaurant, PickAndPlace).
Start the sensor layer FIRST::

    ros2 launch vision_bringup vision_driver.launch.py    # pan-tilt + Orbbec + FFS

Then this perception layer. Two always-on core nodes come up bare; the rest are
gated per task (all task flags default OFF — opt into the one task you are
running)::

    ros2 launch vision_bringup vision_bringup.launch.py                    # core only
    ros2 launch vision_bringup vision_bringup.launch.py enable_hri:=true
    ros2 launch vision_bringup vision_bringup.launch.py enable_gpsr:=true
    ros2 launch vision_bringup vision_bringup.launch.py enable_restaurant:=true

Sets ``FASTRTPS_DEFAULT_PROFILES_FILE`` to the SHM profile so these camera
subscribers negotiate shared memory with the driver's Orbbec publisher (~30 Hz
vs ~3 Hz over UDP — src/tk26_vision/CAMERA_BRINGUP.md). FoundationStereo is NOT
here (it is a driver-layer node and must run outside the SHM profile); see
``docs/vision-bringup-design.md`` for the full node-selection rationale.

Always-on core (default ON, ungated by task)
--------------------------------------------
- ``enable_generalist`` (true)  generalist_node  → /object_detection_generalist
- ``enable_door``       (true)  door_detection   → /door_detection_srv

Per-task groups (default OFF)
-----------------------------
- ``enable_hri``        HRI + Follow (the two are one task): yolo_seg_node,
  person_track_server, waving_person_server, feature_recognition,
  feature_matching, seat_recommend_bbox, follow_head.
- ``enable_gpsr``       GPSR: yolo_seg_node, person_track_server,
  waving_person_server, feature_recognition, get_image.
- ``enable_restaurant`` Restaurant: waving_person_server, follow_head.
- ``enable_pick_place`` (alias ``enable_pnp``) PickAndPlace: object_scan
  (batched labels-only VLM table scan). Plus the always-on core (generalist +
  door) it shares.

Shared nodes spawn once even with several task flags on (each is gated by the OR
of the tasks that need it):
  yolo_seg_node        ← hri OR gpsr
  person_track_server  ← hri OR gpsr
  waving_person_server ← hri OR gpsr OR restaurant
  feature_recognition  ← hri OR gpsr
  follow_head          ← hri OR restaurant

API keys: the kimi_api nodes (feature_recognition, feature_matching,
seat_recommend_bbox) raise at init without OPENROUTER_API_KEY / DASHSCOPE_API_KEY.
They load ``.env`` from the launch CWD upward, so launch from the workspace root.
Each node runs via its own re-shebang'd entry-point script (vision nodes ->
.venv-vision-main); mixed venvs are fine.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
    PythonExpression,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _if(arg):
    """Condition: single enable_* flag is true."""
    return IfCondition(LaunchConfiguration(arg))


def _if_any(*args):
    """Condition: any of the enable_* flags is true (OR)."""
    expr = []
    for i, arg in enumerate(args):
        if i:
            expr.append(' or ')
        expr += ["'", LaunchConfiguration(arg), "' == 'true'"]
    return IfCondition(PythonExpression(expr))


def _node(package, executable, condition, **kwargs):
    return Node(
        package=package, executable=executable, output='screen',
        condition=condition, **kwargs,
    )


def generate_launch_description():
    pkg_share = FindPackageShare('vision_bringup')
    fastdds_profile = PathJoinSubstitution([pkg_share, 'config', 'fastdds_shm.xml'])
    pan_tilt_cfg = PathJoinSubstitution([
        FindPackageShare('pan_tilt'), 'config', 'pan_tilt.yaml',
    ])

    args = [
        # Always-on core (default ON).
        DeclareLaunchArgument('enable_generalist', default_value='true'),
        DeclareLaunchArgument('enable_door', default_value='true'),
        # Per-task groups (default OFF).
        DeclareLaunchArgument('enable_hri', default_value='false',
                              description='HRI + Follow (one task).'),
        DeclareLaunchArgument('enable_gpsr', default_value='false'),
        DeclareLaunchArgument('enable_restaurant', default_value='false'),
        DeclareLaunchArgument('enable_pick_place', default_value='false',
                              description='PickAndPlace: object_scan '
                                          '(batched labels-only VLM table scan).'),
        # Alias so `enable_pnp:=true` (operator shorthand) works identically.
        DeclareLaunchArgument('enable_pnp', default_value='false',
                              description='Alias for enable_pick_place.'),
    ]

    # SHM profile for these camera subscribers (matches the driver's publisher).
    set_dds = SetEnvironmentVariable(
        'FASTRTPS_DEFAULT_PROFILES_FILE', fastdds_profile,
    )

    nodes = [
        # --- always-on core ---
        _node('object_detection_generalist', 'generalist_node',
              _if('enable_generalist')),
        _node('vision_util', 'door_detection',
              _if('enable_door')),
        # --- shared across tasks (OR-gated, spawn once) ---
        _node('object_detection_new', 'yolo_seg_node',
              _if_any('enable_hri', 'enable_gpsr')),
        _node('vision_track', 'person_track_server',
              _if_any('enable_hri', 'enable_gpsr')),
        _node('tk_vision_specialized', 'waving_person_server',
              _if_any('enable_hri', 'enable_gpsr', 'enable_restaurant'),
              # Force show_window off, same as detect_waving.launch.py: the
              # node's True default SIGABRTs headless (cv2's Qt has no
              # "offscreen" platform plugin in .venv-vision-main).
              parameters=[{'show_window': False}]),
        _node('kimi_api', 'feature_recognition',
              _if_any('enable_hri', 'enable_gpsr')),
        _node('pan_tilt', 'follow_head',
              _if_any('enable_hri', 'enable_restaurant'),
              parameters=[pan_tilt_cfg]),
        # --- HRI-only ---
        _node('kimi_api', 'feature_matching', _if('enable_hri')),
        _node('kimi_api', 'seat_recommend_bbox', _if('enable_hri')),
        # --- GPSR-only ---
        # Legacy camera service names are owned by the subscription-free
        # provider compatibility bridge during cutover.
        _node('camera_server', 'camera_compat_bridge', _if('enable_gpsr')),
        # --- PickAndPlace-only ---
        _node('kimi_api', 'object_scan',
              _if_any('enable_pick_place', 'enable_pnp')),
    ]

    return LaunchDescription(args + [set_dds] + nodes)
