"""Launch the handeye_web eye-in-hand calibration server.

  ros2 launch handeye_calib handeye_web.launch.py port:=8766 robot_name:=tinker2

Brings up the handeye_web node (FastAPI UI + rclpy). The RealSense camera
(realsense2_camera, camera_name:=xarm_camera) must be launched separately —
the UI shows 'no camera' until color frames arrive.
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

_STR = [
    ("bind", "127.0.0.1"),
    # Default mirrors $ROBOT_NAME so the ``robot_name`` param can't silently
    # contradict the env var the node's _resolve_robot_name() prefers. An
    # explicit ``robot_name:=…`` launch arg still overrides. Without this the
    # param defaulted to '' and `ros2 param get /handeye_web robot_name` read
    # empty even with ROBOT_NAME exported — a misleading "unset" signal.
    ("robot_name", os.environ.get("ROBOT_NAME", "")),
    # Matches the realsense2_camera launch convention used in this workspace
    # (namespace=/camera, node=xarm_camera) — topics land at
    # /camera/xarm_camera/color/*. Override via launch arg if the wrist camera
    # is brought up under a different namespace.
    ("color_image_topic", "/camera/xarm_camera/color/image_raw"),
    ("camera_info_topic", "/camera/xarm_camera/color/camera_info"),
    ("base_frame", "link_base"),
    ("eef_frame", "link_eef"),
    ("aruco_dict", "DICT_5X5_100"),
    # Canonical action name as of 2026-06-20 (see commit 78cd535 in tk25
    # for the BT migration). The legacy /xarm/joint_move endpoint was
    # retired by tk25_manipulation — verified empty (0 servers) via
    # `ros2 action info /xarm/joint_move`.
    ("jointmove_action", "joint_move_action"),
    # Internal camera geometry T_camera_link -> {color,ir}_optical the solve
    # composes through to recover the camera_link mount. D435 vendor-URDF values:
    # color_frame is +15 mm in Y then the optical rotation; left-IR is coincident
    # with camera_link (the depth origin) then the SAME optical rotation. (These
    # are the REAL defaults — previously "0 0 0", which mis-wrote color_optical
    # into the camera_link joint.)
    ("mount_to_color_xyz", "0 0.015 0"),
    ("mount_to_color_rpy", "-1.5707963267948966 0 -1.5707963267948966"),
    ("mount_to_ir_xyz", "0 0 0"),
    ("mount_to_ir_rpy", "-1.5707963267948966 0 -1.5707963267948966"),
    # FoundationStereo metric-depth service (color-aligned). Override if the
    # foundation_stereo node runs under a non-default namespace.
    ("ffs_service", "/foundation_stereo/get_depth"),
    # Observation frame: 'color' (default) or 'ir' (left-IR == camera_link, native
    # FFS depth). Runtime-switchable in the web UI (Info tab).
    ("calib_frame", "color"),
    # Left-IR observation topics + native-IR FFS depth stream (run FFS with
    # stream_enabled:=true stream_align_to_color:=false; or point ffs_ir_depth_topic
    # at /camera/xarm_camera/depth/image_rect_raw for RealSense-native IR depth).
    ("ir_image_topic", "/camera/xarm_camera/infra1/image_rect_raw"),
    ("ir_info_topic", "/camera/xarm_camera/infra1/camera_info"),
    ("ffs_ir_depth_topic", "/foundation_stereo/depth/image_rect_raw"),
    # Camera node whose depth_module.emitter_enabled param the IR-emitter toggle
    # flips (disable for IR-frame capture; the dot pattern corrupts ChArUco).
    ("camera_node_name", "/camera/xarm_camera"),
]
# Booleans must be passed with value_type=bool — a raw string "false" would be
# truthy in the node's bool() read and silently keep FFS depth enabled.
_BOOL = [
    ("use_ffs_depth", "true"),
]
_INT = [
    ("port", "8766"),
    ("squares_x", "5"), ("squares_y", "5"),
    # StabilityTracker window — see gates.StabilityTracker docstring.
    ("stability_window", "5"),
    # FFS depth: local median half-window, min valid corners to use depth, and
    # consecutive depth-less captures before the one-time IR-stream WARN.
    ("depth_win", "2"), ("depth_min_corners", "3"), ("ffs_depth_warn_after", "5"),
]
_FLOAT = [
    ("square_len_m", "0.04"), ("marker_len_m", "0.03"),
    # FFS depth knobs. depth_weight=1.0 makes FFS metric depth a co-equal
    # constraint (pins the optical-axis DOF) without out-voting the sub-pixel
    # reprojection that owns rotation — raise only if FFS is metrically
    # validated on this robot. depth_sigma_m = assumed stereo noise; depth_z_*
    # = valid-depth band (m). ffs_*_s = service wait/call timeouts.
    ("depth_weight", "1.0"), ("depth_sigma_m", "0.005"),
    ("depth_z_min", "0.05"), ("depth_z_max", "2.0"),
    ("ffs_wait_for_service_s", "1.0"), ("ffs_call_timeout_s", "10.0"),
    # min_diversity_deg lowered 2026-06-21 from 30 → 5. The all-vs-all 30°
    # rule rejected ~half of any 20+ waypoint set due to SO(3) packing
    # limits — wrong semantic for hand-eye, which only needs the *set* to
    # span rotation. 5° still dedups camera-shake duplicates. Set to 0 to
    # disable the gate entirely (gate.is_diverse short-circuits).
    ("min_diversity_deg", "5.0"),
    # Settle-gate thresholds. Defaults tuned for camera-only ChArUco PnP at
    # ~30-60 cm. Tighten only if your optical conditions are dramatically
    # quieter than typical (or your noise budget is dramatically smaller).
    ("stability_rot_tol_deg", "0.5"),
    ("stability_trans_tol_m", "0.003"),
]


def generate_launch_description():
    decls = [DeclareLaunchArgument(n, default_value=d)
             for n, d in (_STR + _BOOL + _INT + _FLOAT)]
    params = {}
    for n, _ in _STR:
        params[n] = ParameterValue(LaunchConfiguration(n), value_type=str)
    for n, _ in _BOOL:
        params[n] = ParameterValue(LaunchConfiguration(n), value_type=bool)
    for n, _ in _INT:
        params[n] = ParameterValue(LaunchConfiguration(n), value_type=int)
    for n, _ in _FLOAT:
        params[n] = ParameterValue(LaunchConfiguration(n), value_type=float)
    web = Node(
        package="handeye_calib",
        executable="handeye_web",
        name="handeye_web",
        output="screen",
        parameters=[params],
    )
    return LaunchDescription(decls + [web])
