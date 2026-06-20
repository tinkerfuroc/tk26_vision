"""Launch the handeye_web eye-in-hand calibration server.

  ros2 launch handeye_calib handeye_web.launch.py port:=8766 robot_name:=tinker2

Brings up the handeye_web node (FastAPI UI + rclpy). The RealSense camera
(realsense2_camera, camera_name:=xarm_camera) must be launched separately —
the UI shows 'no camera' until color frames arrive.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

_STR = [
    ("bind", "127.0.0.1"),
    ("robot_name", ""),
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
    ("mount_to_color_xyz", "0 0 0"),
    ("mount_to_color_rpy", "0 0 0"),
]
_INT = [
    ("port", "8766"),
    ("squares_x", "5"), ("squares_y", "5"),
    # StabilityTracker window — see gates.StabilityTracker docstring.
    ("stability_window", "5"),
]
_FLOAT = [
    ("square_len_m", "0.04"), ("marker_len_m", "0.03"), ("min_diversity_deg", "30.0"),
    # Settle-gate thresholds. Defaults tuned for camera-only ChArUco PnP at
    # ~30-60 cm. Tighten only if your optical conditions are dramatically
    # quieter than typical (or your noise budget is dramatically smaller).
    ("stability_rot_tol_deg", "0.5"),
    ("stability_trans_tol_m", "0.003"),
]


def generate_launch_description():
    decls = [DeclareLaunchArgument(n, default_value=d) for n, d in (_STR + _INT + _FLOAT)]
    params = {}
    for n, _ in _STR:
        params[n] = ParameterValue(LaunchConfiguration(n), value_type=str)
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
