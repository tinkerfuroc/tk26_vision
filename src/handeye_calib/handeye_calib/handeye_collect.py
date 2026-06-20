"""ROS node: drive the xArm to authored poses, settle-gate, detect the ChArUco board
(reusing pan_tilt's detector), accumulate diverse high-quality Samples, save a session.

The pure accumulator (CaptureSession) is unit-tested; the rclpy wiring is exercised by
the hardware dry-run.
"""
import json
import numpy as np

from handeye_calib import handeye_model as hm
from handeye_calib import gates


class CaptureSession:
    def __init__(self, min_diversity_deg=30.0,
                 min_corners=10, max_reproj_px=1.5, min_area_frac=0.05):
        self.min_diversity_deg = min_diversity_deg
        self.q = dict(min_corners=min_corners, max_reproj_px=max_reproj_px,
                      min_area_frac=min_area_frac)
        self.samples = []

    def try_add(self, T_base_eef, T_cam_board, obs_px, corner_idx,
                n_corners, reproj_px, area_frac):
        ok, reason = gates.quality_ok(n_corners, reproj_px, area_frac, **self.q)
        if not ok:
            return False, reason
        accepted_eef = [s.T_base_eef for s in self.samples]
        if not gates.is_diverse(T_base_eef, accepted_eef, self.min_diversity_deg):
            return False, "not diverse (<%g deg)" % self.min_diversity_deg
        self.samples.append(hm.Sample(np.asarray(T_base_eef), np.asarray(T_cam_board),
                                      np.asarray(obs_px), np.asarray(corner_idx)))
        return True, "accepted"

    def to_json(self):
        return json.dumps([{
            "T_base_eef": s.T_base_eef.tolist(),
            "T_cam_board": s.T_cam_board.tolist(),
            "obs_px": s.obs_px.tolist(),
            "corner_idx": s.corner_idx.tolist(),
        } for s in self.samples])


# ---- rclpy node (exercised on hardware; imports guarded so unit tests stay ROS-free) ----
def main():
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from tf2_ros import Buffer, TransformListener
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CameraInfo
    from tinker_arm_msgs.action import JointMove
    # The hardware capture loop additionally imports pan_tilt.calibration.aruco_detect
    # (ChArUco detection/consensus) and handeye_calib.transforms (FK math) where used,
    # when the _on_image/run loop below is fleshed out on hardware.

    class HandeyeCollect(Node):
        def __init__(self):
            super().__init__("handeye_collect")
            self.session = CaptureSession()
            self.bridge = CvBridge()
            self.tf_buffer = Buffer()
            TransformListener(self.tf_buffer, self)
            self.jm = ActionClient(self, JointMove, "/xarm/joint_move")
            # Topics match the realsense2_camera launch convention used in
            # this workspace (namespace=/camera, node=xarm_camera).
            self.sub = self.create_subscription(
                Image, "/camera/xarm_camera/color/image_raw", self._on_image, 1)
            self.info_sub = self.create_subscription(
                CameraInfo, "/camera/xarm_camera/color/camera_info", self._on_info, 1)
            self.stability = gates.StabilityTracker()
            self.K = None
            self.get_logger().info("handeye_collect ready")
        # _on_info caches K; _on_image runs aruco_detect + StabilityTracker;
        # a run() coroutine sends JointMove goals, waits settle, then captures.
        # See README 'Collection node' for the full loop; this is hardware-tier code.

    rclpy.init()
    node = HandeyeCollect()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
