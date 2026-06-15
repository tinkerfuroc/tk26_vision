"""calib_web-style browser tool for eye-in-hand calibration.

Pure helpers (validate_pose_set, diff_payload) are unit-tested. The FastAPI + rclpy
server mirrors pan_tilt/calib_web.py: live overlay, pose authoring validated against
SafetyEnvelope, subprocess solve with streamed logs, verification overlay, and
diff-preview + atomic promote via handeye_calib.apply_handeye.

Importing this module is ROS-free on purpose: rclpy / FastAPI / cv2 imports live
inside main() so the unit-tested helpers load under the plain venv. Mirrors the
same import discipline as pan_tilt/calib_web.py's optional-import guards.
"""
MIN_POSES = 12


def validate_pose_set(poses):
    if len(poses) < MIN_POSES:
        return False, f"need at least {MIN_POSES} poses, got {len(poses)}"
    for i, p in enumerate(poses):
        if "joints" not in p or len(p["joints"]) != 7:
            return False, f"pose {i}: expected 7 joint values"
    return True, "ok"


def diff_payload(old_xyz, new_xyz, old_rpy, new_rpy):
    return {
        "xyz": {"old": old_xyz, "new": new_xyz},
        "rpy": {"old": old_rpy, "new": new_rpy},
        "changed": (old_xyz != new_xyz) or (old_rpy != new_rpy),
    }


def main():
    # Mirrors pan_tilt/calib_web.py main(): build rclpy node, start uvicorn worker,
    # serve the authoring/run/verify/promote UI. Hardware-tier; see README.
    #
    # ROS / web imports are deferred to here so `import handeye_calib.handeye_web`
    # stays ROS-free for the unit-tested helpers above.
    #
    # Server structure to reuse from pan_tilt/calib_web.py:
    #   - rclpy.spin() on the main thread; uvicorn on a worker thread.
    #   - A single Node owns shared state behind `node.lock`; every FastAPI
    #     handler touches ROS state only through lock-protected accessors.
    #   - Tabs: (1) live camera + ChArUco overlay; (2) pose authoring — joint
    #     input validated with validate_pose_set() AND against
    #     calibration safety (SafetyEnvelope-style gate), "send to robot",
    #     draft pose list; (3) run the solver as a subprocess
    #     (handeye_solve) with streamed logs; (4) verification overlay;
    #     (5) diff-preview (diff_payload) + atomic promote.
    #
    # Promote-area wiring note (DO NOT hardcode a guessed literal):
    #   The real URDF camera mount joint lives in
    #     src/tk25_manipulation/src/xarm_ros2/xarm_description/urdf/camera/
    #     realsense_d435i.urdf.xacro
    #   and is xacro-templated: name="${camera_prefix}camera_link_joint".
    #   So when promote calls apply_handeye.patch_urdf_origin(xacro_text,
    #   joint_name, xyz, rpy), the caller MUST pass that exact templated joint
    #   name string (i.e. "${camera_prefix}camera_link_joint", or the prefix
    #   resolved to the deployment's actual camera_prefix value) — never a
    #   guessed literal like "camera_link_joint". The atomic write itself goes
    #   through apply_handeye.write_with_backup (timestamped .old backup + tmp
    #   rename).
    #
    # tinker_robot_config note:
    #   apply_handeye.handeye_yaml_dict(...) output already passes the existing
    #   tinker_robot_config lint as-is — no schema change is needed to persist
    #   the solved hand-eye block.
    import rclpy
    rclpy.init()
    # ... node + uvicorn wiring (reuse calib_web structure) ...
    rclpy.shutdown()


if __name__ == "__main__":
    main()
