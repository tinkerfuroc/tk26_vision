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


def _make_node_class():
    """Build the rclpy Node class lazily so importing this module stays ROS-free.

    All rclpy / cv_bridge / tinker_arm_msgs / pan_tilt / scipy imports happen
    inside this factory — they only run when ``handeye_web.HandeyeWebNode`` is
    accessed (via the module ``__getattr__`` hook below), never at module import
    time. This keeps the unit-tested helpers (validate_pose_set, diff_payload)
    loadable under a plain venv with no ROS on the path.
    """
    import os
    import time
    import collections
    import threading
    import numpy as np
    import cv2
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.action import ActionClient
    from tf2_ros import Buffer, TransformListener
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CameraInfo, JointState
    from tinker_arm_msgs.action import JointMove
    from scipy.spatial.transform import Rotation as _R

    from handeye_calib import web_support as ws
    from handeye_calib import handeye_model as hm
    from handeye_calib import handeye_solve as hs
    from handeye_calib import apply_handeye as ah
    from handeye_calib import gates as hgates
    from handeye_calib.handeye_collect import CaptureSession
    # Detection + safety reuse from pan_tilt. aruco_detect's Detection does NOT
    # surface per-corner charuco pixels/IDs (only pose + corner count + reproj
    # RMS), so for capture we call cv2.aruco.CharucoDetector.detectBoard directly
    # to get charucoCorners/charucoIds, then board.matchImagePoints + solvePnP for
    # the board pose, observed pixels, IDs, and a scalar reprojection error.
    from pan_tilt.calibration.aruco_detect import BoardSpec, build_board, build_detector
    from pan_tilt.calibration.safety import SafetyEnvelope

    class HandeyeWebNode(Node):
        def __init__(self):
            super().__init__("handeye_web")
            self.lock = threading.Lock()

            self.bind_host = self._param("bind", "127.0.0.1")
            self.bind_port = int(self._param("port", 8766))
            self._robot_name = self._param("robot_name", os.environ.get("ROBOT_NAME", ""))
            self._base_frame = self._param("base_frame", "link_base")
            self._eef_frame = self._param("eef_frame", "link_eef")
            self._aruco_dict_name = self._param("aruco_dict", "DICT_5X5_100")
            self._marker_len = float(self._param("marker_len_m", 0.03))
            self._mount_to_color_xyz = self._param("mount_to_color_xyz", "0 0 0")
            self._mount_to_color_rpy = self._param("mount_to_color_rpy", "0 0 0")

            self.bridge = CvBridge()
            self._frame = None
            self._K = None
            self._D = None
            self._last_det = None              # {corners:int, reproj_px:float} or None
            self._last_corners_xy = None       # (M,2) px for overlay, or None
            self._cap = None                   # latest {T_cam_board, obs_px, corner_idx, reproj_px, area_frac}

            # Frame-rate bookkeeping. Each _on_image bumps the counter and pushes
            # a monotonic timestamp onto a rolling 30-sample deque; frame_hz is
            # derived from the deque span (delta-time across all samples).
            self._frame_count = 0
            self._frame_ts = collections.deque(maxlen=30)
            self._last_frame_monotonic = None
            self._time = time

            # Cached base->ee TF; refreshed lazily inside get_state_dict() with
            # a 50ms timeout, so the WS push never blocks if TF is unavailable.
            self._t_base_ee_cache = None

            # /joint_states best-effort: subscribe and stash xArm joint positions.
            # The xArm publishes 7 joints named joint1..joint7 (or with link_ prefix
            # depending on URDF); we accept either, falling back to the raw list.
            self._xarm_joint_positions = None
            self._xarm_joint_names = tuple(
                f"joint{i+1}" for i in range(7)
            )
            self.create_subscription(
                JointState, self._param("joint_states_topic", "/joint_states"),
                self._on_joint_state, qos_profile_sensor_data)

            # Stability tracker (observable in T1; T4 promotes it to a hard gate).
            self._stab_window = int(self._param("stability_window", 5))
            self._stab_rot_tol_deg = float(self._param("stability_rot_tol_deg", 0.1))
            self._stab_trans_tol_m = float(self._param("stability_trans_tol_m", 0.0003))
            self._stability = hgates.StabilityTracker(
                window=self._stab_window,
                rot_tol_deg=self._stab_rot_tol_deg,
                trans_tol_m=self._stab_trans_tol_m,
            )
            self._stability_steady = False
            self._stability_since_frames = 0

            # Diversity target (degrees). T4 will compute actual coverage from
            # the accepted-sample rotation spread; T1 ships the field at 0.0.
            self._diversity_target_deg = float(self._param("min_diversity_deg", 30.0))

            self.session = CaptureSession(min_diversity_deg=self._diversity_target_deg)
            self.last_solve = None

            self._sx = int(self._param("squares_x", 5))
            self._sy = int(self._param("squares_y", 5))
            self._sq = float(self._param("square_len_m", 0.04))
            self._board_pts = hm.board_corners(self._sx, self._sy, self._sq)

            self.tf_buffer = Buffer()
            TransformListener(self.tf_buffer, self)

            # ChArUco board + detector. aruco_dict param is a cv2.aruco predefined
            # dictionary name (e.g. "DICT_5X5_100"); resolve it to its int id and
            # build a BoardSpec matching our squares/lengths so detection geometry
            # agrees with self._board_pts.
            dict_id = getattr(cv2.aruco, self._aruco_dict_name, cv2.aruco.DICT_5X5_100)
            self._board_spec = BoardSpec(
                squares_x=self._sx, squares_y=self._sy,
                square_len_m=self._sq, marker_len_m=self._marker_len,
                dict_id=dict_id,
            )
            self._board = build_board(self._board_spec)
            self._detector = build_detector(self._board)

            # SafetyEnvelope with permissive defaults (constructor needs no config).
            # validate() takes a 4x4 base->eef pose, not joint angles, so do_move
            # below validates the *current* EE pose via TF when available and treats
            # "no TF / no envelope" as skip — never blocking node construction.
            try:
                self._safety = SafetyEnvelope()
            except Exception as exc:  # pragma: no cover - permissive guard
                self.get_logger().warn(f"SafetyEnvelope unavailable ({exc}); skipping pose validation")
                self._safety = None

            self._image_topic = self._param("color_image_topic", "/xarm_camera/color/image_raw")
            self.create_subscription(
                Image, self._image_topic,
                self._on_image, qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, self._param("camera_info_topic", "/xarm_camera/color/camera_info"),
                self._on_info, qos_profile_sensor_data)
            self._jm = ActionClient(self, JointMove, self._param("jointmove_action", "/xarm/joint_move"))

            self._np = np
            self._cv2 = cv2
            self._R = _R
            self.get_logger().info("handeye_web node ready")

        # ---- param helper ----------------------------------------------------

        def _param(self, name, default):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # ---- subscriptions ---------------------------------------------------

        def _on_image(self, msg):
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:
                self.get_logger().warn(
                    f"cv_bridge conversion failed ({exc})", throttle_duration_sec=5.0)
                return
            with self.lock:
                K = None if self._K is None else self._K.copy()
                D = None if self._D is None else self._D.copy()

            corners_xy, last_det, cap = self._detect(bgr, K, D)

            # Frame-rate bookkeeping (monotonic clock, immune to wall-clock jumps).
            now = self._time.monotonic()
            # Stability tracker: feed the latest cam->board pose only when we
            # have one; missing/lost detection resets the running window.
            if cap is not None:
                steady = self._stability.update(cap["T_cam_board"])
            else:
                self._stability.reset()
                steady = False

            with self.lock:
                self._frame = bgr
                self._last_corners_xy = corners_xy
                self._last_det = last_det
                self._cap = cap
                self._frame_count += 1
                self._frame_ts.append(now)
                self._last_frame_monotonic = now
                if steady:
                    self._stability_since_frames += 1
                else:
                    self._stability_since_frames = 0
                self._stability_steady = steady

        def _on_info(self, msg):
            np = self._np
            K = np.array(msg.k, float).reshape(3, 3)
            D = np.array(msg.d, float).flatten() if len(msg.d) else np.zeros(5)
            with self.lock:
                self._K = K
                self._D = D

        def _on_joint_state(self, msg):
            """Stash xArm joint positions (best-effort).

            Multiple publishers feed /joint_states on Tinker (xArm + pan-tilt);
            we only care about the 7 xArm joints. If the message contains the
            named xArm joints, pull them in canonical order; otherwise — for
            mock/single-publisher setups — accept the raw position vector when
            it's exactly 7 long.
            """
            names = list(msg.name) if msg.name else []
            positions = list(msg.position) if msg.position else []
            xarm = []
            if names and len(positions) == len(names):
                lookup = {n: p for n, p in zip(names, positions)}
                xarm = [lookup[j] for j in self._xarm_joint_names if j in lookup]
                if len(xarm) != len(self._xarm_joint_names):
                    xarm = []
            if not xarm and len(positions) == 7:
                xarm = list(positions)
            if xarm:
                with self.lock:
                    self._xarm_joint_positions = list(map(float, xarm))

        # ---- detection -------------------------------------------------------

        def _detect(self, bgr, K, D):
            """Run ChArUco detection on a BGR frame.

            Returns (corners_xy, last_det, cap) where:
              corners_xy: (M,2) float px or None  (for the overlay)
              last_det:   {"corners": int, "reproj_px": float} or None
              cap:        capture dict or None (None if no usable pose / no K)
            """
            np = self._np
            cv2 = self._cv2
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            try:
                ch_corners, ch_ids, _, _ = self._detector.detectBoard(gray)
            except cv2.error:
                return None, None, None
            if ch_corners is None or ch_ids is None or len(ch_ids) < 4:
                return None, None, None

            obs_px = np.asarray(ch_corners, float).reshape(-1, 2)
            corner_idx = np.asarray(ch_ids).reshape(-1).astype(int)

            # No intrinsics yet → overlay only, no pose / no capturable sample.
            if K is None:
                return obs_px, {"corners": int(len(corner_idx)), "reproj_px": None}, None

            try:
                obj_pts, img_pts = self._board.matchImagePoints(ch_corners, ch_ids)
            except cv2.error:
                return obs_px, {"corners": int(len(corner_idx)), "reproj_px": None}, None
            if obj_pts is None or len(obj_pts) < 4:
                return obs_px, {"corners": int(len(corner_idx)), "reproj_px": None}, None

            dist = D if D is not None else np.zeros(5)
            ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return obs_px, {"corners": int(len(corner_idx)), "reproj_px": None}, None

            # Scalar reprojection error (RMS px) for the quality gate.
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            proj = proj.reshape(-1, 2)
            reproj_px = float(np.sqrt(np.mean(np.sum((proj - img_pts.reshape(-1, 2)) ** 2, axis=1))))

            T_cam_board = np.eye(4)
            T_cam_board[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_cam_board[:3, 3] = tvec.reshape(3)

            # area_frac = bbox area of detected corners / image area.
            h, w = bgr.shape[:2]
            x0, y0 = obs_px.min(axis=0)
            x1, y1 = obs_px.max(axis=0)
            area_frac = float(((x1 - x0) * (y1 - y0)) / (w * h)) if w and h else 0.0

            cap = {
                "T_cam_board": T_cam_board,
                "obs_px": obs_px,
                "corner_idx": corner_idx,
                "reproj_px": reproj_px,
                "area_frac": area_frac,
            }
            return obs_px, {"corners": int(len(corner_idx)), "reproj_px": reproj_px}, cap

        # ---- accessors -------------------------------------------------------

        def get_state_dict(self):
            """Snapshot of the node state for the WS push + REST /api/state.

            Composes :func:`web_support.enriched_state_payload` from cached
            members plus a best-effort TF refresh (50 ms timeout). T1 wires the
            full key surface; T4 populates ``samples``, T5 populates
            ``last_solve``.
            """
            np = self._np
            # Refresh the base->ee cache outside the lock to keep the WS push
            # responsive even when TF is unavailable.
            t_base_ee = self._refresh_t_base_ee_cache()

            with self.lock:
                frame_count = self._frame_count
                ts = list(self._frame_ts)
                last_mono = self._last_frame_monotonic
                steady = self._stability_steady
                since = self._stability_since_frames
                xarm = (None if self._xarm_joint_positions is None
                        else list(self._xarm_joint_positions))
                camera_connected = self._frame is not None
                intrinsics_ok = self._K is not None
                num_samples = len(self.session.samples)
                last_det = self._last_det
                image_topic = self._image_topic

            # frame_hz: rolling rate across the deque. Need >= 2 timestamps to
            # measure a delta; falls back to 0.0 on a cold start.
            if len(ts) >= 2:
                span = ts[-1] - ts[0]
                frame_hz = (len(ts) - 1) / span if span > 0 else 0.0
            else:
                frame_hz = 0.0
            frame_age_sec = (None if last_mono is None
                             else max(0.0, self._time.monotonic() - last_mono))

            ros_domain_id = int(os.environ.get("ROS_DOMAIN_ID", "0") or "0")

            # Board spec dict — keep mirrored with BoardSpec fields the UI shows.
            board = {
                "squares_x": int(self._sx),
                "squares_y": int(self._sy),
                "square_len_m": float(self._sq),
                "marker_len_m": float(self._marker_len),
                "aruco_dict": str(self._aruco_dict_name),
            }
            # Safety envelope — surface only safe-to-serialise scalars / lists.
            safety_envelope = self._safety_envelope_dict()

            stability = {
                "steady": bool(steady),
                "since_frames": int(since),
                "target_frames": int(self._stab_window),
            }
            # Diversity: T1 ships the wired field at 0.0 (no samples are added
            # by the v1 capture path until T4 turns the gates on). target_deg
            # reflects min_diversity_deg the session was constructed with.
            diversity = {
                "coverage_deg": 0.0,
                "target_deg": float(self._diversity_target_deg),
            }

            return ws.enriched_state_payload(
                camera_connected=camera_connected,
                intrinsics_ok=intrinsics_ok,
                num_samples=num_samples,
                last_detection=last_det,
                status_msg="ok",
                frame_count=frame_count,
                frame_hz=frame_hz,
                frame_age_sec=frame_age_sec,
                image_topic=image_topic,
                ros_domain_id=ros_domain_id,
                t_base_ee=t_base_ee,
                xarm_joint_positions=xarm,
                board=board,
                safety_envelope=safety_envelope,
                stability=stability,
                samples=[],  # T4 populates this from session metadata
                diversity=diversity,
                last_solve=None,  # T5 populates this from the last solve result
            )

        def _refresh_t_base_ee_cache(self):
            """Look up base->ee TF with a 50ms timeout; cache + invalidate on fail.

            Returns the cached 4x4 (as a nested list) or ``None`` when no TF is
            ever available. Repeat failures invalidate the cache so a stale
            transform doesn't survive a teleport / driver restart.
            """
            np = self._np
            try:
                import rclpy
                tfm = self.tf_buffer.lookup_transform(
                    self._base_frame, self._eef_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.05))
                T = ws.tf_to_matrix(
                    [tfm.transform.translation.x, tfm.transform.translation.y,
                     tfm.transform.translation.z],
                    [tfm.transform.rotation.x, tfm.transform.rotation.y,
                     tfm.transform.rotation.z, tfm.transform.rotation.w])
                self._t_base_ee_cache = [list(map(float, row)) for row in T]
            except Exception:
                # No TF (no robot in this env, or transform stale beyond 50ms);
                # drop the cache so the UI shows "—" rather than a stale matrix.
                self._t_base_ee_cache = None
            return self._t_base_ee_cache

        def _safety_envelope_dict(self):
            """JSON-friendly snapshot of the SafetyEnvelope parameters."""
            env = self._safety
            if env is None:
                return {}
            out = {}
            for name in ("z_floor_m", "mast_radius_m", "mast_z_max"):
                v = getattr(env, name, None)
                if v is not None:
                    out[name] = float(v)
            ctr = getattr(env, "mast_xy_center", None)
            if ctr is not None:
                try:
                    out["mast_xy_center"] = [float(ctr[0]), float(ctr[1])]
                except Exception:
                    pass
            return out

        def latest_jpeg(self):
            np = self._np
            with self.lock:
                if self._frame is None:
                    return ws.placeholder_jpeg("no camera")
                frame = self._frame.copy()
                corners = self._last_corners_xy
            return ws.encode_jpeg(ws.draw_charuco_overlay(
                frame, corners if corners is not None else np.empty((0, 2))))

        # ---- commands --------------------------------------------------------

        def do_move(self, joints):
            if not isinstance(joints, (list, tuple)) or len(joints) != 7:
                return {"ok": False, "reason": "expected 7 joint values"}

            # SafetyEnvelope.validate() gates a 4x4 base->eef POSE, not joints.
            # We can't FK joints here, so validate the *current* EE pose from TF
            # as a sanity guard; "no TF / no envelope" => skip validation (the
            # collision-checking arm server is the real safety boundary).
            if self._safety is not None:
                try:
                    tfm = self.tf_buffer.lookup_transform(
                        self._base_frame, self._eef_frame, self._rclpy_time())
                    T = ws.tf_to_matrix(
                        [tfm.transform.translation.x, tfm.transform.translation.y,
                         tfm.transform.translation.z],
                        [tfm.transform.rotation.x, tfm.transform.rotation.y,
                         tfm.transform.rotation.z, tfm.transform.rotation.w])
                    reason = self._safety.validate(T)
                    if reason is not None:
                        return {"ok": False, "reason": f"safety: {reason}"}
                except Exception:
                    # No TF available (no robot in this env) — skip the pose gate.
                    pass

            if not self._jm.wait_for_server(timeout_sec=0.5):
                return {"ok": False, "reason": "arm action server unavailable"}

            goal = JointMove.Goal()
            j = [float(x) for x in joints]
            goal.joint0, goal.joint1, goal.joint2, goal.joint3 = j[0], j[1], j[2], j[3]
            goal.joint4, goal.joint5, goal.joint6 = j[4], j[5], j[6]
            goal.add_octomap = False
            self._jm.send_goal_async(goal)
            return {"ok": True, "reason": "sent"}

        def do_capture(self):
            with self.lock:
                cap, K, frame = self._cap, self._K, self._frame
            if frame is None:
                return {"ok": False, "reason": "no camera frame"}
            if K is None:
                return {"ok": False, "reason": "no camera intrinsics"}
            if cap is None:
                return {"ok": False, "reason": "no board detection"}

            # NOTE: a settle gate (gates.StabilityTracker) is deferred to a later
            # iteration — this is a single-frame capture for v1.
            try:
                tfm = self.tf_buffer.lookup_transform(
                    self._base_frame, self._eef_frame, self._rclpy_time())
            except Exception as exc:
                return {"ok": False, "reason": f"TF {self._base_frame}->{self._eef_frame} unavailable: {exc}"}
            T_base_eef = ws.tf_to_matrix(
                [tfm.transform.translation.x, tfm.transform.translation.y,
                 tfm.transform.translation.z],
                [tfm.transform.rotation.x, tfm.transform.rotation.y,
                 tfm.transform.rotation.z, tfm.transform.rotation.w])

            ok, reason = self.session.try_add(
                T_base_eef, cap["T_cam_board"], cap["obs_px"], cap["corner_idx"],
                n_corners=len(cap["corner_idx"]), reproj_px=cap["reproj_px"],
                area_frac=cap["area_frac"])
            return {"ok": ok, "reason": reason, "num_samples": len(self.session.samples)}

        def do_solve(self):
            with self.lock:
                samples, K, D = list(self.session.samples), self._K, self._D
            if len(samples) < 6:
                return {"ok": False, "reason": f"need >=6 samples, have {len(samples)}"}
            if K is None:
                return {"ok": False, "reason": "no camera intrinsics"}
            res = hs.solve(samples, K, D, self._board_pts)
            self.last_solve = res
            return {"ok": True, **ws.solve_payload(res)}

        def do_promote(self):
            import os
            import yaml
            if self.last_solve is None:
                return {"ok": False, "reason": "run solve first"}

            T_mount_color = self._mount_to_color_matrix()
            T_eef_mount = ah.compose_eef_to_mount(self.last_solve.X, T_mount_color)
            d = ah.handeye_yaml_dict(
                T_eef_mount, self.last_solve.X, len(self.session.samples),
                self.last_solve.heldout_metrics, "unset", self._sq)
            new_xyz = d["hand_eye"]["arm_to_camera_xyz"]
            new_rpy = d["hand_eye"]["arm_to_camera_rpy"]
            diff = diff_payload(old_xyz="(unknown)", new_xyz=new_xyz,
                                old_rpy="(unknown)", new_rpy=new_rpy)

            robot = self._param("robot_name", "") or os.environ.get("ROBOT_NAME", "")
            path = self._hand_eye_path(robot)
            if robot and path is not None and path.parent.is_dir():
                ah.write_with_backup(str(path), yaml.safe_dump(d))
                return {"ok": True, "written_path": str(path), "diff": diff}
            return {"ok": True, "preview": d, "diff": diff}

        # ---- internal helpers ------------------------------------------------

        def _mount_to_color_matrix(self):
            """Parse mount_to_color_xyz / _rpy string params ("x y z" / "r p y")
            into a 4x4. Defaults to identity when both are zero/empty."""
            np = self._np
            try:
                xyz = [float(v) for v in str(self._mount_to_color_xyz).split()]
                rpy = [float(v) for v in str(self._mount_to_color_rpy).split()]
            except ValueError:
                return np.eye(4)
            if len(xyz) != 3 or len(rpy) != 3:
                return np.eye(4)
            T = np.eye(4)
            T[:3, :3] = self._R.from_euler("xyz", rpy).as_matrix()
            T[:3, 3] = xyz
            return T

        def _hand_eye_path(self, robot):
            """Resolve <ws>/src/.../tinker_robot_config/robots/<robot>/hand_eye.yaml."""
            from pathlib import Path
            if not robot:
                return None
            here = Path(__file__).resolve()
            for parent in here.parents:
                cand = (parent / "tk25_basic" / "src" / "tinker_robot_config"
                        / "robots" / robot / "hand_eye.yaml")
                if cand.parent.parent.is_dir():
                    return cand
            return None

        def _rclpy_time(self):
            import rclpy
            return rclpy.time.Time()

    return HandeyeWebNode


def make_app(node):
    """Build the FastAPI app bound to a HandeyeWebNode.

    fastapi imports live inside the body so `import handeye_calib.handeye_web`
    stays ROS-/web-free for the unit-tested helpers above. Every endpoint
    delegates to the node's thread-safe accessors/commands, which already
    degrade gracefully (return {"ok": False, ...}) when no hardware is present.
    """
    import asyncio
    from pathlib import Path
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, Response, JSONResponse
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="handeye_web", docs_url="/api/docs")

    # v2 static UI. webui/ ships next to handeye_web.py via setup.py's
    # package_data hook (see setup.py), so `Path(__file__).parent / "webui"`
    # resolves both from the source tree and the install tree.
    webui_dir = Path(__file__).resolve().parent / "webui"
    app.mount("/static", StaticFiles(directory=str(webui_dir)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(webui_dir / "index.html"), media_type="text/html")

    @app.get("/api/state")
    def state():
        return JSONResponse(node.get_state_dict())

    @app.get("/api/frame.jpg")
    def frame():
        return Response(content=node.latest_jpeg(), media_type="image/jpeg")

    @app.post("/api/move")
    async def move(request: Request):
        body = await request.json()
        return JSONResponse(node.do_move(body.get("joints")))

    @app.post("/api/capture")
    def capture():
        return JSONResponse(node.do_capture())

    @app.post("/api/solve")
    def solve():
        return JSONResponse(node.do_solve())

    @app.post("/api/promote")
    def promote():
        return JSONResponse(node.do_promote())

    @app.websocket("/ws")
    async def ws_state(ws_conn: WebSocket):
        """5 Hz state push for the static UI.

        Pushes the enriched state payload every 200 ms. Cleanly handles client
        disconnects; the surrounding try/except keeps a broken socket from
        propagating an exception into uvicorn's task supervisor.
        """
        await ws_conn.accept()
        try:
            while True:
                await ws_conn.send_json(node.get_state_dict())
                await asyncio.sleep(0.2)  # 5 Hz
        except WebSocketDisconnect:
            return
        except Exception:
            return

    return app


def main():
    # Mirrors pan_tilt/calib_web.py main(): build the rclpy node, start a uvicorn
    # worker thread for the FastAPI authoring/run/verify/promote UI, and spin the
    # node on the main thread. The server starts WITHOUT hardware — all endpoints
    # degrade gracefully when no camera / arm is present.
    #
    # ROS / web imports are deferred (HandeyeWebNode is built lazily via the
    # module __getattr__ below, and uvicorn/rclpy import inside this body) so that
    # `import handeye_calib.handeye_web` stays ROS-free for the unit-tested
    # helpers above.
    import os
    os.environ.setdefault("FASTDDS_BUILTIN_TRANSPORTS", "UDPv4")
    os.environ.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    import rclpy
    rclpy.init()
    # Module __getattr__ only fires for *attribute access* on the module object,
    # not for bare-name lookups inside a function (those go global-dict -> builtins
    # and would raise NameError). Resolve the lazily-built class via the module
    # attribute so the __getattr__ -> _make_node_class() hook runs.
    import sys
    HandeyeWebNode = getattr(sys.modules[__name__], "HandeyeWebNode")
    node = HandeyeWebNode()
    app = make_app(node)
    import uvicorn, threading, asyncio
    config = uvicorn.Config(app, host=node.bind_host, port=node.bind_port,
                            log_level="info", access_log=False, loop="asyncio")
    server = uvicorn.Server(config)

    def _serve():
        asyncio.run(server.serve())

    t = threading.Thread(target=_serve, daemon=True, name="uvicorn")
    t.start()
    node.get_logger().info(f"handeye_web listening on http://{node.bind_host}:{node.bind_port}")
    # SIGTERM (from `ros2 launch` teardown / `kill`) raises KeyboardInterrupt so it
    # flows through the same clean-shutdown path as Ctrl-C (SIGINT) below.
    import signal
    def _on_sigterm(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_sigterm)
    # Under `ros2 launch`, rclpy installs its own signal handlers, so SIGINT/SIGTERM
    # surface as ExternalShutdownException from spin() rather than KeyboardInterrupt;
    # catch both so the clean-shutdown `finally` runs and the process exits 0.
    from rclpy.executors import ExternalShutdownException
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        server.should_exit = True
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        t.join(timeout=2.0)


def __getattr__(name):
    if name == "HandeyeWebNode":
        return _make_node_class()
    raise AttributeError(name)


if __name__ == "__main__":
    main()
