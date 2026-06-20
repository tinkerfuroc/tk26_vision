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
    from handeye_calib import waypoints as hwp
    from handeye_calib.waypoints import WaypointStore
    # Detection + safety reuse from pan_tilt. aruco_detect's Detection does NOT
    # surface per-corner charuco pixels/IDs (only pose + corner count + reproj
    # RMS), so for capture we call cv2.aruco.CharucoDetector.detectBoard directly
    # to get charucoCorners/charucoIds, then board.matchImagePoints + solvePnP for
    # the board pose, observed pixels, IDs, and a scalar reprojection error.
    from pan_tilt.calibration.aruco_detect import BoardSpec, build_board, build_detector
    from pan_tilt.calibration.safety import SafetyEnvelope

    def _downscale(bgr, target_w: int = 960, cv2_mod=cv2):
        """Downsize a BGR frame to ``target_w`` on the long edge for bandwidth.

        Mirrors ``pan_tilt/pan_tilt/calib_web.py::_downscale`` byte-for-byte —
        keep the two in sync if either tool's bandwidth budget changes. No-ops
        when the image is already <= ``target_w`` wide.
        """
        h, w = bgr.shape[:2]
        if w <= target_w:
            return bgr
        scale = target_w / w
        return cv2_mod.resize(bgr, (target_w, int(h * scale)))

    class CaptureSequenceRunner:
        """T3-seq: auto-capture state machine — move → settle → capture loop.

        Owns its own daemon thread + stop event + state dict + log deque. All
        moves go through ``self.node.do_move`` and all captures through
        ``self.node.do_capture`` — the runner is a pure coordinator and never
        reimplements the SafetyEnvelope check or the StabilityTracker hard
        gate.

        Threading model
        ---------------
        ``HandeyeWebNode.main()`` spins the node on the main thread (single-
        threaded). ``send_goal_async`` / ``get_result_async`` futures resolve
        on that spin thread independent of this runner's daemon thread, which
        mirrors the proven pattern in ``pan_tilt/calib_web.py::_run_action``:
        the runner uses a ``time.sleep(0.05)`` polling loop on the futures so
        it never blocks rclpy's executor and the stop event is observed at
        every poll tick.

        Cancel semantics
        ----------------
        ``cancel()`` sets the stop event AND (best-effort) cancels the
        in-flight ``JointMove`` goal handle via ``cancel_goal_async`` — both
        steps fire, in that order, so the arm doesn't keep completing a
        now-stale target after the operator hits Cancel. The runner thread
        then exits at its next stop-check tick.
        """

        STEP_IDLE = "idle"
        STEP_STARTING = "starting"
        STEP_MOVING = "moving"
        STEP_SETTLING = "settling"
        STEP_CAPTURING = "capturing"
        STEP_DONE = "done"
        STEP_CANCELLED = "cancelled"
        STEP_ERROR = "error"

        # Total budget for a single waypoint move (goal accept + execution).
        MOVE_DEADLINE_S = 30.0
        # Wait for goal acceptance before declaring "send timed out".
        ACCEPT_DEADLINE_S = 5.0
        # Settle poll period (10 Hz per the brief).
        SETTLE_POLL_S = 0.1
        # Need this many consecutive steady ticks before we call it settled.
        SETTLE_STEADY_TICKS = 3

        def __init__(self, node):
            self.node = node
            self._stop = threading.Event()
            self._lock = threading.Lock()
            self._thread = None
            self._inflight_handle = None  # latest JointMove goal handle
            self._state = {
                "running": False,
                "dry_run": False,
                "current_idx": None,
                "total": 0,
                "current_step": self.STEP_IDLE,
            }
            self._log = collections.deque(maxlen=20)

        # ---- public API --------------------------------------------------

        def state_dict(self):
            """Snapshot of runner state for the WS push (copied under lock)."""
            with self._lock:
                return {**self._state, "log": list(self._log)}

        def start(self, dry_run: bool = False, settle_timeout_s: float = 5.0):
            """Spawn the daemon thread that runs the move/settle/capture loop.

            Returns ``{ok: True}`` immediately; the loop body runs on the
            spawned thread. Refuses with ``{ok: False, reason: ...}`` if a
            prior run is still live or if the waypoint store is empty (the
            latter is also guarded one level up in ``do_start_sequence``).
            """
            with self._lock:
                if self._state["running"]:
                    return {"ok": False, "reason": "sequence already running"}
                with self.node.lock:
                    wps = self.node.waypoint_store.list()
                if not wps:
                    return {"ok": False, "reason": "no waypoints recorded"}
                self._stop.clear()
                self._state.update({
                    "running": True,
                    "dry_run": bool(dry_run),
                    "current_idx": None,
                    "total": len(wps),
                    "current_step": self.STEP_STARTING,
                })
                self._log.clear()
                self._append_log_locked(
                    f"starting sequence ({len(wps)} waypoints, "
                    f"dry_run={bool(dry_run)})")
            self._thread = threading.Thread(
                target=self._run,
                args=(wps, bool(dry_run), float(settle_timeout_s)),
                daemon=True, name="capture-sequence")
            self._thread.start()
            return {"ok": True}

        def cancel(self):
            """Request shutdown: cancel in-flight goal FIRST, then set stop event.

            The order matters — set the stop flag alone and the arm completes
            its current goal before the runner notices. Calling cancel on a
            done/idle runner is a harmless no-op (idempotent)."""
            # Best-effort goal cancel — fire first so the arm doesn't keep
            # moving toward the now-stale target while the loop spins.
            handle = self._inflight_handle
            if handle is not None:
                try:
                    handle.cancel_goal_async()
                except Exception:
                    pass
            self._stop.set()
            self._append_log("cancel requested")
            return {"ok": True}

        # ---- internals ---------------------------------------------------

        def _set_state(self, **kwargs):
            with self._lock:
                self._state.update(kwargs)

        def _append_log(self, msg):
            with self._lock:
                self._append_log_locked(msg)

        def _append_log_locked(self, msg):
            ts = time.strftime("%H:%M:%S")
            self._log.append(f"[{ts}] {msg}")

        def _await_future(self, future, deadline_s: float) -> bool:
            """Poll-wait for ``future`` to complete, respecting the stop event.

            Returns True when ``future.done()``; False on timeout or on stop.
            The rclpy executor runs on the main spin thread, so we just sleep
            briefly between checks instead of calling
            ``spin_until_future_complete`` (which would re-enter the executor
            and deadlock here — mirrors the calib_web ``_run_action`` pattern).
            """
            t0 = time.monotonic()
            while not future.done():
                if self._stop.is_set():
                    return False
                if time.monotonic() - t0 >= deadline_s:
                    return False
                time.sleep(0.05)
            return True

        def _do_move_wait(self, joints):
            """Send a JointMove goal and wait for completion.

            Returns ``{ok: bool, reason: str}``. Uses the same goal shape
            ``do_move`` does (so we inherit the field name contract); the
            SafetyEnvelope pre-check is reused via ``self.node.do_move`` ONLY
            for its parameter validation — we then construct + send our own
            goal so we can capture the goal handle for cancellation.
            """
            jm = self.node._jm
            if not jm.wait_for_server(timeout_sec=0.5):
                return {"ok": False, "reason": "arm action server unavailable"}
            # SafetyEnvelope pre-check on current EE pose (best-effort,
            # mirrors do_move; skipped silently when no TF / no envelope).
            if self.node._safety is not None:
                try:
                    tfm = self.node.tf_buffer.lookup_transform(
                        self.node._base_frame, self.node._eef_frame,
                        self.node._rclpy_time())
                    T = ws.tf_to_matrix(
                        [tfm.transform.translation.x,
                         tfm.transform.translation.y,
                         tfm.transform.translation.z],
                        [tfm.transform.rotation.x,
                         tfm.transform.rotation.y,
                         tfm.transform.rotation.z,
                         tfm.transform.rotation.w])
                    reason = self.node._safety.validate(T)
                    if reason is not None:
                        return {"ok": False, "reason": f"safety: {reason}"}
                except Exception:
                    pass
            goal = JointMove.Goal()
            j = [float(x) for x in joints]
            goal.joint0, goal.joint1, goal.joint2, goal.joint3 = j[0], j[1], j[2], j[3]
            goal.joint4, goal.joint5, goal.joint6 = j[4], j[5], j[6]
            goal.add_octomap = False
            send_fut = jm.send_goal_async(goal)
            if not self._await_future(send_fut, self.ACCEPT_DEADLINE_S):
                if self._stop.is_set():
                    return {"ok": False, "reason": "cancelled before goal acceptance"}
                return {"ok": False, "reason": "send_goal timed out"}
            goal_handle = send_fut.result()
            if goal_handle is None or not getattr(goal_handle, "accepted", False):
                return {"ok": False, "reason": "goal rejected"}
            self._inflight_handle = goal_handle
            try:
                result_fut = goal_handle.get_result_async()
                if not self._await_future(result_fut, self.MOVE_DEADLINE_S):
                    if self._stop.is_set():
                        # Best-effort cancel — cancel() already fired its
                        # cancel_goal_async, but re-issue here defensively
                        # in case the stop event was set externally.
                        try:
                            goal_handle.cancel_goal_async()
                        except Exception:
                            pass
                        return {"ok": False, "reason": "cancelled mid-move"}
                    try:
                        goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                    return {"ok": False,
                            "reason": f"move timed out after {self.MOVE_DEADLINE_S:.0f}s"}
                wrapped = result_fut.result()
                result = getattr(wrapped, "result", wrapped)
                ok = bool(getattr(result, "success", False))
                return {"ok": ok,
                        "reason": "ok" if ok else "arm reported success=False"}
            finally:
                self._inflight_handle = None

        def _wait_for_settle(self, settle_timeout_s: float) -> bool:
            """Poll the cached StabilityTracker verdict at 10 Hz.

            Returns True when steady for ``SETTLE_STEADY_TICKS`` consecutive
            ticks; False on timeout or on stop. Reuses ``_stability_steady``
            written by ``_on_image`` — no fresh threshold tuning here per the
            brief (T4 owns those thresholds)."""
            t0 = time.monotonic()
            consec = 0
            while True:
                if self._stop.is_set():
                    return False
                if time.monotonic() - t0 >= settle_timeout_s:
                    return False
                with self.node.lock:
                    steady = self.node._stability_steady
                if steady:
                    consec += 1
                    if consec >= self.SETTLE_STEADY_TICKS:
                        return True
                else:
                    consec = 0
                time.sleep(self.SETTLE_POLL_S)

        def _run(self, waypoints, dry_run: bool, settle_timeout_s: float):
            """The state-machine body. Runs on a daemon thread.

            Iteration: ``moving`` → ``settling`` → ``capturing`` (skipped on
            dry-run). Per-step failures (move failed, settle timeout) log +
            ``continue`` to the next waypoint rather than abort the whole
            sequence; an unhandled exception transitions to ``error``."""
            try:
                for idx, wp in enumerate(waypoints):
                    if self._stop.is_set():
                        break
                    self._set_state(current_idx=idx, current_step=self.STEP_MOVING)
                    self._append_log(
                        f"#{idx}: moving to "
                        f"[{wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}, ...]")
                    move_res = self._do_move_wait(wp)
                    if self._stop.is_set():
                        break
                    if not move_res["ok"]:
                        self._append_log(
                            f"#{idx}: move failed ({move_res['reason']}); skipping")
                        continue

                    self._set_state(current_step=self.STEP_SETTLING)
                    self._append_log(f"#{idx}: settling …")
                    settled = self._wait_for_settle(settle_timeout_s)
                    if self._stop.is_set():
                        break
                    if not settled:
                        self._append_log(
                            f"#{idx}: settle timeout after {settle_timeout_s:.1f}s; skipping")
                        continue

                    if dry_run:
                        self._append_log(
                            f"#{idx}: dry-run — settled but skipping capture")
                        continue

                    self._set_state(current_step=self.STEP_CAPTURING)
                    cap = self.node.do_capture()
                    if cap.get("ok"):
                        self._append_log(f"#{idx}: captured")
                    else:
                        self._append_log(
                            f"#{idx}: capture skipped ({cap.get('reason', '?')})")
                final_step = (self.STEP_CANCELLED if self._stop.is_set()
                              else self.STEP_DONE)
                self._set_state(running=False, current_step=final_step)
                self._append_log(f"sequence {final_step}")
            except Exception as exc:  # pragma: no cover - defensive
                self._set_state(running=False, current_step=self.STEP_ERROR)
                self._append_log(f"runner crashed: {exc!r}")

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

            # T4: per-sample sidecars (kept parallel to ``self.session.samples``).
            # ``_thumbs[idx]`` is a downscaled (~320 px wide) JPEG of the
            # frame+overlay at capture time, served via /api/samples/{idx}/thumb.jpg.
            # ``_sample_joints[idx]`` is the xArm joint snapshot at capture time
            # (or None when /joint_states hasn't arrived yet).
            # ``_sample_ts[idx]`` is a monotonic timestamp for the gallery row.
            # All three are recompacted on delete so the keys stay 0..N-1.
            self._thumbs = {}
            self._sample_joints = {}
            self._sample_ts = {}
            # reproj_px + area_frac are scalars from the per-frame detection;
            # the Sample dataclass doesn't keep them. Stash here for the
            # gallery row so the operator can see quality at a glance.
            self._sample_reproj_px = {}
            self._sample_area_frac = {}

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

            self._image_topic = self._param("color_image_topic", "/camera/xarm_camera/color/image_raw")
            self.create_subscription(
                Image, self._image_topic,
                self._on_image, qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, self._param("camera_info_topic", "/camera/xarm_camera/color/camera_info"),
                self._on_info, qos_profile_sensor_data)
            self._jm = ActionClient(self, JointMove, self._param("jointmove_action", "/xarm/joint_move"))

            self._np = np
            self._cv2 = cv2
            self._R = _R

            # Waypoint store — best-effort load on startup.
            self.waypoint_store = WaypointStore()
            try:
                result = self.do_reload_waypoints()
                if result["ok"]:
                    self.get_logger().info(
                        f"loaded {result['count']} waypoints from {result.get('path', '?')}")
            except Exception as exc:
                self.get_logger().warn(f"waypoint startup load skipped: {exc}")

            # T3-seq: auto-capture state machine. Lazily constructed on the
            # first ``do_start_sequence`` so the idle state.sequence stays the
            # canonical SEQUENCE_IDLE_DEFAULT (no leftover state from a prior
            # run) and node construction stays cheap.
            self.sequence_runner = None

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
                samples_snapshot = list(self.session.samples)
                num_samples = len(samples_snapshot)
                last_det = self._last_det
                image_topic = self._image_topic
                # T4: paired sidecars at snapshot time so per-sample metadata
                # stays consistent with the samples list under concurrent ops.
                joints_by_idx = dict(self._sample_joints)
                ts_by_idx = dict(self._sample_ts)
                reproj_by_idx = dict(self._sample_reproj_px)
                area_by_idx = dict(self._sample_area_frac)

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
            # T4: diversity = max pairwise rotation between any two accepted
            # T_base_eef rotations across the session. target_deg reflects
            # min_diversity_deg the session was constructed with.
            diversity = {
                "coverage_deg": float(ws.compute_diversity_deg(samples_snapshot)),
                "target_deg": float(self._diversity_target_deg),
            }

            # T4: per-sample metadata list for the Capture-tab gallery. Each
            # entry mirrors web_support.sample_metadata's keys; the n_corners/
            # reproj_px/area_frac fields aren't stored on Sample so we re-derive
            # them from the array shapes / cap dict where possible.
            samples_metadata = []
            for i, s in enumerate(samples_snapshot):
                prev = samples_snapshot[i - 1] if i > 0 else None
                samples_metadata.append(ws.sample_metadata(
                    i, s, prev_sample=prev,
                    n_corners=int(len(s.corner_idx)),
                    reproj_px=reproj_by_idx.get(i),
                    area_frac=area_by_idx.get(i),
                    joint_positions=joints_by_idx.get(i),
                    ts=ts_by_idx.get(i),
                ))

            # T3: server-evaluated SafetyEnvelope check against the cached EE
            # pose, surfaced as state.safety_preview so the Move tab doesn't
            # have to duplicate the math in JS.
            safety_preview = self.safety_preview()

            with self.lock:
                waypoints_list = self.waypoint_store.list()
            waypoints_meta = [ws.waypoint_metadata(i, w)
                              for i, w in enumerate(waypoints_list)]

            # T3-seq: auto-capture state. Idle default when no runner has
            # been instantiated yet so the UI sees a stable shape from t=0.
            sequence_state = (self.sequence_runner.state_dict()
                              if self.sequence_runner is not None
                              else dict(ws.SEQUENCE_IDLE_DEFAULT))

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
                samples=samples_metadata,
                diversity=diversity,
                last_solve=None,  # T5 populates this from the last solve result
                safety_preview=safety_preview,
                waypoints=waypoints_meta,
                sequence=sequence_state,
            )

        def safety_preview(self):
            """Live SafetyEnvelope verdict on the cached base->ee pose.

            Returns ``{"safe": bool|None, "detail": str}``. ``safe`` is:
              * ``True``  if the cached pose passes ``SafetyEnvelope.validate``;
              * ``False`` if it fails (``detail`` carries the rejection reason);
              * ``None``  when we can't decide — no cached TF, or no envelope
                instance (``SafetyEnvelope`` construction failed at init).

            This runs off the cached ``_t_base_ee_cache`` (refreshed on every
            WS push by :func:`get_state_dict`) so the WS push stays cheap; it
            never blocks on TF here. The UI reads this verbatim into the Move
            tab's ``#move-safety-status`` line — same wording as the pan_tilt
            ``evaluateSafetyEnvelope`` JS helper so the two tools speak the
            same language.
            """
            np = self._np
            env = self._safety
            cache = self._t_base_ee_cache
            if env is None:
                return {"safe": None, "detail": "safety envelope unavailable"}
            if cache is None:
                return {"safe": None, "detail": "TF unavailable"}
            try:
                T = np.asarray(cache, float)
                reason = env.validate(T)
            except Exception as exc:  # pragma: no cover - permissive guard
                return {"safe": None, "detail": f"safety check error: {exc}"}
            if reason is None:
                z = float(T[2, 3])
                dx = float(T[0, 3] - env.mast_xy_center[0])
                dy = float(T[1, 3] - env.mast_xy_center[1])
                import math
                r = math.hypot(dx, dy)
                return {
                    "safe": True,
                    "detail": f"safe (z={z:.3f} m, r_mast={r:.3f} m)",
                }
            return {"safe": False, "detail": f"VIOLATION: {reason}"}

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

        def latest_jpeg(self, raw: bool = False):
            """Latest camera frame encoded as JPEG.

            ``raw=False`` (default): annotated overlay — green corner dots,
            integer corner IDs, header bar with RMS + image_topic. ``raw=True``:
            the raw BGR frame, no annotations. Always downscaled to 960 px wide
            before encoding (mirrors pan_tilt's ``_downscale``) so the UI's
            ~3 Hz polling stays cheap. Placeholder JPEG returned when no frame
            has arrived yet.
            """
            np = self._np
            cv2 = self._cv2
            with self.lock:
                if self._frame is None:
                    return ws.placeholder_jpeg("no camera")
                frame = self._frame.copy()
                corners = self._last_corners_xy
                last_det = self._last_det
                cap = self._cap
                image_topic = self._image_topic

            if raw:
                return ws.encode_jpeg(_downscale(frame, 960, cv2))

            rms_px = None
            if isinstance(last_det, dict):
                rms_px = last_det.get("reproj_px")
            ids = cap["corner_idx"] if (cap is not None and "corner_idx" in cap) else None
            annotated = ws.draw_charuco_overlay(
                frame,
                corners if corners is not None else np.empty((0, 2)),
                ids=ids,
                rms_px=rms_px,
                image_topic=image_topic,
            )
            return ws.encode_jpeg(_downscale(annotated, 960, cv2))

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

        def is_steady(self):
            """T4: latest StabilityTracker verdict (cached by ``_on_image``)."""
            with self.lock:
                return bool(self._stability_steady)

        def do_capture(self):
            # T4 HARD GATE: must run BEFORE the v1 "no camera / no K / no cap"
            # branches so the operator sees a settle rejection even when a
            # board pose is in fact available but not yet steady. This closes
            # the v1 deferral noted in the prior comment.
            with self.lock:
                steady = self._stability_steady
                since = self._stability_since_frames
                target = self._stab_window
                cap, K, frame = self._cap, self._K, self._frame
                joints_snapshot = (None if self._xarm_joint_positions is None
                                   else list(self._xarm_joint_positions))
                now_mono = self._last_frame_monotonic
            if not steady:
                return {
                    "ok": False,
                    "reason": (f"not stable yet ({int(since)}/{int(target)} "
                               f"steady frames)"),
                    "num_samples": len(self.session.samples),
                }
            if frame is None:
                return {"ok": False, "reason": "no camera frame",
                        "num_samples": len(self.session.samples)}
            if K is None:
                return {"ok": False, "reason": "no camera intrinsics",
                        "num_samples": len(self.session.samples)}
            if cap is None:
                return {"ok": False, "reason": "no board detection",
                        "num_samples": len(self.session.samples)}

            try:
                tfm = self.tf_buffer.lookup_transform(
                    self._base_frame, self._eef_frame, self._rclpy_time())
            except Exception as exc:
                return {"ok": False,
                        "reason": (f"TF {self._base_frame}->{self._eef_frame}"
                                   f" unavailable: {exc}"),
                        "num_samples": len(self.session.samples)}
            T_base_eef = ws.tf_to_matrix(
                [tfm.transform.translation.x, tfm.transform.translation.y,
                 tfm.transform.translation.z],
                [tfm.transform.rotation.x, tfm.transform.rotation.y,
                 tfm.transform.rotation.z, tfm.transform.rotation.w])

            ok, reason = self.session.try_add(
                T_base_eef, cap["T_cam_board"], cap["obs_px"], cap["corner_idx"],
                n_corners=len(cap["corner_idx"]), reproj_px=cap["reproj_px"],
                area_frac=cap["area_frac"])
            if ok:
                # Cache a downscaled JPEG of the overlayed frame + the joint
                # snapshot for the new sample. Index matches session.samples[-1].
                idx = len(self.session.samples) - 1
                try:
                    annotated = ws.draw_charuco_overlay(
                        frame, cap["obs_px"], ids=cap["corner_idx"],
                        rms_px=cap["reproj_px"], image_topic=self._image_topic)
                    jpg = ws.encode_jpeg(_downscale(annotated, 320, self._cv2))
                except Exception as exc:  # pragma: no cover - thumb is best-effort
                    self.get_logger().warn(
                        f"thumb encode failed for sample {idx} ({exc})")
                    jpg = None
                with self.lock:
                    if jpg is not None:
                        self._thumbs[idx] = jpg
                    self._sample_joints[idx] = joints_snapshot
                    self._sample_ts[idx] = (float(now_mono) if now_mono is not None
                                            else float(self._time.monotonic()))
                    self._sample_reproj_px[idx] = float(cap["reproj_px"])
                    self._sample_area_frac[idx] = float(cap["area_frac"])
            return {"ok": ok, "reason": reason,
                    "num_samples": len(self.session.samples)}

        def do_delete_sample(self, idx):
            """Pop sample idx + recompact thumbnail/joint sidecars.

            ``num_samples`` is returned even on failure so the UI can refresh
            its counter unconditionally. Recompacts the sidecar dicts so all
            keys stay contiguous 0..N-1 after the pop — the gallery never has
            to handle "holes" in the index space.
            """
            with self.lock:
                samples = self.session.samples
                if not isinstance(idx, int):
                    try:
                        idx = int(idx)
                    except (TypeError, ValueError):
                        return {"ok": False,
                                "reason": f"bad idx {idx!r}",
                                "num_samples": len(samples)}
                if idx < 0 or idx >= len(samples):
                    return {"ok": False,
                            "reason": (f"idx {idx} out of range "
                                       f"(0..{len(samples) - 1})"),
                            "num_samples": len(samples)}
                samples.pop(idx)
                # Recompact thumbnails + joint snapshots + ts: shift down
                # every key > idx, drop the popped one.
                self._thumbs = {
                    (k if k < idx else k - 1): v
                    for k, v in self._thumbs.items() if k != idx
                }
                self._sample_joints = {
                    (k if k < idx else k - 1): v
                    for k, v in self._sample_joints.items() if k != idx
                }
                self._sample_ts = {
                    (k if k < idx else k - 1): v
                    for k, v in self._sample_ts.items() if k != idx
                }
                self._sample_reproj_px = {
                    (k if k < idx else k - 1): v
                    for k, v in self._sample_reproj_px.items() if k != idx
                }
                self._sample_area_frac = {
                    (k if k < idx else k - 1): v
                    for k, v in self._sample_area_frac.items() if k != idx
                }
                num = len(samples)
            return {"ok": True, "num_samples": num}

        def sample_thumb(self, idx):
            """Return the cached JPEG bytes for sample ``idx`` or ``None``."""
            with self.lock:
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    return None
                return self._thumbs.get(idx)

        def do_solve(self, method: str = "auto"):
            """Run the hand-eye solve with optional method picker.

            ``method`` is one of ``"auto"`` (default — sweep all five OpenCV
            methods and pick the lowest reproj RMS) or one of the canonical
            method names: ``TSAI`` / ``PARK`` / ``HORAUD`` / ``ANDREFF`` /
            ``DANIILIDIS``. Unknown values fall back to ``auto`` (the brief's
            preferred-over-error stance, mirroring the rest of this node's
            degrade-rather-than-422 philosophy). On success returns
            ``{"ok": True, **solve_payload_v2(...)}``; on degraded paths returns
            ``{"ok": False, "reason": ...}``.
            """
            with self.lock:
                samples, K, D = list(self.session.samples), self._K, self._D
            if len(samples) < 6:
                return {"ok": False, "reason": f"need >=6 samples, have {len(samples)}"}
            if K is None:
                return {"ok": False, "reason": "no camera intrinsics"}
            method_str = str(method or "auto").strip()
            if method_str.lower() == "auto":
                methods_subset = None
            elif method_str in hs._METHODS:
                methods_subset = {method_str: hs._METHODS[method_str]}
            else:
                # Unknown method: degrade to auto rather than 422; same
                # philosophy as do_capture's "no detection" fallthrough.
                methods_subset = None
            res = hs.solve(samples, K, D, self._board_pts, methods=methods_subset)
            self.last_solve = res
            return {"ok": True, **ws.solve_payload_v2(res, samples, K, D, self._board_pts)}

        # ---- T1-wp: waypoint CRUD + per-robot YAML persistence ---------------

        def do_add_waypoint(self):
            """Append the current xArm joint positions as a new waypoint.

            Returns ``{ok: False, reason: "no current joints"}`` when
            ``_xarm_joint_positions`` is not yet cached (no /joint_states).
            Returns ``{ok: True, count: N}`` on success.
            """
            with self.lock:
                joints = (None if self._xarm_joint_positions is None
                          else list(self._xarm_joint_positions))
            if joints is None:
                return {"ok": False, "reason": "no current joints — /joint_states not received yet"}
            with self.lock:
                idx = self.waypoint_store.add(joints)
                count = len(self.waypoint_store.list())
            return {"ok": True, "count": count, "idx": idx}

        def do_delete_waypoint(self, idx: int):
            """Delete waypoint at ``idx``.

            Returns ``{ok: True|False, count: N, reason?}``.
            """
            with self.lock:
                ok = self.waypoint_store.delete(int(idx))
                count = len(self.waypoint_store.list())
            if ok:
                return {"ok": True, "count": count}
            return {"ok": False, "count": count,
                    "reason": f"idx {idx} out of range (store has {count} waypoints)"}

        def do_save_waypoints(self):
            """Persist the in-memory waypoint list to the per-robot YAML.

            Refuses (mirrors T6 promote pattern) when ``ROBOT_NAME`` is unset.
            Returns ``{ok: True, path: str, count: N}`` on success.
            """
            robot = self._resolve_robot_name()
            if not robot:
                return {"ok": False,
                        "reason": "ROBOT_NAME unset — cannot save per-robot waypoints"}
            basic_root = self._tk25_basic_repo_root()
            if basic_root is None:
                return {"ok": False,
                        "reason": "tk25_basic repo root not resolvable — cannot save waypoints"}
            from pathlib import Path
            path = hwp.resolve_waypoints_path(robot, basic_root)
            with self.lock:
                store_snapshot = self.waypoint_store
                count = len(store_snapshot.list())
                try:
                    store_snapshot.save_yaml(path, recorded_for_robot=robot)
                except Exception as exc:
                    return {"ok": False, "reason": f"save failed: {exc}"}
            return {"ok": True, "path": str(path), "count": count}

        def do_reload_waypoints(self):
            """Re-read the per-robot YAML; replaces the in-memory list.

            Returns ``{ok: True, count: N, path: str}`` on success,
            ``{ok: False, reason: str}`` on failure (silently logged on startup).
            """
            robot = self._resolve_robot_name()
            if not robot:
                return {"ok": False,
                        "reason": "ROBOT_NAME unset — no per-robot waypoints to load"}
            basic_root = self._tk25_basic_repo_root()
            if basic_root is None:
                return {"ok": False,
                        "reason": "tk25_basic repo root not resolvable"}
            from pathlib import Path
            path = hwp.resolve_waypoints_path(robot, basic_root)
            if not path.exists():
                return {"ok": False,
                        "reason": f"no waypoints file at {path}"}
            with self.lock:
                try:
                    count = self.waypoint_store.load_yaml(path)
                except Exception as exc:
                    return {"ok": False, "reason": f"load failed: {exc}"}
            return {"ok": True, "count": count, "path": str(path)}

        def do_promote(self):
            """Back-compat shim — POST /api/promote = apply_promote(which='both').

            Older callers (tests, scripts) that POST'd ``/api/promote`` get the
            new ``apply_promote`` shape; the brief's split into
            ``/api/promote/diff`` + ``/api/promote/apply`` is the canonical
            path going forward.
            """
            return self.apply_promote(which="both")

        # ---- T3-seq: auto-capture state machine ------------------------------

        def do_start_sequence(self, dry_run: bool = False,
                              settle_timeout_s: float = 5.0):
            """Kick off the auto-capture state machine.

            Refuses on empty waypoint list with ``{ok: False, reason: "no
            waypoints recorded"}``. Otherwise lazily (re)constructs the
            runner so prior state can't leak across runs, then delegates to
            ``runner.start``. Returns immediately; the runner spins on its
            own daemon thread.
            """
            with self.lock:
                wp_count = len(self.waypoint_store.list())
            if wp_count == 0:
                return {"ok": False, "reason": "no waypoints recorded"}
            # If a prior runner is still alive (running), refuse with the
            # runner's own reason; otherwise rebuild fresh.
            existing = self.sequence_runner
            if existing is not None and existing.state_dict().get("running"):
                return {"ok": False, "reason": "sequence already running"}
            self.sequence_runner = CaptureSequenceRunner(self)
            return self.sequence_runner.start(
                dry_run=bool(dry_run),
                settle_timeout_s=float(settle_timeout_s),
            )

        def do_cancel_sequence(self):
            """Cancel the in-flight sequence (idempotent no-op if none running).

            Always returns ``{ok: True}``; ``CaptureSequenceRunner.cancel`` is
            itself idempotent (set-event semantics + best-effort goal cancel).
            """
            runner = self.sequence_runner
            if runner is None:
                return {"ok": True}
            return runner.cancel()

        # ---- T6: per-robot xacro override + yaml-diff ------------------------

        def _resolve_robot_name(self):
            """ROBOT_NAME env var wins; falls back to the ``robot_name`` param.

            Empty string => unset (UI shows the yaml-only banner). The env-var
            precedence mirrors the existing ``_param("robot_name", os.environ
            .get("ROBOT_NAME", ""))`` init pattern; we re-read both at call
            time so the operator can ``export ROBOT_NAME=…`` and reload
            without re-launching.
            """
            import os
            return (os.environ.get("ROBOT_NAME", "")
                    or self._param("robot_name", "") or "")

        def _tk25_basic_repo_root(self):
            """Locate the ``tk25_basic`` ROS package on disk.

            Resolution order (first hit wins):

            1. Walk parents of THIS file, checking both ``<parent>/tk25_basic``
               and ``<parent>/src/tk25_basic`` at each level. Covers source-tree
               runs (handeye_web.py at ``src/tk26_vision/src/handeye_calib/...``)
               where ``src/`` is a parent and ``src/tk25_basic`` sits beside us.
            2. Walk parents of CWD with the same two prefixes. Covers
               ``ros2 launch`` invoked from the workspace root.
            3. Use ``ament_index_python`` to find the install share of
               ``tinker_robot_config``, then walk up to ``install/``'s parent
               (the workspace root) and check ``src/tk25_basic/...``. Covers
               install-tree runs where this file lives in
               ``install/handeye_calib/lib/.../site-packages/handeye_calib/``
               — no parent of which contains ``tk25_basic`` directly.

            Returns the ``Path`` to ``tk25_basic`` (the ROS package root, NOT
            its inner ``src/``) or ``None`` if every resolver fails.
            """
            from pathlib import Path

            def _check(parent):
                for prefix in ("", "src"):
                    base = parent / prefix if prefix else parent
                    cand = base / "tk25_basic" / "src" / "tinker_robot_config"
                    if cand.is_dir():
                        return (base / "tk25_basic").resolve()
                return None

            # (1) file-relative
            here = Path(__file__).resolve()
            for parent in here.parents:
                hit = _check(parent)
                if hit is not None:
                    return hit
            # (2) cwd-relative
            cwd = Path.cwd().resolve()
            for parent in (cwd, *cwd.parents):
                hit = _check(parent)
                if hit is not None:
                    return hit
            # (3) ament_index → workspace root
            try:
                from ament_index_python.packages import get_package_share_directory
                share = Path(get_package_share_directory("tinker_robot_config")).resolve()
                for parent in share.parents:
                    if parent.name == "install":
                        ws_root = parent.parent
                        hit = _check(ws_root)
                        if hit is not None:
                            return hit
                        break
            except Exception:
                pass
            return None

        def _mount_joint_name(self):
            """Joint name carrying the wrist-camera mount in the URDF.

            Per the brief: ``camera_link_joint`` in the d435i vendor xacro is
            the joint we override per-robot. Exposed as a launch param so a
            future arm/camera swap (different joint name) only needs a
            parameter override, not a code change.
            """
            return self._param("mount_joint_name", "camera_link_joint")

        def _calibration_date_or_unset(self):
            """T6: ISO date stamp for the yaml's calibration_date field.

            Defaults to today (UTC). The v1 do_promote hard-coded
            ``"unset"``; surfacing a real date lets the operator see staleness
            at a glance.
            """
            import datetime
            return datetime.date.today().isoformat()

        def _format_xyz_str(self, T):
            return " ".join(f"{float(v):.9g}" for v in T[:3, 3])

        def _format_rpy_str(self, T):
            from scipy.spatial.transform import Rotation as _R
            rpy = _R.from_matrix(T[:3, :3]).as_euler('xyz')
            return " ".join(f"{float(v):.9g}" for v in rpy)

        def _latest_backup_for(self, path):
            """Glob ``<path>.old-*`` and return the newest, or ``None``."""
            from pathlib import Path
            p = Path(path)
            cands = sorted(p.parent.glob(p.name + ".old-*"))
            return str(cands[-1]) if cands else None

        def _yaml_half_for_diff(self):
            """Build the yaml side of the promote diff (always computed)."""
            import difflib
            import yaml
            T_mount_color = self._mount_to_color_matrix()
            T_eef_mount = ah.compose_eef_to_mount(self.last_solve.X, T_mount_color)
            proposed_dict = ah.handeye_yaml_dict(
                T_eef_mount, self.last_solve.X, len(self.session.samples),
                self.last_solve.heldout_metrics,
                date=self._calibration_date_or_unset(),
                square_len_m=self._sq,
            )
            proposed_yaml = yaml.safe_dump(proposed_dict, sort_keys=False)
            robot = self._resolve_robot_name()
            yaml_target = self._hand_eye_path(robot) if robot else None
            current_yaml = (yaml_target.read_text()
                            if yaml_target and yaml_target.exists() else "")
            fromfile = (str(yaml_target) if yaml_target
                        else "(set ROBOT_NAME=tinker1|tinker2 to resolve target)")
            yaml_diff = "".join(difflib.unified_diff(
                current_yaml.splitlines(keepends=True),
                proposed_yaml.splitlines(keepends=True),
                fromfile=fromfile, tofile="proposed", lineterm=""))
            return {
                "target_path": str(yaml_target) if yaml_target else "",
                "current_text": current_yaml,
                "proposed_text": proposed_yaml,
                "diff": yaml_diff,
                "mode": "patch",
            }, T_eef_mount

        def _xacro_half_for_diff(self, T_eef_mount):
            """Build the xacro side of the promote diff.

            Returns either the populated half dict, or ``None`` when
            ``ROBOT_NAME`` is unset / the basic repo root can't be located /
            the resolved target points at the shared vendor xacro (in which
            case we still return a dict, but mark it ``mode='refuse-vendor'``
            so the UI can render a warning banner without enabling Apply).
            """
            import difflib
            robot = self._resolve_robot_name()
            if not robot:
                return None
            basic_root = self._tk25_basic_repo_root()
            if basic_root is None:
                return {
                    "target_path": "",
                    "current_text": "",
                    "proposed_text": "",
                    "diff": "",
                    "mode": "refuse-vendor",
                    "warning": "tk25_basic repo root not resolvable",
                }
            xacro_target = ah.resolve_robot_xacro_path(robot, basic_root)
            if xacro_target is None:
                return None
            joint_name = self._mount_joint_name()
            xyz_str = self._format_xyz_str(T_eef_mount)
            rpy_str = self._format_rpy_str(T_eef_mount)
            # Vendor-path refusal: even if some bug resolved us at the shared
            # d435i xacro, refuse to seed/patch it. The Apply endpoint does
            # the same check belt-and-suspenders.
            if "xarm_description/urdf/camera/realsense_d435i.urdf.xacro" in str(xacro_target):
                return {
                    "target_path": str(xacro_target),
                    "current_text": "",
                    "proposed_text": "",
                    "diff": "",
                    "mode": "refuse-vendor",
                    "warning": (f"refusing to write shared vendor xacro — "
                                f"set up per-robot override at {xacro_target}"),
                }
            if xacro_target.exists():
                current_xacro = xacro_target.read_text()
                try:
                    proposed_xacro = ah.patch_urdf_origin(
                        current_xacro, joint_name,
                        xyz_str.split(), rpy_str.split())
                    mode = "patch"
                except ValueError:
                    # joint not in existing override file → re-seed it
                    proposed_xacro = ah.seed_handeye_override_xacro(
                        joint_name, xyz_str, rpy_str)
                    mode = "seed"
            else:
                current_xacro = ""
                proposed_xacro = ah.seed_handeye_override_xacro(
                    joint_name, xyz_str, rpy_str)
                mode = "seed"
            xacro_diff = "".join(difflib.unified_diff(
                current_xacro.splitlines(keepends=True),
                proposed_xacro.splitlines(keepends=True),
                fromfile=str(xacro_target), tofile="proposed", lineterm=""))
            return {
                "target_path": str(xacro_target),
                "current_text": current_xacro,
                "proposed_text": proposed_xacro,
                "diff": xacro_diff,
                "mode": mode,
            }

        def compute_promote_diff(self):
            """T6: build both yaml + xacro halves of the promote unified-diff.

            Returns one of:
              * ``{"ok": False, "reason": "run solve first"}`` if no solve yet.
              * ``{"ok": True, "yaml": {...}, "xacro": {...}|None,
                   "robot_name": "tinker2"|None}``

            ``xacro`` is ``None`` when ``ROBOT_NAME`` is unset (the UI shows a
            yaml-only banner). Each half dict has keys
            ``target_path / current_text / proposed_text / diff / mode``
            (``mode ∈ {"patch", "seed", "refuse-vendor"}``) plus an optional
            ``warning`` for vendor-path refusal so the UI can render a red
            banner.
            """
            if self.last_solve is None:
                return {"ok": False, "reason": "run solve first"}
            yaml_half, T_eef_mount = self._yaml_half_for_diff()
            xacro_half = self._xacro_half_for_diff(T_eef_mount)
            robot = self._resolve_robot_name()
            return {"ok": True, "yaml": yaml_half, "xacro": xacro_half,
                    "robot_name": robot or None}

        def apply_promote(self, which="both"):
            """T6: apply the promote diff to disk, per-half, with backups.

            ``which`` is one of:
              * ``"yaml"`` — write the per-robot ``hand_eye.yaml`` only.
              * ``"xacro"`` — write the per-robot ``wrist_camera.xacro`` only.
              * ``"both"`` (default) — try yaml first, then xacro. If the yaml
                write succeeds and the xacro write fails (e.g. vendor-path
                refusal), the response is the partial-success shape
                ``{ok: True, yaml: {...}, xacro: {ok: False, reason: ...}}``
                so the operator sees both halves and isn't lied to about a
                "successful" promote that only wrote half the state.

            Per-half result shape:
              * Success: ``{written_path, backup_path}`` (``backup_path`` may
                be ``None`` if no prior file existed).
              * Failure: ``{ok: False, reason: str}``.
            """
            if self.last_solve is None:
                return {"ok": False, "reason": "run solve first"}
            diff = self.compute_promote_diff()
            if not diff["ok"]:
                return diff
            out = {"ok": True}

            # Single-half "which" gets a top-level ok=False when that one
            # half can't be written (e.g. ``which='xacro'`` with ROBOT_NAME
            # unset). ``which='both'`` keeps top-level ok=True even on
            # partial-failure so the operator sees both halves' results.

            if which in ("yaml", "both") and diff["yaml"]:
                y = diff["yaml"]
                if not y["target_path"]:
                    out["yaml"] = {
                        "ok": False,
                        "reason": ("ROBOT_NAME unset — cannot resolve "
                                   "hand_eye.yaml target path"),
                    }
                else:
                    try:
                        backup = ah.write_with_backup(
                            y["target_path"], y["proposed_text"])
                        out["yaml"] = {"written_path": y["target_path"],
                                       "backup_path": backup}
                    except Exception as exc:
                        out["yaml"] = {
                            "ok": False,
                            "reason": f"yaml write failed: {exc}",
                        }

            if which in ("xacro", "both"):
                if diff["xacro"] is None:
                    out["xacro"] = {
                        "ok": False,
                        "reason": ("ROBOT_NAME unset — cannot write per-robot "
                                   "xacro override"),
                    }
                else:
                    x = diff["xacro"]
                    if x.get("mode") == "refuse-vendor":
                        out["xacro"] = {
                            "ok": False,
                            "reason": (x.get("warning") or
                                       "refusing to write shared vendor xacro"),
                        }
                    elif ("xarm_description/urdf/camera/realsense_d435i.urdf.xacro"
                          in x["target_path"]):
                        # Belt-and-suspenders: even if compute_promote_diff
                        # didn't catch it, refuse here too.
                        out["xacro"] = {
                            "ok": False,
                            "reason": ("refusing to write shared vendor xacro — "
                                       "set up per-robot override at "
                                       + x["target_path"]),
                        }
                    else:
                        try:
                            backup = ah.write_with_backup(
                                x["target_path"], x["proposed_text"])
                            out["xacro"] = {"written_path": x["target_path"],
                                            "backup_path": backup}
                        except Exception as exc:
                            out["xacro"] = {
                                "ok": False,
                                "reason": f"xacro write failed: {exc}",
                            }

            # Single-half failure ⇒ surface at top-level so callers that only
            # asked for one half don't have to dig into the sub-dict to find
            # an ok:False. ``both`` keeps top-level ok=True so partial-success
            # is visible (the brief's explicit shape).
            if which == "yaml":
                y_res = out.get("yaml", {})
                if isinstance(y_res, dict) and y_res.get("ok") is False:
                    return {"ok": False, "reason": y_res.get("reason", "yaml write failed"),
                            "yaml": y_res}
            if which == "xacro":
                x_res = out.get("xacro", {})
                if isinstance(x_res, dict) and x_res.get("ok") is False:
                    return {"ok": False, "reason": x_res.get("reason", "xacro write failed"),
                            "xacro": x_res}
            return out

        def reload_promote(self):
            """T6: clear the cached solve so the operator can re-run solve.

            The diff/apply endpoints both read ``self.last_solve``; this
            reset lets the operator "reload from disk" without restarting
            the node. (The yaml/xacro on disk are the source of truth after
            a successful promote — there's nothing to reload back into RAM.)
            """
            with self.lock:
                self.last_solve = None
            return {"ok": True, "reason": "cached solve cleared"}

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
            """Resolve <ws>/src/.../tinker_robot_config/robots/<robot>/hand_eye.yaml.

            Delegates to ``_tk25_basic_repo_root`` so the source-tree /
            install-tree / cwd resolution is uniform across save endpoints.
            """
            if not robot:
                return None
            basic_root = self._tk25_basic_repo_root()
            if basic_root is None:
                return None
            return (basic_root / "src" / "tinker_robot_config"
                    / "robots" / robot / "hand_eye.yaml")

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
    def frame(raw: int = 0):
        """Latest camera frame as JPEG. ``?raw=1`` skips the overlay."""
        return Response(content=node.latest_jpeg(raw=bool(raw)), media_type="image/jpeg")

    @app.post("/api/move")
    async def move(request: Request):
        body = await request.json()
        return JSONResponse(node.do_move(body.get("joints")))

    @app.post("/api/capture")
    def capture():
        return JSONResponse(node.do_capture())

    @app.get("/api/samples/{idx}/thumb.jpg")
    def sample_thumb(idx: int):
        """T4: per-sample gallery thumbnail (downscaled JPEG of the frame+overlay).

        Returns 404 when the index is unknown (deleted or never captured) so
        the UI can drop stale gallery entries on the next render.
        """
        jpg = node.sample_thumb(idx)
        if jpg is None:
            return Response(status_code=404)
        return Response(content=jpg, media_type="image/jpeg")

    @app.delete("/api/samples/{idx}")
    def delete_sample(idx: int):
        """T4: remove a captured sample by index; recompact sidecar dicts.

        Always returns 200 with ``{ok, num_samples, reason?}`` so the UI's
        delete handler doesn't have to special-case HTTP errors against the
        usual ``{ok: False, reason}`` flow.
        """
        return JSONResponse(node.do_delete_sample(idx))

    @app.post("/api/solve")
    async def solve(request: Request):
        # T5: accept an optional ``{method: "auto"|"TSAI"|"PARK"|"HORAUD"|
        # "ANDREFF"|"DANIILIDIS"}`` body and forward to ``do_solve``. Missing
        # or empty body falls back to ``method="auto"``.
        try:
            body = await request.json()
        except Exception:
            body = {}
        return JSONResponse(node.do_solve(method=(body or {}).get("method", "auto")))

    @app.post("/api/promote")
    def promote():
        # Back-compat shim: equivalent to apply_promote(which="both"). The
        # T6 brief's canonical surface is the three split routes below.
        return JSONResponse(node.do_promote())

    @app.get("/api/promote/diff")
    def promote_diff():
        """T6: unified-diff preview for both yaml + xacro halves.

        Returns ``{ok: False, reason: ...}`` until ``last_solve`` is set.
        On success returns both halves; ``xacro`` is ``None`` when
        ``ROBOT_NAME`` is unset (UI shows a yaml-only banner).
        """
        return JSONResponse(node.compute_promote_diff())

    @app.post("/api/promote/apply")
    async def promote_apply(request: Request):
        """T6: write yaml and/or xacro to disk with timestamped backups.

        Body: ``{which: "yaml"|"xacro"|"both"}`` (defaults to ``"both"``).
        Partial-success is surfaced explicitly when one half succeeds and
        the other fails; the operator must see both halves so a vendor-path
        refusal never silently shadows a successful yaml write.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        which = str((body or {}).get("which", "both")).lower()
        if which not in ("yaml", "xacro", "both"):
            return JSONResponse(
                {"ok": False, "reason": f"unknown which={which!r}"})
        return JSONResponse(node.apply_promote(which=which))

    @app.post("/api/promote/reload")
    def promote_reload():
        """T6: reset the cached ``last_solve`` so the operator can rerun."""
        return JSONResponse(node.reload_promote())

    # ---- T1-wp: waypoint CRUD + per-robot YAML persistence ------------------

    @app.get("/api/waypoints")
    def waypoints_list():
        """GET /api/waypoints → {count, items: [waypoint_metadata(...)]}."""
        from handeye_calib import web_support as _ws
        items = [_ws.waypoint_metadata(i, w)
                 for i, w in enumerate(node.waypoint_store.list())]
        return JSONResponse({"count": len(items), "items": items})

    @app.post("/api/waypoints")
    def waypoints_add():
        """POST /api/waypoints {} → do_add_waypoint()."""
        return JSONResponse(node.do_add_waypoint())

    @app.delete("/api/waypoints/{idx}")
    def waypoints_delete(idx: int):
        """DELETE /api/waypoints/{idx} → do_delete_waypoint(idx)."""
        return JSONResponse(node.do_delete_waypoint(idx))

    @app.post("/api/waypoints/save")
    def waypoints_save():
        """POST /api/waypoints/save → do_save_waypoints()."""
        return JSONResponse(node.do_save_waypoints())

    @app.post("/api/waypoints/reload")
    def waypoints_reload():
        """POST /api/waypoints/reload → do_reload_waypoints()."""
        return JSONResponse(node.do_reload_waypoints())

    # ---- T3-seq: auto-capture sequence endpoints ---------------------------

    @app.post("/api/sequence/start")
    async def sequence_start(request: Request):
        """POST /api/sequence/start {dry_run?: bool} → do_start_sequence(...).

        Body is optional; missing/malformed JSON is treated as the default
        ``{dry_run: false}`` per the same degrade-rather-than-422 stance the
        rest of the API uses."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        return JSONResponse(node.do_start_sequence(
            dry_run=bool(body.get("dry_run", False))))

    @app.post("/api/sequence/cancel")
    def sequence_cancel():
        """POST /api/sequence/cancel → do_cancel_sequence() (idempotent no-op)."""
        return JSONResponse(node.do_cancel_sequence())

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
