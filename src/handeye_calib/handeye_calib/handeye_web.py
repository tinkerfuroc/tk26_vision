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
    from rclpy.callback_groups import ReentrantCallbackGroup
    from tf2_ros import Buffer, TransformListener
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image, CameraInfo, JointState
    from tinker_arm_msgs.action import JointMove
    from tinker_vision_msgs_26.srv import FoundationStereoDepth
    from scipy.spatial.transform import Rotation as _R

    from handeye_calib import web_support as ws
    from handeye_calib import handeye_model as hm
    from handeye_calib import handeye_solve as hs
    from handeye_calib import apply_handeye as ah
    from handeye_calib import gates as hgates
    from handeye_calib.handeye_collect import CaptureSession
    from handeye_calib import waypoints as hwp
    from handeye_calib.waypoints import WaypointStore
    from handeye_calib import handeye_sessions as hsx
    from handeye_calib.placement_state import PlacementState, make_placement, slug_id  # noqa: F401
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
        # These class defaults are overridable per-instance from the node's ROS
        # params (see HandeyeWebNode init: ``settle_steady_ticks``).
        SETTLE_STEADY_TICKS = 3

        def __init__(self, node):
            self.node = node
            self._stop = threading.Event()
            self._lock = threading.Lock()
            self._thread = None
            self._inflight_handle = None  # latest JointMove goal handle
            # Per-instance overrides — pulled from the node's ROS params so an
            # operator can ``-p settle_steady_ticks:=8 -p settle_poll_s:=0.2``
            # without rebuilding. Class-level constants remain as fallbacks.
            self.steady_ticks = int(getattr(node, "_settle_steady_ticks",
                                            self.SETTLE_STEADY_TICKS))
            self.poll_s = float(getattr(node, "_settle_poll_s",
                                        self.SETTLE_POLL_S))
            # Frame-based settle floor shared with the manual capture gate: the
            # cached steady verdict must have HELD for this many consecutive
            # frames before settle is declared (in addition to the poll hold).
            self.min_steady_frames = int(getattr(node,
                                         "_capture_min_steady_frames", 5))
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
            # SafetyEnvelope pre-check REMOVED 2026-06-20 at operator request
            # (see do_move docstring). The arm driver / MoveIt's collision
            # checker remains the real safety boundary; safety_preview still
            # publishes informational verdict to state.safety_preview.
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
            """Poll the cached StabilityTracker verdict at ``1/poll_s`` Hz.

            Returns True when steady for ``self.steady_ticks`` consecutive
            ticks; False on timeout or on stop. Both knobs are pulled from
            the node's ROS params at construction so operators can dial
            without rebuilding. Reuses ``_stability_steady`` written by
            ``_on_image``."""
            t0 = time.monotonic()
            consec = 0
            while True:
                if self._stop.is_set():
                    return False
                if time.monotonic() - t0 >= settle_timeout_s:
                    return False
                with self.node.lock:
                    steady = self.node._stability_steady
                    since = self.node._stability_since_frames
                if steady:
                    consec += 1
                    # Settle requires BOTH the poll hold (steady_ticks polls)
                    # AND the camera-frame floor (>= min_steady_frames held
                    # frames), so a slow camera can't satisfy the poll count
                    # with too few actual stable frames.
                    if (consec >= self.steady_ticks
                            and since >= self.min_steady_frames):
                        return True
                else:
                    consec = 0
                time.sleep(self.poll_s)

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
            # Declare robot_name at startup so `ros2 param get/set /handeye_web
            # robot_name` works immediately; the default mirrors $ROBOT_NAME so
            # the param can't contradict the env var. Do NOT cache the resolved
            # name on self — _resolve_robot_name() (env-var-first, re-read per
            # request) is the single source of truth; a startup snapshot would
            # go stale after a live `ros2 param set`.
            self._param("robot_name", os.environ.get("ROBOT_NAME", ""))
            self._base_frame = self._param("base_frame", "link_base")
            self._eef_frame = self._param("eef_frame", "link_eef")
            self._aruco_dict_name = self._param("aruco_dict", "DICT_5X5_100")
            self._marker_len = float(self._param("marker_len_m", 0.03))
            # Internal camera geometry: T_camera_link -> {color,ir}_optical, the
            # FIXED factory transforms the hand-eye solve composes THROUGH to
            # recover T_eef->camera_link (the stored mount). Defaults are the
            # D435 vendor URDF values (realsense_d435i.urdf.xacro): color_frame
            # is +15 mm in Y off camera_link then the optical rotation; left-IR
            # frame is coincident with camera_link (0 0 0) then the SAME optical
            # rotation. (Previously both defaulted to identity, which silently
            # wrote T_eef->color_optical into the camera_link joint — the reason
            # the deployed tinker2 xacro needed a manual correction. Real
            # geometry => the tool now writes the correct camera_link directly.)
            self._mount_to_color_xyz = self._param("mount_to_color_xyz", "0 0.015 0")
            self._mount_to_color_rpy = self._param("mount_to_color_rpy", "-1.5707963267948966 0 -1.5707963267948966")
            self._mount_to_ir_xyz = self._param("mount_to_ir_xyz", "0 0 0")
            self._mount_to_ir_rpy = self._param("mount_to_ir_rpy", "-1.5707963267948966 0 -1.5707963267948966")
            # Observation frame for the board: 'color' (default) or 'ir' (left-IR,
            # the native FFS depth frame == camera_link). Runtime-switchable via
            # /api/config; switching clears the (frame-specific) sample set.
            self._calib_frame = str(self._param("calib_frame", "color")).strip().lower()
            if self._calib_frame not in ("color", "ir"):
                self._calib_frame = "color"
            # IR emitter control: the wrist RealSense node exposes the projector
            # as the settable param ``depth_module.emitter_enabled``; we flip it
            # via an async SetParameters client (no device contention). Disable it
            # for IR-frame capture (the dot pattern corrupts ChArUco corners; FFS
            # is passive so depth survives). ``_ir_emitter_enabled`` tracks the
            # last value WE set (None = unknown / never set).
            self._camera_node_name = str(self._param("camera_node_name", "/camera/xarm_camera"))
            self._ir_emitter_enabled = None
            self._emitter_wait_s = 1.0

            # ---- pan-tilt HEAD Orbbec warm-start anchor ---------------------
            # The head is already calibrated (~3 mm/0.5deg in base_link). It
            # observes the SAME fixed board and supplies T_base_board, used ONLY
            # as a basin-immune SEED for the wrist solve (handeye_solve.solve
            # anchor_Tbb=...). Tbb stays FREE in the bundle adjust, so the head's
            # absolute bias never enters the final X. Disabled until the first
            # successful anchor; head camera defaults are the Orbbec /camera ns.
            self._head_image_topic = str(self._param("head_image_topic", "/camera/color/image_raw"))
            self._head_info_topic = str(self._param("head_info_topic", "/camera/color/camera_info"))
            self._head_optical_frame = str(self._param("head_optical_frame", "camera_color_optical_frame"))
            self._head_frame = None
            self._head_frame_stamp = None
            self._head_K = None
            self._head_D = None

            self.bridge = CvBridge()
            self._frame = None
            self._frame_stamp = None   # ROS stamp of the most recent image
            self._K = None             # ACTIVE intrinsics (color or IR, per calib_frame)
            self._D = None
            # Per-source intrinsic caches (the active _K/_D mirror one of these).
            self._color_K = None
            self._color_D = None
            self._ir_K = None
            self._ir_D = None
            # Latest cached native-IR FoundationStereo depth (float32 metres) +
            # stamp, filled by _on_ir_depth from the FFS native-IR stream. Used
            # for the depth residual when calib_frame='ir'.
            self._ir_depth = None
            self._ir_depth_stamp = None
            self._last_det = None              # {corners:int, reproj_px:float} or None
            self._last_corners_xy = None       # (M,2) px for overlay, or None
            self._cap = None                   # latest {T_cam_board, obs_px, corner_idx, reproj_px, area_frac}

            # Rolling buffer of recent (ids, px) detections for multi-frame
            # consensus at capture time (pan_tilt parity: cluster_consensus).
            # Only pushed while a board pose is present; reset on lost detection.
            self._consensus_frames = int(self._param("consensus_frames", 10))
            self._consensus_min_frac = float(self._param("consensus_min_frac", 0.6))
            self._det_history = collections.deque(maxlen=self._consensus_frames)

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
            # /joint_states publishers (joint_state_publisher / xarm driver) use
            # the default RELIABLE / VOLATILE QoS — match it with depth=10
            # instead of qos_profile_sensor_data (BEST_EFFORT). FastDDS sometimes
            # fails to match a BEST_EFFORT sub to a RELIABLE pub on low-rate
            # topics, surfacing as a permanent "joint_states not yet received"
            # state. pan_tilt/calib_web uses the same depth=10 pattern.
            self.create_subscription(
                JointState, self._param("joint_states_topic", "/joint_states"),
                self._on_joint_state, 10)

            # Stability tracker (observable in T1; T4 promotes it to a hard gate).
            self._stab_window = int(self._param("stability_window", 5))
            # Defaults loosened 2026-06-20 from 0.1° / 0.3 mm to physically
            # realistic camera-PnP noise floors at typical calibration
            # distances. See gates.StabilityTracker docstring for the
            # underlying noise-budget rationale.
            self._stab_rot_tol_deg = float(self._param("stability_rot_tol_deg", 0.5))
            self._stab_trans_tol_m = float(self._param("stability_trans_tol_m", 0.003))
            self._stability = hgates.StabilityTracker(
                window=self._stab_window,
                rot_tol_deg=self._stab_rot_tol_deg,
                trans_tol_m=self._stab_trans_tol_m,
            )
            self._stability_steady = False
            self._stability_since_frames = 0

            # Diversity threshold for the per-sample dedup gate (gates.is_diverse).
            # Default DISABLED (0°) 2026-06-28 at operator request: every
            # authored, settled, detected position must be RECORDED — the dedup
            # gate was silently dropping ~half of a 20+ waypoint set (its 30°->5°
            # history below is the same SO(3)-packing problem in miniature).
            # Redundant / inconsistent poses are now culled where they belong:
            # at solve time by the per-axis MAD rejection + the per-sample
            # residual view, not silently at capture. Set min_diversity_deg>0 to
            # re-enable the old camera-shake dedup (5° catches duplicates of the
            # same authored pose; 30° was the original, over-aggressive default).
            self._diversity_target_deg = float(self._param("min_diversity_deg", 0.0))

            # Per-sample quality gates (gates.quality_ok). Exposed as ROS
            # params so the operator can dial without rebuilding when the
            # board / camera geometry differs from the 5x5-40mm @ 1280x720
            # baseline the defaults assume.
            self._min_corners = int(self._param("min_corners", 10))
            self._max_reproj_px = float(self._param("max_reproj_px", 1.5))
            self._min_area_frac = float(self._param("min_area_frac", 0.01))

            # Auto-capture sequence settle knobs (CaptureSequenceRunner).
            # Defaults bumped 2026-06-21 at operator request for "longer
            # settle". Override per run via ROS params:
            #   settle_timeout_s     — wall-clock budget waiting for steady
            #   settle_steady_ticks  — N consecutive steady reads to confirm
            #   settle_poll_s        — poll period (10 Hz default)
            self._settle_timeout_s = float(self._param("settle_timeout_s", 10.0))
            self._settle_steady_ticks = int(self._param("settle_steady_ticks", 5))
            self._settle_poll_s = float(self._param("settle_poll_s", 0.1))
            # Camera-frame settle FLOOR (pan-tilt parity): require the
            # StabilityTracker verdict to HOLD for this many CONSECUTIVE steady
            # frames before a capture is accepted, instead of firing on the very
            # first steady verdict. Mirrors the pan-tilt calibration's
            # "held for a duration" image-stability gate and gives the wrist
            # mount extra frames to stop micro-settling after the
            # StabilityTracker window first agrees. Applies to BOTH the manual
            # /api/capture path and the automated waypoint sweep. Raise it for a
            # longer settle (5 frames ~= 0.17 s at 30 Hz, ON TOP of the 5-frame
            # StabilityTracker window).
            self._capture_min_steady_frames = int(
                self._param("capture_min_steady_frames", 5))

            # ---- FoundationStereo (FFS) metric depth ------------------------
            # The per-view T_cam_board is otherwise monocular planar PnP, whose
            # optical-axis translation is the weakest DOF. At capture time we
            # call the FFS get_depth service (color-aligned, same color
            # intrinsics the solver uses), sample depth at the detected corner
            # pixels, and store the deprojected metric points on the Sample;
            # the solver's depth residual then pins the scale/standoff of the
            # rigidly-mounted camera. Reuses the proven object_seg_yolo client
            # pattern (threading.Event + add_done_callback on a Reentrant group
            # so the blocking wait — which runs on the FastAPI/sequence thread,
            # never the rclpy spin thread — doesn't deadlock the executor).
            #
            # Default ON; degrades gracefully to monocular when FFS is down
            # (service missing / timeout / non-zero status / shape mismatch) so
            # an FFS hiccup never blocks a capture.
            self._use_ffs_depth = bool(self._param("use_ffs_depth", True))
            self._ffs_service = str(self._param("ffs_service", "/foundation_stereo/get_depth"))
            self._ffs_wait_for_service_s = float(self._param("ffs_wait_for_service_s", 1.0))
            self._ffs_call_timeout_s = float(self._param("ffs_call_timeout_s", 10.0))
            # Depth residual knobs forwarded to handeye_solve.solve. depth_weight
            # 2.0 makes FFS depth (at depth_sigma_m metres of stereo noise) a
            # dominant-but-safe scale constraint while sub-pixel reprojection
            # keeps owning rotation; dial up to trust depth more, 0 to disable.
            # depth_weight lowered 2026-06-22 (review) from 2.0 -> 1.0: at 2.0 the
            # depth block out-votes the sub-pixel reprojection ~2:1, so a realistic
            # systematic FFS metric scale bias (~0.5-1% from stereo baseline/
            # rectification) could drag an otherwise-good calibration off and past
            # the PASS gate. 1.0 keeps depth a co-equal constraint that pins the
            # optical-axis DOF without steamrolling reprojection; raise it if FFS
            # depth on this robot is validated metrically trustworthy.
            self._depth_weight = float(self._param("depth_weight", 1.0))
            self._depth_sigma_m = float(self._param("depth_sigma_m", 0.005))
            self._depth_win = int(self._param("depth_win", 2))
            # Valid-depth band (m) + minimum valid corners to use depth for a
            # pose — exposed for parity with depth_win so a different board
            # standoff doesn't need a rebuild.
            self._depth_z_min = float(self._param("depth_z_min", 0.05))
            self._depth_z_max = float(self._param("depth_z_max", 2.0))
            self._depth_min_corners = int(self._param("depth_min_corners", 3))
            self._ffs_cb_group = ReentrantCallbackGroup()
            self._ffs_cli = None
            self._last_depth_source = None  # 'ffs' | 'unavailable' | 'shape-mismatch' | ...
            # One-time "FFS enabled but never delivering depth" hint (the usual
            # cause is the wrist RealSense brought up without infra1/infra2 IR
            # streams FoundationStereo needs). See _note_ffs_depth_outcome.
            self._ffs_depth_miss_streak = 0
            self._ffs_depth_warned = False
            self._ffs_depth_warn_after = int(self._param("ffs_depth_warn_after", 5))

            # --- multi-placement state ---
            _default_label = "default"
            _default_id = "default"
            self._placements = {
                _default_id: make_placement(
                    _default_label,
                    min_diversity_deg=self._diversity_target_deg,
                    min_corners=self._min_corners,
                    max_reproj_px=self._max_reproj_px,
                    min_area_frac=self._min_area_frac,
                )
            }
            self._active_placement_id = _default_id
            self.last_solve = None
            # JSON solve payload (not the raw SolveResult) cached so the WS push
            # can rehydrate the Solve tab on reconnect/reload, and live MAD
            # rejection progress streamed from hs.solve's progress callback.
            self._last_solve_payload = None
            self._solve_progress = {
                "running": False, "phase": "idle", "n_orig": 0, "n_active": 0,
                "min_keep": 0, "iteration": 0, "rejection_log": [],
                "last_drop": None, "solve_ts": 0.0,
            }

            # Name of the on-disk capture session the live samples persist into
            # (``<HANDEYE_DUMP_DIR|calibration_data>/wrist_handeye_sessions/<name>``).
            # Lazily created at the first capture of an empty set and reset to
            # None whenever the set is cleared, so each capture run is its own
            # browsable history entry; set to an existing name by do_load_session.
            self._session_name = None

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
            # Left-IR observation path (calib_frame='ir'): the rectified IR image
            # + its rectified intrinsics (P-block, zero distortion) + the native-IR
            # FFS depth stream (run FFS with stream_enabled:=true
            # stream_align_to_color:=false; or point ffs_ir_depth_topic at the
            # RealSense-native /camera/xarm_camera/depth/image_rect_raw). All three
            # are subscribed unconditionally; only the active frame drives detection.
            self._ir_image_topic = self._param(
                "ir_image_topic", "/camera/xarm_camera/infra1/image_rect_raw")
            self._ir_info_topic = self._param(
                "ir_info_topic", "/camera/xarm_camera/infra1/camera_info")
            self._ffs_ir_depth_topic = self._param(
                "ffs_ir_depth_topic", "/foundation_stereo/depth/image_rect_raw")
            self._ir_depth_max_age_s = float(self._param("ir_depth_max_age_s", 1.0))
            self.create_subscription(
                Image, self._ir_image_topic, self._on_ir_image, qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, self._ir_info_topic, self._on_ir_info, qos_profile_sensor_data)
            self.create_subscription(
                Image, self._ffs_ir_depth_topic, self._on_ir_depth, qos_profile_sensor_data)
            self.create_subscription(
                Image, self._head_image_topic, self._on_head_image,
                qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, self._head_info_topic, self._on_head_info, 10)
            self._jm = ActionClient(self, JointMove, self._param("jointmove_action", "joint_move_action"))

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

        # ---- multi-placement properties -------------------------------------

        @property
        def _active_placement(self) -> PlacementState:
            return self._placements[self._active_placement_id]

        @property
        def session(self):
            return self._active_placement.session

        @property
        def _thumbs(self):
            return self._active_placement.thumbs

        @property
        def _sample_joints(self):
            return self._active_placement.sample_joints

        @property
        def _sample_ts(self):
            return self._active_placement.sample_ts

        @property
        def _sample_reproj_px(self):
            return self._active_placement.sample_reproj_px

        @property
        def _sample_area_frac(self):
            return self._active_placement.sample_area_frac

        @property
        def _sample_depth_source(self):
            return self._active_placement.sample_depth_source

        @property
        def _anchor_obs(self):
            return self._active_placement.anchor_obs

        @_anchor_obs.setter
        def _anchor_obs(self, v):
            self._active_placement.anchor_obs = v

        @property
        def _tbb_head(self):
            return self._active_placement.tbb_head

        @_tbb_head.setter
        def _tbb_head(self, v):
            self._active_placement.tbb_head = v

        @property
        def _anchor_scatter(self):
            return self._active_placement.anchor_scatter

        @_anchor_scatter.setter
        def _anchor_scatter(self, v):
            self._active_placement.anchor_scatter = v

        # ---- placement management -------------------------------------------

        def do_add_placement(self, label: str) -> dict:
            """Create and activate a new empty placement. Returns {ok, id, label, count}."""
            label = str(label).strip()
            if not label:
                return {"ok": False, "reason": "label must be non-empty"}
            with self.lock:
                pid = slug_id(label, set(self._placements.keys()))
                self._placements[pid] = make_placement(
                    label,
                    min_diversity_deg=self._diversity_target_deg,
                    min_corners=self._min_corners,
                    max_reproj_px=self._max_reproj_px,
                    min_area_frac=self._min_area_frac,
                )
                self._active_placement_id = pid
                count = len(self._placements)
            return {"ok": True, "id": pid, "label": label, "count": count}

        def do_activate_placement(self, pid: str) -> dict:
            """Switch active placement. Returns {ok, id}."""
            with self.lock:
                if pid not in self._placements:
                    return {"ok": False, "reason": f"placement {pid!r} not found"}
                self._active_placement_id = pid
            return {"ok": True, "id": pid}

        def do_rename_placement(self, pid: str, new_label: str) -> dict:
            """Rename a placement label. Returns {ok, id, label}."""
            new_label = str(new_label).strip()
            if not new_label:
                return {"ok": False, "reason": "label must be non-empty"}
            with self.lock:
                if pid not in self._placements:
                    return {"ok": False, "reason": f"placement {pid!r} not found"}
                self._placements[pid].label = new_label
            return {"ok": True, "id": pid, "label": new_label}

        def do_delete_placement(self, pid: str) -> dict:
            """Remove a placement. Cannot remove the only one. Returns {ok, deleted}."""
            with self.lock:
                if len(self._placements) <= 1:
                    return {"ok": False, "reason": "cannot delete the only placement"}
                if pid not in self._placements:
                    return {"ok": False, "reason": f"placement {pid!r} not found"}
                del self._placements[pid]
                if self._active_placement_id == pid:
                    self._active_placement_id = next(iter(self._placements))
            return {"ok": True, "deleted": pid}

        # ---- param helper ----------------------------------------------------

        def _param(self, name, default):
            if not self.has_parameter(name):
                self.declare_parameter(name, default)
            return self.get_parameter(name).value

        # ---- subscriptions ---------------------------------------------------

        def _ingest_frame(self, bgr, K, D):
            """Run detection on ``bgr`` (with intrinsics K,D) and update the SHARED
            active-frame state. Called only by the ACTIVE frame's image callback
            (color OR IR per ``calib_frame``), so ``self._K``/``self._frame``/
            ``self._cap`` always reflect the frame currently being calibrated."""
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
                self._K = K       # active intrinsics for do_solve/do_capture
                self._D = D
                # Consensus ring write — kept under the SAME lock as the
                # do_capture read so the mutual exclusion is real, not just
                # GIL-incidental.
                if cap is not None:
                    self._det_history.append(
                        (self._np.asarray(cap["corner_idx"]).copy(),
                         self._np.asarray(cap["obs_px"], float).copy()))
                else:
                    self._det_history.clear()
                # Stamp the image with ROS-time AT RECEIPT, NOT msg.header.stamp.
                # realsense2_camera (and many other drivers) populate
                # header.stamp from the camera HW clock, which drifts seconds
                # from ROS time on this workstation (observed: image stamp
                # 1.82s ahead of /tf latest, breaking TF lookup with
                # "extrapolation into future" errors). ROS-now() at receipt
                # carries ~5-20ms of USB+ROS-pub latency, which is the right
                # answer for "when was this image captured, in TF clock".
                self._frame_stamp = self.get_clock().now().to_msg()
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

        def _on_image(self, msg):
            """Color image. Drives detection only when calib_frame=='color'."""
            if self._calib_frame != "color":
                return
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:
                self.get_logger().warn(
                    f"cv_bridge conversion failed ({exc})", throttle_duration_sec=5.0)
                return
            with self.lock:
                K = None if self._color_K is None else self._color_K.copy()
                D = None if self._color_D is None else self._color_D.copy()
            self._ingest_frame(bgr, K, D)

        def _on_ir_image(self, msg):
            """Left-IR image (mono8, rectified). Drives detection only when
            calib_frame=='ir'. Converted gray->BGR so the overlay + jpeg path is
            uniform; ``_detect`` re-grays it."""
            if self._calib_frame != "ir":
                return
            try:
                gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
            except Exception as exc:
                self.get_logger().warn(
                    f"cv_bridge IR conversion failed ({exc})", throttle_duration_sec=5.0)
                return
            bgr = self._cv2.cvtColor(gray, self._cv2.COLOR_GRAY2BGR)
            with self.lock:
                K = None if self._ir_K is None else self._ir_K.copy()
                D = None if self._ir_D is None else self._ir_D.copy()
            self._ingest_frame(bgr, K, D)

        def _on_info(self, msg):
            """Color camera_info -> cache color intrinsics (and mirror to active)."""
            np = self._np
            K = np.array(msg.k, float).reshape(3, 3)
            D = np.array(msg.d, float).flatten() if len(msg.d) else np.zeros(5)
            with self.lock:
                self._color_K = K
                self._color_D = D
                if self._calib_frame == "color":
                    self._K = K
                    self._D = D

        def _on_ir_info(self, msg):
            """IR camera_info -> cache rectified IR intrinsics. Prefer the P-block
            (rectified projection K, zero distortion) over msg.k, matching how
            realsense2_camera publishes infra1's rect intrinsics."""
            np = self._np
            P = np.array(msg.p, float).reshape(3, 4) if len(msg.p) >= 12 else None
            if P is not None and np.any(P[:3, :3]):
                K = P[:3, :3].copy()
            else:
                K = np.array(msg.k, float).reshape(3, 3)
            D = np.zeros(5)  # rectified -> no distortion
            with self.lock:
                self._ir_K = K
                self._ir_D = D
                if self._calib_frame == "ir":
                    self._K = K
                    self._D = D

        def _on_head_image(self, msg):
            """Cache the latest HEAD Orbbec color frame (warm-start anchor only;
            does NOT drive the wrist detection/stability path)."""
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"head cv_bridge failed ({exc})",
                                       throttle_duration_sec=10.0)
                return
            with self.lock:
                self._head_frame = bgr
                self._head_frame_stamp = self.get_clock().now().to_msg()

        def _on_head_info(self, msg):
            np = self._np
            K = np.array(msg.k, float).reshape(3, 3)
            D = np.array(msg.d, float).flatten() if len(msg.d) else np.zeros(5)
            with self.lock:
                self._head_K = K
                self._head_D = D

        def do_anchor_board(self):
            """Observe the fixed board with the HEAD Orbbec and record one
            T_base_board sample for the warm-start. Call multiple times (ideally
            from a few different pan/tilt head poses) to average down the head's
            pose-dependent bias; the running mean + scatter is stored on
            ``self._tbb_head`` / ``self._anchor_scatter``. Degrades to ok:False
            (never 500) when the head frame / intrinsics / TF / detection are
            missing."""
            from pan_tilt.calibration.aruco_detect import detect_pose
            with self.lock:
                bgr = None if self._head_frame is None else self._head_frame.copy()
                K = None if self._head_K is None else self._head_K.copy()
                D = None if self._head_D is None else self._head_D.copy()
                stamp = self._head_frame_stamp
                n_obs = len(self._anchor_obs)
            if bgr is None or K is None:
                return {"ok": False, "reason": "no head camera frame/intrinsics yet",
                        "n_anchor_obs": n_obs}
            det = detect_pose(bgr, K, D, board=self._board, detector=self._detector)
            if not det.success:
                return {"ok": False, "reason": "head saw no usable board",
                        "n_anchor_obs": n_obs}
            # detect_pose returns the board pose in the head OPTICAL frame.
            from rclpy.time import Time as _RclpyTime
            tf_time = (_RclpyTime.from_msg(stamp) if stamp is not None
                       else self._rclpy_time())
            try:
                tfm = self.tf_buffer.lookup_transform(
                    self._base_frame, self._head_optical_frame, tf_time)
            except Exception:
                try:
                    tfm = self.tf_buffer.lookup_transform(
                        self._base_frame, self._head_optical_frame, self._rclpy_time())
                except Exception as exc2:
                    return {"ok": False,
                            "reason": (f"TF {self._base_frame}->"
                                       f"{self._head_optical_frame} unavailable: {exc2}"),
                            "n_anchor_obs": n_obs}
            T_base_headopt = ws.tf_to_matrix(
                [tfm.transform.translation.x, tfm.transform.translation.y,
                 tfm.transform.translation.z],
                [tfm.transform.rotation.x, tfm.transform.rotation.y,
                 tfm.transform.rotation.z, tfm.transform.rotation.w])
            Tbb_obs = T_base_headopt @ det.pose_optical
            with self.lock:
                self._anchor_obs.append(Tbb_obs)
                mean, scatter = hs.average_board_anchors(self._anchor_obs)
                self._tbb_head = mean
                self._anchor_scatter = scatter
                n_obs = len(self._anchor_obs)
            return {"ok": True, "n_anchor_obs": n_obs, "scatter": scatter,
                    "reproj_px": float(det.reprojection_rms_px)}

        def do_clear_anchor(self):
            with self.lock:
                self._anchor_obs = []
                self._tbb_head = None
                self._anchor_scatter = None
            return {"ok": True, "n_anchor_obs": 0}

        def _on_ir_depth(self, msg):
            """Cache the latest native-IR FFS depth (float32 metres). The stream
            may publish 16UC1 (mm) or 32FC1 (m); coerce both to metres."""
            try:
                d = self.bridge.imgmsg_to_cv2(msg, "passthrough")
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"IR depth decode failed: {exc}",
                                       throttle_duration_sec=10.0)
                return
            np = self._np
            d = np.asarray(d)
            d = (d.astype(np.float32) / 1000.0 if d.dtype == np.uint16
                 else d.astype(np.float32, copy=False))
            with self.lock:
                self._ir_depth = d
                self._ir_depth_stamp = self.get_clock().now().to_msg()

        def _get_ir_depth(self):
            """Latest cached native-IR FFS depth (float32 m) or None if none yet."""
            with self.lock:
                return None if self._ir_depth is None else self._ir_depth

        def _ir_depth_too_stale(self, ref_stamp):
            """True when the cached native-IR depth lags the captured IR frame by
            more than ``ir_depth_max_age_s`` — guards against a frozen / dead depth
            stream feeding geometry from a stale instant (the do_capture steady
            recheck only proves the ARM is steady NOW, not that the depth frame is
            contemporaneous). Unknown stamps => NOT stale (lenient; the staleness
            guard targets an OLD stamp, not a missing one)."""
            with self.lock:
                ds = self._ir_depth_stamp
            if ds is None or ref_stamp is None:
                return False

            def _sec(s):
                return float(s.sec) + float(s.nanosec) * 1e-9
            return abs(_sec(ref_stamp) - _sec(ds)) > float(self._ir_depth_max_age_s)

        def _set_calib_frame(self, frame):
            """Switch the observation frame ('color'|'ir'). On an ACTUAL change,
            discard the frame-specific sample set + sidecars (each sample carries
            that frame's intrinsics — mixing color-K and IR-K samples corrupts the
            solve) and repoint the active intrinsics. Returns the resolved frame.
            A bad value or a same-frame call is a no-op (samples preserved)."""
            frame = str(frame).strip().lower()
            if frame not in ("color", "ir") or frame == self._calib_frame:
                return self._calib_frame
            with self.lock:
                self._calib_frame = frame
                self.session.samples.clear()
                self._thumbs.clear()
                self._sample_joints.clear()
                self._sample_ts.clear()
                self._sample_reproj_px.clear()
                self._sample_area_frac.clear()
                self._sample_depth_source.clear()
                # Detach from the current on-disk session (left intact as
                # history) so the next capture starts a fresh session entry.
                self._session_name = None
                self._cap = None
                self._last_corners_xy = None
                self._last_det = None
                self._stability.reset()
                self._stability_steady = False
                self._stability_since_frames = 0
                self._frame = None  # force a fresh frame from the new source
                self._K = self._ir_K if frame == "ir" else self._color_K
                self._D = self._ir_D if frame == "ir" else self._color_D
            self.get_logger().info(
                f"calib_frame -> {frame} (samples cleared; observing the "
                f"{'left-IR (native depth == camera_link)' if frame == 'ir' else 'color'} frame)")
            return self._calib_frame

        def _set_ir_emitter(self, enabled):
            """Set the wrist RealSense IR projector via the driver's settable
            param ``depth_module.emitter_enabled`` (async SetParameters client to
            the camera node). Returns ``{ok, reason}``. Degrades gracefully — a
            missing camera node / timeout / error just reports ``ok:False`` and
            never raises into the config request."""
            from rcl_interfaces.srv import SetParameters
            from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
            enabled = bool(enabled)
            svc = self._camera_node_name.rstrip("/") + "/set_parameters"
            try:
                cli = self.create_client(SetParameters, svc,
                                         callback_group=self._ffs_cb_group)
                try:
                    if not cli.wait_for_service(timeout_sec=float(self._emitter_wait_s)):
                        return {"ok": False,
                                "reason": f"camera node param service {svc} unavailable"}
                    req = SetParameters.Request()
                    req.parameters = [Parameter(
                        name="depth_module.emitter_enabled",
                        value=ParameterValue(type=ParameterType.PARAMETER_BOOL,
                                             bool_value=enabled))]
                    fut = cli.call_async(req)
                    ev = threading.Event()
                    fut.add_done_callback(lambda _f: ev.set())
                    if not ev.wait(timeout=float(self._emitter_wait_s) + 1.0):
                        return {"ok": False, "reason": "emitter set timed out"}
                    resp = fut.result()
                    results = list(getattr(resp, "results", []) or [])
                    ok = bool(results and all(r.successful for r in results))
                    self._ir_emitter_enabled = enabled if ok else self._ir_emitter_enabled
                    reason = ("ok" if ok else
                              (results[0].reason if results else "no result"))
                    return {"ok": ok, "reason": reason, "enabled": enabled}
                finally:
                    try:
                        self.destroy_client(cli)
                    except Exception:  # noqa: BLE001
                        pass
            except Exception as exc:  # noqa: BLE001 — never break the config call
                return {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}

        def do_set_config(self, **kw):
            """Live-apply runtime config from the web UI. All knobs optional;
            unknown keys ignored. Returns ``{ok, ...}`` (top-level ok reflects the
            non-emitter knobs, which always apply; the emitter result — which may
            fail if the camera node is absent — is a separate ``emitter`` sub-dict
            so a projector miss never reads as a total failure)."""
            def _num(key, cast, lo=None, hi=None):
                if key not in kw or kw[key] is None:
                    return None
                try:
                    v = cast(kw[key])
                except (TypeError, ValueError):
                    return None
                if lo is not None:
                    v = max(lo, v)
                if hi is not None:
                    v = min(hi, v)
                return v

            out = {"ok": True}
            if "calib_frame" in kw and kw["calib_frame"] is not None:
                out["calib_frame"] = self._set_calib_frame(kw["calib_frame"])
            if "use_ffs_depth" in kw and kw["use_ffs_depth"] is not None:
                self._use_ffs_depth = bool(kw["use_ffs_depth"])
            for key, cast, lo, hi in (
                ("depth_weight", float, 0.0, None),
                ("depth_sigma_m", float, 1e-4, None),
                ("depth_win", int, 0, None),
                ("depth_min_corners", int, 1, None),
                ("depth_z_min", float, 0.0, None),
                ("depth_z_max", float, 0.0, None),
            ):
                v = _num(key, cast, lo, hi)
                if v is not None:
                    setattr(self, "_" + key, v)
            if "ir_emitter_enabled" in kw and kw["ir_emitter_enabled"] is not None:
                out["emitter"] = self._set_ir_emitter(kw["ir_emitter_enabled"])
            self.get_logger().info(
                f"config applied: calib_frame={self._calib_frame} "
                f"use_ffs_depth={self._use_ffs_depth} depth_weight={self._depth_weight}")
            return out

        def config_dict(self):
            """Current runtime config for the WS state push (state.config)."""
            return {
                "calib_frame": self._calib_frame,
                "use_ffs_depth": bool(self._use_ffs_depth),
                "depth_weight": float(self._depth_weight),
                "depth_sigma_m": float(self._depth_sigma_m),
                "depth_win": int(self._depth_win),
                "depth_min_corners": int(self._depth_min_corners),
                "depth_z_min": float(self._depth_z_min),
                "depth_z_max": float(self._depth_z_max),
                "ir_emitter_enabled": self._ir_emitter_enabled,
                "ffs_ir_depth_topic": self._ffs_ir_depth_topic,
            }

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
            # SOLVEPNP_ITERATIVE uses DLT under the hood and requires >=6 point
            # correspondences. With fewer points cv2.solvePnP throws an
            # unrecoverable cv2.error that would otherwise propagate out of the
            # image callback and kill the node — keep the overlay alive but
            # skip the pose / capturable-sample branch.
            if obj_pts is None or len(obj_pts) < 6:
                return obs_px, {"corners": int(len(corner_idx)), "reproj_px": None}, None

            dist = D if D is not None else np.zeros(5)
            try:
                ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
            except cv2.error as exc:
                # Belt-and-suspenders: cv2 can still throw on degenerate point
                # configurations even with >=6 corners (e.g. all collinear).
                self.get_logger().warn(
                    f"solvePnP failed ({exc})", throttle_duration_sec=5.0)
                return obs_px, {"corners": int(len(corner_idx)), "reproj_px": None}, None
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

        # ---- IPPE-seeded re-PnP (consensus helper) ---------------------------

        def _pnp_ippe_refine(self, obj_pts, img_pts, K, D):
            """IPPE-seeded ITERATIVE PnP (planar two-fold-ambiguity safe).

            Returns (T_cam_board 4x4, reproj_px) or (None, None) on failure.
            Mirrors pan_tilt.aruco_detect._solve_iterative: IPPE seed picks the
            correct planar branch, ITERATIVE refines it."""
            np = self._np
            cv2 = self._cv2
            obj_pts = np.asarray(obj_pts, float).reshape(-1, 1, 3)
            img_pts = np.asarray(img_pts, float).reshape(-1, 1, 2)
            if len(obj_pts) < 6:
                return None, None
            try:
                n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    obj_pts, img_pts, K, D, flags=cv2.SOLVEPNP_IPPE)
                if not n_sol:
                    return None, None
                rvec, tvec = rvecs[0], tvecs[0]
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, K, D, rvec, tvec, useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE)
                if not ok:
                    return None, None
            except cv2.error:
                return None, None
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
            reproj = float(np.sqrt(np.mean(np.sum(
                (proj.reshape(-1, 2) - img_pts.reshape(-1, 2)) ** 2, axis=1))))
            T = np.eye(4)
            T[:3, :3] = cv2.Rodrigues(rvec)[0]
            T[:3, 3] = tvec.reshape(3)
            return T, reproj

        # ---- FoundationStereo depth -----------------------------------------

        def _try_ffs_depth(self):
            """Call the FFS get_depth service; return float32 depth (meters,
            color-aligned) or ``None`` on any failure.

            Mirrors ``object_detection_new.object_seg_yolo._try_ffs_depth``. The
            safety invariant here is NOT the callback group (this node spins a
            single-threaded executor, under which a ReentrantCallbackGroup grants
            no parallelism — it's kept only as future-proofing if the executor
            ever becomes multi-threaded). Safety comes purely from WHERE this
            runs: ``do_capture`` is only ever called from the FastAPI
            ``/api/capture`` worker or the CaptureSequenceRunner daemon — never
            the rclpy spin thread — so blocking the calling thread on a
            ``threading.Event`` (set from ``add_done_callback``) leaves the lone
            spin thread free to process the response and resolve the future. (If
            do_capture were ever called from a subscription/timer callback, this
            would deadlock the single-threaded executor instantly.)
            Returns ``None`` for: disabled, service unavailable within the wait
            window, call timeout, future exception, non-zero status, or decode
            failure — every one of which falls back to a monocular capture.
            """
            if not self._use_ffs_depth:
                return None
            # Outer guard: ANY rclpy-side failure — create_client on a malformed
            # ffs_service name, or wait_for_service / call_async hitting an
            # invalidated handle during a node-teardown race — must degrade to a
            # monocular capture, never raise out of do_capture (which would 500
            # the /api/capture request or flip the auto-capture sequence to
            # STEP_ERROR and abort the whole run). Review finding #2.
            try:
                return self._try_ffs_depth_inner()
            except Exception as exc:  # noqa: BLE001 — degrade to monocular
                self.get_logger().warn(
                    f"FFS depth acquisition failed "
                    f"({type(exc).__name__}: {exc}); monocular fallback",
                    throttle_duration_sec=10.0)
                return None

        def _try_ffs_depth_inner(self):
            np = self._np
            svc = self._ffs_service
            if self._ffs_cli is None or self._ffs_cli.srv_name != svc:
                try:
                    if self._ffs_cli is not None:
                        self.destroy_client(self._ffs_cli)
                except Exception:  # noqa: BLE001 — best-effort swap
                    pass
                self._ffs_cli = self.create_client(
                    FoundationStereoDepth, svc, callback_group=self._ffs_cb_group)
            if not self._ffs_cli.wait_for_service(timeout_sec=self._ffs_wait_for_service_s):
                return None
            req = FoundationStereoDepth.Request()
            req.align_to_color = True  # depth in the color optical frame == K's frame
            fut = self._ffs_cli.call_async(req)
            event = threading.Event()
            fut.add_done_callback(lambda _f: event.set())
            if not event.wait(timeout=self._ffs_call_timeout_s):
                try:
                    self._ffs_cli.remove_pending_request(fut)
                except Exception:  # noqa: BLE001
                    pass
                return None
            try:
                resp = fut.result()
            except Exception as exc:  # noqa: BLE001 — log + monocular fallback
                self.get_logger().warn(f"FFS call raised: {exc}",
                                       throttle_duration_sec=10.0)
                return None
            if resp is None or resp.status != 0:
                return None
            try:
                depth = self.bridge.imgmsg_to_cv2(resp.depth_image, "passthrough")
            except Exception as exc:  # noqa: BLE001 — bad encoding -> fallback
                self.get_logger().warn(f"FFS depth decode failed: {exc}",
                                       throttle_duration_sec=10.0)
                return None
            # Service guarantees 32FC1 meters; coerce defensively.
            return np.asarray(depth).astype(np.float32, copy=False)

        def _note_ffs_depth_outcome(self, depth_source):
            """One-time operator hint when FFS is enabled but never yields depth.

            The most common cause is the wrist RealSense being launched WITHOUT
            the infra1/infra2 IR streams FoundationStereo needs — it then returns
            status=1 ('no synced stereo frame') on every call, ``_try_ffs_depth``
            returns ``None``, and every capture silently records
            ``depth_source='unavailable'`` while the solve runs monocular. Fires
            one WARN after ``ffs_depth_warn_after`` consecutive depth-less
            captures so the operator isn't silently downgraded (review #5)."""
            if depth_source == "ffs":
                self._ffs_depth_miss_streak = 0
                return
            self._ffs_depth_miss_streak += 1
            if (not self._ffs_depth_warned
                    and self._ffs_depth_miss_streak >= self._ffs_depth_warn_after):
                self._ffs_depth_warned = True
                if self._calib_frame == "ir":
                    advice = (
                        "Most likely the native-IR FFS depth stream isn't running: "
                        "launch foundation_stereo with 'stream_enabled:=true "
                        "stream_align_to_color:=false' (publishing on "
                        f"{self._ffs_ir_depth_topic}), or point ffs_ir_depth_topic "
                        "at the RealSense-native /camera/xarm_camera/depth/image_rect_raw.")
                else:
                    advice = (
                        "Most likely the wrist RealSense was launched without the "
                        "infra1/infra2 IR streams FoundationStereo needs: relaunch it "
                        "with 'enable_infra1:=true enable_infra2:=true' and confirm the "
                        "foundation_stereo node is up (ros2 launch foundation_stereo "
                        "foundation_stereo.launch.py).")
                self.get_logger().warn(
                    f"use_ffs_depth=True but the last {self._ffs_depth_miss_streak} "
                    f"captures got NO FFS depth (last source='{depth_source}', "
                    f"calib_frame={self._calib_frame}) — the solve is running MONOCULAR. "
                    f"{advice} Set use_ffs_depth:=false to silence this.")

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
                # Active intrinsics for the Coverage canvas (3x3 -> JSON list).
                K_snapshot = (None if self._K is None
                              else [[float(v) for v in row] for row in self._K])
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
                depth_src_by_idx = dict(self._sample_depth_source)
                # Cached solve payload (rehydrates the Solve tab on reconnect) +
                # live MAD progress snapshot (shallow-copy rejection_log list).
                last_solve_payload = self._last_solve_payload
                solve_progress = dict(self._solve_progress)
                solve_progress["rejection_log"] = list(
                    solve_progress.get("rejection_log") or [])

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
                    depth_source=depth_src_by_idx.get(i),
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

            payload = ws.enriched_state_payload(
                camera_connected=camera_connected,
                intrinsics_ok=intrinsics_ok,
                num_samples=num_samples,
                last_detection=last_det,
                status_msg="ok",
                frame_count=frame_count,
                frame_hz=frame_hz,
                frame_age_sec=frame_age_sec,
                image_topic=(self._ir_image_topic if self._calib_frame == "ir"
                             else image_topic),
                ros_domain_id=ros_domain_id,
                t_base_ee=t_base_ee,
                xarm_joint_positions=xarm,
                board=board,
                safety_envelope=safety_envelope,
                stability=stability,
                samples=samples_metadata,
                diversity=diversity,
                last_solve=last_solve_payload,  # cached JSON payload (rehydrate)
                safety_preview=safety_preview,
                waypoints=waypoints_meta,
                sequence=sequence_state,
                K=K_snapshot,
                solve_progress=solve_progress,
            )
            # Runtime config (calib_frame + depth knobs + emitter) for the
            # Settings controls on the Info tab.
            payload["config"] = self.config_dict()
            with self.lock:
                payload["anchor"] = {
                    "have": self._tbb_head is not None,
                    "n_obs": len(self._anchor_obs),
                    "scatter": self._anchor_scatter,
                }
                # Multi-placement metadata: summary list + active id for the UI.
                payload["active_placement_id"] = self._active_placement_id
                payload["placements"] = [
                    {
                        "id": pid,
                        "label": p.label,
                        "n_samples": len(p.session.samples),
                        "anchor_have": p.tbb_head is not None,
                    }
                    for pid, p in self._placements.items()
                ]
            return payload

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

            # SafetyEnvelope pre-check REMOVED 2026-06-20 at operator request —
            # the calibration workflow needs poses that the envelope's
            # mast-cylinder / z-floor heuristics flag as violations (looking
            # down at a board near the base, EE behind the mast for diversity,
            # etc.). The arm driver / MoveIt's collision checker remains the
            # real safety boundary. `safety_preview` still publishes the
            # informational verdict to state.safety_preview so the operator
            # can glance at the verdict in the left-dock — it just no longer
            # blocks. To re-enable the block, restore the validate() branch.

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
                target = self._capture_min_steady_frames
                cap, K, frame = self._cap, self._K, self._frame
                frame_stamp = self._frame_stamp
                snap_frame = self._calib_frame   # guard against a mid-capture switch
                joints_snapshot = (None if self._xarm_joint_positions is None
                                   else list(self._xarm_joint_positions))
                now_mono = self._last_frame_monotonic
            if not steady or since < target:
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

            # Look up EEF TF. The xArm TF publisher runs at ~4 Hz so the
            # precise image-timestamp lookup almost always fails with
            # "extrapolation into future" (image stamp = ROS-now at receipt,
            # but latest TF data is 150–250 ms behind). Since the arm is
            # stationary at capture the latest available transform IS correct.
            # We only emit a warning when the lag exceeds 1 s, which would
            # indicate the arm was still settling or the TF source died.
            from rclpy.time import Time as _RclpyTime
            tf_time = (_RclpyTime.from_msg(frame_stamp) if frame_stamp is not None
                       else self._rclpy_time())
            try:
                tfm = self.tf_buffer.lookup_transform(
                    self._base_frame, self._eef_frame, tf_time)
            except Exception:
                try:
                    tfm = self.tf_buffer.lookup_transform(
                        self._base_frame, self._eef_frame, self._rclpy_time())
                except Exception as exc2:
                    return {"ok": False,
                            "reason": (f"TF {self._base_frame}->{self._eef_frame}"
                                       f" unavailable: {exc2}"),
                            "num_samples": len(self.session.samples)}
                if frame_stamp is not None:
                    tf_s = (tfm.header.stamp.sec
                            + tfm.header.stamp.nanosec * 1e-9)
                    img_s = (frame_stamp.sec
                             + frame_stamp.nanosec * 1e-9)
                    lag_ms = (img_s - tf_s) * 1000.0
                    if lag_ms > 1000.0:
                        self.get_logger().warn(
                            f"TF is {lag_ms:.0f} ms stale at capture — "
                            "arm may still be settling; sample quality suspect",
                            throttle_duration_sec=5.0)
            T_base_eef = ws.tf_to_matrix(
                [tfm.transform.translation.x, tfm.transform.translation.y,
                 tfm.transform.translation.z],
                [tfm.transform.rotation.x, tfm.transform.rotation.y,
                 tfm.transform.rotation.z, tfm.transform.rotation.w])

            # Multi-frame consensus (pan_tilt parity): average the last N steady
            # detections' corners and re-PnP, replacing the single-shot cap so
            # the stored obs_px AND T_cam_board are denoised. Falls back to the
            # single-frame cap when consensus can't reach quorum.
            np = self._np
            with self.lock:
                hist = list(self._det_history)
            cons_ids, cons_px = hs.consensus_corners(
                hist, min_frac=self._consensus_min_frac)
            n_consensus = 0
            if cons_ids is not None and K is not None:
                try:
                    obj_pts, img_pts = self._board.matchImagePoints(
                        cons_px.reshape(-1, 1, 2).astype(np.float32),
                        cons_ids.reshape(-1, 1).astype(np.int32))
                except Exception:
                    obj_pts = None
                if obj_pts is not None and len(obj_pts) >= 6:
                    T_c, reproj_c = self._pnp_ippe_refine(
                        obj_pts, img_pts, K, (self._D if self._D is not None
                                              else np.zeros(5)))
                    if T_c is not None:
                        h, w = frame.shape[:2]
                        xs, ys = cons_px[:, 0], cons_px[:, 1]
                        area_frac = (float((xs.max() - xs.min()) *
                                           (ys.max() - ys.min())) / float(h * w))
                        cap = {"T_cam_board": T_c, "obs_px": cons_px,
                               "corner_idx": cons_ids, "reproj_px": reproj_c,
                               "area_frac": area_frac}
                        n_consensus = len(hist)

            # FFS metric depth at the detected corners. The arm is settled +
            # static here, so the FFS stereo view matches the cached color frame
            # — but get_depth can block up to ffs_call_timeout_s, so we RE-CHECK
            # steadiness after it returns and drop the depth if the pose moved
            # during the call (the fresh stereo view would no longer correspond
            # to the cached corners). Degrades to monocular on any failure —
            # depth is a refinement, never an admission gate.
            obs_xyz_cam, obs_xyz_valid = None, None
            depth_source = "disabled"
            if self._use_ffs_depth:
                # IR mode samples the cached native-IR FFS stream; color mode
                # calls the color-aligned get_depth service. Either way the depth
                # is in the ACTIVE frame, matching self._K used to deproject.
                _stale = False
                if snap_frame == "ir":
                    depth = self._get_ir_depth()
                    if depth is not None and self._ir_depth_too_stale(frame_stamp):
                        # A frozen/dead native-IR depth stream would otherwise feed
                        # geometry from a stale instant; drop it -> monocular.
                        self.get_logger().warn(
                            "native-IR depth stale vs the IR frame; dropping depth "
                            "(monocular this pose)", throttle_duration_sec=10.0)
                        depth, _stale = None, True
                else:
                    depth = self._try_ffs_depth()
                if depth is None:
                    depth_source = "ir-depth-stale" if _stale else "unavailable"
                elif depth.shape[:2] != frame.shape[:2]:
                    depth_source = "shape-mismatch"
                    self.get_logger().warn(
                        f"FFS depth {depth.shape[:2]} != color {frame.shape[:2]}; "
                        "capturing monocular this pose",
                        throttle_duration_sec=10.0)
                else:
                    with self.lock:
                        still_steady = self._stability_steady
                    if not still_steady:
                        depth_source = "moved-during-ffs"
                        self.get_logger().warn(
                            "pose moved during the FFS depth call; dropping depth "
                            "(monocular this pose)", throttle_duration_sec=10.0)
                    else:
                        from handeye_calib import depth_sample as _hds
                        xyz, valid = _hds.deproject_corners(
                            cap["obs_px"], depth, K, win=self._depth_win,
                            z_min=self._depth_z_min, z_max=self._depth_z_max)
                        n_valid = int(np.count_nonzero(valid))
                        if n_valid >= self._depth_min_corners:
                            obs_xyz_cam, obs_xyz_valid = xyz, valid
                            depth_source = "ffs"
                            self.get_logger().info(
                                f"capture: FFS depth {n_valid}/{len(valid)} "
                                f"corners valid")
                        else:
                            depth_source = "ffs-too-sparse"
                            self.get_logger().info(
                                f"capture: FFS depth only {n_valid}/{len(valid)} "
                                f"corners valid (<{self._depth_min_corners}); "
                                "monocular this pose")
                # One-time operator hint if FFS is on but never delivers depth.
                self._note_ffs_depth_outcome(depth_source)

            # Commit ATOMICALLY under the lock, re-checking that calib_frame
            # hasn't switched since the snapshot. A concurrent /api/config frame
            # switch clears the session under the same lock; without this guard
            # the (stale-frame) sample could land in the freshly-cleared
            # new-frame session — the exact color-K/IR-K mixing the clear
            # prevents. try_add + the sidecar writes share one critical section
            # so the index can't drift either.
            with self.lock:
                if self._calib_frame != snap_frame:
                    return {"ok": False,
                            "reason": "calib_frame switched during capture; sample dropped",
                            "num_samples": len(self.session.samples)}
                self._last_depth_source = depth_source
                ok, reason = self.session.try_add(
                    T_base_eef, cap["T_cam_board"], cap["obs_px"], cap["corner_idx"],
                    n_corners=len(cap["corner_idx"]), reproj_px=cap["reproj_px"],
                    area_frac=cap["area_frac"],
                    obs_xyz_cam=obs_xyz_cam, obs_xyz_valid=obs_xyz_valid)
                if ok:
                    # Downscaled JPEG of the overlayed frame + the joint snapshot
                    # for the new sample. Index matches session.samples[-1].
                    idx = len(self.session.samples) - 1
                    try:
                        annotated = ws.draw_charuco_overlay(
                            frame, cap["obs_px"], ids=cap["corner_idx"],
                            rms_px=cap["reproj_px"],
                            image_topic=(self._ir_image_topic if snap_frame == "ir"
                                         else self._image_topic))
                        jpg = ws.encode_jpeg(_downscale(annotated, 320, self._cv2))
                    except Exception as exc:  # pragma: no cover - thumb is best-effort
                        self.get_logger().warn(
                            f"thumb encode failed for sample {idx} ({exc})")
                        jpg = None
                    if jpg is not None:
                        self._thumbs[idx] = jpg
                    self._sample_joints[idx] = joints_snapshot
                    self._sample_ts[idx] = (float(now_mono) if now_mono is not None
                                            else float(self._time.monotonic()))
                    self._sample_reproj_px[idx] = float(cap["reproj_px"])
                    self._sample_area_frac[idx] = float(cap["area_frac"])
                    self._sample_depth_source[idx] = depth_source
                num = len(self.session.samples)
            if ok:
                # Persist the capture to disk immediately (outside the lock —
                # _persist_session re-acquires it) so a restart/crash can't lose
                # it and the history browser can re-open it.
                self._persist_session()
            return {"ok": ok, "reason": reason, "depth_source": depth_source,
                    "n_consensus_frames": n_consensus,
                    "num_samples": num}

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
                self._active_placement.thumbs = {
                    (k if k < idx else k - 1): v
                    for k, v in self._thumbs.items() if k != idx
                }
                self._active_placement.sample_joints = {
                    (k if k < idx else k - 1): v
                    for k, v in self._sample_joints.items() if k != idx
                }
                self._active_placement.sample_ts = {
                    (k if k < idx else k - 1): v
                    for k, v in self._sample_ts.items() if k != idx
                }
                self._active_placement.sample_reproj_px = {
                    (k if k < idx else k - 1): v
                    for k, v in self._sample_reproj_px.items() if k != idx
                }
                self._active_placement.sample_area_frac = {
                    (k if k < idx else k - 1): v
                    for k, v in self._sample_area_frac.items() if k != idx
                }
                self._active_placement.sample_depth_source = {
                    (k if k < idx else k - 1): v
                    for k, v in self._sample_depth_source.items() if k != idx
                }
                num = len(samples)
            # Re-persist (outside the lock) so session.json + thumbs/ on disk
            # match the recompacted 0..N-1 indexing after the delete.
            self._persist_session()
            return {"ok": True, "num_samples": num}

        def sample_thumb(self, idx):
            """Return the cached JPEG bytes for sample ``idx`` or ``None``."""
            with self.lock:
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    return None
                return self._thumbs.get(idx)

        def do_solve(self, method: str = "auto", reject_sigma="default"):
            """Run the hand-eye solve with optional method picker.

            ``method`` is one of ``"auto"`` (default — sweep all five OpenCV
            methods and pick the lowest reproj RMS) or one of the canonical
            method names: ``TSAI`` / ``PARK`` / ``HORAUD`` / ``ANDREFF`` /
            ``DANIILIDIS``. Unknown values fall back to ``auto`` (the brief's
            preferred-over-error stance, mirroring the rest of this node's
            degrade-rather-than-422 philosophy). On success returns
            ``{"ok": True, **solve_payload_v2(...)}``; on degraded paths returns
            ``{"ok": False, "reason": ...}``.

            ``reject_sigma``: sentinel ``"default"`` (default) lets
            :func:`handeye_solve.solve` use its own default (2.5 — default-on
            per-axis MAD rejection).  Pass ``None`` to disable rejection
            entirely; pass a float to override the threshold.
            """
            with self.lock:
                samples, K, D = list(self.session.samples), self._K, self._D
                anchor = None if self._tbb_head is None else self._tbb_head.copy()
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
            # Translate the sentinel to a keyword for hs.solve:
            #   "default" → omit reject_sigma (solver uses its own default 2.5)
            #   None      → explicitly disable rejection
            #   float     → use that threshold
            if reject_sigma == "default":
                rs_kwarg = {}
            elif reject_sigma is None:
                rs_kwarg = {"reject_sigma": None}
            else:
                rs_kwarg = {"reject_sigma": float(reject_sigma)}
            # Live MAD-rejection progress: reset, then stream each drop into
            # _solve_progress (read by get_state_dict -> WS push) so the UI shows
            # rejections AS THEY HAPPEN. _cb runs on this (executor) thread; the
            # state read runs on the loop thread; self.lock serializes both.
            with self.lock:
                self._solve_progress = {
                    "running": True, "phase": "start", "n_orig": len(samples),
                    "n_active": len(samples), "min_keep": 0, "iteration": 0,
                    "rejection_log": [], "last_drop": None, "solve_ts": 0.0,
                }

            def _cb(ev):
                with self.lock:
                    sp = self._solve_progress
                    sp["running"] = True
                    sp["phase"] = ev.get("phase", sp["phase"])
                    for k in ("n_orig", "n_active", "min_keep", "iteration"):
                        if ev.get(k) is not None:
                            sp[k] = ev[k]
                    if ev.get("phase") == "rejecting" and ev.get("last_drop"):
                        sp["last_drop"] = ev["last_drop"]
                        sp["rejection_log"] = sp["rejection_log"] + [ev["last_drop"]]

            try:
                res = hs.solve(samples, K, D, self._board_pts,
                               methods=methods_subset,
                               **rs_kwarg,
                               depth_weight=self._depth_weight,
                               depth_sigma_m=self._depth_sigma_m,
                               anchor_Tbb=anchor,
                               progress_cb=_cb)
            except Exception as exc:
                # seed_handeye raises when every OpenCV method fails (e.g. all
                # samples colinear); bundle_adjust may also blow up on a
                # singular Jacobian. Surface the failure as a JSON-safe ok:False
                # rather than a 500 the UI would render as a JSON.parse crash.
                with self.lock:
                    self._solve_progress["running"] = False
                    self._solve_progress["phase"] = "error"
                return {"ok": False,
                        "reason": f"solve failed ({type(exc).__name__}): {exc} — "
                                  "try collecting more diverse poses (vary EE rotation, "
                                  "not just translation)"}
            if not (np.all(np.isfinite(res.X)) and np.all(np.isfinite(res.Tbb))):
                # Numerically valid (no exception) but the recovered transform
                # is non-finite — JSONResponse would render plain-text 500.
                # Don't cache it (Promote tab must not diff against NaN).
                with self.lock:
                    self._solve_progress["running"] = False
                    self._solve_progress["phase"] = "error"
                return {"ok": False,
                        "reason": "solve produced non-finite transform — add more "
                                  "diverse waypoints (vary EE rotation, not just "
                                  "translation) and re-solve"}
            self.last_solve = res
            self.get_logger().info(
                f"solve: seed_used={res.seed_used!r}, status={res.status}"
                + ("" if anchor is None else " (head anchor available)"))
            payload = ws.solve_payload_v2(res, samples, K, D, self._board_pts)
            # Surface outlier-rejection results at the top level — the
            # per_method_summary projection only carries (name, reproj_px),
            # so the rejected_indices field gets dropped silently.
            rej_idx = []
            for m in (res.per_method or []):
                if m.get("name") == "rejected_indices":
                    rej_idx = list(m.get("rejected_indices") or [])
                    break
            payload["rejected_indices"] = rej_idx
            # Per-sample diagnostic to stderr so the operator can spot data-side
            # outliers (one bad sample dragging RMSE up) without UI changes.
            # Format: "[handeye] per-sample reproj_px: [1.2, 8.5, ...]" — easy
            # to eyeball; an outlier > 5x median is almost certainly the cause.
            try:
                ps = payload.get("per_sample_reproj_px", [])
                fmt = ", ".join(f"{v:.2f}" for v in ps if v is not None and np.isfinite(v))
                # Per-sample camera-to-board distance (norm of T_cam_board
                # translation). When board geometry matches the configured
                # square_len_m, this clusters tightly around the physical
                # distance (~0.5m typical). When square_len is wrong, PnP
                # returns scale-wrong poses whose magnitude varies sample-
                # to-sample with the detected corner subset — values will
                # spread 20%+. Operator-facing root-cause hint.
                dists = [float(np.linalg.norm(s.T_cam_board[:3, 3])) for s in samples]
                d_fmt = ", ".join(f"{d:.3f}" for d in dists)
                d_min, d_max = (min(dists), max(dists)) if dists else (0, 0)
                d_spread_pct = 100.0 * (d_max - d_min) / max(d_min, 1e-6)
                # Pan-tilt-style per-drop "why" lines (residual + robust-z).
                rej_log = payload.get("rejection_log") or []
                if rej_log:
                    for r in rej_log:
                        self.get_logger().info(
                            f"  MAD reject sample #{r['idx']}: residual "
                            f"{r['trans_mm']:.1f} mm / {r['rot_deg']:.2f} deg / "
                            f"{r['reproj_px']:.1f} px (robust-z trans {r['z_trans']:.2f}, "
                            f"rot {r['z_rot']:.2f}, reproj {r['z_reproj']:.2f})")
                rej_str = (f" | rejected_indices={payload['rejected_indices']}"
                           if payload.get("rejected_indices") else "")
                # FFS-depth usage: how many samples carried metric depth, and the
                # depth-grounded accuracy when present (in-sample — no split).
                n_depth_samples = sum(
                    1 for s in samples if getattr(s, "obs_xyz_cam", None) is not None)
                depth_rmse = payload["metrics_mm_deg"].get("depth_point_rmse_mm")
                if n_depth_samples == 0:
                    depth_str = f" | depth: 0/{len(samples)} samples (MONOCULAR solve)"
                else:
                    hd = (f"{depth_rmse:.2f}mm" if depth_rmse is not None else "n/a")
                    depth_str = (f" | depth: {n_depth_samples}/{len(samples)} samples, "
                                 f"depth_rmse={hd} (w={self._depth_weight})")
                self.get_logger().info(
                    f"solve: N={len(samples)} method={method_str} "
                    f"trans={payload['metrics_mm_deg'].get('trans_rmse_mm', float('nan')):.2f}mm "
                    f"rot={payload['metrics_mm_deg'].get('rot_rmse_deg', float('nan')):.2f}deg "
                    f"reproj={payload['metrics_mm_deg'].get('reproj_px', float('nan')):.2f}px "
                    f"(over {len(samples) - len(payload.get('rejected_indices') or [])} kept) | "
                    f"per-sample reproj_px=[{fmt}] | "
                    f"cam-board-dist_m=[{d_fmt}] (spread {d_spread_pct:.1f}%)"
                    f"{depth_str}{rej_str}")
            except Exception:
                pass  # diagnostic only — never fail the solve over a log line
            try:
                self._dump_solve(res, payload)
            except Exception as exc:  # persistence is best-effort, never blocks
                self.get_logger().warn(f"solve dump failed: {exc}")
            # Fold the solve result into the browsable on-disk session too, so the
            # history entry shows its verdict + per-sample residuals.
            self._persist_session(res, payload)
            # Cache the JSON payload + stamp it so the WS push can rehydrate the
            # Solve tab on reconnect/reload (not just on the POST response), and
            # mark progress done.
            solve_ts = float(self._time.monotonic())
            cached = dict(payload)
            cached["solve_ts"] = solve_ts
            with self.lock:
                self._last_solve_payload = cached
                self._solve_progress["running"] = False
                self._solve_progress["phase"] = "done"
                self._solve_progress["solve_ts"] = solve_ts
            return {"ok": True, **payload}

        def _build_session_dict(self, res=None, payload=None, combined_res=None, combined_payload=None):
            """Serialize all placements (+ optional solve result) into the
            canonical v2 session dict written to disk. ``res``/``payload`` None at
            capture time (samples only); populated at solve time for the active
            placement. ``combined_res``/``combined_payload`` carry the result of a
            multi-placement bundle-adjust (T5+)."""
            import time
            np = self._np

            def m2l(M):
                return None if M is None else np.asarray(M, float).tolist()

            placements_out = []
            with self.lock:
                placements_snapshot = list(self._placements.items())
                active_pid = self._active_placement_id

            for pid, p in placements_snapshot:
                samp = []
                for i, s in enumerate(p.session.samples):
                    samp.append({
                        "idx": i,
                        "T_base_eef": m2l(s.T_base_eef),
                        "T_cam_board": m2l(s.T_cam_board),
                        "obs_px": m2l(s.obs_px),
                        "corner_idx": np.asarray(s.corner_idx).astype(int).tolist(),
                        "obs_xyz_cam": m2l(getattr(s, "obs_xyz_cam", None)),
                        "capture_reproj_px": p.sample_reproj_px.get(i),
                        "depth_source": p.sample_depth_source.get(i),
                        "joints": p.sample_joints.get(i),
                    })
                # Per-placement result: only if this is the active placement and res is provided
                placement_result = None
                if pid == active_pid and res is not None:
                    pp = payload or {}
                    placement_result = {
                        "status": res.status,
                        "seed_used": getattr(res, "seed_used", ""),
                        "rejected_sample_indices": list(getattr(res, "rejected_indices", None) or []),
                        "rejection_log": list(getattr(res, "rejection_log", None) or []),
                        "X_eef_cam": m2l(res.X),
                        "Tbb_base_board": m2l(res.Tbb),
                        "metrics": res.metrics,
                        "per_sample_reproj_px": pp.get("per_sample_reproj_px"),
                        "per_sample_trans_mm": pp.get("per_sample_trans_mm"),
                        "per_sample_rot_deg": pp.get("per_sample_rot_deg"),
                        "observability": pp.get("observability"),
                        "X_xyz_mm": pp.get("X_xyz_mm"),
                        "X_rpy_deg": pp.get("X_rpy_deg"),
                    }
                placements_out.append({
                    "id": pid,
                    "label": p.label,
                    "anchor_have": p.tbb_head is not None,
                    "Tbb_head": m2l(p.tbb_head),
                    "samples": samp,
                    "result": placement_result,
                })

            with self.lock:
                out = {
                    "schema": "wrist_handeye_session/2",
                    "timestamp": time.strftime("%Y%m%d_%H%M%S"),
                    "robot": self._robot_name,
                    "calib_frame": self._calib_frame,
                    "board": {"squares_x": self._sx, "squares_y": self._sy,
                              "square_len_m": self._sq,
                              "aruco_dict": self._aruco_dict_name},
                    "K": m2l(self._K), "D": m2l(self._D),
                    "active_placement": active_pid,
                    "placements": placements_out,
                }
            # Combined result (from multi-placement solve)
            if combined_res is not None:
                cp = combined_payload or {}
                out["combined_result"] = {
                    "status": combined_res.status,
                    "X_eef_cam": m2l(combined_res.X),
                    "combined_metrics": combined_res.combined_metrics,
                    "seed_placement_id": combined_res.seed_placement_id,
                }
            else:
                out["combined_result"] = None
            return out

        def _persist_session(self, res=None, payload=None, combined_res=None, combined_payload=None):
            """Write all placements (+ optional solve) to the on-disk session
            dir so a restart/crash never loses captures and the history browser
            can re-open them. Lazily names the session at the first persisted
            capture. Best-effort: a disk hiccup must never break capture/solve."""
            try:
                import os as _os
                import time
                with self.lock:
                    has_any_samples = any(len(p.session.samples) > 0
                                          for p in self._placements.values())
                    if not has_any_samples and self._session_name is None:
                        return  # nothing to persist yet
                    if self._session_name is None:
                        # Timestamped to the second; dedup against existing dirs
                        # so a clear+recapture within one second can't collide.
                        base_name = hsx.new_session_name(time.localtime())
                        name = base_name
                        k = 1
                        while _os.path.isdir(hsx.session_dir(name)):
                            k += 1
                            name = f"{base_name}_{k}"
                        self._session_name = name
                    name = self._session_name
                    placements_snapshot = list(self._placements.items())
                # _build_session_dict acquires self.lock internally for its own snapshot
                out = ws.json_safe(self._build_session_dict(res, payload, combined_res, combined_payload))
                hsx.write_session(name, out)
                for pid, p in placements_snapshot:
                    hsx.rewrite_placement_thumbs(name, pid, p.thumbs)
                return name
            except Exception as exc:  # pragma: no cover - persistence is best-effort
                self.get_logger().warn(f"session persist failed: {exc}")
                return None

        def _dump_solve(self, res, payload):
            """Flat per-solve replay archive (one timestamped file under
            ``$HANDEYE_DUMP_DIR``/``calibration_data`` ``/wrist_handeye_dumps``),
            kept alongside the browsable session dir for offline re-analysis.
            Best-effort — the caller guards exceptions."""
            import os
            import json
            import time
            root = os.environ.get("HANDEYE_DUMP_DIR", "calibration_data")
            ddir = os.path.join(root, "wrist_handeye_dumps")
            os.makedirs(ddir, exist_ok=True)
            out = self._build_session_dict(res, payload)
            path = os.path.join(ddir, f"solve_{time.strftime('%Y%m%d_%H%M%S')}.json")
            with open(path, "w") as f:
                json.dump(ws.json_safe(out), f, indent=2)
            self.get_logger().info(f"solve dump saved: {os.path.abspath(path)}")

        def do_load_session(self, name):
            """Re-hydrate a historical capture session into the LIVE state so it
            can be inspected and RE-SOLVED with the current solver. Restores all
            placements with their samples, per-sample sidecars, thumbnails,
            intrinsics/board, and head anchors from disk, and points the active
            session name at ``name`` so a subsequent solve folds its result back
            into the same history entry. Handles both v1 (single-placement flat)
            and v2 (multi-placement) session schemas."""
            import os as _os
            np = self._np
            try:
                data = hsx.read_session(name)
            except Exception as exc:
                return {"ok": False, "reason": f"load failed: {exc}"}

            def l2m(v):
                return None if v is None else np.asarray(v, float)

            # V1 compat: synthesize v2 structure from legacy flat format
            if "placements" not in data:
                data = {**data, "schema": "wrist_handeye_session/2",
                        "placements": [{"id": "default", "label": "default",
                                        "anchor_have": data.get("anchor_have", False),
                                        "Tbb_head": data.get("Tbb_head"),
                                        "samples": data.get("samples") or [],
                                        "result": data.get("result")}],
                        "active_placement": "default"}

            cf = data.get("calib_frame")
            t0 = float(self._time.monotonic())
            new_placements = {}
            for pd in data["placements"]:
                pid = pd.get("id", "default")
                label = pd.get("label", pid)
                raw_samples = pd.get("samples") or []
                rebuilt = []
                for s in raw_samples:
                    rebuilt.append(hm.Sample(
                        np.asarray(s["T_base_eef"], float),
                        np.asarray(s["T_cam_board"], float),
                        np.asarray(s["obs_px"], float),
                        np.asarray(s["corner_idx"], int),
                        obs_xyz_cam=l2m(s.get("obs_xyz_cam")),
                        obs_xyz_valid=None))
                p = make_placement(label,
                                   min_diversity_deg=self._diversity_target_deg,
                                   min_corners=self._min_corners,
                                   max_reproj_px=self._max_reproj_px,
                                   min_area_frac=self._min_area_frac)
                p.session.samples = rebuilt
                for i, s in enumerate(raw_samples):
                    p.sample_reproj_px[i] = s.get("capture_reproj_px")
                    p.sample_depth_source[i] = s.get("depth_source")
                    p.sample_joints[i] = s.get("joints")
                    p.sample_ts[i] = t0 + i
                    try:
                        # Placement-aware path first; fall back to v1 legacy flat path
                        tp = hsx.placement_thumb_path(name, pid, i)
                        if not _os.path.isfile(tp):
                            tp = hsx.thumb_path(name, i)
                        if _os.path.isfile(tp):
                            with open(tp, "rb") as f:
                                p.thumbs[i] = f.read()
                    except Exception:
                        pass  # a missing thumb just shows the placeholder
                if pd.get("Tbb_head") is not None:
                    p.tbb_head = np.asarray(pd["Tbb_head"], float)
                new_placements[pid] = p

            with self.lock:
                self._placements = new_placements
                self._active_placement_id = data.get(
                    "active_placement", next(iter(new_placements)))
                # Restore intrinsics + board from the session so an immediate
                # re-solve uses the geometry the samples were captured with.
                b = data.get("board") or {}
                if b:
                    self._sx = int(b.get("squares_x", self._sx))
                    self._sy = int(b.get("squares_y", self._sy))
                    self._sq = float(b.get("square_len_m", self._sq))
                    self._board_pts = hm.board_corners(self._sx, self._sy, self._sq)
                if data.get("K") is not None:
                    self._K = np.asarray(data["K"], float)
                self._D = l2m(data.get("D"))
                if cf in ("color", "ir"):
                    self._calib_frame = cf
                self._session_name = name
                self.last_solve = None  # stale: belongs to a different capture

            total_samples = sum(len(p.session.samples) for p in new_placements.values())
            self.get_logger().info(
                f"loaded session {name!r}: {total_samples} samples across "
                f"{len(new_placements)} placement(s) (calib_frame={cf})")
            return {"ok": True, "name": name, "num_samples": total_samples,
                    "n_placements": len(new_placements), "calib_frame": cf}

        def do_delete_session(self, name):
            """Delete a historical session directory. Detaches the live session
            name if it pointed at the deleted one."""
            try:
                existed = hsx.delete_session(name)
            except Exception as exc:
                return {"ok": False, "reason": f"delete failed: {exc}"}
            with self.lock:
                if self._session_name == name:
                    self._session_name = None
            return {"ok": bool(existed), "deleted": name}

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

        def do_clear_waypoints(self):
            """Remove ALL waypoints from the in-memory store.

            Returns ``{ok: True, cleared: N}``.
            """
            with self.lock:
                n = len(self.waypoint_store.list())
                self.waypoint_store.clear()
            return {"ok": True, "cleared": n}

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
                              settle_timeout_s: float = None):
            """Kick off the auto-capture state machine.

            ``settle_timeout_s=None`` (default) uses the ``settle_timeout_s``
            ROS param (default 10.0). Pass an explicit float to override
            per-call. Refuses on empty waypoint list with ``{ok: False,
            reason: "no waypoints recorded"}``. Otherwise lazily (re)
            constructs the runner so prior state can't leak across runs,
            then delegates to ``runner.start``.
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
            timeout = (float(settle_timeout_s) if settle_timeout_s is not None
                       else float(self._settle_timeout_s))
            self.sequence_runner = CaptureSequenceRunner(self)
            return self.sequence_runner.start(
                dry_run=bool(dry_run),
                settle_timeout_s=timeout,
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

        # ---- calibration cross-check: board pose consistency across poses ----

        def do_board_pose_check(self, pose_indices=None, n_poses: int = 5,
                                settle_timeout_s: float = None):
            """Verify calibration by checking if the board's pose in ``link_base``
            looks the same from multiple arm positions.

            For each requested waypoint i, captures ``T_cam_board_i`` and looks
            up ``T_base_eef_i``, then computes:
                T_base_board_i = T_base_eef_i · X · T_cam_board_i⁻¹
            where ``X = last_solve.X`` (current in-memory calibration).
            The board is physically stationary, so ``T_base_board_i`` MUST be
            constant across all i if X is correct. The spread of T_base_board
            translations and rotations is the calibration's real-world
            accuracy — much more diagnostic than train/heldout residuals.

            ``pose_indices`` (optional): explicit list of waypoint indices to
            visit. Defaults to ``n_poses`` indices evenly spaced through the
            waypoint store (so a 25-waypoint list with n_poses=5 visits
            indices [0, 5, 10, 15, 20]).

            Returns ``{ok, n_poses, t_std_mm, t_max_offset_mm, r_std_deg,
            r_max_deg, per_pose: [...]}``.
            """
            with self.lock:
                waypoints = list(self.waypoint_store.list())
            if not waypoints:
                return {"ok": False, "reason": "no waypoints recorded"}
            # Prefer last_solve.X (in-memory, freshest); fall back to the
            # deployed per-robot ``hand_eye.yaml`` so the check still runs
            # against the calibration on disk after a server restart.
            if self.last_solve is not None:
                X = np.asarray(self.last_solve.X)
                X_source = "last_solve"
            else:
                robot = self._resolve_robot_name()
                if not robot:
                    return {"ok": False, "reason":
                            "no last_solve and ROBOT_NAME unset — can't load deployed X"}
                basic_root = self._tk25_basic_repo_root()
                if basic_root is None:
                    return {"ok": False, "reason":
                            "tk25_basic repo root not resolvable"}
                yaml_path = self._hand_eye_path(robot)
                if not yaml_path.exists():
                    return {"ok": False, "reason":
                            f"no last_solve and no hand_eye.yaml at {yaml_path}"}
                try:
                    import yaml as _yaml
                    with open(yaml_path) as f:
                        doc = _yaml.safe_load(f) or {}
                    he = (doc or {}).get("hand_eye") or {}
                    xyz_s = he.get("color_optical_xyz", "").split()
                    rpy_s = he.get("color_optical_rpy", "").split()
                    if len(xyz_s) != 3 or len(rpy_s) != 3:
                        raise ValueError(
                            "hand_eye.yaml missing color_optical_xyz/rpy")
                    xyz = [float(v) for v in xyz_s]
                    rpy = [float(v) for v in rpy_s]
                    X = np.eye(4)
                    X[:3, :3] = _R.from_euler('xyz', rpy).as_matrix()
                    X[:3, 3] = xyz
                    X_source = f"yaml@{yaml_path}"
                except Exception as exc:
                    return {"ok": False, "reason":
                            f"failed to load X from {yaml_path}: {exc}"}

            if pose_indices is None:
                n = max(2, min(int(n_poses), len(waypoints)))
                pose_indices = [int(round(i * (len(waypoints) - 1) / (n - 1)))
                                for i in range(n)]
            pose_indices = [int(p) for p in pose_indices]
            for p in pose_indices:
                if p < 0 or p >= len(waypoints):
                    return {"ok": False, "reason":
                            f"pose_idx {p} out of range (store has {len(waypoints)})"}

            timeout = (float(settle_timeout_s) if settle_timeout_s is not None
                       else float(self._settle_timeout_s))
            runner = CaptureSequenceRunner(self)

            per_pose = []
            skipped = []  # list of {pose_idx, reason}
            from rclpy.time import Time as _RclpyTime
            for pose_idx in pose_indices:
                joints = list(waypoints[pose_idx])
                mv = runner._do_move_wait(joints)
                if not mv["ok"]:
                    skipped.append({"pose_idx": int(pose_idx),
                                    "reason": f"move failed: {mv['reason']}"})
                    self.get_logger().warn(
                        f"board_pose_check: skip pose {pose_idx} — move failed: {mv['reason']}")
                    continue
                if not runner._wait_for_settle(timeout):
                    skipped.append({"pose_idx": int(pose_idx),
                                    "reason": f"settle timed out after {timeout:.1f}s"})
                    self.get_logger().warn(
                        f"board_pose_check: skip pose {pose_idx} — settle timed out")
                    continue
                with self.lock:
                    cap = self._cap
                    frame_stamp = self._frame_stamp
                if cap is None:
                    skipped.append({"pose_idx": int(pose_idx),
                                    "reason": "no board detection"})
                    self.get_logger().warn(
                        f"board_pose_check: skip pose {pose_idx} — no board detection")
                    continue
                tf_time = (_RclpyTime.from_msg(frame_stamp) if frame_stamp is not None
                           else self._rclpy_time())
                try:
                    tfm = self.tf_buffer.lookup_transform(
                        self._base_frame, self._eef_frame, tf_time)
                except Exception:
                    try:
                        tfm = self.tf_buffer.lookup_transform(
                            self._base_frame, self._eef_frame, self._rclpy_time())
                    except Exception as exc:
                        skipped.append({"pose_idx": int(pose_idx),
                                        "reason": f"TF lookup failed: {exc}"})
                        self.get_logger().warn(
                            f"board_pose_check: skip pose {pose_idx} — TF lookup failed: {exc}")
                        continue
                T_base_eef = ws.tf_to_matrix(
                    [tfm.transform.translation.x, tfm.transform.translation.y,
                     tfm.transform.translation.z],
                    [tfm.transform.rotation.x, tfm.transform.rotation.y,
                     tfm.transform.rotation.z, tfm.transform.rotation.w])
                T_cam_board = np.asarray(cap["T_cam_board"])
                # T_base_board = T_base_eef · X · T_cam_board.
                # OpenCV PnP returns T_cam_board such that board-frame points
                # transform to camera-frame coords via T_cam_board (i.e. it's
                # the pose of the board IN the camera frame), so chaining
                # T_base_eef · X (= T_base_cam) · T_cam_board yields T_base_board
                # with NO inverse. Mirrors the solver's `_reproj_rms` constraint
                # T_cam_board = inv(T_base_eef·X) · T_base_board, solved for Tbb.
                T_base_board = T_base_eef @ X @ T_cam_board
                per_pose.append({
                    "pose_idx": int(pose_idx),
                    "T_base_board": T_base_board,
                    "reproj_px": float(cap["reproj_px"]),
                })

            # Need at least 2 successful poses to compute a meaningful spread.
            if len(per_pose) < 2:
                return {"ok": False,
                        "reason": (f"only {len(per_pose)} of {len(pose_indices)} "
                                   f"poses succeeded — need ≥2 for spread"),
                        "skipped": skipped}

            # Spread: each pose's T_base_board vs the mean translation + first rotation
            trans = np.asarray([p["T_base_board"][:3, 3] for p in per_pose])
            t_mean = trans.mean(axis=0)
            t_offsets = np.linalg.norm(trans - t_mean, axis=1)
            R0 = per_pose[0]["T_base_board"][:3, :3]
            def _rot_angle(Ra, Rb):
                return float(np.degrees(np.linalg.norm(
                    _R.from_matrix(Ra.T @ Rb).as_rotvec())))
            rot_devs = [_rot_angle(R0, np.asarray(p["T_base_board"])[:3, :3])
                        for p in per_pose]

            self.get_logger().info(
                f"board_pose_check: X_source={X_source} "
                f"used={len(per_pose)}/{len(pose_indices)} (skipped {len(skipped)}) | "
                f"t_std={t_offsets.std()*1000:.2f}mm "
                f"t_max_offset={t_offsets.max()*1000:.2f}mm | "
                f"r_std={float(np.std(rot_devs)):.3f}deg "
                f"r_max={float(np.max(rot_devs)):.3f}deg")
            return {
                "ok": True,
                "X_source": X_source,
                "n_poses": len(per_pose),
                "n_skipped": len(skipped),
                "skipped": skipped,
                "pose_indices": pose_indices,
                "t_std_mm": float(t_offsets.std() * 1000.0),
                "t_max_offset_mm": float(t_offsets.max() * 1000.0),
                "r_std_deg": float(np.std(rot_devs)),
                "r_max_deg": float(np.max(rot_devs)),
                "per_pose": [{
                    "pose_idx": int(p["pose_idx"]),
                    "T_base_board_xyz_mm": [
                        float(p["T_base_board"][0, 3] * 1000),
                        float(p["T_base_board"][1, 3] * 1000),
                        float(p["T_base_board"][2, 3] * 1000),
                    ],
                    "offset_from_mean_mm": float(np.linalg.norm(
                        p["T_base_board"][:3, 3] - t_mean) * 1000.0),
                    "rot_from_first_deg": float(d),
                    "reproj_px": p["reproj_px"],
                } for p, d in zip(per_pose, rot_devs)],
            }

        # ---- mount rigidity test ---------------------------------------------

        def do_mount_test(self, pose_idx: int = 0, n_visits: int = 5,
                          scramble_pose_idx: int = None,
                          settle_timeout_s: float = None):
            """Verify whether the camera mount is rigid.

            Moves the arm repeatedly to the same target pose (waypoint
            ``pose_idx``), interleaving each visit with a scramble pose
            (default: roughly antipodal in the waypoint list) so the wrist
            has to traverse different orientations between visits. Captures
            ``T_cam_board`` at each visit and reports the spread.

            Interpretation: if the camera is rigidly bolted to the arm and
            the board is fixed in base frame, ``T_base_eef`` repeats exactly
            (FK is deterministic) so ``T_cam_board`` must repeat too —
            within sub-pixel PnP noise (~0.2-0.5 mm translation, ~0.05 deg
            rotation). Any larger spread is direct evidence of mechanical
            compliance somewhere in the chain (mount, bracket, board flex).

            Returns ``{ok, trans_std_mm, trans_max_offset_mm, rot_std_deg,
            rot_max_deg, per_visit}``. ``per_visit`` carries each visit's
            T_cam_board translation + reproj_px so the operator can dig in.
            """
            with self.lock:
                waypoints = list(self.waypoint_store.list())
            if not waypoints:
                return {"ok": False, "reason": "no waypoints recorded"}
            if pose_idx < 0 or pose_idx >= len(waypoints):
                return {"ok": False, "reason":
                        f"pose_idx {pose_idx} out of range (store has {len(waypoints)})"}
            if scramble_pose_idx is None:
                scramble_pose_idx = (pose_idx + len(waypoints) // 2) % len(waypoints)
                if scramble_pose_idx == pose_idx and len(waypoints) >= 2:
                    scramble_pose_idx = (pose_idx + 1) % len(waypoints)
            if scramble_pose_idx < 0 or scramble_pose_idx >= len(waypoints):
                return {"ok": False, "reason":
                        f"scramble_pose_idx {scramble_pose_idx} out of range"}
            n_visits = max(2, int(n_visits))
            timeout = (float(settle_timeout_s) if settle_timeout_s is not None
                       else float(self._settle_timeout_s))
            # WaypointStore.list() returns list[list[float]] (raw joint vectors).
            target_joints = list(waypoints[pose_idx])
            scramble_joints = list(waypoints[scramble_pose_idx])

            # Reuse CaptureSequenceRunner's helpers (move + settle) so we
            # don't re-implement the rclpy async patterns. The runner stores
            # nothing on session.samples — captures are pulled directly from
            # node._cap after each settle.
            runner = CaptureSequenceRunner(self)

            visits = []
            for i in range(n_visits):
                # Scramble first so the arm always APPROACHES target from
                # a different starting orientation (catches direction-
                # dependent backlash too).
                move_res = runner._do_move_wait(scramble_joints)
                if not move_res["ok"]:
                    return {"ok": False, "reason":
                            f"visit {i}: scramble move failed: {move_res['reason']}"}
                if not runner._wait_for_settle(timeout):
                    return {"ok": False, "reason":
                            f"visit {i}: scramble settle timed out after {timeout:.1f}s"}
                # Move to target and capture
                move_res = runner._do_move_wait(target_joints)
                if not move_res["ok"]:
                    return {"ok": False, "reason":
                            f"visit {i}: target move failed: {move_res['reason']}"}
                if not runner._wait_for_settle(timeout):
                    return {"ok": False, "reason":
                            f"visit {i}: target settle timed out after {timeout:.1f}s"}
                with self.lock:
                    cap = self._cap
                if cap is None:
                    return {"ok": False, "reason":
                            f"visit {i}: no board detection at target"}
                T = np.asarray(cap["T_cam_board"])
                visits.append({
                    "trans_m": T[:3, 3].tolist(),
                    "R": T[:3, :3].tolist(),
                    "reproj_px": float(cap["reproj_px"]),
                })

            # Spread metrics — translation std (3D distance from mean) +
            # rotation spread (angular distance from first visit, since
            # there's no straightforward SO(3) mean for small N).
            trans = np.asarray([v["trans_m"] for v in visits])
            trans_mean = trans.mean(axis=0)
            trans_offsets_m = np.linalg.norm(trans - trans_mean, axis=1)
            R0 = np.asarray(visits[0]["R"])
            # Angular distance via rotvec — numerically stable vs arccos-of-trace.
            def _rot_angle(Ra, Rb):
                return float(np.degrees(np.linalg.norm(
                    _R.from_matrix(Ra.T @ Rb).as_rotvec())))
            rot_devs_deg = [_rot_angle(R0, np.asarray(v["R"])) for v in visits]
            log = self.get_logger()
            log.info(
                f"mount_test: pose={pose_idx} scramble={scramble_pose_idx} "
                f"N={n_visits} | "
                f"trans_std={trans_offsets_m.std()*1000:.2f}mm "
                f"trans_max_offset={trans_offsets_m.max()*1000:.2f}mm | "
                f"rot_std={float(np.std(rot_devs_deg)):.3f}deg "
                f"rot_max={float(np.max(rot_devs_deg)):.3f}deg")
            return {
                "ok": True,
                "pose_idx": int(pose_idx),
                "scramble_pose_idx": int(scramble_pose_idx),
                "n_visits": int(n_visits),
                "trans_std_mm": float(trans_offsets_m.std() * 1000.0),
                "trans_max_offset_mm": float(trans_offsets_m.max() * 1000.0),
                "rot_std_deg": float(np.std(rot_devs_deg)),
                "rot_max_deg": float(np.max(rot_devs_deg)),
                "per_visit": [{
                    "trans_mm": [v["trans_m"][0]*1000, v["trans_m"][1]*1000,
                                 v["trans_m"][2]*1000],
                    "reproj_px": v["reproj_px"],
                    "offset_from_mean_mm": float(np.linalg.norm(
                        np.asarray(v["trans_m"]) - trans_mean) * 1000.0),
                    "rot_from_first_deg": float(d),
                } for v, d in zip(visits, rot_devs_deg)],
            }

        # ---- T6: per-robot xacro override + yaml-diff ------------------------

        def _resolve_robot_name(self):
            """ROBOT_NAME env var wins; falls back to the ``robot_name`` param.

            Empty string => unset (UI shows the yaml-only banner). Both inputs
            are re-read at call time, so a runtime ``ros2 param set /handeye_web
            robot_name <name>`` takes effect on the next reload without
            re-launching.

            The env var, by contrast, canNOT be changed at runtime: a process's
            ``os.environ`` is frozen at spawn, so ``export ROBOT_NAME=…`` in
            another shell has no effect on an already-running node. Set the env
            var *before* launch, or use ``ros2 param set`` to override live.
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
            """Build the yaml side of the promote diff (always computed).

            Frame-agnostic output: ``self.last_solve.X`` is ``T_eef->observed_optical``
            (color OR left-IR, per calib_frame). Compose back to the camera body
            via the matching internal transform — ``T_eef->camera_link = X .
            inv(T_camera_link->observed_optical)`` — and ALWAYS derive the
            color_optical reference from camera_link, so the yaml reports the same
            ``arm_to_camera`` (camera_link mount) + ``color_optical`` regardless of
            which frame was observed. The xacro mount joint (camera_link_joint) is
            the single deployment target either way."""
            import difflib
            import yaml
            T_mount_observed = self._mount_to_optical_matrix()
            T_eef_mount = ah.compose_eef_to_mount(self.last_solve.X, T_mount_observed)
            # color_optical reference, always derived from camera_link:
            #   T_eef->color_optical = T_eef->camera_link . T_camera_link->color_optical
            T_eef_color = np.asarray(T_eef_mount) @ self._mount_to_color_matrix()
            proposed_dict = ah.handeye_yaml_dict(
                T_eef_mount, T_eef_color, len(self.session.samples),
                self.last_solve.metrics,
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
            # Always write the full property-redefinition form — the ONLY
            # form xarm_description/urdf/camera/realsense_d435i.urdf.xacro's
            # <xacro:include> actually consumes (redefines handeye_xyz/
            # handeye_rpy). The prior <joint>-patch path silently never took
            # effect on the real URDF (the vendor xacro's <joint> already
            # existed and read its origin from those two properties, so a
            # sibling <joint> block in the include was inert) — tinker2's
            # deployed override had to be hand-copied because of this bug.
            current_xacro = (xacro_target.read_text()
                              if xacro_target.exists() else "")
            proposed_xacro = ah.seed_handeye_override_xacro(
                robot, xyz_str, rpy_str)
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
                            out["xacro"] = {
                                "written_path": x["target_path"],
                                "backup_path": backup,
                                "message": (
                                    "Rebuild + relaunch to pick this up: "
                                    "tkbuild tk25_basic --packages-select "
                                    "tinker_robot_config, then relaunch the "
                                    "arm bringup (robot_state_publisher) so "
                                    "the new wrist-camera mount is loaded."
                                ),
                            }
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

        def _xyz_rpy_to_matrix(self, xyz_str, rpy_str):
            """Parse "x y z" / "r p y" strings into a 4x4 (URDF fixed-axis rpy).
            Returns identity on malformed input (defensive — never crash a solve)."""
            np = self._np
            try:
                xyz = [float(v) for v in str(xyz_str).split()]
                rpy = [float(v) for v in str(rpy_str).split()]
            except ValueError:
                return np.eye(4)
            if len(xyz) != 3 or len(rpy) != 3:
                return np.eye(4)
            T = np.eye(4)
            T[:3, :3] = self._R.from_euler("xyz", rpy).as_matrix()
            T[:3, 3] = xyz
            return T

        def _mount_to_color_matrix(self):
            """T_camera_link -> color_optical (factory geometry, +15 mm + optical rot)."""
            return self._xyz_rpy_to_matrix(self._mount_to_color_xyz, self._mount_to_color_rpy)

        def _mount_to_ir_optical_matrix(self):
            """T_camera_link -> left_ir_optical (factory geometry).

            On a D435 the left-IR / depth frame is COINCIDENT with camera_link
            (vendor URDF left_ir joint origin "0 0 0"), so this is a pure optical
            rotation with zero translation — observing in IR measures the camera
            body (camera_link) directly."""
            return self._xyz_rpy_to_matrix(self._mount_to_ir_xyz, self._mount_to_ir_rpy)

        def _mount_to_optical_matrix(self):
            """T_camera_link -> the CURRENTLY-OBSERVED optical frame (per calib_frame).
            This is the transform the solve composes through to recover camera_link."""
            return (self._mount_to_ir_optical_matrix() if self._calib_frame == "ir"
                    else self._mount_to_color_matrix())

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
    import os
    from pathlib import Path
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, Response
    from fastapi.responses import JSONResponse as _RawJSONResponse
    from fastapi.staticfiles import StaticFiles
    from handeye_calib import web_support as ws
    from handeye_calib import handeye_sessions as hsx

    def JSONResponse(content, *args, **kwargs):
        # Boundary scrub: Starlette renders with ``allow_nan=False`` so a
        # stray NaN/Inf anywhere in the payload triggers a plain-text 500
        # the browser reports as ``JSON.parse: unexpected character at
        # line 1 column 1``. Sanitize to ``None`` so the UI shows a blank
        # field instead. Cheap (O(payload size), pure-python walk).
        return _RawJSONResponse(ws.json_safe(content), *args, **kwargs)

    app = FastAPI(title="handeye_web", docs_url="/api/docs")

    # v2 static UI. webui/ ships next to handeye_web.py via setup.py's
    # package_data hook (see setup.py), so `Path(__file__).parent / "webui"`
    # resolves both from the source tree and the install tree.
    webui_dir = Path(__file__).resolve().parent / "webui"
    app.mount("/static", StaticFiles(directory=str(webui_dir), html=False), name="static")

    # Serve the two files that change between sessions with no-store so the
    # browser never serves stale JS or HTML after an update.
    @app.get("/static/app.js")
    def _app_js():
        return FileResponse(str(webui_dir / "app.js"), media_type="application/javascript",
                            headers={"Cache-Control": "no-store"})

    @app.get("/static/index.html")
    def _static_index():
        return FileResponse(str(webui_dir / "index.html"), media_type="text/html",
                            headers={"Cache-Control": "no-store"})

    @app.get("/")
    def index():
        return FileResponse(str(webui_dir / "index.html"), media_type="text/html",
                            headers={"Cache-Control": "no-store"})

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
        body_d = body or {}
        # Optional outlier rejection: sentinel "default" when reject_sigma is
        # absent from the body (solver picks its own default, currently 2.5 —
        # default-on per-axis MAD rejection).  Explicit null → None (disables).
        # Any number → float threshold.  Invalid values fall back to "default".
        if "reject_sigma" not in body_d:
            rs = "default"
        else:
            raw = body_d["reject_sigma"]
            try:
                rs = float(raw) if raw is not None else None
            except (TypeError, ValueError):
                rs = "default"
        # Offload the (blocking, multi-second, iterative) solve to a worker
        # thread so the asyncio event loop stays free to keep pushing /ws state
        # frames — that is what lets the UI show MAD rejections AS THEY HAPPEN
        # (do_solve streams progress into _solve_progress, surfaced by the push).
        # Mirrors the board_pose_check / mount_test / set_config offload below.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: node.do_solve(method=body_d.get("method", "auto"),
                                  reject_sigma=rs))
        return JSONResponse(result)

    @app.post("/api/anchor")
    async def anchor(request: Request):
        return JSONResponse(ws.json_safe(node.do_anchor_board()))

    @app.post("/api/anchor/clear")
    async def anchor_clear(request: Request):
        return JSONResponse(ws.json_safe(node.do_clear_anchor()))

    # ---- Capture-session history (browse/re-open past captures) -------------
    # Mirrors pan_tilt/calib_web.py's session browser: every capture is persisted
    # to <HANDEYE_DUMP_DIR|calibration_data>/wrist_handeye_sessions/<name>/ so the
    # operator can list past captures, inspect their gallery + solve verdict, and
    # re-load one into the live session to re-solve with the current solver.
    @app.get("/api/sessions")
    def sessions_list():
        return JSONResponse(ws.json_safe({"ok": True, "sessions": hsx.list_sessions()}))

    @app.get("/api/sessions/{name}")
    def sessions_detail(name: str):
        try:
            data = hsx.read_session(name)
        except Exception as exc:
            return JSONResponse({"ok": False, "reason": f"no such session: {exc}"},
                                status_code=404)
        return JSONResponse(ws.json_safe({"ok": True, **data}))

    @app.get("/api/sessions/{name}/samples/{idx}/thumb.jpg")
    def sessions_thumb(name: str, idx: int):
        try:
            p = hsx.thumb_path(name, idx)
        except Exception:
            p = None
        if p and os.path.isfile(p):
            with open(p, "rb") as f:
                return Response(content=f.read(), media_type="image/jpeg")
        return Response(content=ws.placeholder_jpeg("no thumb"),
                        media_type="image/jpeg")

    @app.post("/api/sessions/{name}/load")
    def sessions_load(name: str):
        return JSONResponse(ws.json_safe(node.do_load_session(name)))

    @app.delete("/api/sessions/{name}")
    def sessions_delete(name: str):
        return JSONResponse(ws.json_safe(node.do_delete_session(name)))

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

    @app.delete("/api/waypoints")
    def waypoints_clear():
        """DELETE /api/waypoints → remove ALL waypoints from the in-memory store."""
        return JSONResponse(node.do_clear_waypoints())

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
        # ``settle_timeout_s`` is optional; if absent or non-numeric the node
        # falls back to its ROS-param default (default 10s).
        try:
            settle = (None if "settle_timeout_s" not in body
                      else float(body.get("settle_timeout_s")))
        except (TypeError, ValueError):
            settle = None
        return JSONResponse(node.do_start_sequence(
            dry_run=bool(body.get("dry_run", False)),
            settle_timeout_s=settle))

    @app.post("/api/sequence/cancel")
    def sequence_cancel():
        """POST /api/sequence/cancel → do_cancel_sequence() (idempotent no-op)."""
        return JSONResponse(node.do_cancel_sequence())

    @app.post("/api/board_pose_check")
    async def board_pose_check(request: Request):
        """Calibration cross-check: visit multiple waypoints, compute the
        board's pose in link_base from each view, report the spread.

        Body (all optional):
          {pose_indices: list[int] = evenly spaced through waypoints,
           n_poses: int = 5,
           settle_timeout_s: float = None}

        Returns ``{t_std_mm, t_max_offset_mm, r_std_deg, r_max_deg, per_pose}``.
        ``t_std_mm`` is the headline number: if X is correct, this is the
        calibration's real-world accuracy (a single number you can ship to
        a downstream task as the error budget).
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        kwargs = {}
        if body.get("pose_indices") is not None:
            kwargs["pose_indices"] = body["pose_indices"]
        if body.get("n_poses") is not None:
            kwargs["n_poses"] = int(body["n_poses"])
        if body.get("settle_timeout_s") is not None:
            try:
                kwargs["settle_timeout_s"] = float(body["settle_timeout_s"])
            except (TypeError, ValueError):
                pass
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None,
            lambda: node.do_board_pose_check(**kwargs))
        return JSONResponse(result)

    @app.post("/api/mount_test")
    async def mount_test(request: Request):
        """Camera-mount rigidity diagnostic.

        Body (all optional):
          {pose_idx: int = 0,              # target waypoint index
           n_visits: int = 5,              # visits to the target
           scramble_pose_idx: int = None,  # waypoint to scramble through;
                                           # defaults to ~antipodal in list
           settle_timeout_s: float = None} # falls back to ROS param

        Blocks the request for ~(n_visits * 2 * settle_s) seconds. Runs
        the actual work in a worker thread so the asyncio event loop
        (uvicorn) stays responsive for WS state push + other clients.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        kwargs = {
            "pose_idx": int(body.get("pose_idx", 0)),
            "n_visits": int(body.get("n_visits", 5)),
        }
        if body.get("scramble_pose_idx") is not None:
            try:
                kwargs["scramble_pose_idx"] = int(body["scramble_pose_idx"])
            except (TypeError, ValueError):
                pass
        if body.get("settle_timeout_s") is not None:
            try:
                kwargs["settle_timeout_s"] = float(body["settle_timeout_s"])
            except (TypeError, ValueError):
                pass
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None,
            lambda: node.do_mount_test(**kwargs))
        return JSONResponse(result)

    @app.post("/api/config")
    async def set_config(request: Request):
        """Runtime config: calib_frame (color|ir), depth knobs, IR emitter.

        Body (all optional): ``{calib_frame, use_ffs_depth, depth_weight,
        depth_sigma_m, depth_win, depth_min_corners, depth_z_min, depth_z_max,
        ir_emitter_enabled}``. Switching calib_frame discards the (frame-specific)
        captured samples. The emitter set runs off the event loop (parameter
        client to the camera node) so it can't stall the UI."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        if not isinstance(body, dict):
            body = {}
        # Whitelist known keys before **-expanding into do_set_config — a stray
        # body key (e.g. "self", or a non-identifier string) would otherwise raise
        # a TypeError and surface as a plain-text 500 the UI can't parse.
        _allowed = {"calib_frame", "use_ffs_depth", "depth_weight", "depth_sigma_m",
                    "depth_win", "depth_min_corners", "depth_z_min", "depth_z_max",
                    "ir_emitter_enabled"}
        body = {k: v for k, v in body.items() if k in _allowed}
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: node.do_set_config(**body))
        return JSONResponse(result)

    @app.websocket("/ws")
    async def ws_state(ws_conn: WebSocket):
        """10 Hz state push for the static UI.

        Pushes the enriched state payload every 100 ms. The previous 5 Hz
        rate made waypoint-list / sequence-progress updates feel sluggish
        (up to 200 ms lag between an operator click and the UI rendering
        the new server state). 100 ms is below the human-perception
        threshold and bandwidth is negligible (~5-10 KB JSON × 10 Hz).
        Cleanly handles client disconnects; the surrounding try/except
        keeps a broken socket from propagating an exception into uvicorn's
        task supervisor.
        """
        await ws_conn.accept()
        try:
            while True:
                # Same NaN/Inf scrub as the HTTP responses — Starlette's
                # ``send_json`` uses ``allow_nan=False``, so a non-finite
                # in stability/diversity/safety would otherwise drop the
                # socket every cycle.
                await ws_conn.send_json(ws.json_safe(node.get_state_dict()))
                await asyncio.sleep(0.1)  # 10 Hz
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
