import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse
from tinker_vision_msgs_26.action import DetectWaving
from sensor_msgs.msg import Image, RegionOfInterest
from cv_bridge import CvBridge
import cv2
import os
import time
import queue
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import threading
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FutureTimeoutError
from geometry_msgs.msg import PointStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Shared logger
from vision_util.action_queue import QueuedActionGate
from vision_util.camera_intake import CameraIntake, IntakeConfig, StreamSpec
from vision_util.depth_reproject import waving_optical_points
from vision_util.tf_lookup import TransformHelper
from vision_util.vision_logging import VisionLogger
from vision_util.weights_cache import resolve_weights
from ._waving_vlm import (
    request_waving_persons_chain,
    build_provider_models,
    has_provider_key,
    should_wait_for_vlm,
    resolve_effective_mode,
    WavingVlmError,
)
from ._waving_geometry import is_duplicate_box, centroid_from_box


FRAME_MAX_AGE_S = 1.0
FRAME_WAIT_TIMEOUT_S = 2.0
CANCEL_STATE_TIMEOUT_S = 0.1
CANCEL_STATE_POLL_S = 0.005


class _GoalCanceled(Exception):
    """Internal control flow for cooperative action cancellation."""


class DetectWavingPersonsNode(Node):
    def __init__(self):
        super().__init__('detect_waving_persons_node')

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('sync_slop_sec', 0.1)
        color_topic = (
            self.get_parameter('color_topic').get_parameter_value().string_value)
        depth_topic = (
            self.get_parameter('depth_topic').get_parameter_value().string_value)
        camera_info_topic = (
            self.get_parameter('camera_info_topic')
            .get_parameter_value().string_value)
        sync_slop_sec = float(self.get_parameter('sync_slop_sec').value)

        self.bridge = CvBridge()
        self.action_cb_group = MutuallyExclusiveCallbackGroup()
        self.intake_cb_group = MutuallyExclusiveCallbackGroup()
        self.camera_intake = self._create_camera_intake(
            color_topic,
            depth_topic,
            camera_info_topic,
            sync_slop_sec,
        )
        self.transform_helper = TransformHelper(self)
        self.declare_parameter('model_path', 'yolo11m-seg.pt')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.yolo = YOLO(str(resolve_weights(model_path)))
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        # static_image_mode=True: each YOLO ROI is independent; the default
        # (False) builds a video tracker that pollutes subsequent crops.
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
        )

        # show_window=true pops up a real cv2.imshow window with per-person
        # bounding boxes, fed from a bounded queue by a dedicated background
        # thread (_cv2_window_loop) so imshow/waitKey are always called from
        # the same thread, not from whichever ROS callback-group thread
        # happens to process a given request.
        self.declare_parameter('show_window', True)
        self.show_window = (
            self.get_parameter('show_window').get_parameter_value().bool_value)
        self.declare_parameter('min_person_conf', 0.4)
        self.min_person_conf = float(self.get_parameter('min_person_conf').value)
        # rqt_image_view subscribes via image_transport, which hard-codes
        # `rmw_qos_profile_default` (RELIABLE, VOLATILE, KEEP_LAST=10) and
        # offers no GUI/CLI override (rqt_image_view#54, image_common#156).
        # Mirror that profile here: the integer 10 means depth=10 with rcl
        # defaults -- exactly what the subscriber expects. depth=10 also
        # prevents the outbound queue from overflowing on back-to-back
        # action goals (which is what dropped frames at depth=1).
        self.debug_image_pub = self.create_publisher(
            Image, '/detect_waving_debug_image', 10,
        )

        # Late-subscriber recovery. rqt_image_view subscribes VOLATILE and its
        # GUI/Qt cold start (~3-5 s) often outlives the launch's 1 s grace
        # period, so service publishes that fire during that window are
        # dropped forever. Cache the last published Image msg and republish
        # at 2 Hz from a separate callback group -- the next tick latches
        # onto whatever subscriber appeared in the meantime, so the most
        # recent annotated frame always reaches rqt within ~0.5 s. Identical
        # content on republish => no rqt flicker.
        self._last_debug_msg_lock = threading.Lock()
        self._last_debug_msg = None
        # 10 Hz republish -- gives rqt a steady ~100 ms cadence. Header stamp
        # is rewritten to `now()` on each tick (see _republish_last_debug)
        # because image_transport drops frames whose stamp is older than
        # the most recently displayed one; reusing the original capture
        # stamp made republishes look intermittent.
        self._republish_timer = self.create_timer(
            0.1,
            self._republish_last_debug,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        # Which detector decides who is waving:
        #   'vlm'       -> VLM is the sole waver source (MediaPipe skipped);
        #                  YOLO still runs for person masks -> 3D centroid.
        #   'hybrid'    -> MediaPipe wavers + VLM augmentation gated by the
        #                  request's min_waving_persons (legacy 2026-07-03).
        #   'mediapipe' -> MediaPipe only, VLM never called.
        # 'vlm'/'hybrid' degrade to 'mediapipe' at call time when no provider
        # key is configured (see resolve_effective_mode) so offline/no-key
        # boxes keep working unchanged. enable_vlm_fallback=False forces
        # 'mediapipe' regardless (hard kill-switch).
        self.declare_parameter('waving_detector', 'vlm')
        self.waving_detector = str(
            self.get_parameter('waving_detector').value).lower()
        self.declare_parameter('enable_vlm_fallback', True)
        self.declare_parameter('vlm_provider', 'qwen')
        self.declare_parameter('vlm_fallback_provider', 'gemini')
        self.declare_parameter('vlm_model_qwen', 'qwen3-vl-plus')
        self.declare_parameter('vlm_model_gemini', 'google/gemini-2.5-pro')
        self.declare_parameter('vlm_timeout_s', 20.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('vlm_dedup_iou', 0.3)
        self.enable_vlm_fallback = (
            self.get_parameter('enable_vlm_fallback').value)
        self.vlm_provider = self.get_parameter('vlm_provider').value
        self.vlm_fallback_provider = (
            self.get_parameter('vlm_fallback_provider').value)
        self.vlm_model_qwen = self.get_parameter('vlm_model_qwen').value
        self.vlm_model_gemini = self.get_parameter('vlm_model_gemini').value
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.vlm_max_retries = int(self.get_parameter('vlm_max_retries').value)
        self.vlm_dedup_iou = float(self.get_parameter('vlm_dedup_iou').value)
        self._vlm_chain = self._resolve_provider_chain()
        # Dedicated pool for the concurrent VLM waving-fallback call. 2
        # workers, not 1: an abandoned call from an early-exited request (see
        # _start_vlm_call / _run_detect_waving) can still be finishing in
        # the background when the NEXT request submits its own call -- with
        # only 1 worker, that submission would queue behind the abandoned one
        # and silently reintroduce the wait this whole change exists to avoid.
        self._vlm_executor = ThreadPoolExecutor(max_workers=2)
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self._vision_logger = VisionLogger(
            self,
            self.get_parameter('vision_logging_enabled').get_parameter_value().bool_value,
            self.get_parameter('vision_log_folder').get_parameter_value().string_value,
        )

        self._frame_queue = queue.Queue(maxsize=2)
        self._window_shutdown = threading.Event()
        self._window_thread = None
        if self.show_window:
            self._window_thread = threading.Thread(
                target=self._cv2_window_loop, name='waving_cv2_window',
                daemon=True,
            )
            self._window_thread.start()

        self._action_gate = QueuedActionGate()
        self.action_server = self._create_action_server()

        self.get_logger().info(
            f'Detect Waving Persons node started (show_window={self.show_window}, '
            f'waving_detector={self.waving_detector!r}, '
            f'enable_vlm_fallback={self.enable_vlm_fallback})')

    def _create_camera_intake(
        self,
        color_topic,
        depth_topic,
        camera_info_topic,
        sync_slop_s,
    ):
        return CameraIntake(
            self,
            IntakeConfig(
                camera='orbbec',
                color=StreamSpec(
                    color_topic, best_effort=False, qos_depth=10),
                depth=StreamSpec(
                    depth_topic, best_effort=False, qos_depth=10),
                camera_info=StreamSpec(
                    camera_info_topic, best_effort=False, qos_depth=10),
                sync_queue=10,
                sync_slop_s=sync_slop_s,
                age_source='stamp',
            ),
            callback_group=self.intake_cb_group,
            bridge=self.bridge,
        )

    def _create_action_server(self):
        return ActionServer(
            self,
            DetectWaving,
            'detect_waving_persons',
            execute_callback=self.detect_waving_execute_callback,
            cancel_callback=self.detect_waving_cancel_callback,
            handle_accepted_callback=self.detect_waving_handle_accepted_callback,
            callback_group=self.action_cb_group,
            result_timeout=0,
        )

    def _cv2_window_loop(self):
        """Own a single cv2.imshow window for the life of the node.

        Runs on a dedicated thread so imshow/waitKey are always called from
        the same thread -- required for the Qt/GTK backend to behave, and
        not guaranteed if called directly from a ROS callback-group thread
        (_run_detect_waving may run on a different worker for each goal).
        Pulls the latest annotated frame from _frame_queue; a get() timeout
        keeps waitKey() pumping the window's event loop even when no new
        frame has arrived, so the window stays responsive/movable between
        detections instead of freezing.
        """
        # Match the operator's real X display. Only sets it if the process
        # doesn't already have one -- never clobber an explicit launch-time
        # DISPLAY. Verified live (2026-07-03): a bare cv2.namedWindow +
        # imshow + waitKey round-trip against DISPLAY=:0 succeeds in this
        # venv (Qt5 GUI backend), including from a background thread fed by
        # a producer/consumer queue exactly like the one below.
        os.environ.setdefault('DISPLAY', ':0')
        window_name = 'Waving Detection'
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)
        except Exception as exc:  # noqa: BLE001 -- no GUI backend available
            self.get_logger().error(
                f'show_window=true but cv2 has no GUI backend available '
                f'(DISPLAY={os.environ.get("DISPLAY")!r}, error: {exc}); no '
                f'popup window will appear. The debug image is still '
                f'published on /detect_waving_debug_image.'
            )
            return
        # Paint something immediately instead of waiting for the first real
        # detection frame -- a namedWindow with nothing shown yet can sit
        # unmapped/unpainted on some window managers until the first imshow,
        # which made the window appear not to "pop up" at all until the
        # first action goal arrived. This also proves the window is alive
        # independent of whether a detection has happened yet.
        placeholder = np.zeros((540, 960, 3), dtype=np.uint8)
        cv2.putText(
            placeholder, 'Waiting for detect_waving_persons goals...',
            (20, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2,
        )
        self._show_frame(window_name, placeholder)
        self.get_logger().info(
            f'cv2 popup window ready (DISPLAY={os.environ.get("DISPLAY")!r}).'
        )
        while not self._window_shutdown.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                cv2.waitKey(1)
                continue
            try:
                self._show_frame(window_name, frame)
            except Exception as exc:  # noqa: BLE001 -- keep the node alive
                self.get_logger().error(
                    f'cv2 imshow/waitKey failed ({exc}); disabling the '
                    f'popup window for the rest of this run.'
                )
                return
        # No explicit cv2.destroyWindow() here: the process is exiting
        # anyway (this only runs from destroy_node's shutdown path), and
        # destroying a Qt-backed window from here produced a real (if
        # non-fatal) "QObject::killTimer: Timers cannot be stopped from
        # another thread" warning in testing -- the OS/window manager
        # reclaims the window on process exit regardless.

    @staticmethod
    def _show_frame(window_name: str, frame) -> None:
        """imshow + repeated waitKey pumps, raising the window on each update.

        A single waitKey(1) right after imshow is the textbook pattern, but
        proved unreliable for actually materializing/repainting the window
        promptly on some window managers -- a few short pumps in quick
        succession (still ~a few ms total, no meaningful latency added) are
        far more consistent at forcing the frame to actually paint.

        Since this runs exactly once per new frame (plus the startup
        placeholder), it's also the hook for bringing the window to the front
        whenever the image updates: toggle WND_PROP_TOPMOST on before the
        paint (raises the window) then back off after (releases the
        always-on-top flag, so the operator can cover it again between
        detections). Both setWindowProperty calls are guarded because
        WND_PROP_TOPMOST needs a Qt/GTK highgui backend and isn't present on
        every OpenCV build -- a missing constant/backend must only skip the
        raise, never kill the window loop or block the paint.
        """
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1.0)
        except Exception:  # noqa: BLE001 -- no Qt/GTK backend; paint anyway
            pass
        cv2.imshow(window_name, frame)
        for _ in range(3):
            cv2.waitKey(1)
        try:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0.0)
            cv2.waitKey(1)
        except Exception:  # noqa: BLE001 -- raise is best-effort only
            pass

    def destroy_node(self):
        self._vlm_executor.shutdown(wait=False)
        self._window_shutdown.set()
        if self._window_thread is not None:
            self._window_thread.join(timeout=2.0)
        return super().destroy_node()

    def detect_waving_handle_accepted_callback(self, goal_handle) -> None:
        """Queue an accepted action goal for serialized execution."""
        self._action_gate.accept(goal_handle)

    def detect_waving_cancel_callback(self, goal_handle):
        """Accept cancellation and retain intent for a queued goal."""
        self._action_gate.cancel_queued(goal_handle)
        return CancelResponse.ACCEPT

    def _should_cancel(self, goal_handle) -> bool:
        return self._action_gate.should_cancel(goal_handle)

    def _raise_if_canceled(self, goal_handle) -> None:
        if self._should_cancel(goal_handle):
            raise _GoalCanceled

    def _publish_feedback(
        self,
        goal_handle,
        *,
        stage: str,
        message: str,
        delay_limit: float,
    ) -> None:
        self._raise_if_canceled(goal_handle)
        feedback = DetectWaving.Feedback()
        feedback.status = 0
        feedback.delay_limit = float(delay_limit)
        feedback.stage = stage
        feedback.message = message
        goal_handle.publish_feedback(feedback)

    def _vlm_delay_limit(self) -> float:
        retries = max(0, int(self.vlm_max_retries))
        per_provider_s = retries * max(0.0, float(self.vlm_timeout_s))
        return max(1.0, len(self._vlm_chain) * per_provider_s + 5.0)

    def _canceled_result(self, goal_handle):
        result = DetectWaving.Result()
        result.status = 1
        result.error_msg = 'Waving detection canceled.'

        deadline = time.monotonic() + CANCEL_STATE_TIMEOUT_S
        while not bool(getattr(goal_handle, 'is_cancel_requested', False)):
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                result.status = -1
                result.error_msg = (
                    'Waving detection cancellation-state error: cancel '
                    'request did not become visible.'
                )
                self.get_logger().error(result.error_msg)
                goal_handle.abort()
                return result
            time.sleep(min(CANCEL_STATE_POLL_S, remaining_s))

        goal_handle.canceled()
        return result

    def detect_waving_execute_callback(self, goal_handle):
        """Execute one queued waving-detection goal."""
        try:
            self._raise_if_canceled(goal_handle)
            result = self._run_detect_waving(goal_handle)
            self._raise_if_canceled(goal_handle)
        except _GoalCanceled:
            return self._canceled_result(goal_handle)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'Unhandled waving detection failure: {exc}'
            )
            result = DetectWaving.Result()
            result.status = -1
            result.error_msg = f'Waving detection failed: {exc}.'
            goal_handle.abort()
            return result
        else:
            goal_handle.succeed()
            return result
        finally:
            self._action_gate.notify_finished(goal_handle)

    def _annotate_frame(self, rgb_bgr, waving_annotations, waving_centroids):
        frame = rgb_bgr.copy()
        for idx, ((x1, y1, x2, y2, landmarks), point_stamped) in enumerate(
            zip(waving_annotations, waving_centroids), start=1
        ):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            label = f'waving #{idx} z={point_stamped.point.z:.2f}m'
            cv2.putText(
                frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2,
            )
            if landmarks is not None:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    self.mp_draw.draw_landmarks(
                        roi, landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def _annotate_all_persons(self, rgb_bgr, person_annotations):
        """Draw only the persons judged to be waving, boxed in red.

        Non-waving persons are intentionally left unboxed so the debug
        window/image highlights just the wavers. Empty/still scenes still
        emit the raw RGB (via the empty/failure paths) so operators can
        confirm the pipeline ran."""
        frame = rgb_bgr.copy()
        red = (0, 0, 255)
        wave_idx = 0
        for x1, y1, x2, y2, landmarks, is_wave in person_annotations:
            if not is_wave:
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), red, 2)
            wave_idx += 1
            label = f'waving #{wave_idx}'
            cv2.putText(
                frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, red, 2,
            )
            if landmarks is not None:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    self.mp_draw.draw_landmarks(
                        roi, landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def _publish_debug_image(self, image, header, *,
                             persons, waving, status_text=None,
                             already_annotated=False):
        """Publish an annotated debug frame to /detect_waving_debug_image.

        Called from every action exit path that has an `rgb_image`
        in hand (success, TF failure, depth failure, post-success transform
        failure). Publishing on failure is what lets the operator see *what
        the camera was looking at when the request was rejected* -- without
        it, debugging "why did my action fail" requires re-running
        the camera capture.

        `already_annotated=True` skips re-drawing (used by the success path,
        which has already drawn the red waver bboxes via
        `_annotate_all_persons`). For failure paths we just stamp a status
        banner on the raw frame."""
        if not already_annotated:
            image = image.copy()
        cv2.putText(
            image,
            f'persons={persons} waving={waving}',
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
        )
        if status_text:
            cv2.putText(
                image, status_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
            )
        try:
            msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            msg.header = header
            self.debug_image_pub.publish(msg)
            with self._last_debug_msg_lock:
                self._last_debug_msg = msg
            n_subs = self.debug_image_pub.get_subscription_count()
            self.get_logger().info(
                f'/detect_waving_debug_image published '
                f'({image.shape[1]}x{image.shape[0]}, subs={n_subs}, '
                f'persons={persons}, waving={waving}'
                f'{f", status={status_text}" if status_text else ""})'
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f'debug_image publish failed: {exc}')

    def _republish_last_debug(self):
        """Re-emit the most recent annotated frame so late-joining VOLATILE
        subscribers (rqt_image_view) eventually display it.

        Stamp is rewritten to wall-clock now() on every tick -- rqt /
        image_transport drops frames whose stamp is non-monotonic vs the
        last displayed one, so reusing the original capture stamp produced
        an uneven display cadence."""
        with self._last_debug_msg_lock:
            msg = self._last_debug_msg
        if msg is None:
            return
        try:
            msg.header.stamp = self.get_clock().now().to_msg()
            self.debug_image_pub.publish(msg)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().debug(f'debug_image republish failed: {exc}')

    MIN_VISIBILITY = 0.5
    ELBOW_TOL_NORM = 0.1

    def _resolve_provider_chain(self):
        """Build the (provider, model) chain; [] if fallback off or no keys.

        Non-fatal: a missing API key disables the VLM fallback (the node still
        serves MediaPipe-only) instead of raising at init.
        """
        if not self.enable_vlm_fallback:
            self.get_logger().info('VLM waving fallback disabled by param.')
            return []

        def model_for(provider):
            return (self.vlm_model_qwen if provider == 'qwen'
                    else self.vlm_model_gemini)

        chain = build_provider_models(
            self.vlm_provider, self.vlm_fallback_provider,
            has_key=has_provider_key, model_for=model_for,
            logger=self.get_logger())
        if not chain:
            self.get_logger().warn(
                'VLM fallback enabled but no provider API key found; '
                'serving MediaPipe-only.')
        else:
            self.get_logger().info(
                f'Waving VLM chain: {[p for p, _ in chain]}')
        return chain

    def _start_vlm_call(
        self,
        rgb_image,
        min_waving_persons: int,
        *,
        force: bool = False,
        should_abort=None,
    ):
        """Launch the VLM waving-fallback call on the dedicated executor.

        Returns None immediately if the fallback is disabled or no provider
        has a key configured. Otherwise, when `force` is False, also returns
        None unless the caller opted in (min_waving_persons > 0 -- the
        request-level default, so GPSR/EGPSR and any other caller that never
        sets this field keep today's fast-only behavior in 'hybrid' mode).
        `force=True` (VLM-only mode) bypasses the min_waving_persons gate so
        the VLM always runs. Returns a Future whose .result() is a
        WavingVlmResult, or raises WavingVlmError on hard failure (matching
        request_waving_persons_chain's own contract).
        """
        if not self.enable_vlm_fallback or not self._vlm_chain:
            return None
        if not force and min_waving_persons <= 0:
            return None
        return self._vlm_executor.submit(
            request_waving_persons_chain, rgb_image,
            provider_models=self._vlm_chain,
            timeout_s=self.vlm_timeout_s, max_retries=self.vlm_max_retries,
            logger=self.get_logger(),
            should_abort=should_abort,
        )

    def _log_discarded_vlm_result(self, future: Future):
        """Done-callback for an abandoned (early-exited) VLM future.

        Never raises: swallows whatever the call eventually produced (result
        or exception) so an abandoned call finishing later doesn't surface as
        an unhandled-exception warning from the executor thread.
        """
        try:
            future.result()
            self.get_logger().debug(
                'Discarded VLM waving result (CV already found enough wavers).')
        except Exception as exc:  # noqa: BLE001 -- intentionally swallowed
            self.get_logger().debug(f'Discarded VLM waving call failed: {exc}')

    def _merge_vlm_result(self, vlm_result, points, validmask_points, header,
                          request, person_records, waving_persons_centroids,
                          waving_annotations, waving_masks, waving_sources):
        """Fold a completed WavingVlmResult into the CV-found waver lists.

        Mutates the four aligned waver lists in place. Returns
        (n_added, provider_used). Same dedup/centroid logic the old
        _vlm_augment had, just taking an already-computed result instead of
        fetching it itself.
        """
        existing_boxes = [(a[0], a[1], a[2], a[3]) for a in waving_annotations]
        n_added = 0
        for box in vlm_result.boxes:
            if is_duplicate_box(box, existing_boxes,
                                iou_thresh=self.vlm_dedup_iou):
                continue
            out = centroid_from_box(points, validmask_points, box,
                                    person_records)
            if out is None:
                self.get_logger().info(
                    f'VLM box {box} skipped: no usable depth.')
                continue
            centroid, used_mask = out
            if (request.threshold_meters > 0
                    and centroid[2] > request.threshold_meters):
                self.get_logger().info(
                    f'VLM waver dropped: depth {centroid[2]:.2f}m > threshold '
                    f'{request.threshold_meters:.2f}m')
                continue
            point_stamped = PointStamped()
            point_stamped.header = header
            point_stamped.point.x = float(centroid[0])
            point_stamped.point.y = float(centroid[1])
            point_stamped.point.z = float(centroid[2])
            x1, y1, x2, y2 = box
            waving_persons_centroids.append(point_stamped)
            waving_annotations.append((x1, y1, x2, y2, None))
            waving_masks.append(used_mask)
            waving_sources.append('vlm')
            existing_boxes.append(box)
            n_added += 1
        return n_added, vlm_result.provider

    def is_waving(self, pose_landmarks, person_roi):
        if pose_landmarks is None:
            return False

        landmarks = pose_landmarks.landmark
        PL = mp.solutions.pose.PoseLandmark
        nose = landmarks[PL.NOSE]
        rh = landmarks[PL.RIGHT_WRIST]
        re = landmarks[PL.RIGHT_ELBOW]
        rs = landmarks[PL.RIGHT_SHOULDER]
        lh = landmarks[PL.LEFT_WRIST]
        le = landmarks[PL.LEFT_ELBOW]
        ls = landmarks[PL.LEFT_SHOULDER]

        if nose.visibility < self.MIN_VISIBILITY:
            self.get_logger().debug(
                f'is_waving: nose not visible ({nose.visibility:.2f}); skip'
            )
            return False

        right_visible = (
            min(rh.visibility, re.visibility, rs.visibility) >= self.MIN_VISIBILITY)
        left_visible = (
            min(lh.visibility, le.visibility, ls.visibility) >= self.MIN_VISIBILITY)
        if not (right_visible or left_visible):
            self.get_logger().debug(
                f'is_waving: neither arm visible '
                f'(R_min={min(rh.visibility, re.visibility, rs.visibility):.2f}, '
                f'L_min={min(lh.visibility, le.visibility, ls.visibility):.2f}); skip'
            )
            return False

        right_wave = right_visible and (
            rh.y < nose.y
            or (rh.y < re.y and re.y <= rs.y + self.ELBOW_TOL_NORM)
        )
        left_wave = left_visible and (
            lh.y < nose.y
            or (lh.y < le.y and le.y <= ls.y + self.ELBOW_TOL_NORM)
        )

        self.get_logger().debug(
            f'is_waving: R_vis={right_visible} L_vis={left_visible} | '
            f'nose.y={nose.y:.2f} | '
            f'rh.y={rh.y:.2f} rs.y={rs.y:.2f} re.y={re.y:.2f} | '
            f'lh.y={lh.y:.2f} ls.y={ls.y:.2f} le.y={le.y:.2f} | '
            f'R_wave={right_wave} L_wave={left_wave}'
        )

        gesture = right_wave or left_wave
        if gesture:
            self.get_logger().info(
                f'Wave gesture detected (ROI {person_roi.shape[1]}x{person_roi.shape[0]}, '
                f'side={"R" if right_wave else "L"})'
            )
        return gesture

    def _run_detect_waving(self, goal_handle):
        _t0 = time.perf_counter()
        request = goal_handle.request
        response = DetectWaving.Result()
        self.get_logger().info('Detect waving request received. Detecting persons now...')

        self._publish_feedback(
            goal_handle,
            stage='acquiring_frame',
            message='Acquiring a fresh synchronized color and depth frame.',
            delay_limit=FRAME_WAIT_TIMEOUT_S,
        )
        frame = self.camera_intake.wait_fresh(
            max_age_s=FRAME_MAX_AGE_S,
            timeout_s=FRAME_WAIT_TIMEOUT_S,
            poll_s=0.05,
            on_timeout='stale',
        )
        if frame is None:
            response.status = -1
            response.error_msg = 'No image, depth data received yet'
            self.get_logger().error(response.error_msg)
            return response
        if frame.K is None:
            response.status = -1
            response.error_msg = 'No camera info received yet'
            self.get_logger().error(response.error_msg)
            return response

        try:
            rgb_image = frame.color_bgr().copy()
            depth_m = frame.depth_m()
        except Exception as exc:  # noqa: BLE001
            response.status = -1
            response.error_msg = f'camera frame decode failed: {exc}'
            self.get_logger().error(response.error_msg)
            return response
        header = frame.header
        camera_k = frame.K
        self._raise_if_canceled(goal_handle)

        # Resolve which detector this call uses. 'vlm'/'hybrid' fall back to
        # 'mediapipe' when the VLM chain is unavailable (no key / kill-switch)
        # so offline boxes keep working; only key-present boxes are truly
        # VLM-only.
        effective_mode, degraded_from = resolve_effective_mode(
            self.waving_detector, self.enable_vlm_fallback,
            bool(self._vlm_chain))
        if degraded_from is not None:
            self.get_logger().warn(
                f'waving_detector={degraded_from!r} requested but VLM '
                f'unavailable (enable_vlm_fallback={self.enable_vlm_fallback}, '
                f'chain={[p for p, _ in self._vlm_chain]}); using MediaPipe '
                f'for this call.')
        run_mediapipe = effective_mode in ('mediapipe', 'hybrid')

        # Launch the VLM now, in parallel with the depth conversion + YOLO +
        # (optional) MediaPipe pass below -- it only needs rgb_image, which is
        # already available. In 'vlm' mode it always runs (force=True bypasses
        # the min_waving_persons gate); in 'hybrid' mode the request's
        # min_waving_persons > 0 opt-in still gates it. See _start_vlm_call /
        # the merge-or-discard logic after the CV loop.
        if effective_mode in ('vlm', 'hybrid'):
            self._publish_feedback(
                goal_handle,
                stage='vlm_call',
                message='Starting VLM waving detection.',
                delay_limit=self._vlm_delay_limit(),
            )
            vlm_future = self._start_vlm_call(
                rgb_image, request.min_waving_persons,
                force=(effective_mode == 'vlm'),
                should_abort=lambda: self._should_cancel(goal_handle),
            )
        else:
            vlm_future = None

        self.get_logger().info('Data copied for processing. Starting detection...')
        transform = None
        if request.target_frame and request.target_frame != header.frame_id:
            # Snapshot once at the start of the callback. Latest-available
            # TF, generous 5 s budget -- the pan-tilt + base chain is fixed
            # while the service runs, so a one-time snapshot is correct
            # for every centroid below.
            self._publish_feedback(
                goal_handle,
                stage='transforming',
                message='Snapshotting the latest target transform.',
                delay_limit=5.0,
            )
            transform = self.transform_helper.wait_lookup(
                request.target_frame,
                header.frame_id,
                deadline_s=5.0,
                latest=True,
                poll_s=0.02,
            )
            self._raise_if_canceled(goal_handle)
            if transform is None:
                response.status = -1
                response.error_msg = (
                    f'Failed to lookup transform from {header.frame_id} '
                    f'to {request.target_frame} within 5.0s'
                )
                self.get_logger().error(response.error_msg)
                self._publish_debug_image(
                    rgb_image, header, persons=0, waving=0,
                    status_text=(
                        f'TF FAILED ({header.frame_id} -> {request.target_frame})'),
                )
                if vlm_future is not None:
                    vlm_future.add_done_callback(self._log_discarded_vlm_result)
                return response

        self.get_logger().info(
            'Transform lookup successful (if needed). '
            'Processing depth image and running YOLO...')

        self._publish_feedback(
            goal_handle,
            stage='detecting',
            message='Running person segmentation and waving detection.',
            delay_limit=max(5.0, self._vlm_delay_limit()),
        )
        try:
            points, validmask_points = waving_optical_points(
                depth_m, camera_k)
        except Exception as exc:  # noqa: BLE001 -- bad frame shouldn't kill the executor
            response.status = -1
            response.error_msg = f'depth conversion failed: {exc}'
            self.get_logger().error(response.error_msg)
            self._publish_debug_image(
                rgb_image, header, persons=0, waving=0,
                status_text=f'DEPTH FAILED: {exc}',
            )
            if vlm_future is not None:
                vlm_future.add_done_callback(self._log_discarded_vlm_result)
            return response
        self._raise_if_canceled(goal_handle)
        yolo_results = self.yolo(rgb_image, conf=self.min_person_conf, verbose=False)
        self._raise_if_canceled(goal_handle)

        boxes = yolo_results[0].boxes
        masks = yolo_results[0].masks  # None if model has no seg head or no instances
        total_boxes = 0 if boxes is None else len(boxes)
        self.get_logger().info(
            f'YOLO inference done. Found {total_boxes} candidate box(es).')

        waving_persons_centroids = []
        waving_annotations = []
        waving_masks = []
        waving_sources = []  # 'mp' or 'vlm', kept aligned with the lists above
        person_records = []  # (x1, y1, x2, y2, seg_mask_or_None) for every person
        all_person_annotations = []  # (x1, y1, x2, y2, landmarks, is_wave) for every person
        person_candidates = 0
        if boxes is not None:
            for i, box in enumerate(boxes):
                self._raise_if_canceled(goal_handle)
                if self.yolo.names[int(box.cls[0])] == 'person':
                    person_candidates += 1
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    self.get_logger().info(
                        f'Detecting person candidate #{person_candidates} '
                        f'at bbox=({x1}, {y1}, {x2}, {y2})')
                    person_roi = rgb_image[y1:y2, x1:x2]

                    if person_roi.size == 0:
                        self.get_logger().info(
                            f'Person candidate #{person_candidates} skipped: empty ROI')
                        continue

                    # In 'vlm' mode the VLM is the sole waver source, so skip
                    # the MediaPipe pose pass entirely (saves the per-ROI
                    # inference); YOLO masks below still feed VLM centroids.
                    if run_mediapipe:
                        pose_results = self.pose.process(
                            cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                        landmarks = pose_results.pose_landmarks
                        is_wave = self.is_waving(landmarks, person_roi)
                    else:
                        landmarks = None
                        is_wave = False
                    all_person_annotations.append(
                        (x1, y1, x2, y2, landmarks, is_wave)
                    )
                    rec_mask = None
                    if masks is not None and i < len(masks.data):
                        seg = masks.data[i].cpu().numpy().astype(np.uint8)
                        if seg.shape != rgb_image.shape[:2]:
                            seg = cv2.resize(
                                seg,
                                (rgb_image.shape[1], rgb_image.shape[0]),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        rec_mask = seg.astype(bool)
                    person_records.append((x1, y1, x2, y2, rec_mask))

                    if is_wave:
                        # Prefer YOLO seg silhouette over the rectangular bbox so the
                        # centroid mean isn't pulled toward background pixels visible
                        # inside the bbox. Fall back to bbox if seg is unavailable.
                        if masks is not None and i < len(masks.data):
                            seg = masks.data[i].cpu().numpy().astype(np.uint8)
                            if seg.shape != rgb_image.shape[:2]:
                                seg = cv2.resize(
                                    seg,
                                    (rgb_image.shape[1], rgb_image.shape[0]),
                                    interpolation=cv2.INTER_NEAREST,
                                )
                            person_mask = seg.astype(bool)
                        else:
                            person_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
                            person_mask[y1:y2, x1:x2] = True

                        combined_mask = (
                            person_mask.astype(float) * validmask_points.astype(float))

                        # Distant persons can have a seg mask too sparse to hit 10
                        # valid depth pixels -- retry once with the bbox mask before
                        # dropping the candidate. Mirrors object_seg_yolo.py:854-858.
                        if combined_mask.sum() < 10:
                            self.get_logger().info(
                                f'Person candidate #{person_candidates}: seg mask too sparse '
                                f'({int(combined_mask.sum())} valid px); retrying with bbox.'
                            )
                            bbox_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
                            bbox_mask[y1:y2, x1:x2] = True
                            person_mask = bbox_mask
                            combined_mask = (
                                bbox_mask.astype(float) * validmask_points.astype(float))

                        if combined_mask.sum() < 10:
                            self.get_logger().info(
                                f'Person candidate #{person_candidates} skipped: '
                                f'no usable depth ({int(combined_mask.sum())} valid px).'
                            )
                            continue

                        person_points = points[np.nonzero(combined_mask)]

                        if person_points.shape[0] > 0:
                            centroid = np.mean(person_points, axis=0)
                            centroid[2] = np.median(person_points[:, 2])

                            if (request.threshold_meters > 0
                                    and centroid[2] > request.threshold_meters):
                                self.get_logger().info(
                                    f'Person candidate #{person_candidates} dropped: '
                                    f'depth {centroid[2]:.2f}m > threshold '
                                    f'{request.threshold_meters:.2f}m'
                                )
                            if (request.threshold_meters <= 0
                                    or centroid[2] <= request.threshold_meters):
                                point_stamped = PointStamped()
                                point_stamped.header = header
                                point_stamped.point.x = float(centroid[0])
                                point_stamped.point.y = float(centroid[1])
                                point_stamped.point.z = float(centroid[2])
                                waving_persons_centroids.append(point_stamped)
                                waving_annotations.append(
                                    (x1, y1, x2, y2, landmarks))
                                waving_masks.append(person_mask)
                                waving_sources.append('mp')
                                self.get_logger().info(
                                    f'Detected waving person '
                                    f'#{len(waving_persons_centroids)} '
                                    f'at ({point_stamped.point.x:.3f}, '
                                    f'{point_stamped.point.y:.3f}, '
                                    f'{point_stamped.point.z:.3f})'
                                )
        self.get_logger().info(f'Person candidates checked: {person_candidates}')

        n_vlm_added = 0
        vlm_provider_used = ''
        if vlm_future is not None:
            # 'vlm' mode always waits (the VLM is the only waver source);
            # 'hybrid' mode keeps the CV-found-enough short-circuit.
            if effective_mode == 'vlm' or should_wait_for_vlm(
                    len(waving_persons_centroids), request.min_waving_persons):
                try:
                    vlm_result = vlm_future.result(timeout=self.vlm_timeout_s)
                except (WavingVlmError, FutureTimeoutError) as exc:
                    self._raise_if_canceled(goal_handle)
                    self.get_logger().warn(
                        f'VLM waving fallback unavailable: {exc}')
                    vlm_result = None
                    vlm_future.add_done_callback(self._log_discarded_vlm_result)
                if vlm_result is not None:
                    self._raise_if_canceled(goal_handle)
                    n_vlm_added, vlm_provider_used = self._merge_vlm_result(
                        vlm_result, points, validmask_points, header, request,
                        person_records, waving_persons_centroids,
                        waving_annotations, waving_masks, waving_sources,
                    )
                    self.get_logger().info(
                        f'VLM fallback added {n_vlm_added} waver(s) '
                        f'(provider={vlm_provider_used or "none"}).')
                    self._raise_if_canceled(goal_handle)
            else:
                vlm_future.add_done_callback(self._log_discarded_vlm_result)
                self.get_logger().info(
                    f'CV already found {len(waving_persons_centroids)} '
                    f'waver(s) (>= request.min_waving_persons='
                    f'{request.min_waving_persons}); discarding VLM call '
                    f'without waiting.'
                )

        self._publish_feedback(
            goal_handle,
            stage='judging',
            message='Finalizing waving detections and centroids.',
            delay_limit=3.0,
        )

        # sort waving person centroids from closest to farthest (keep annotations + masks aligned)
        if waving_persons_centroids:
            quads = sorted(
                zip(waving_persons_centroids, waving_annotations,
                    waving_masks, waving_sources),
                key=lambda t: t[0].point.z,
            )
            waving_persons_centroids = [p for p, _, _, _ in quads]
            waving_annotations = [a for _, a, _, _ in quads]
            waving_masks = [m for _, _, m, _ in quads]
            waving_sources = [s for _, _, _, s in quads]

        # Always render an annotated frame -- only wavers are boxed (red).
        # Non-waving persons are left unboxed; empty scenes still emit the raw
        # RGB so operators can confirm the pipeline ran.
        annotated = self._annotate_all_persons(rgb_image, all_person_annotations)
        for (x1, y1, x2, y2, _lm), src in zip(waving_annotations, waving_sources):
            if src != 'vlm':
                continue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, 'waving (vlm)', (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        self._publish_debug_image(
            annotated, header,
            persons=person_candidates,
            waving=len(waving_persons_centroids),
            already_annotated=True,
        )
        if self.show_window:
            try:
                self._frame_queue.put_nowait(annotated)
            except queue.Full:
                # Drop the stale frame, keep the newest -- the window loop
                # would rather show the latest detection than fall behind.
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(annotated)
                except queue.Full:
                    pass

        if self._vision_logger.enabled:
            detections = []
            for (x1, y1, x2, y2, _lm), pt, person_mask, src in zip(
                waving_annotations, waving_persons_centroids,
                waving_masks, waving_sources
            ):
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'mask': person_mask,
                    'cls_name': ('waving_person_vlm' if src == 'vlm'
                                 else 'waving_person'),
                    'conf': 1.0,
                    'centroid': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'centroid_3d': [float(pt.point.x), float(pt.point.y),
                                    float(pt.point.z)],
                })
            self._vision_logger.write(
                rgb_image, detections,
                request_ctx={
                    'target_frame': request.target_frame,
                    'threshold_meters': float(request.threshold_meters),
                    'min_waving_persons': int(request.min_waving_persons),
                },
                branch='detect_waving',
                extras={'n_person_candidates': person_candidates,
                        'n_vlm_added': n_vlm_added,
                        'vlm_provider': vlm_provider_used},
                timings={'detect_waving': time.perf_counter() - _t0},
            )

        # Build image-space boxes 1:1 with the (sorted) centroid list so the
        # consumer (re-seed) can pick a waver and recover its 2D box. Both the
        # MediaPipe and VLM-fallback paths populate waving_annotations in the
        # same index order as waving_persons_centroids, and the sort above keeps
        # them aligned -- so iterating waving_annotations is alignment-safe.
        waving_boxes = []
        for x1, y1, x2, y2, _lm in waving_annotations:
            roi = RegionOfInterest()
            roi.x_offset = max(0, int(x1))
            roi.y_offset = max(0, int(y1))
            roi.width = max(0, int(x2) - int(x1))
            roi.height = max(0, int(y2) - int(y1))
            roi.do_rectify = False
            waving_boxes.append(roi)

        if request.target_frame and waving_persons_centroids:
            if request.target_frame != header.frame_id:
                try:
                    transformed_points = []
                    for point in waving_persons_centroids:
                        self._raise_if_canceled(goal_handle)
                        transformed = self.transform_helper.transform_point(
                            point, transform)
                        if transformed is None:
                            raise RuntimeError('point transform failed')
                        transformed_points.append(transformed)
                    response.waving_persons = transformed_points
                    response.waving_boxes = waving_boxes
                except Exception as e:  # noqa: BLE001
                    self._raise_if_canceled(goal_handle)
                    response.status = -1
                    response.error_msg = (
                        f'Failed to transform point from {header.frame_id} '
                        f'to {request.target_frame}: {e}')
                    self.get_logger().error(response.error_msg)
                    # Detection actually ran on this path -- preserve the
                    # red waver overlay rather than overwriting the
                    # cached frame with the raw rgb_image.
                    self._publish_debug_image(
                        annotated, header, persons=person_candidates,
                        waving=len(waving_persons_centroids),
                        status_text=f'POINT TRANSFORM FAILED: {e}',
                        already_annotated=True,
                    )
                    return response
            else:
                response.waving_persons = waving_persons_centroids
                response.waving_boxes = waving_boxes
        else:
            response.waving_persons = waving_persons_centroids
            response.waving_boxes = waving_boxes

        if response.waving_persons:
            response.status = 0
            response.error_msg = f'Detected {len(response.waving_persons)} waving person(s).'
            self.get_logger().info(response.error_msg)
        else:
            response.status = 1
            response.error_msg = 'No waving persons detected'
            self.get_logger().info(response.error_msg)

        self.get_logger().info(
            f'Detect waving request processing complete in '
            f'{time.perf_counter() - _t0:.3f} seconds.')

        return response


def main(args=None):
    rclpy.init(args=args)
    node = DetectWavingPersonsNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
