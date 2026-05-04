import rclpy
from rclpy.node import Node
from tinker_vision_msgs_26.srv import DetectWaving
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
import os
import shutil
import subprocess
import time
import queue
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import threading
from geometry_msgs.msg import PointStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import tf2_ros
import tf2_geometry_msgs

# Shared logger
from vision_util.vision_logging import VisionLogger
from vision_util.weights_cache import resolve_weights

def depth_image_to_points(
    depth_msg: Image, cam_K: np.ndarray, bridge: CvBridge,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project a registered depth Image to a (H, W, 3) XYZ grid in the
    camera optical frame, plus a (H, W) bool valid mask. Mirrors the math in
    `_process_realsense_data` (object_seg_yolo.py:407–431) but uses the
    standard pinhole convention (u=col↔fx,cx; v=row↔fy,cy) so the output
    matches the Orbbec depth_registered/points frame the rest of this node
    expects."""
    depth_img = bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(float) / 1000.0
    H, W = depth_img.shape
    fx, fy, cx, cy = cam_K[0, 0], cam_K[1, 1], cam_K[0, 2], cam_K[1, 2]

    valid_mask = (depth_img > 1e-6) & (depth_img < 10.0)
    depth_img = np.clip(depth_img, 0.0, 10.0)

    u = np.arange(W, dtype=float)[None, :]   # column index → x
    v = np.arange(H, dtype=float)[:, None]   # row index    → y
    x = (u - cx) * depth_img / fx
    y = (v - cy) * depth_img / fy

    points = np.stack([x, y, depth_img], axis=2)
    return points, valid_mask

class DetectWavingPersonsNode(Node):
    def __init__(self):
        super().__init__('detect_waving_persons_node')
        self.srv = self.create_service(DetectWaving, 'detect_waving_persons', self.detect_waving_callback, callback_group=MutuallyExclusiveCallbackGroup())

        self.declare_parameter('color_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('sync_slop_sec', 0.1)
        color_topic = self.get_parameter('color_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        sync_slop_sec = float(self.get_parameter('sync_slop_sec').value)

        image_sub = Subscriber(self, Image, color_topic)
        depth_image_sub = Subscriber(self, Image, depth_topic)

        self.ts = ApproximateTimeSynchronizer([image_sub, depth_image_sub], queue_size=10, slop=sync_slop_sec)
        self.ts.registerCallback(self.image_callback)

        self.create_subscription(CameraInfo, camera_info_topic, self.camera_info_callback, 10)

        self.bridge = CvBridge()
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

        self.img_lock = threading.Lock()
        self.intrinsiscs_lock = threading.Lock()
        self.rgb_image = None
        self.depth_image = None
        self.header = None
        self.camera_k = None
        self.received_intrinsics = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # show_window=true spawns rqt_image_view as a subprocess subscribed to
        # /detect_waving_debug_image. cv2.imshow is unreliable here because the system has
        # opencv-python-headless installed alongside opencv-python, and the
        # headless wheel wins import resolution → cv2.waitKey raises
        # "not implemented. Rebuild the library with GTK+ 2.x". rqt_image_view
        # uses Qt directly and works regardless of the cv2 wheel.
        self.declare_parameter('show_window', True)
        self.show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        self.declare_parameter('min_person_conf', 0.4)
        self.min_person_conf = float(self.get_parameter('min_person_conf').value)
        # rqt_image_view subscribes via image_transport, which hard-codes
        # `rmw_qos_profile_default` (RELIABLE, VOLATILE, KEEP_LAST=10) and
        # offers no GUI/CLI override (rqt_image_view#54, image_common#156).
        # Mirror that profile here: the integer 10 means depth=10 with rcl
        # defaults — exactly what the subscriber expects. depth=10 also
        # prevents the outbound queue from overflowing on back-to-back
        # service calls (which is what dropped frames at depth=1).
        self.debug_image_pub = self.create_publisher(
            Image, '/detect_waving_debug_image', 10,
        )

        # Late-subscriber recovery. rqt_image_view subscribes VOLATILE and its
        # GUI/Qt cold start (~3-5 s) often outlives the launch's 1 s grace
        # period, so service publishes that fire during that window are
        # dropped forever. Cache the last published Image msg and republish
        # at 2 Hz from a separate callback group — the next tick latches
        # onto whatever subscriber appeared in the meantime, so the most
        # recent annotated frame always reaches rqt within ~0.5 s. Identical
        # content on republish ⇒ no rqt flicker.
        self._last_debug_msg_lock = threading.Lock()
        self._last_debug_msg = None
        # 10 Hz republish — gives rqt a steady ~100 ms cadence. Header stamp
        # is rewritten to `now()` on each tick (see _republish_last_debug)
        # because image_transport drops frames whose stamp is older than
        # the most recently displayed one; reusing the original capture
        # stamp made republishes look intermittent.
        self._republish_timer = self.create_timer(
            0.1,
            self._republish_last_debug,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self._vision_logger = VisionLogger(
            self,
            self.get_parameter('vision_logging_enabled').get_parameter_value().bool_value,
            self.get_parameter('vision_log_folder').get_parameter_value().string_value,
        )

        self._viewer_proc = None
        if self.show_window:
            self._spawn_image_viewer()

        self.get_logger().info(f'Detect Waving Persons node started (show_window={self.show_window})')

    def _spawn_image_viewer(self):
        """Launch rqt_image_view subscribed to the debug_image topic.

        Falls back to a warning if rqt_image_view isn't installed — the
        debug_image topic is still published, so the operator can run any
        viewer manually."""
        topic = '/detect_waving_debug_image'
        if shutil.which('ros2') is None:
            self.get_logger().warn('show_window=true but `ros2` not on PATH; skipping viewer spawn.')
            return
        env = os.environ.copy()
        env.setdefault('DISPLAY', ':0')
        try:
            self._viewer_proc = subprocess.Popen(
                ['ros2', 'run', 'rqt_image_view', 'rqt_image_view', topic],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.get_logger().info(
                f'Spawned rqt_image_view (pid={self._viewer_proc.pid}) on {topic}'
            )
        except FileNotFoundError as exc:
            self.get_logger().warn(f'Failed to spawn rqt_image_view: {exc}')

    def destroy_node(self):
        if self._viewer_proc is not None and self._viewer_proc.poll() is None:
            try:
                self._viewer_proc.terminate()
                self._viewer_proc.wait(timeout=2.0)
            except Exception:  # noqa: BLE001
                self._viewer_proc.kill()
        return super().destroy_node()

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
                    self.mp_draw.draw_landmarks(roi, landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def _annotate_all_persons(self, rgb_bgr, person_annotations):
        """Draw every detected person, color-coded by waving verdict.

        Red bbox = waving, green bbox = still. Renders on every service call
        so the debug window/image is populated even when no wave fires."""
        frame = rgb_bgr.copy()
        wave_idx = 0
        for x1, y1, x2, y2, landmarks, is_wave in person_annotations:
            color = (0, 0, 255) if is_wave else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if is_wave:
                wave_idx += 1
                label = f'waving #{wave_idx}'
            else:
                label = 'still'
            cv2.putText(
                frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
            if landmarks is not None:
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    self.mp_draw.draw_landmarks(roi, landmarks, self.mp_pose.POSE_CONNECTIONS)
        return frame

    def camera_info_callback(self, msg):
        if self.received_intrinsics:
            return
        self.intrinsiscs_lock.acquire()
        try:
            self.camera_k = np.array(msg.k).reshape((3, 3))
            self.get_logger().info('Camera info received.')
            self.received_intrinsics = True
        finally:
            self.intrinsiscs_lock.release()

    def _snapshot_latest_transform(self, target_frame, source_frame,
                                    wait_seconds=1.0, poll_period=0.02):
        """Snapshot the latest available TF for (target<-source) at call time.

        Called once at the *start* of the service callback so that all per-
        centroid transforms later in the callback use the same fixed pose —
        this both removes the "TF moved between centroid #1 and centroid #N"
        race and sidesteps the "Lookup would require extrapolation into the
        future" error you get when the image stamp is a few ms newer than
        the most recent TF (the camera+TF pipeline lag a tick behind images).
        We pass `rclpy.time.Time()` (a default-constructed Time = epoch 0,
        which tf2 interprets as "give me the latest"), so no extrapolation
        ever happens — we always read what's already in the buffer.

        For slow-changing chains like base_link↔camera_color_optical_frame
        on the pan-tilt this is sub-mm accurate. If the chain is empty
        (TF listener hasn't received any frames yet on first request),
        poll up to `wait_seconds` for one to arrive. Returns TransformStamped
        or None on total failure."""
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=wait_seconds)
        latest = rclpy.time.Time()  # tf2 magic value for "latest"
        while True:
            if self.tf_buffer.can_transform(target_frame, source_frame, latest):
                try:
                    return self.tf_buffer.lookup_transform(
                        target_frame, source_frame, latest,
                    )
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                        tf2_ros.ExtrapolationException) as exc:
                    self.get_logger().error(
                        f'TF lookup raced after can_transform passed '
                        f'({target_frame}<-{source_frame}): {exc}'
                    )
                    return None
            if self.get_clock().now() >= deadline:
                self.get_logger().error(
                    f'TF for {target_frame}<-{source_frame} not available '
                    f'within {wait_seconds:.2f}s'
                )
                return None
            time.sleep(poll_period)

    def _publish_debug_image(self, image, header, *,
                              persons, waving, status_text=None,
                              already_annotated=False):
        """Publish an annotated debug frame to /detect_waving_debug_image.

        Called from every service-callback exit path that has an `rgb_image`
        in hand (success, TF failure, depth failure, post-success transform
        failure). Publishing on failure is what lets the operator see *what
        the camera was looking at when the request was rejected* — without
        it, debugging "why did my service call fail" requires re-running
        the camera capture.

        `already_annotated=True` skips re-drawing (used by the success path,
        which has already drawn red/green person bboxes via
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

        Stamp is rewritten to wall-clock now() on every tick — rqt /
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

    def image_callback(self, rgb_msg, depth_image_msg):
        # Convert outside the lock — CvBridge can raise on a malformed image
        # message, and a leaked img_lock would deadlock every subsequent
        # image callback and service request under MultiThreadedExecutor.
        try:
            cv_img = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
        except Exception as exc:  # noqa: BLE001 — drop bad frame, keep node alive
            self.get_logger().warn(f'imgmsg_to_cv2 failed: {exc}; dropping frame')
            return
        self.img_lock.acquire()
        try:
            self.rgb_image = cv_img
            self.depth_image = depth_image_msg
            self.header = rgb_msg.header
        finally:
            self.img_lock.release()

    # Tuned against detect_waving_test/ (41 images) on 2026-05-04: per-side
    # visibility gate (each arm trusted independently if its 3 joints all
    # exceed MIN_VISIBILITY) + shoulder/elbow tolerances of 0.1 in normalized
    # image-y units yield 24/30 wave recall with 0/11 false alarms
    # (~85% accuracy). A *global* min-visibility gate over all 6 joints was
    # too strict — single-arm waves naturally have the resting/occluded arm
    # below 0.5 visibility, which collapsed recall on real wave gestures.
    MIN_VISIBILITY = 0.5
    SHOULDER_TOL_NORM = 0.1
    ELBOW_TOL_NORM = 0.1

    def is_waving(self, pose_landmarks, person_roi):
        if pose_landmarks is None:
            return False

        landmarks = pose_landmarks.landmark
        PL = mp.solutions.pose.PoseLandmark
        rh, re, rs = landmarks[PL.RIGHT_WRIST], landmarks[PL.RIGHT_ELBOW], landmarks[PL.RIGHT_SHOULDER]
        lh, le, ls = landmarks[PL.LEFT_WRIST],  landmarks[PL.LEFT_ELBOW],  landmarks[PL.LEFT_SHOULDER]

        # Per-side visibility: an arm is "trusted" only if all 3 of its joints
        # have visibility ≥ MIN_VISIBILITY. We then evaluate each trusted arm
        # independently — a wave on one arm fires even if the other arm is
        # occluded (typical when only one hand is raised).
        right_visible = min(rh.visibility, re.visibility, rs.visibility) >= self.MIN_VISIBILITY
        left_visible  = min(lh.visibility, le.visibility, ls.visibility) >= self.MIN_VISIBILITY
        if not (right_visible or left_visible):
            self.get_logger().debug(
                f'is_waving: neither arm visible '
                f'(R_min={min(rh.visibility, re.visibility, rs.visibility):.2f}, '
                f'L_min={min(lh.visibility, le.visibility, ls.visibility):.2f}); skip'
            )
            return False

        # All landmark .y values are normalized [0, 1] from image top.
        right_wave = right_visible and (
            rh.y <= rs.y + self.SHOULDER_TOL_NORM
            or (rh.y < re.y and re.y <= rs.y + self.ELBOW_TOL_NORM)
        )
        left_wave = left_visible and (
            lh.y <= ls.y + self.SHOULDER_TOL_NORM
            or (lh.y < le.y and le.y <= ls.y + self.ELBOW_TOL_NORM)
        )

        self.get_logger().debug(
            f'is_waving: R_vis={right_visible} L_vis={left_visible} | '
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


    def detect_waving_callback(self, request, response):
        _t0 = time.perf_counter()
        self.get_logger().info('Detect waving request received. Detecting persons now...')
        
        self.img_lock.acquire()
        try:
            if self.rgb_image is None or self.depth_image is None:
                response.status = -1
                response.error_msg = 'No image, depth data received yet'
                self.get_logger().error(response.error_msg)
                return response
            if self.camera_k is None:
                response.status = -1
                response.error_msg = 'No camera info received yet'
                self.get_logger().error(response.error_msg)
                return response

            rgb_image = self.rgb_image.copy()
            depth_image = self.depth_image
            header = self.header
            camera_k = self.camera_k
        finally:
            self.img_lock.release()

        self.get_logger().info('Data copied for processing. Starting detection...')
        transform = None
        if request.target_frame and request.target_frame != header.frame_id:
            # Snapshot once at the start of the callback. Latest-available
            # TF, generous 5 s budget — the pan-tilt + base chain is fixed
            # while the service runs, so a one-time snapshot is correct
            # for every centroid below.
            transform = self._snapshot_latest_transform(
                request.target_frame, header.frame_id,
                wait_seconds=5.0,
            )
            if transform is None:
                response.status = -1
                response.error_msg = (
                    f'Failed to lookup transform from {header.frame_id} '
                    f'to {request.target_frame} within 5.0s'
                )
                self.get_logger().error(response.error_msg)
                self._publish_debug_image(
                    rgb_image, header, persons=0, waving=0,
                    status_text=f'TF FAILED ({header.frame_id} -> {request.target_frame})',
                )
                return response

        self.get_logger().info('Transform lookup successful (if needed). Processing depth image and running YOLO...')

        try:
            points, validmask_points = depth_image_to_points(depth_image, camera_k, self.bridge)
        except Exception as exc:  # noqa: BLE001 — bad frame shouldn't kill the executor
            response.status = -1
            response.error_msg = f'depth conversion failed: {exc}'
            self.get_logger().error(response.error_msg)
            self._publish_debug_image(
                rgb_image, header, persons=0, waving=0,
                status_text=f'DEPTH FAILED: {exc}',
            )
            return response
        yolo_results = self.yolo(rgb_image, conf=self.min_person_conf, verbose=False)

        boxes = yolo_results[0].boxes
        masks = yolo_results[0].masks  # None if model has no seg head or returned no instances
        total_boxes = 0 if boxes is None else len(boxes)
        self.get_logger().info(f'YOLO inference done. Found {total_boxes} candidate box(es).')

        waving_persons_centroids = []
        waving_annotations = []
        waving_masks = []
        all_person_annotations = []  # (x1, y1, x2, y2, landmarks, is_wave) for every person
        person_candidates = 0
        if boxes is not None:
            for i, box in enumerate(boxes):
                if self.yolo.names[int(box.cls[0])] == 'person':
                    person_candidates += 1
                    x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                    self.get_logger().info(f'Detecting person candidate #{person_candidates} at bbox=({x1}, {y1}, {x2}, {y2})')
                    person_roi = rgb_image[y1:y2, x1:x2]

                    if person_roi.size == 0:
                        self.get_logger().info(f'Person candidate #{person_candidates} skipped: empty ROI')
                        continue

                    pose_results = self.pose.process(cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB))
                    is_wave = self.is_waving(pose_results.pose_landmarks, person_roi)
                    all_person_annotations.append(
                        (x1, y1, x2, y2, pose_results.pose_landmarks, is_wave)
                    )

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

                        combined_mask = person_mask.astype(float) * validmask_points.astype(float)

                        # Distant persons can have a seg mask too sparse to hit 10
                        # valid depth pixels — retry once with the bbox mask before
                        # dropping the candidate. Mirrors object_seg_yolo.py:854–858.
                        if combined_mask.sum() < 10:
                            self.get_logger().info(
                                f'Person candidate #{person_candidates}: seg mask too sparse '
                                f'({int(combined_mask.sum())} valid px); retrying with bbox.'
                            )
                            bbox_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
                            bbox_mask[y1:y2, x1:x2] = True
                            person_mask = bbox_mask
                            combined_mask = bbox_mask.astype(float) * validmask_points.astype(float)

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

                            if request.threshold_meters > 0 and centroid[2] > request.threshold_meters:
                                self.get_logger().info(
                                    f'Person candidate #{person_candidates} dropped: '
                                    f'depth {centroid[2]:.2f}m > threshold {request.threshold_meters:.2f}m'
                                )
                            if request.threshold_meters <= 0 or centroid[2] <= request.threshold_meters:
                                point_stamped = PointStamped()
                                point_stamped.header = header
                                point_stamped.point.x = float(centroid[0])
                                point_stamped.point.y = float(centroid[1])
                                point_stamped.point.z = float(centroid[2])
                                waving_persons_centroids.append(point_stamped)
                                waving_annotations.append((x1, y1, x2, y2, pose_results.pose_landmarks))
                                waving_masks.append(person_mask)
                                self.get_logger().info(
                                    f'Detected waving person #{len(waving_persons_centroids)} '
                                    f'at ({point_stamped.point.x:.3f}, {point_stamped.point.y:.3f}, {point_stamped.point.z:.3f})'
                                )
        self.get_logger().info(f'Person candidates checked: {person_candidates}')
        # sort waving person centroids from closest to farthest (keep annotations + masks aligned)
        if waving_persons_centroids:
            triples = sorted(
                zip(waving_persons_centroids, waving_annotations, waving_masks),
                key=lambda t: t[0].point.z,
            )
            waving_persons_centroids = [p for p, _, _ in triples]
            waving_annotations = [a for _, a, _ in triples]
            waving_masks = [m for _, _, m in triples]

        # Always render an annotated frame — every person candidate is drawn
        # (red=waving, green=still). Empty scenes still emit the raw RGB so
        # operators can confirm the pipeline ran.
        annotated = self._annotate_all_persons(rgb_image, all_person_annotations)
        self._publish_debug_image(
            annotated, header,
            persons=person_candidates,
            waving=len(waving_persons_centroids),
            already_annotated=True,
        )
        if self.show_window and self._frame_queue is not None:
            try:
                self._frame_queue.put_nowait(annotated)
            except queue.Full:
                pass

        if self._vision_logger.enabled:
            detections = []
            for (x1, y1, x2, y2, _lm), pt, person_mask in zip(
                waving_annotations, waving_persons_centroids, waving_masks
            ):
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'mask': person_mask,
                    'cls_name': 'waving_person',
                    'conf': 1.0,
                    'centroid': [(x1 + x2) // 2, (y1 + y2) // 2],
                    'centroid_3d': [float(pt.point.x), float(pt.point.y), float(pt.point.z)],
                })
            self._vision_logger.write(
                rgb_image, detections,
                request_ctx={
                    'target_frame': request.target_frame,
                    'threshold_meters': float(request.threshold_meters),
                },
                branch='detect_waving',
                extras={'n_person_candidates': person_candidates},
                timings={'detect_waving': time.perf_counter() - _t0},
            )

        if request.target_frame and waving_persons_centroids:
            if request.target_frame != header.frame_id:
                try:
                    transformed_points = []
                    for point in waving_persons_centroids:
                        transformed_points.append(tf2_geometry_msgs.do_transform_point(point, transform))
                    response.waving_persons = transformed_points
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                    response.status = -1
                    response.error_msg = f"Failed to transform point from {header.frame_id} to {request.target_frame}: {e}"
                    self.get_logger().error(response.error_msg)
                    # Detection actually ran on this path — preserve the
                    # red/green person overlay rather than overwriting the
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
        else:
            response.waving_persons = waving_persons_centroids

        if response.waving_persons:
            response.status = 0
            response.error_msg = f"Detected {len(response.waving_persons)} waving person(s)."
            self.get_logger().info(response.error_msg)
        else:
            response.status = 1
            response.error_msg = "No waving persons detected"
            self.get_logger().info(response.error_msg)
        
        self.get_logger().info(f'Detect waving request processing complete in {time.perf_counter() - _t0:.3f} seconds.')

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
