import rclpy
from rclpy.node import Node
from tinker_vision_msgs_26.srv import DetectWaving
from sensor_msgs.msg import Image, CameraInfo
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2
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

        # OpenCV imshow from a non-main thread is fragile on Linux+Qt builds
        # of opencv-python (silent no-show, QSocketNotifier warnings). Default
        # off; opt in via -p show_window:=true at your own risk. The
        # ROS-canonical view is `rqt_image_view /detect_waving_persons/debug_image`.
        self.declare_parameter('show_window', False)
        self.show_window = self.get_parameter('show_window').get_parameter_value().bool_value
        self.declare_parameter('min_person_conf', 0.4)
        self.min_person_conf = float(self.get_parameter('min_person_conf').value)
        self.debug_image_pub = self.create_publisher(Image, '~/debug_image', 1)

        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self._vision_logger = VisionLogger(
            self,
            self.get_parameter('vision_logging_enabled').get_parameter_value().bool_value,
            self.get_parameter('vision_log_folder').get_parameter_value().string_value,
        )

        self._frame_queue = None
        self._display_thread = None
        if self.show_window:
            self._frame_queue = queue.Queue(maxsize=2)
            self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
            self._display_thread.start()

        self.get_logger().info(f'Detect Waving Persons node started (show_window={self.show_window})')

    def _display_loop(self):
        while rclpy.ok():
            try:
                frame = self._frame_queue.get(timeout=0.1)
                cv2.imshow('waving_persons', frame)
            except queue.Empty:
                pass
            cv2.waitKey(1)
        cv2.destroyAllWindows()

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

    # Tuned against detect_waving_test/ on 2026-05-04: visibility filter at
    # 0.5, shoulder/elbow tolerances at 0.1 in normalized image-y units yield
    # 12/18 wave recall with 0/7 false alarms (~76% accuracy). The remaining
    # FNs are the far/occluded set where MediaPipe visibility is genuinely
    # below 0.5 — correctly suppressed rather than guessed.
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

        # MediaPipe-pose landmarks carry .visibility ∈ [0, 1]; values < 0.5 mean
        # the model isn't confident the joint is in-frame, so the (x, y) is
        # unreliable — refuse to classify rather than guess.
        if min(lm.visibility for lm in (rh, re, rs, lh, le, ls)) < self.MIN_VISIBILITY:
            self.get_logger().debug(
                f'is_waving: visibility too low '
                f'(min={min(lm.visibility for lm in (rh, re, rs, lh, le, ls)):.2f}); skip'
            )
            return False

        # All landmark .y values are normalized [0, 1] from image top.
        rh_above_sh = rh.y <= rs.y + self.SHOULDER_TOL_NORM
        lh_above_sh = lh.y <= ls.y + self.SHOULDER_TOL_NORM
        rh_above_el = rh.y < re.y
        lh_above_el = lh.y < le.y
        re_above_sh = re.y <= rs.y + self.ELBOW_TOL_NORM
        le_above_sh = le.y <= ls.y + self.ELBOW_TOL_NORM

        self.get_logger().debug(
            f'is_waving: rh.y={rh.y:.2f} rs.y={rs.y:.2f} re.y={re.y:.2f} | '
            f'lh.y={lh.y:.2f} ls.y={ls.y:.2f} le.y={le.y:.2f} | '
            f'rh^sh={rh_above_sh} lh^sh={lh_above_sh} '
            f'rh^el={rh_above_el} lh^el={lh_above_el} '
            f're^sh={re_above_sh} le^sh={le_above_sh}'
        )

        gesture = (rh_above_sh or lh_above_sh
                   or (rh_above_el and re_above_sh)
                   or (lh_above_el and le_above_sh))
        if gesture:
            self.get_logger().info(f'Wave gesture detected (ROI {person_roi.shape[1]}x{person_roi.shape[0]})')
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
            try:
                transform = self.tf_buffer.lookup_transform(
                    request.target_frame,
                    header.frame_id,
                    header.stamp,
                    timeout=rclpy.duration.Duration(seconds=1.0)
                )
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                response.status = -1
                response.error_msg = f"Failed to lookup transform from {header.frame_id} to {request.target_frame}: {e}"
                self.get_logger().error(response.error_msg)
                return response
            
        self.get_logger().info('Transform lookup successful (if needed). Processing depth image and running YOLO...')

        try:
            points, validmask_points = depth_image_to_points(depth_image, camera_k, self.bridge)
        except Exception as exc:  # noqa: BLE001 — bad frame shouldn't kill the executor
            response.status = -1
            response.error_msg = f'depth conversion failed: {exc}'
            self.get_logger().error(response.error_msg)
            return response
        yolo_results = self.yolo(rgb_image, conf=self.min_person_conf, verbose=False)

        boxes = yolo_results[0].boxes
        masks = yolo_results[0].masks  # None if model has no seg head or returned no instances
        total_boxes = 0 if boxes is None else len(boxes)
        self.get_logger().info(f'YOLO inference done. Found {total_boxes} candidate box(es).')

        waving_persons_centroids = []
        waving_annotations = []
        waving_masks = []
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

                    if self.is_waving(pose_results.pose_landmarks, person_roi):
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

        if waving_persons_centroids:
            annotated = self._annotate_frame(rgb_image, waving_annotations, waving_persons_centroids)
            # Always publish the annotated debug image — robust across all GUI
            # backends, viewable via `rqt_image_view /detect_waving_persons/debug_image`.
            try:
                msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
                msg.header = header
                self.debug_image_pub.publish(msg)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f'debug_image publish failed: {exc}')
            # cv2.imshow path is opt-in (Linux+Qt-fragile, see __init__).
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
