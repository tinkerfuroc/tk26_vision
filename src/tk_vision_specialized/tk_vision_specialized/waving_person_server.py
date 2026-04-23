import rclpy
from rclpy.node import Node
from tinker_vision_msgs_26.srv import DetectWaving
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
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

def get_array_from_points(points: PointCloud2, cam_K: np.array) -> tuple[np.array, np.array]:
    h, w = 720, 1280
    arr = np.frombuffer(points.data, dtype='<f4')
    N = len(arr) // 5
    points = arr.reshape((N, 5))[:, [0, 1, 2]]
    points_homo = points / np.repeat(points[:, 2: 3], 3, axis=1)
    coor_homo = (cam_K @ points_homo.T).T
    coor = np.rint(coor_homo[:, :2]).astype(int)

    depth_img = np.zeros((h, w, 3))
    valid_coor = (coor[:, 0] >= 0) & (coor[:, 0] < w) & (coor[:, 1] >= 0) & (coor[:, 1] < h)
    coor = coor[valid_coor]
    points = points[valid_coor]
    depth_img[coor[:, 1], coor[:, 0], :] = points
    mask = (depth_img[:, :, 2] > 1e-3)
    return depth_img, mask

class DetectWavingPersonsNode(Node):
    def __init__(self):
        super().__init__('detect_waving_persons_node')
        self.srv = self.create_service(DetectWaving, 'detect_waving_persons', self.detect_waving_callback, callback_group=MutuallyExclusiveCallbackGroup())

        image_sub = Subscriber(self, Image, '/camera/color/image_raw')
        point_cloud_sub = Subscriber(self, PointCloud2, '/camera/depth_registered/points')

        self.ts = ApproximateTimeSynchronizer([image_sub, point_cloud_sub], queue_size=10, slop=0.2)
        self.ts.registerCallback(self.image_callback)

        self.create_subscription(CameraInfo, '/camera/color/camera_info', self.camera_info_callback, 10)

        self.bridge = CvBridge()
        self.yolo = YOLO('yolov8s.pt')
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.lock = threading.Lock()
        self.rgb_image = None
        self.depth_points = None
        self.header = None
        self.camera_k = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.declare_parameter('show_window', True)
        self.show_window = self.get_parameter('show_window').get_parameter_value().bool_value

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
        if self.camera_k is None:
            self.camera_k = np.array(msg.k).reshape((3, 3))
            self.get_logger().info('Camera info received.')

    def image_callback(self, rgb_msg, depth_msg):
        with self.lock:
            self.rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            self.depth_points = depth_msg
            self.header = rgb_msg.header
            #self.get_logger().info('Image info received.')

    def is_waving(self, pose_landmarks, person_roi):
        if pose_landmarks is None:
            return False

        landmarks = pose_landmarks.landmark

        img_h, img_w, _ = person_roi.shape

        right_hand = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

        left_hand = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]

        nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]

        right_hand_above_shoulder = right_hand.y <= right_shoulder.y
        left_hand_above_shoulder = left_hand.y <= left_shoulder.y

        right_hand_above_elbow = right_hand.y < right_elbow.y
        left_hand_above_elbow = left_hand.y < left_elbow.y

        right_elbow_above_shoulder = right_elbow.y <= (right_shoulder.y + int(img_h * 0.1))
        left_elbow_above_shoulder = left_elbow.y <= (left_shoulder.y + int(img_h * 0.1))

        # log in separate lines, both pose landmarks and boolean values
        self.get_logger().info(f"nose: {nose}")
        self.get_logger().info(f"right_hand: {right_hand}")
        self.get_logger().info(f"right_elbow: {right_elbow}")
        self.get_logger().info(f"right_shoulder: {right_shoulder}")
        self.get_logger().info(f"left_hand: {left_hand}")
        self.get_logger().info(f"left_elbow: {left_elbow}")
        self.get_logger().info(f"left_shoulder: {left_shoulder}")
        self.get_logger().info(f"right_hand_above_shoulder: {right_hand_above_shoulder}")
        self.get_logger().info(f"left_hand_above_shoulder: {left_hand_above_shoulder}")
        self.get_logger().info(f"right_hand_above_elbow: {right_hand_above_elbow}")
        self.get_logger().info(f"right_elbow_above_shoulder: {right_elbow_above_shoulder}")
        self.get_logger().info(f"left_hand_above_elbow: {left_hand_above_elbow}")
        self.get_logger().info(f"left_elbow_above_shoulder: {left_elbow_above_shoulder}")

        is_waving_gesture = (right_hand_above_shoulder or left_hand_above_shoulder or
                             (right_hand_above_elbow and right_elbow_above_shoulder) or
                             (left_hand_above_elbow and left_elbow_above_shoulder))

        return is_waving_gesture


    def detect_waving_callback(self, request, response):
        _t0 = time.perf_counter()
        self.get_logger().info('Detect waving request received. Detecting persons now...')
        with self.lock:
            if self.rgb_image is None or self.depth_points is None or self.camera_k is None:
                response.status = -1
                response.error_msg = 'No image, depth data, or camera info received yet'
                return response

            rgb_image = self.rgb_image.copy()
            depth_points = self.depth_points
            header = self.header
            camera_k = self.camera_k

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

        points, validmask_points = get_array_from_points(depth_points, camera_k)
        yolo_results = self.yolo(rgb_image)

        boxes = yolo_results[0].boxes
        total_boxes = 0 if boxes is None else len(boxes)
        self.get_logger().info(f'YOLO inference done. Found {total_boxes} candidate box(es).')

        waving_persons_centroids = []
        waving_annotations = []
        person_candidates = 0
        if boxes is not None:
            for box in boxes:
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
                        person_mask = np.zeros(rgb_image.shape[:2], dtype=bool)
                        person_mask[y1:y2, x1:x2] = True

                        combined_mask = person_mask & validmask_points

                        if np.any(combined_mask):
                            person_points = points[combined_mask]

                            if person_points.shape[0] > 0:
                                centroid = np.mean(person_points, axis=0)

                                if request.threshold_meters <= 0 or centroid[2] <= request.threshold_meters:
                                    point_stamped = PointStamped()
                                    point_stamped.header = header
                                    point_stamped.point.x = float(centroid[0])
                                    point_stamped.point.y = float(centroid[1])
                                    point_stamped.point.z = float(centroid[2])
                                    waving_persons_centroids.append(point_stamped)
                                    waving_annotations.append((x1, y1, x2, y2, pose_results.pose_landmarks))
                                    self.get_logger().info(
                                        f'Detected waving person #{len(waving_persons_centroids)} '
                                        f'at ({point_stamped.point.x:.3f}, {point_stamped.point.y:.3f}, {point_stamped.point.z:.3f})'
                                    )
        self.get_logger().info(f'Person candidates checked: {person_candidates}')
        # sort waving person centroids from closest to farthest (keep annotations aligned)
        if waving_persons_centroids:
            paired = sorted(
                zip(waving_persons_centroids, waving_annotations),
                key=lambda pair: pair[0].point.z,
            )
            waving_persons_centroids = [p for p, _ in paired]
            waving_annotations = [a for _, a in paired]

        if self.show_window and self._frame_queue is not None and waving_persons_centroids:
            annotated = self._annotate_frame(rgb_image, waving_annotations, waving_persons_centroids)
            try:
                self._frame_queue.put_nowait(annotated)
            except queue.Full:
                pass

        if self._vision_logger.enabled:
            detections = []
            for (x1, y1, x2, y2, _lm), pt in zip(
                waving_annotations, waving_persons_centroids
            ):
                mask = np.zeros(rgb_image.shape[:2], dtype=bool)
                mask[y1:y2, x1:x2] = True
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'mask': mask,
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
