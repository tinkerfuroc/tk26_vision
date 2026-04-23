#!/usr/bin/env python3

import argparse
import json
import os
import time
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from vision_util.weights_cache import resolve_weights


class WavingInferenceCheckNode(Node):
    def __init__(
        self,
        max_runs: int,
        interval_sec: float,
        image_topic: str,
        model_path: str,
        output_root: str,
        node_name: str,
    ):
        super().__init__(node_name)

        self.bridge = CvBridge()
        self.yolo = YOLO(str(resolve_weights(model_path)))
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

        self.latest_image = None
        self.latest_stamp = None
        self.locked = False

        self.max_runs = max_runs
        self.run_count = 0

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = os.path.join(
            os.path.expanduser(output_root),
            f'waving_check_{timestamp}',
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.timer = self.create_timer(interval_sec, self.run_one_check)

        self.get_logger().info(f'Output dir: {self.output_dir}')
        self.get_logger().info(
            f'Config: max_runs={self.max_runs}, interval_sec={interval_sec}, '
            f'image_topic={image_topic}, model={model_path}'
        )
        self.get_logger().info('Waiting for image frames...')

    def image_callback(self, msg: Image):
        if self.locked:
            return
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        self.latest_stamp = msg.header.stamp

    @staticmethod
    def _get_landmark_dict(landmarks, idx):
        lm = landmarks[idx]
        return {
            'x': float(lm.x),
            'y': float(lm.y),
            'z': float(lm.z),
            'visibility': float(lm.visibility),
        }

    def _is_waving(self, landmarks):
        right_hand = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

        left_hand = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]

        right_hand_above_shoulder = right_hand.y <= right_shoulder.y
        left_hand_above_shoulder = left_hand.y <= left_shoulder.y
        right_hand_above_elbow = right_hand.y < right_elbow.y
        left_hand_above_elbow = left_hand.y < left_elbow.y
        right_elbow_above_shoulder = right_elbow.y <= right_shoulder.y
        left_elbow_above_shoulder = left_elbow.y <= left_shoulder.y

        waving = (
            right_hand_above_shoulder
            or left_hand_above_shoulder
            or (right_hand_above_elbow and right_elbow_above_shoulder)
            or (left_hand_above_elbow and left_elbow_above_shoulder)
        )
        return bool(waving)

    def run_one_check(self):
        if self.run_count >= self.max_runs:
            self.get_logger().info(f'Completed {self.max_runs} checks, shutting down.')
            self.timer.cancel()
            rclpy.shutdown()
            return

        if self.latest_image is None:
            self.get_logger().info('No image yet, waiting...')
            return

        self.locked = True
        frame = self.latest_image.copy()
        stamp = self.latest_stamp
        self.locked = False

        self.run_count += 1
        idx = self.run_count

        raw_path = os.path.join(self.output_dir, f'{idx:02d}_raw.jpg')
        ann_path = os.path.join(self.output_dir, f'{idx:02d}_annotated.jpg')
        json_path = os.path.join(self.output_dir, f'{idx:02d}_result.json')

        cv2.imwrite(raw_path, frame)
        annotated = frame.copy()

        t0 = time.perf_counter()
        yolo_results = self.yolo(frame, verbose=False)
        yolo_ms = (time.perf_counter() - t0) * 1000.0

        boxes = yolo_results[0].boxes
        yolo_records = []
        mediapipe_records = []

        person_idx = 0
        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = self.yolo.names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]

                yolo_records.append(
                    {
                        'class_id': cls_id,
                        'class_name': cls_name,
                        'confidence': conf,
                        'bbox_xyxy': [x1, y1, x2, y2],
                    }
                )

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    annotated,
                    f'{cls_name}:{conf:.2f}',
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                if cls_name != 'person':
                    continue

                person_idx += 1
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    mediapipe_records.append(
                        {
                            'person_index': person_idx,
                            'bbox_xyxy': [x1, y1, x2, y2],
                            'status': 'empty_roi',
                        }
                    )
                    continue

                mp_t0 = time.perf_counter()
                pose_result = self.pose.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                mp_ms = (time.perf_counter() - mp_t0) * 1000.0

                person_record = {
                    'person_index': person_idx,
                    'bbox_xyxy': [x1, y1, x2, y2],
                    'mediapipe_ms': mp_ms,
                    'has_landmarks': pose_result.pose_landmarks is not None,
                    'is_waving': False,
                }

                if pose_result.pose_landmarks is not None:
                    lm = pose_result.pose_landmarks.landmark
                    person_record['is_waving'] = self._is_waving(lm)
                    person_record['keypoints'] = {
                        'nose': self._get_landmark_dict(lm, self.mp_pose.PoseLandmark.NOSE),
                        'right_wrist': self._get_landmark_dict(
                            lm, self.mp_pose.PoseLandmark.RIGHT_WRIST
                        ),
                        'right_elbow': self._get_landmark_dict(
                            lm, self.mp_pose.PoseLandmark.RIGHT_ELBOW
                        ),
                        'right_shoulder': self._get_landmark_dict(
                            lm, self.mp_pose.PoseLandmark.RIGHT_SHOULDER
                        ),
                        'left_wrist': self._get_landmark_dict(
                            lm, self.mp_pose.PoseLandmark.LEFT_WRIST
                        ),
                        'left_elbow': self._get_landmark_dict(
                            lm, self.mp_pose.PoseLandmark.LEFT_ELBOW
                        ),
                        'left_shoulder': self._get_landmark_dict(
                            lm, self.mp_pose.PoseLandmark.LEFT_SHOULDER
                        ),
                    }

                    roi_draw = annotated[y1:y2, x1:x2]
                    self.mp_draw.draw_landmarks(
                        roi_draw,
                        pose_result.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                    )

                    wave_txt = 'WAVING' if person_record['is_waving'] else 'NOT_WAVING'
                    wave_color = (0, 0, 255) if person_record['is_waving'] else (255, 255, 0)
                    cv2.putText(
                        annotated,
                        wave_txt,
                        (x1, min(annotated.shape[0] - 10, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        wave_color,
                        2,
                    )

                mediapipe_records.append(person_record)

        cv2.imwrite(ann_path, annotated)

        payload = {
            'check_index': idx,
            'stamp': {
                'sec': None if stamp is None else int(stamp.sec),
                'nanosec': None if stamp is None else int(stamp.nanosec),
            },
            'yolo_ms': yolo_ms,
            'yolo_detections': yolo_records,
            'mediapipe_results': mediapipe_records,
            'raw_image': raw_path,
            'annotated_image': ann_path,
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        self.get_logger().info(
            f'[{idx}/{self.max_runs}] saved: {os.path.basename(raw_path)}, '
            f'{os.path.basename(ann_path)}, {os.path.basename(json_path)} | '
            f'YOLO {yolo_ms:.1f} ms'
        )


def main(args=None):
    parser = argparse.ArgumentParser(description='Run N waving-person checks and save artifacts.')
    parser.add_argument('--runs', type=int, default=10, help='Number of checks to run (default: 10)')
    parser.add_argument(
        '--interval',
        type=float,
        default=1.0,
        help='Seconds between checks (default: 1.0)',
    )
    parser.add_argument(
        '--image-topic',
        type=str,
        default='/camera/color/image_raw',
        help='Image topic to subscribe (default: /camera/color/image_raw)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8s.pt',
        help='YOLO model path or name (default: yolov8s.pt)',
    )
    parser.add_argument(
        '--output-root',
        type=str,
        default='~/tk25_ws/src/tk26_vision/log_vision',
        help='Root directory to store results (default: ~/tk25_ws/src/tk26_vision/log_vision)',
    )
    parser.add_argument(
        '--node-name',
        type=str,
        default='waving_inference_check_node',
        help='ROS node name (default: waving_inference_check_node)',
    )
    parsed = parser.parse_args(args=args)

    max_runs = max(1, int(parsed.runs))
    interval_sec = max(0.05, float(parsed.interval))

    rclpy.init(args=args)
    node = WavingInferenceCheckNode(
        max_runs=max_runs,
        interval_sec=interval_sec,
        image_topic=parsed.image_topic,
        model_path=parsed.model,
        output_root=parsed.output_root,
        node_name=parsed.node_name,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
