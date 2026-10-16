"""YOLO26 bounding-box detection followed by MobileSAM segmentation."""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor

from vision_util.weights_cache import resolve_weights

from .sam_mask import SamPredictor
from .yolo26_seg_node import Yolo26SegNode, select_labels


def _box_iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def drop_duplicate_boxes(bboxes, confs, iou_threshold):
    """Indices to keep when one object gets several labels.

    Class-agnostic: of any boxes overlapping at IoU >= iou_threshold only the
    highest-confidence one survives. Kept indices stay in model order;
    iou_threshold <= 0 disables the filter.
    """
    if iou_threshold <= 0:
        return list(range(len(bboxes)))
    kept = []
    for i in sorted(range(len(bboxes)), key=lambda i: -confs[i]):
        if all(_box_iou(bboxes[i], bboxes[j]) < iou_threshold for j in kept):
            kept.append(i)
    return sorted(kept)


class Yolo26SamNode(Yolo26SegNode):
    """Fixed-label YOLO26 detector whose boxes are segmented by MobileSAM."""

    _default_model_path = 'yolo26s.pt'
    _default_service_name = 'object_detection_yolo26_sam'
    _expected_model_task = 'detect'
    _pipeline_description = 'YOLO26 bbox + MobileSAM'

    def __init__(self, node_name='yolo26_sam_node', parameter_overrides=None):
        super().__init__(
            node_name=node_name,
            parameter_overrides=parameter_overrides,
        )
        self._sam = SamPredictor(
            weights_path=str(resolve_weights(self.sam_weights)),
            device=self.device,
            logger=self.get_logger(),
        )
        self._sam_lock = threading.Lock()
        self.get_logger().info(
            f'YOLO26 bbox + MobileSAM ready: detector={self.model_path}, '
            f'sam={self.sam_weights}'
        )

    def _declare_parameters(self):
        super()._declare_parameters()
        self.declare_parameter('sam_weights', 'mobile_sam.pt')
        # Same object, several labels: keep the most confident (0 disables).
        self.declare_parameter('duplicate_iou', 0.7)

    def _load_parameters(self):
        super()._load_parameters()
        self.sam_weights = self.get_parameter('sam_weights').value
        self.duplicate_iou = float(self.get_parameter('duplicate_iou').value)

    def _detect_bboxes(self, rgb_img, selected_labels):
        """Return boxes, confidences, labels and class IDs in model order."""
        h, w = rgb_img.shape[:2]
        h_pad = ((h + 31) // 32) * 32
        w_pad = ((w + 31) // 32) * 32
        if (h, w) == (h_pad, w_pad):
            padded = rgb_img
        else:
            padded = cv2.copyMakeBorder(
                rgb_img, 0, h_pad - h, 0, w_pad - w,
                cv2.BORDER_CONSTANT, value=0,
            )

        class_ids = None if selected_labels is None else [
            cls_id for cls_id, name in self.model.names.items()
            if name in selected_labels
        ]
        started = time.perf_counter()
        with self._yolo_model_lock:
            results = self.model(
                padded,
                imgsz=(h_pad, w_pad),
                verbose=False,
                classes=class_ids,
                conf=self.conf_threshold,
            )
        elapsed = time.perf_counter() - started

        bboxes = []
        confs = []
        labels = []
        detected_class_ids = []
        for result in results or []:
            boxes = getattr(result, 'boxes', None)
            if boxes is None or boxes.xyxy is None:
                continue
            for i in range(len(boxes)):
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                cls_name = self.model.names[cls_id]
                if conf < self.conf_threshold:
                    continue
                if cls_name in self.excluded_classes:
                    continue
                if (
                    selected_labels is not None
                    and cls_name not in selected_labels
                ):
                    continue
                x1, y1, x2, y2 = boxes.xyxy[i].detach().cpu().tolist()
                bbox = (
                    max(0, min(int(round(x1)), w - 1)),
                    max(0, min(int(round(y1)), h - 1)),
                    max(0, min(int(round(x2)), w - 1)),
                    max(0, min(int(round(y2)), h - 1)),
                )
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    continue
                bboxes.append(bbox)
                confs.append(conf)
                labels.append(cls_name)
                detected_class_ids.append(cls_id)
        keep = drop_duplicate_boxes(bboxes, confs, self.duplicate_iou)
        if len(keep) < len(bboxes):
            dropped = [
                f'{labels[i]} conf={confs[i]:.2f}'
                for i in range(len(bboxes)) if i not in keep
            ]
            self.get_logger().info(
                f'dropped duplicate label(s) on the same object: '
                f'{", ".join(dropped)}'
            )
            bboxes, confs, labels, detected_class_ids = (
                [seq[i] for i in keep]
                for seq in (bboxes, confs, labels, detected_class_ids)
            )
        self.get_logger().info(
            f'YOLO26 bbox inference: {elapsed * 1000:.2f} ms, '
            f'{len(bboxes)} detection(s)'
        )
        return bboxes, confs, labels, detected_class_ids, elapsed

    def _yolo_pipeline(self, *, rgb_img, points, valid_mask, prompt,
                       camera, header, sort_mode, return_segments,
                       target_frame='') -> dict:
        """Run YOLO26 bbox inference, then box-prompted MobileSAM."""
        self._last_detection_info = []
        self._last_detection_info_all = []
        self._last_rgb_img = rgb_img.copy()
        try:
            selected = select_labels(prompt, self.model.names)
            bboxes, confs, labels, class_ids, detector_elapsed = (
                self._detect_bboxes(rgb_img, selected)
            )
            if not bboxes:
                result = self._empty_result('yolo')
                result['detector_elapsed'] = detector_elapsed
                return result

            with self._sam_lock:
                masks, sam_elapsed = self._sam.segment(rgb_img, bboxes)

            objects, segments, closest_distances = (
                self._build_fallback_objects(
                    prompt, bboxes, masks, points, valid_mask, camera,
                    return_segments=return_segments,
                    confs=confs,
                    labels=labels,
                    class_ids=class_ids,
                    header=header,
                    target_frame=target_frame,
                )
            )
            objects, segments, closest_distances = (
                self._apply_realsense_range_gate(
                    objects, segments, camera, closest_distances,
                )
            )
            if objects:
                objects, segments = self._sort_objects_and_segments(
                    objects, segments, sort_mode,
                    camera=camera,
                    source_frame=header.frame_id,
                    header=header,
                    closest_distances=closest_distances,
                )

            detections = [
                {
                    'bbox': bbox,
                    'mask': mask,
                    'cls_name': label,
                    'conf': conf,
                }
                for bbox, mask, label, conf
                in zip(bboxes, masks, labels, confs)
                if mask is not None and np.any(mask)
            ]
            self._last_detection_info = detections
            self._last_detection_info_all = list(detections)
            if self.visualization and detections:
                self._visualize_all_detections(
                    rgb_img, detections, displaying_all=True,
                )

            return {
                'source': 'yolo',
                'objects': objects,
                'segments': segments,
                'bboxes': bboxes,
                'masks': masks,
                'confs': confs,
                'world_elapsed': 0.0,
                'vlm_elapsed': 0.0,
                'sam_elapsed': sam_elapsed,
                'detector_elapsed': detector_elapsed,
                'error': None,
            }
        except Exception as exc:  # keep service alive after model errors
            self.get_logger().error(f'YOLO26 bbox + MobileSAM failed: {exc}')
            return self._empty_result('yolo', error=str(exc))

    def _log_debug(self, started, request, prompt, rgb_img, result):
        self._write_debug_artifacts(
            rgb_img,
            self._last_detection_info,
            request_ctx={
                'service': self.get_parameter('service_name').value,
                'model_path': self.model_path,
                'sam_weights': self.sam_weights,
                'prompt': prompt,
                'camera': request.camera,
                'target_frame': request.target_frame,
                'sort_closest': request.sort_closest,
                'sort_highest': request.sort_highest,
                'detection_source': 'yolo' if result['objects'] else 'none',
                'n_objects': len(result['objects']),
                'error': result.get('error'),
                'depth_source': getattr(self, '_last_depth_source', 'native'),
            },
            branch='yolo_sam',
            timings={
                'total': time.perf_counter() - started,
                'yolo': result.get('detector_elapsed', 0.0),
                'sam': result.get('sam_elapsed', 0.0),
            },
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=8)
    try:
        node = Yolo26SamNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if node is not None:
            if node.visualization:
                cv2.destroyAllWindows()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
