"""Checkpoint-label YOLO26 segmentation with the generalist ROS service ABI."""

from __future__ import annotations

import re
import time

import cv2
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

from object_detection_new.object_seg_yolo import YOLOSegmentationNode

from .generalist_node import GeneralistDetectionNode


def select_labels(prompt: str, names: dict[int, str]) -> frozenset[str] | None:
    """Resolve exact labels; None means all checkpoint labels."""
    prompt = (prompt or '').strip()
    if not prompt:
        return None
    by_name = {name.casefold(): name for name in names.values()}
    # An exact full-label match takes precedence over list separators.
    if prompt.casefold() in by_name:
        return frozenset([by_name[prompt.casefold()]])
    parts = [
        part.strip() for part in re.split(r'\s+\.\s+|[,，;；|\n]+', prompt)
        if part.strip()
    ]
    if not parts:
        raise ValueError('prompt contains no labels')
    unknown = [part for part in parts if part.casefold() not in by_name]
    if unknown:
        raise ValueError(
            f'Unknown checkpoint label(s): {", ".join(unknown)}. '
            'Inspect the labels ROS parameter for supported names.'
        )
    return frozenset(by_name[part.casefold()] for part in parts)


class Yolo26SegNode(GeneralistDetectionNode):
    """Reuse generalist geometry with one segmentation checkpoint."""

    _checkpoint_only = True
    _default_model_path = 'yolo26s-seg.pt'
    _default_service_name = 'object_detection_yolo26'
    _expected_model_task = 'segment'
    _pipeline_description = 'YOLO26 segmentation'

    def __init__(self, node_name='yolo26_seg_node', parameter_overrides=None):
        defaults = {
            'model_path': Parameter(
                'model_path', value=self._default_model_path),
            'service_name': Parameter(
                'service_name', value=self._default_service_name),
            'confidence_threshold': Parameter(
                'confidence_threshold', value=0.25),
        }
        defaults.update({p.name: p for p in parameter_overrides or []})
        # Skip GeneralistDetectionNode.__init__: no SAM, VLM or World models.
        YOLOSegmentationNode.__init__(
            self, node_name=node_name,
            parameter_overrides=list(defaults.values()),
        )
        self._yolo_class_names = set(self.model.names.values())
        self.declare_parameter(
            'labels', [self.model.names[i] for i in sorted(self.model.names)],
            descriptor=ParameterDescriptor(
                read_only=True,
                description='Labels embedded in the checkpoint',
            ),
        )
        self._orbbec_response_depth_intake = None
        self._init_orbbec_response_depth_intake()
        self.get_logger().info(
            f'{self._pipeline_description} checkpoint loaded: '
            f'{self.model_path}; '
            f'labels={self.model.names}'
        )

    def _declare_parameters(self):
        YOLOSegmentationNode._declare_parameters(self)
        self.declare_parameter('realsense_max_distance_m', 1.0)
        self.declare_parameter(
            'orbbec_depth_image_topic', '/camera/depth/image_raw')

    def _load_parameters(self):
        YOLOSegmentationNode._load_parameters(self)
        self.realsense_max_distance_m = self.get_parameter(
            'realsense_max_distance_m').value

    def _init_model(self):
        # Generalist's loader puts the checkpoint on CPU without a warm-up.
        GeneralistDetectionNode._init_model(self)
        if self.model.task != self._expected_model_task:
            raise ValueError(
                f'model_path must be a {self._expected_model_task} '
                'checkpoint, '
                f'got {self.model.task}'
            )
        if not self.model.names:
            raise ValueError('The checkpoint contains no labels')
        self.model.to(self.device)
        self._yolo_loaded_device = self.device

    def _detect_objects(self, *args, **kwargs):
        with self._yolo_model_lock:
            return YOLOSegmentationNode._detect_objects(self, *args, **kwargs)

    def _yolo_pipeline(self, **ctx):
        # No stale artifacts when a request is rejected before inference.
        self._last_detection_info = []
        try:
            ctx['prompt'] = select_labels(ctx['prompt'], self.model.names)
            return GeneralistDetectionNode._yolo_pipeline(self, **ctx)
        except Exception as exc:
            self.get_logger().error(f'YOLO26 detection failed: {exc}')
            return self._empty_result('yolo', error=str(exc))

    def _log_debug(self, started, request, prompt, rgb_img, result):
        self._write_debug_artifacts(
            rgb_img, self._last_detection_info,
            request_ctx={
                'service': self.get_parameter('service_name').value,
                'model_path': self.model_path,
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
            branch='yolo',
            timings={'total': time.perf_counter() - started},
        )


def main(args=None):
    rclpy.init(args=args)
    node = None
    executor = MultiThreadedExecutor(num_threads=8)
    try:
        node = Yolo26SegNode()
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
