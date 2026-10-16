"""Checkpoint-label YOLO26 segmentation with the generalist ROS service ABI."""

from __future__ import annotations

import re
import time

import cv2
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import MultiThreadedExecutor

from object_detection_new.object_seg_yolo import YOLOSegmentationNode

from .generalist_node import GeneralistDetectionNode


def flatten_labels(names: dict[int, str]) -> dict[int, str]:
    """Lowercase checkpoint labels; reject labels that differ only by case."""
    flat = {i: name.lower() for i, name in names.items()}
    if len(set(flat.values())) != len(flat):
        raise ValueError(
            f'Checkpoint labels collide when lowercased: {names}')
    return flat


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
        # Skip GeneralistDetectionNode.__init__: no SAM, VLM or World models.
        YOLOSegmentationNode.__init__(
            self, node_name=node_name,
            parameter_overrides=parameter_overrides or [],
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

    def declare_parameter(self, name, value=None, *args, **kwargs):
        # Swap in this node's defaults at declare time. Passing them as
        # parameter_overrides would beat --ros-args -p values.
        defaults = {
            'model_path': self._default_model_path,
            'service_name': self._default_service_name,
            'confidence_threshold': 0.25,
        }
        value = defaults.get(name, value)
        return super().declare_parameter(name, value, *args, **kwargs)

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
        # Results, the labels param and grasp labels all read model.names;
        # lowercase once here so callers never see checkpoint casing.
        self.model.model.names = flatten_labels(self.model.names)
        self.model.to(self.device)
        self._yolo_loaded_device = self.device

    def _wait_for_recent_frame(self, camera, **kwargs):
        frames = YOLOSegmentationNode._wait_for_recent_frame(
            self, camera, **kwargs)
        logged = self.__dict__.setdefault('_first_frame_logged', set())
        if frames is not None and camera not in logged:
            logged.add(camera)
            cfg = self._camera_intakes[camera].cfg
            source = (cfg.provider_endpoint if cfg.backend == 'service'
                      else cfg.color.topic)
            color, depth = frames
            stamp = color.header.stamp.sec + color.header.stamp.nanosec * 1e-9
            age = self.get_clock().now().nanoseconds * 1e-9 - stamp
            self.get_logger().info(
                f'first {camera} frame from {source} ({cfg.backend}): '
                f'color {color.width}x{color.height} {color.encoding}, '
                f'depth {depth.width}x{depth.height} {depth.encoding}, '
                f'frame={color.header.frame_id!r}, age={age * 1000:.0f} ms'
            )
        return frames

    def _generalist_service_callback(self, request, response):
        # Console trace of every request and its outcome, on all exit paths.
        started = time.perf_counter()
        self.get_logger().info(
            f'request: camera={request.camera!r} prompt={request.prompt!r} '
            f'target_frame={request.target_frame!r} '
            f'sort_closest={request.sort_closest} '
            f'sort_highest={request.sort_highest} '
            f'return_segments={request.return_segments}'
        )
        try:
            response = GeneralistDetectionNode._generalist_service_callback(
                self, request, response)
        except Exception as exc:
            self.get_logger().error(f'request failed with exception: {exc}')
            raise
        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.objects:
            found = ', '.join(
                f'{o.cls} conf={o.conf:.2f} at ({o.centroid.x:.3f}, '
                f'{o.centroid.y:.3f}, {o.centroid.z:.3f})'
                for o in response.objects
            )
            self.get_logger().info(
                f'detected {len(response.objects)} object(s) in '
                f'{elapsed_ms:.0f} ms [frame={response.header.frame_id!r}]: '
                f'{found}'
            )
        else:
            self.get_logger().warn(
                f'no detections in {elapsed_ms:.0f} ms '
                f'(status={response.status}): {response.error_msg}'
            )
        return response

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
