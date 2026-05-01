"""VLM-only tabletop placing-location service.

Subclasses `YOLOSegmentationNode` from `object_detection_new` to reuse camera
sync, intrinsics, depth-to-3D projection, the TF buffer, and the vision
logger. Replaces the YOLO/YOLO-World/race service path with a single Gemini
call that enumerates clear regions on the visible desktop, ranked best to
worst. Each region's bbox is converted to a 3D point via the parent's
`_calculate_centroid` (with a synthetic rectangular mask covering the bbox
interior — depth median is taken over valid pixels only).

This node is intentionally narrower than `object_detection_generalist`:
no YOLO branch, no FastSAM, no race coordinator. The VLM is the sole
detector, and the response shape (`PlacingLocation.srv`) returns
`PointStamped[]` instead of `Object[]` so the semantics ("place here") are
not overloaded onto the detection schema.
"""

from __future__ import annotations

import copy
import time

import cv2
import numpy as np
import rclpy
import rclpy.duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.parameter import Parameter

import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from tf2_ros import (
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
from tf2_geometry_msgs import do_transform_point

from tinker_vision_msgs_26.msg import BoundingBox
from tinker_vision_msgs_26.srv import PlacingLocation

from object_detection_new.object_seg_yolo import YOLOSegmentationNode

from .placing_vlm import VlmPlacingError, request_placing_bboxes


class PlacingLocationServer(YOLOSegmentationNode):
    """ROS2 service node: VLM-only tabletop placing-location proposals."""

    def __init__(self, node_name='placing_location_server',
                 parameter_overrides=None):
        super().__init__(
            node_name=node_name,
            parameter_overrides=parameter_overrides or [],
        )
        self.get_logger().info(
            f'Placing-location node ready: vlm_model={self.vlm_model}, '
            f'timeout={self.vlm_timeout_s}s, '
            f'default_max_candidates={self.default_max_candidates}'
        )

    # --- parameter wiring -------------------------------------------------

    def _declare_parameters(self):
        super()._declare_parameters()
        # Gemini Pro is the recommended default — placement reasoning
        # (size + spatial context) materially improves over Flash on this
        # task. Flip via -p vlm_model:=google/gemini-2.5-flash if latency
        # matters more than ranking quality.
        self.declare_parameter('vlm_model', 'google/gemini-2.5-pro')
        self.declare_parameter('vlm_timeout_s', 8.0)
        self.declare_parameter('vlm_max_retries', 1)
        self.declare_parameter('default_max_candidates', 5)

    def _load_parameters(self):
        super()._load_parameters()
        self.vlm_model = self.get_parameter('vlm_model').value
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.vlm_max_retries = int(
            self.get_parameter('vlm_max_retries').value
        )
        self.default_max_candidates = int(
            self.get_parameter('default_max_candidates').value
        )

    # --- service advertisement -------------------------------------------

    def _init_service(self):
        service_name = self.get_parameter('service_name').value
        self.detection_srv = self.create_service(
            PlacingLocation,
            service_name,
            self._placing_service_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(
            f'Placing-location service created: {service_name}'
        )

    # --- service callback ------------------------------------------------

    def _placing_service_callback(
        self,
        request: PlacingLocation.Request,
        response: PlacingLocation.Response,
    ) -> PlacingLocation.Response:
        _t0 = time.perf_counter()
        response.header = Header(stamp=self.get_clock().now().to_msg())
        response.status = -1
        response.error_msg = ''
        response.candidate_points = []
        response.candidate_bboxes = []

        item = (request.item_description or '').strip()
        if not item:
            response.error_msg = 'item_description is empty'
            return response

        max_candidates = (
            self.default_max_candidates if request.max_candidates == 0
            else max(1, min(int(request.max_candidates), 10))
        )

        camera = self._select_camera(request.camera)
        rec_msg = self._wait_for_recent_frame(camera)
        if rec_msg is None:
            response.error_msg = (
                f'No {camera} camera data within sync threshold'
            )
            return response

        intrinsic = self._get_intrinsic(camera)
        if intrinsic is None:
            response.error_msg = f'No {camera} camera intrinsics available'
            return response

        try:
            if camera == 'realsense':
                rgb_img, points, valid_mask, header = (
                    self._process_realsense_data(rec_msg[0], rec_msg[1], intrinsic)
                )
            else:
                rgb_img, points, valid_mask, header = (
                    self._process_orbbec_data(rec_msg[0], rec_msg[1], intrinsic)
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'Error processing {camera} data: {exc}'
            )
            response.error_msg = f'camera data processing error: {exc}'
            return response

        response.header = header

        try:
            bboxes, ranks, vlm_elapsed = request_placing_bboxes(
                rgb_img,
                item_description=item,
                max_candidates=max_candidates,
                model=self.vlm_model,
                max_retries=self.vlm_max_retries,
                timeout_s=self.vlm_timeout_s,
                logger=self.get_logger(),
            )
        except VlmPlacingError as exc:
            response.error_msg = f'VLM unavailable: {exc}'
            self._maybe_log(_t0, request, item, rgb_img, [], [], 0.0, error=str(exc))
            return response
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'VLM placing call failed: {exc}')
            response.error_msg = f'VLM error: {exc}'
            self._maybe_log(_t0, request, item, rgb_img, [], [], 0.0, error=str(exc))
            return response

        # --- bbox -> 3D centroid -----------------------------------------
        rgb_h, rgb_w = rgb_img.shape[:2]
        target_frame = (request.target_frame or '').strip()
        source_frame = header.frame_id or ''

        kept_bboxes: list = []
        kept_points: list[PointStamped] = []
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            rect_mask = np.zeros((rgb_h, rgb_w), dtype=bool)
            rect_mask[y1:y2, x1:x2] = True
            centroid = self._calculate_centroid(
                points, rect_mask, valid_mask, bbox, camera,
            )
            if centroid is None:
                self.get_logger().warn(
                    f'placing candidate #{i + 1} bbox={bbox} '
                    'has no valid depth; skipping'
                )
                continue

            stamped = PointStamped()
            stamped.header = Header(
                stamp=header.stamp,
                frame_id=source_frame,
            )
            stamped.point = centroid

            if target_frame and target_frame != source_frame:
                transformed = self._transform_point(
                    stamped, target_frame, source_frame, header.stamp,
                )
                if transformed is None:
                    self.get_logger().warn(
                        f'placing candidate #{i + 1}: TF '
                        f'{source_frame} -> {target_frame} unavailable; '
                        'skipping'
                    )
                    continue
                stamped = transformed

            kept_points.append(stamped)
            bb = BoundingBox()
            bb.xmin = int(x1)
            bb.ymin = int(y1)
            bb.xmax = int(x2)
            bb.ymax = int(y2)
            kept_bboxes.append(bb)

        response.candidate_points = kept_points
        response.candidate_bboxes = kept_bboxes
        response.status = 0 if kept_points else 1
        if not kept_points:
            response.error_msg = (
                'VLM returned no usable placing regions'
                if not bboxes else
                'all VLM regions were rejected (no valid depth or TF failed)'
            )

        if request.return_rgb_image:
            response.rgb_image = rec_msg[0]
        if request.return_debug_overlay:
            response.debug_overlay = self.bridge.cv2_to_imgmsg(
                self._render_overlay(rgb_img, bboxes, ranks),
                encoding='bgr8',
            )

        self._maybe_log(
            _t0, request, item, rgb_img, bboxes, ranks, vlm_elapsed,
            error=response.error_msg or None,
        )
        return response

    # --- helpers ---------------------------------------------------------

    def _select_camera(self, camera_req: str) -> str:
        if 'realsense' in (camera_req or ''):
            return 'realsense'
        if 'orbbec' in (camera_req or ''):
            return 'orbbec'
        self.get_logger().warn(
            f'Unknown camera "{camera_req}", defaulting to orbbec'
        )
        return 'orbbec'

    def _wait_for_recent_frame(self, camera: str):
        call_time = self.get_clock().now()
        for _ in range(self.sync_wait_time_limit):
            with self.lock_msg:
                recent_time = self.recent_publish_time[camera]
            if recent_time is None or (
                (call_time - recent_time).nanoseconds / 1e9
                > self.img_sync_thres
            ):
                time.sleep(0.1)
                continue
            with self.lock_msg:
                return copy.deepcopy(self.recent_sync_msg.get(camera))
        return None

    def _get_intrinsic(self, camera: str):
        with self.lock_info:
            return copy.deepcopy(self.camera_intrinsic.get(camera))

    def _transform_point(
        self,
        stamped: PointStamped,
        target_frame: str,
        source_frame: str,
        stamp,
    ) -> PointStamped | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException,
                ExtrapolationException) as exc:
            self.get_logger().warn(
                f'TF {source_frame} -> {target_frame} failed: {exc}'
            )
            return None
        try:
            return do_transform_point(stamped, transform)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f'do_transform_point failed: {exc}'
            )
            return None

    @staticmethod
    def _render_overlay(rgb_img: np.ndarray, bboxes, ranks) -> np.ndarray:
        vis = rgb_img.copy()
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            color = (0, 200, 0) if i == 0 else (40, 180, 220)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cv2.circle(vis, (cx, cy), 5, (0, 0, 255), -1)
            label = ranks[i] if i < len(ranks) and ranks[i] else f'#{i + 1}'
            cv2.putText(
                vis, label, (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
        return vis

    def _maybe_log(self, _t0, request, item, rgb_img,
                   bboxes, ranks, vlm_elapsed, error=None):
        if not getattr(self, '_vision_logger', None) or \
                not self._vision_logger.enabled:
            return
        detections = []
        rgb_h, rgb_w = rgb_img.shape[:2]
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            rect_mask = np.zeros((rgb_h, rgb_w), dtype=bool)
            rect_mask[y1:y2, x1:x2] = True
            detections.append({
                'bbox': bbox,
                'mask': rect_mask,
                'cls_name': ranks[i] if i < len(ranks) and ranks[i] else f'rank{i + 1}',
                'conf': 1.0,
            })
        request_ctx = {
            'service': 'placing_location',
            'item_description': item,
            'camera': request.camera,
            'target_frame': request.target_frame,
            'max_candidates': int(request.max_candidates),
            'n_regions': len(bboxes),
            'error': error,
        }
        timings = {'total': time.perf_counter() - _t0}
        if vlm_elapsed:
            timings['vlm'] = vlm_elapsed
        self._write_debug_artifacts(
            rgb_img, detections,
            request_ctx=request_ctx,
            branch='placing',
            timings=timings,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PlacingLocationServer(
        node_name='placing_location_server',
        parameter_overrides=[
            Parameter('service_name',
                      Parameter.Type.STRING,
                      'placing_location'),
        ],
    )
    # MultiThreadedExecutor: the service callback blocks ~5-8 s on the VLM
    # HTTP call, and during that window camera / TF / camera_info callbacks
    # must keep running so the next request finds fresh frames + intrinsics.
    import multiprocessing
    num_threads = max(8, multiprocessing.cpu_count())
    executor = rclpy.executors.MultiThreadedExecutor(
        num_threads=num_threads,
    )
    executor.add_node(node)
    node.get_logger().info(
        f'Spinning placing-location node with MultiThreadedExecutor '
        f'(num_threads={num_threads})'
    )
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
