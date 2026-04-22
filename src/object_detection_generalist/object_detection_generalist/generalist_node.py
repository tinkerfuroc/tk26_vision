"""Generalist object detection service.

Subclasses `YOLOSegmentationNode` from `object_detection_new` and overrides
the service registration + callback so the node:

  1. advertises `/object_detection_generalist` under the redesigned
     `tinker_vision_msgs_26/srv/ObjectDetection` with typed boolean flags;
  2. runs the pretrained parent YOLO model for classes it already knows;
  3. falls back to Gemini 2.5 Pro (bounding box) + FastSAM (mask) when the
     requested `prompt` is not in the YOLO class list, the caller opts in
     via `use_vlm_sam_fallback`, or `force_vlm_sam` is set.

Camera synchronization, intrinsics handling, image → 3D point projection,
3D centroid computation, and sort-mode logic are all reused unchanged from
the parent class. No duplication.
"""

from __future__ import annotations

import copy
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.parameter import Parameter

from std_msgs.msg import Header

from tinker_vision_msgs.msg import Object
from tinker_vision_msgs_26.srv import ObjectDetection as GeneralistObjectDetection

from object_detection_new.object_seg_yolo import YOLOSegmentationNode

from .sam_mask import FastSAMPredictor
from .vlm_bbox import VlmBboxError, request_bboxes


class GeneralistDetectionNode(YOLOSegmentationNode):
    """YOLO + VLM/SAM fallback object detection server."""

    def __init__(self, node_name='generalist_detection_node',
                 parameter_overrides=None):
        super().__init__(
            node_name=node_name,
            parameter_overrides=parameter_overrides or [],
        )
        # Parent __init__ finished — device, YOLO model, camera subscribers,
        # TF, locks, and the generalist service are all up. Construct the
        # FastSAM predictor once so weights & CUDA context amortize.
        self._sam = FastSAMPredictor(
            weights_path=self.fastsam_weights,
            device=self.device,
            logger=self.get_logger(),
        )
        self.get_logger().info(
            f'Generalist detection node ready: vlm_model={self.vlm_model}, '
            f'fastsam_weights={self.fastsam_weights}, '
            f'allow_auto_fallback={self.allow_auto_fallback}'
        )

    # --- parameter wiring -------------------------------------------------

    def _declare_parameters(self):
        super()._declare_parameters()
        self.declare_parameter('allow_auto_fallback', True)
        self.declare_parameter('vlm_model', 'google/gemini-2.5-pro')
        self.declare_parameter('vlm_timeout_s', 20.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('fastsam_weights', 'FastSAM-s.pt')

    def _load_parameters(self):
        super()._load_parameters()
        self.allow_auto_fallback = (
            self.get_parameter('allow_auto_fallback').value
        )
        self.vlm_model = self.get_parameter('vlm_model').value
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.vlm_max_retries = int(self.get_parameter('vlm_max_retries').value)
        self.fastsam_weights = self.get_parameter('fastsam_weights').value

    # --- service advertisement -------------------------------------------

    def _init_service(self):
        """Advertise the redesigned ObjectDetection service."""
        service_name = self.get_parameter('service_name').value
        self.detection_srv = self.create_service(
            GeneralistObjectDetection,
            service_name,
            self._generalist_service_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(
            f'Generalist detection service created: {service_name}'
        )

    # --- service callback ------------------------------------------------

    def _generalist_service_callback(
        self,
        request: GeneralistObjectDetection.Request,
        response: GeneralistObjectDetection.Response,
    ) -> GeneralistObjectDetection.Response:
        response.header = Header(stamp=self.get_clock().now().to_msg())
        response.status = 1
        response.person_id = 0
        response.objects = []
        response.detection_source = 'none'

        camera = self._select_camera(request.camera)

        rec_msg = self._wait_for_recent_frame(camera)
        if rec_msg is None:
            response.error_msg = f'No {camera} camera data within sync threshold'
            return response

        intrinsic = self._get_intrinsic(camera)
        if intrinsic is None:
            response.error_msg = f'No {camera} camera intrinsics available'
            return response

        try:
            if camera == 'realsense':
                rgb_img, points, valid_mask, header = self._process_realsense_data(
                    rec_msg[0], rec_msg[1], intrinsic
                )
            else:
                rgb_img, points, valid_mask, header = self._process_orbbec_data(
                    rec_msg[0], rec_msg[1], intrinsic
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Error processing {camera} data: {exc}')
            response.error_msg = f'camera data processing error: {exc}'
            return response

        response.header = header

        sort_mode = self._select_sort_mode(request)
        prompt = (request.prompt or '').strip()
        if not prompt:
            response.error_msg = 'prompt is empty'
            return response

        yolo_known = prompt in set(self.model.names.values())

        # --- branching ----------------------------------------------------
        if request.force_vlm_sam:
            branch = 'vlm_sam'
        elif yolo_known:
            branch = 'yolo'
        elif request.use_vlm_sam_fallback or self.allow_auto_fallback:
            branch = 'vlm_sam'
        else:
            response.error_msg = (
                f'class "{prompt}" not in YOLO names and fallback disabled'
            )
            return response

        objects: list[Object] = []
        segments: list = []
        used_source = 'none'

        if branch == 'yolo':
            yolo_objects_msg, yolo_segments = self._detect_objects(
                rgb_img, points, prompt, valid_mask, header, camera,
                request_segments=request.return_segments,
                sort_mode=sort_mode,
            )
            objects = list(yolo_objects_msg.objects)
            segments = list(yolo_segments)
            if objects:
                used_source = 'yolo'
            elif request.use_vlm_sam_fallback or self.allow_auto_fallback:
                self.get_logger().info(
                    f'YOLO returned no matches for "{prompt}"; falling through '
                    'to VLM+SAM'
                )
                branch = 'vlm_sam'
            else:
                used_source = 'yolo'  # empty-but-authoritative response

        if branch == 'vlm_sam':
            try:
                bboxes = request_bboxes(
                    rgb_img,
                    prompt,
                    model=self.vlm_model,
                    max_retries=self.vlm_max_retries,
                    timeout_s=self.vlm_timeout_s,
                    logger=self.get_logger(),
                )
            except VlmBboxError as exc:
                response.error_msg = f'VLM unavailable: {exc}'
                return response

            if not bboxes:
                response.detection_source = 'none'
                response.error_msg = (
                    f'VLM+SAM produced no detections for "{prompt}"'
                )
                return response

            masks = self._sam.segment(rgb_img, bboxes)
            objects, segments = self._build_vlm_sam_objects(
                prompt, bboxes, masks, points, valid_mask, camera,
                return_segments=request.return_segments,
            )
            used_source = 'vlm_sam' if objects else 'none'

            if objects:
                objects, segments = self._sort_objects_and_segments(
                    objects, segments, sort_mode,
                    camera=camera,
                    source_frame=header.frame_id,
                    header=header,
                )

        # --- response assembly --------------------------------------------
        response.objects = objects
        response.detection_source = used_source
        response.status = 0 if objects else 1
        if not objects:
            response.error_msg = response.error_msg or (
                f'no matches for "{prompt}" via {used_source or branch}'
            )

        if request.return_rgb_image:
            response.rgb_image = rec_msg[0]
        if request.return_depth_image and camera == 'realsense':
            # Only realsense carries a depth Image; orbbec uses PointCloud2
            # which does not fit the sensor_msgs/Image field.
            response.depth_image = rec_msg[1]
        if request.return_segments and segments:
            response.segments = [
                self.bridge.cv2_to_imgmsg(seg, encoding='mono8')
                for seg in segments
            ]

        if self.debug_log_overlays:
            request_ctx = {
                'service': 'generalist_ObjectDetection',
                'prompt': prompt,
                'camera': request.camera,
                'target_frame': request.target_frame,
                'sort_closest': bool(request.sort_closest),
                'sort_highest': bool(request.sort_highest),
                'force_vlm_sam': bool(request.force_vlm_sam),
                'use_vlm_sam_fallback': bool(request.use_vlm_sam_fallback),
                'detection_source': used_source,
            }
            if used_source == 'yolo':
                self._write_debug_artifacts(
                    self._last_rgb_img,
                    self._last_detection_info,
                    request_ctx=request_ctx,
                    branch='yolo',
                )
            elif used_source == 'vlm_sam':
                detections = [
                    {
                        'bbox': bbox,
                        'mask': mask,
                        'cls_name': prompt,
                        'conf': 1.0,
                    }
                    for bbox, mask in zip(bboxes, masks)
                    if mask is not None
                ]
                self._write_debug_artifacts(
                    rgb_img,
                    detections,
                    request_ctx=request_ctx,
                    branch='vlm_sam',
                    vlm_raw=[list(bbox) for bbox in bboxes],
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
                rec_msg_ref = self.recent_sync_msg.get(camera)
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

    @staticmethod
    def _select_sort_mode(request) -> str:
        if request.sort_closest:
            return 'closest'
        if request.sort_highest:
            return 'highest'
        return 'none'

    def _build_vlm_sam_objects(
        self,
        prompt: str,
        bboxes,
        masks,
        points,
        valid_mask,
        camera: str,
        return_segments: bool,
    ):
        """Convert (bbox, mask) pairs into Object[] via parent centroid logic.

        All VLM-path objects report conf=1.0 uniformly — Gemini's JSON schema
        does not emit a comparable per-bbox probability. Callers that want to
        filter on confidence should branch on the response's detection_source
        field (`'yolo'` carries real scores; `'vlm_sam'` does not).
        """
        import numpy as np

        objects: list[Object] = []
        segments: list = []
        for bbox, mask in zip(bboxes, masks):
            if mask is None or mask.sum() == 0:
                continue
            centroid = self._calculate_centroid(
                points, mask, valid_mask, bbox, camera
            )
            if centroid is None:
                continue
            obj = Object()
            obj.conf = 1.0  # see class docstring — uniform for VLM branch
            obj.cls = prompt
            obj.centroid = centroid
            obj.id = 0
            obj.object_id = -1
            obj.similarity = 0.0
            obj.being_pointed = 0
            objects.append(obj)
            if return_segments:
                segments.append(mask.astype(np.uint8) * 255)
        return objects, segments


def main(args=None):
    rclpy.init(args=args)
    node = GeneralistDetectionNode(
        node_name='generalist_detection_node',
        parameter_overrides=[
            Parameter('service_name',
                      Parameter.Type.STRING,
                      'object_detection_generalist'),
            # Clean COCO baseline; callers that want the custom-trained
            # competition model should use /object_detection_yolo instead.
            Parameter('model_path',
                      Parameter.Type.STRING,
                      'yolo11m-seg.pt'),
        ],
    )
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
