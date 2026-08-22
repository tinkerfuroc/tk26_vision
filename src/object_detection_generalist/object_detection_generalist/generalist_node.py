"""Generalist object detection service.

Subclasses `YOLOSegmentationNode` from `object_detection_new` and overrides
the service registration + callback. Open-vocab path uses YOLO-World and/or
Gemini 2.5 Flash, both feeding into MobileSAM for masks; 3D centroids and
sorting are inherited unchanged from the parent class.

Branching (after camera/prompt validation):

  * ``request.force_vlm_sam=True``  → VLM + SAM only (operator override;
    bypasses YOLO-World regardless of node config).
  * ``prompt`` is a YOLO class       → run pretrained YOLO. If non-empty, done;
                                       otherwise fall through to the open-vocab path.
  * ``request.use_vlm_sam_fallback=True``
                                     → race YOLO-World and VLM concurrently.
                                       If YOLO-World returns objects first,
                                       cancel the VLM call: an abandon-event
                                       guarantees SAM + downstream work
                                       are skipped (no zombie GPU/SAM-lock
                                       tail), and we also close the OpenAI
                                       client cross-thread as a best-effort
                                       attempt to interrupt the in-flight
                                       HTTP. Worst-case the VLM thread still
                                       blocks in recv() until vlm_timeout_s,
                                       then exits cleanly. Otherwise wait
                                       for VLM.
  * Auto-fallback (``allow_auto_fallback=True``, no per-request flags)
                                     → single fallback chosen by ``enable_vlm``:
                                       YOLO-World if False (default), VLM if True.

Camera synchronization, intrinsics handling, image → 3D point projection,
3D centroid computation, and sort-mode logic are all reused unchanged from
the parent class. No duplication.
"""

from __future__ import annotations

import copy
import re
import threading
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.parameter import Parameter

from sensor_msgs.msg import Image
from std_msgs.msg import Header

from tinker_vision_msgs_26.msg import Object
from tinker_vision_msgs_26.srv import ObjectDetectionGeneralist

from object_detection_new.object_seg_yolo import YOLOSegmentationNode
from vision_util.camera_intake import (
    CameraIntake,
    IntakeConfig,
    StreamSpec,
    configure_camera_backend,
)
from vision_util.vlm_models import vision_flash_model, vision_qwen_model
from vision_util.weights_cache import resolve_weights

from .sam_mask import SamPredictor
from .vlm_bbox import VlmBboxError, load_env, request_bboxes
from .world_bbox import WorldDetector, WorldDetectorError


class GeneralistDetectionNode(YOLOSegmentationNode):
    """YOLO + VLM/SAM fallback object detection server."""

    def _init_model(self):
        """Initialize the inherited YOLO model without occupying CUDA yet."""
        import torch
        from ultralytics import YOLO

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._yolo_model_lock = threading.Lock()
        self._yolo_loaded_device = 'cpu'
        self.get_logger().info(
            f'Using device: {self.device}; loading YOLO on CPU until needed'
        )
        try:
            model_file = resolve_weights(self.model_path)
            self.model = YOLO(str(model_file))
            self.model.to('cpu')
            self.get_logger().info(
                f'YOLO model loaded from {model_file} on CPU; '
                'no warm-up performed'
            )
        except Exception as e:
            self.get_logger().error(f'Failed to load YOLO model: {e}')
            raise

    def __init__(self, node_name='generalist_detection_node',
                 parameter_overrides=None):
        # Load .env (VISION_* model-id overrides) before the base class's
        # __init__ calls self._declare_parameters() — vlm_model,
        # vlm_fallback_models, and dashscope_qwen_model default from
        # vision_util.vlm_models(), which reads os.environ at declare time.
        load_env()
        super().__init__(
            node_name=node_name,
            parameter_overrides=parameter_overrides or [],
        )
        # Parent __init__ finished — device, YOLO model, camera subscribers,
        # TF, locks, and the generalist service are all up. Construct the
        # SAM predictor once so weights & CUDA context amortize.
        self._sam = SamPredictor(
            weights_path=str(resolve_weights(self.sam_weights)),
            device=self.device,
            logger=self.get_logger(),
        )
        # SAM is shared between the YOLO-World and VLM pipelines, which can
        # run concurrently in the race path. Ultralytics models are not
        # thread-safe; serialize segmentation calls.
        self._sam_lock = threading.Lock()

        # Keep the response depth stream independent of the inherited RGB-D
        # ATS. It intentionally returns the latest depth message regardless
        # of which pair was used for detection.
        self._orbbec_response_depth_intake: CameraIntake | None = None
        self._init_orbbec_response_depth_intake()

        # Cache the pretrained YOLO class-name set for O(1) prompt lookup
        # in the service callback. Names map is fixed once the model is
        # loaded; recomputing the set per call is a wasted allocation.
        self._yolo_class_names = set(self.model.names.values())

        # YOLO-World is the default open-vocab fallback, but it is loaded
        # lazily so force_vlm_sam requests never pull it onto the GPU. If load
        # fails (missing weights / ultralytics), keep the node usable: log
        # loudly and fall through to the VLM path on out-of-vocab prompts
        # (which will surface its own error if the API key is missing too).
        self._world: WorldDetector | None = None
        self._world_load_error: str | None = None
        self._world_lock = threading.Lock()
        self.get_logger().info(
            'YOLO-World will be loaded lazily on first YOLO-World fallback; '
            'no warm-up performed'
        )

        self.get_logger().info(
            f'Generalist detection node ready: enable_vlm={self.enable_vlm}, '
            f'world_weights={self.world_weights}, vlm_model={self.vlm_model}, '
            f'vlm_fallback_models={self.vlm_fallback_models}, '
            f'sam_weights={self.sam_weights}, '
            f'allow_auto_fallback={self.allow_auto_fallback}, '
            f'vlm_timeout_s={self.vlm_timeout_s} '
            f'(per_attempt={self.vlm_per_attempt_timeout_s}s)'
        )

    # --- parameter wiring -------------------------------------------------

    def _declare_parameters(self):
        super()._declare_parameters()
        self.declare_parameter('allow_auto_fallback', True)
        # Selects the fallback model used on the *auto* path (no per-request
        # flags). When True, auto-fallback uses Gemini (VLM) + SAM; when
        # False (default), it uses local YOLO-World + SAM. Per-request
        # flags override this:
        #   - force_vlm_sam=True  → VLM only, regardless of enable_vlm.
        #   - use_vlm_sam_fallback=True → race YOLO-World vs VLM concurrently.
        self.declare_parameter('enable_vlm', False)
        self.declare_parameter('vlm_model', vision_flash_model())
        # Fallback chain. A 'dashscope/' prefix routes the model to Alibaba
        # DashScope's OpenAI-compatible endpoint (separate DASHSCOPE_API_KEY +
        # base URL) instead of OpenRouter — see vlm_bbox._split_provider. qwen
        # is served via DashScope so it stays reachable on networks where
        # openrouter.ai is blocked.
        self.declare_parameter(
            'vlm_fallback_models', [f'dashscope/{vision_qwen_model()}']
        )
        # Convenience switch: make the DashScope qwen model the PRIMARY VLM and
        # drop OpenRouter from the chain entirely. Use this on networks where
        # openrouter.ai is unreachable — it avoids the ~20 s of gemini
        # connection failures the node would otherwise burn before falling back.
        # When True, `vlm_model` becomes `dashscope_qwen_model` and
        # `vlm_fallback_models` is emptied (overriding the two params above).
        #   ros2 run ... --ros-args -p prefer_dashscope_qwen:=true
        self.declare_parameter('prefer_dashscope_qwen', False)
        self.declare_parameter('dashscope_qwen_model', f'dashscope/{vision_qwen_model()}')
        self.declare_parameter('vlm_fallback_on_empty', False)
        self.get_logger().info(
            f'VLM model defaults: flash={vision_flash_model()} '
            f'qwen={vision_qwen_model()} (from .env VISION_*)'
        )
        # vlm_timeout_s is the OVERALL wall-clock budget across all retries
        # and fallback models for one /object_detection_generalist call.
        # vlm_per_attempt_timeout_s is the per-attempt cap forwarded to httpx
        # via client.with_options(timeout=...). On a hung stream, httpx
        # raises ReadTimeout after this many seconds and the retry loop
        # starts a fresh attempt — catches the rare 40 s outlier where one
        # attempt would otherwise eat the entire overall budget.
        self.declare_parameter('vlm_timeout_s', 30.0)
        self.declare_parameter('vlm_per_attempt_timeout_s', 10.0)
        self.declare_parameter('vlm_max_retries', 3)
        # Stream the OpenRouter VLM response (SSE).
        # Flip to False to fall back to a single blocking response.
        self.declare_parameter('vlm_stream', True)
        # Range gate for the realsense (manipulation-arm) camera. Detections
        # whose centroid is farther than this are unreachable by the arm so
        # we drop them before returning. Applied ONLY when the request's
        # camera is 'realsense'; orbbec (head camera) is unaffected. Set to
        # 0.0 (or any non-positive value) to disable.
        self.declare_parameter('realsense_max_distance_m', 1.0)
        self.declare_parameter('sam_weights', 'mobile_sam.pt')
        # YOLO-World weights. yolov8s-worldv2.pt is the smallest v2 variant
        # (~25 MB, auto-downloaded). Bigger v2 variants: m / l / x.
        self.declare_parameter('world_weights', 'yolov8s-worldv2.pt')
        # YOLO-World tends to need a low conf threshold for novel classes;
        # boost via -p world_conf_threshold:=0.10 if you see false positives.
        self.declare_parameter('world_conf_threshold', 0.05)
        self.declare_parameter('world_iou_threshold', 0.5)
        # Orbbec depth Image to surface in the response. Two viable choices:
        #   - /camera/depth/image_raw — depth registered to color,
        #     so size MATCHES rgb_image and segments. Requires the camera
        #     launched with `enable_d2c_viewer:=true`.
        #   - /camera/depth/image_raw — raw depth-sensor resolution
        #     (e.g. 640x576 on Femto Bolt). Won't match rgb (1280x720) or
        #     segments. Use only if the caller does its own alignment.
        # Default to the d2c topic so segments fit by construction.
        self.declare_parameter(
            'orbbec_depth_image_topic', '/camera/depth/image_raw'
        )

    def _load_parameters(self):
        super()._load_parameters()
        self.allow_auto_fallback = (
            self.get_parameter('allow_auto_fallback').value
        )
        self.enable_vlm = bool(self.get_parameter('enable_vlm').value)
        self.vlm_model = self.get_parameter('vlm_model').value
        self.vlm_fallback_models = self._load_string_list_parameter(
            'vlm_fallback_models'
        )
        # prefer_dashscope_qwen wins over vlm_model / vlm_fallback_models:
        # qwen-via-DashScope becomes the sole VLM model, so the node never
        # touches the (possibly blocked) OpenRouter host.
        if bool(self.get_parameter('prefer_dashscope_qwen').value):
            self.vlm_model = self.get_parameter('dashscope_qwen_model').value
            self.vlm_fallback_models = []
            self.get_logger().info(
                f'prefer_dashscope_qwen=True: VLM primary set to '
                f'"{self.vlm_model}" (DashScope), OpenRouter fallbacks dropped'
            )
        self.vlm_fallback_on_empty = bool(
            self.get_parameter('vlm_fallback_on_empty').value
        )
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.vlm_per_attempt_timeout_s = float(
            self.get_parameter('vlm_per_attempt_timeout_s').value
        )
        self.vlm_max_retries = int(self.get_parameter('vlm_max_retries').value)
        self.vlm_stream = bool(self.get_parameter('vlm_stream').value)
        self.realsense_max_distance_m = float(
            self.get_parameter('realsense_max_distance_m').value
        )
        self.sam_weights = self.get_parameter('sam_weights').value
        self.world_weights = self.get_parameter('world_weights').value
        self.world_conf_threshold = float(
            self.get_parameter('world_conf_threshold').value
        )
        self.world_iou_threshold = float(
            self.get_parameter('world_iou_threshold').value
        )

    # --- service advertisement -------------------------------------------

    def _init_service(self):
        """Advertise the redesigned ObjectDetection service."""
        service_name = self.get_parameter('service_name').value
        self.detection_srv = self.create_service(
            ObjectDetectionGeneralist,
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
        request: ObjectDetectionGeneralist.Request,
        response: ObjectDetectionGeneralist.Response,
    ) -> ObjectDetectionGeneralist.Response:
        _t0 = time.perf_counter()
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
            if self._vision_logger.enabled:
                self._log_debug(
                    _t0, request, prompt, rgb_img,
                    self._empty_result('none', error='prompt is empty'),
                )
            return response

        ctx = dict(
            rgb_img=rgb_img, points=points, valid_mask=valid_mask,
            prompt=prompt, camera=camera, header=header,
            sort_mode=sort_mode, return_segments=request.return_segments,
            target_frame=request.target_frame,
        )

        yolo_known = prompt in self._yolo_class_names

        # --- branching ----------------------------------------------------
        if request.force_vlm_sam:
            # Operator override: VLM + SAM only (semantics unchanged from the
            # original generalist behavior; ignores enable_vlm).
            # INVARIANT: force_vlm_sam ⇒ YOLO-World must NOT run. This branch
            # calls _vlm_pipeline directly — never _race_world_vlm or
            # _world_pipeline. Keep this check first in the dispatch chain so
            # a future refactor cannot silently route force_vlm_sam through a
            # world-touching path.
            self.get_logger().info(
                f'force_vlm_sam set; running VLM+SAM only for "{prompt}"'
            )
            result = self._vlm_pipeline(**ctx)
        elif yolo_known:
            result = self._yolo_pipeline(**ctx)
            if not result['objects']:
                if request.use_vlm_sam_fallback:
                    self.get_logger().info(
                        f'YOLO empty for "{prompt}"; racing YOLO-World vs VLM'
                    )
                    result = self._race_world_vlm(**ctx)
                elif self.allow_auto_fallback:
                    fb = 'vlm_sam' if self.enable_vlm else 'yolo_world'
                    self.get_logger().info(
                        f'YOLO empty for "{prompt}"; auto-fallback to {fb}'
                    )
                    result = (
                        self._vlm_pipeline(**ctx) if self.enable_vlm
                        else self._world_pipeline(**ctx)
                    )
                # else: keep the empty YOLO result as authoritative.
        elif request.use_vlm_sam_fallback:
            self.get_logger().info(
                f'OOV prompt "{prompt}"; racing YOLO-World vs VLM'
            )
            result = self._race_world_vlm(**ctx)
        elif self.allow_auto_fallback:
            fb = 'vlm_sam' if self.enable_vlm else 'yolo_world'
            self.get_logger().info(
                f'OOV prompt "{prompt}"; auto-fallback to {fb}'
            )
            result = (
                self._vlm_pipeline(**ctx) if self.enable_vlm
                else self._world_pipeline(**ctx)
            )
        else:
            response.error_msg = (
                f'class "{prompt}" not in YOLO names and fallback disabled'
            )
            if self._vision_logger.enabled:
                self._log_debug(
                    _t0, request, prompt, rgb_img,
                    self._empty_result('none', error=response.error_msg),
                )
            return response

        # --- response assembly --------------------------------------------
        objects = result['objects']
        segments = result['segments']
        response.objects = objects
        response.detection_source = result['source'] if objects else 'none'
        response.status = 0 if objects else 1
        # Object centroids are now in target_frame (parent + fallback both
        # transform). Mirror the frame on the response header so the
        # PointStamped pair the caller builds is internally consistent.
        if (
            request.target_frame
            and self._frame_supports_tf_transform(camera)
        ):
            response.header.frame_id = request.target_frame
        if not objects:
            response.error_msg = result.get('error') or (
                f'no matches for "{prompt}" via {result["source"]}'
            )

        if request.return_rgb_image:
            # Raw camera message at native resolution — no resize/processing.
            response.rgb_image = rec_msg[0]
        if request.return_depth_image:
            # Raw depth Image at native resolution — no resize/processing.
            # Realsense: pulled from the synced rgb+depth pair.
            # Orbbec: use the most recent message from the independent
            # depth-only response intake, without synchronizing it to RGB.
            if camera == 'realsense':
                response.depth_image = rec_msg[1]
            else:
                depth_bundle = (
                    self._orbbec_response_depth_intake.latest()
                    if self._orbbec_response_depth_intake is not None
                    else None
                )
                latest = (
                    depth_bundle.depth_msg
                    if depth_bundle is not None
                    else None
                )
                if latest is not None:
                    response.depth_image = copy.deepcopy(latest)
                else:
                    self.get_logger().warn(
                        'return_depth_image=True for orbbec but no depth '
                        'Image received yet on '
                        f'{self.get_parameter("orbbec_depth_image_topic").value}'
                    )
        if request.return_segments and segments:
            # Segments are produced at rgb_img.shape[:2] by both pipelines:
            # YOLO crops to (h, w) in `_detect_objects`; SAM resizes via
            # INTER_NEAREST in `sam_mask.segment`. Verify here so any future
            # pipeline regression that emits off-size masks fails loudly
            # rather than handing the caller mis-shaped buffers.
            rgb_h, rgb_w = rgb_img.shape[:2]
            for i, seg in enumerate(segments):
                if seg.shape[:2] != (rgb_h, rgb_w):
                    self.get_logger().error(
                        f'segment[{i}] shape {seg.shape[:2]} != '
                        f'rgb {(rgb_h, rgb_w)}; pipeline bug'
                    )
            self._warn_if_depth_mismatch(rgb_h, rgb_w, response, camera)
            response.segments = [
                self.bridge.cv2_to_imgmsg(seg, encoding='8UC1')
                for seg in segments
            ]

        if self._vision_logger.enabled:
            self._log_debug(
                _t0, request, prompt, rgb_img, result,
            )

        return response

    # --- pipeline helpers -------------------------------------------------

    def _apply_realsense_range_gate(
        self, objects, segments, camera, closest_distances=None,
    ):
        """Drop objects beyond reach on the realsense (arm) camera.

        Returns ``(filtered_objects, filtered_segments, filtered_distances)``
        preserving 1:1 positional alignment with the input pair, so the
        surrounding sort and response-assembly logic is unaffected.

        No-op when ``camera != 'realsense'`` or the gate is disabled
        (``realsense_max_distance_m <= 0``). Distance is Euclidean from the
        camera origin: realsense centroids are stored in the camera body
        frame (x=fwd, y=left, z=up) and never TF-transformed (see
        ``_CAMERAS_WITH_UNRELIABLE_FRAME_ID`` in the parent), so
        ``sqrt(x² + y² + z²)`` on ``Object.centroid`` is the right metric.
        """
        if camera != 'realsense' or self.realsense_max_distance_m <= 0.0:
            return objects, segments, closest_distances
        max_d = float(self.realsense_max_distance_m)
        kept_objs: list = []
        kept_segs: list = []
        kept_distances: list = []
        dropped = 0
        has_segments = bool(segments)
        for i, obj in enumerate(objects):
            c = obj.centroid
            dist = (c.x * c.x + c.y * c.y + c.z * c.z) ** 0.5
            if dist <= max_d:
                kept_objs.append(obj)
                if has_segments and i < len(segments):
                    kept_segs.append(segments[i])
                if closest_distances is not None and i < len(closest_distances):
                    kept_distances.append(closest_distances[i])
            else:
                dropped += 1
        if dropped:
            self.get_logger().info(
                f'realsense range gate: dropped {dropped}/{len(objects)} '
                f'object(s) beyond {max_d:.2f} m'
            )
        return kept_objs, kept_segs, (
            kept_distances if closest_distances is not None else None
        )

    @staticmethod
    def _empty_result(source: str, error: str | None = None,
                      world_elapsed: float = 0.0,
                      vlm_elapsed: float = 0.0,
                      sam_elapsed: float = 0.0) -> dict:
        return {
            'source': source,
            'objects': [], 'segments': [],
            'bboxes': [], 'masks': [], 'confs': [],
            'world_elapsed': world_elapsed,
            'vlm_elapsed': vlm_elapsed,
            'sam_elapsed': sam_elapsed,
            'error': error,
        }

    def _yolo_pipeline(self, *, rgb_img, points, valid_mask, prompt,
                       camera, header, sort_mode, return_segments,
                       target_frame='') -> dict:
        """Run pretrained YOLO via the parent's `_detect_objects`."""
        yolo_objects_msg, yolo_segments = self._detect_objects(
            rgb_img, points, prompt, valid_mask, header, camera,
            request_segments=return_segments, sort_mode=sort_mode,
            target_frame=target_frame,
        )
        objects = list(yolo_objects_msg.objects)
        segments = list(yolo_segments)
        # Apply post-build range gate. Parent already sorted, but the gate
        # preserves order so the closest-first contract holds.
        objects, segments, _ = self._apply_realsense_range_gate(
            objects, segments, camera,
        )
        result = self._empty_result('yolo')
        result['objects'] = objects
        result['segments'] = segments
        return result

    def _world_pipeline(self, *, rgb_img, points, valid_mask, prompt,
                        camera, header, sort_mode, return_segments,
                        target_frame='') -> dict:
        """Run YOLO-World + SAM. Returns a result dict (never raises)."""
        if self._world is None:
            return self._empty_result(
                'yolo_world', error='YOLO-World unavailable at node init'
            )
        try:
            bboxes, confs, world_elapsed = self._world.detect(rgb_img, prompt)
        except Exception as exc:  # noqa: BLE001 — model errors stay non-fatal
            self.get_logger().error(f'YOLO-World inference failed: {exc}')
            return self._empty_result(
                'yolo_world', error=f'YOLO-World inference: {exc}'
            )

        if not bboxes:
            return self._empty_result(
                'yolo_world', world_elapsed=world_elapsed,
            )

        with self._sam_lock:
            masks, sam_elapsed = self._sam.segment(rgb_img, bboxes)
        objects, segments, closest_distances = self._build_fallback_objects(
            prompt, bboxes, masks, points, valid_mask, camera,
            return_segments=return_segments, confs=confs,
            header=header, target_frame=target_frame,
        )
        objects, segments, closest_distances = self._apply_realsense_range_gate(
            objects, segments, camera, closest_distances,
        )
        if objects:
            objects, segments = self._sort_objects_and_segments(
                objects, segments, sort_mode,
                camera=camera, source_frame=header.frame_id, header=header,
                closest_distances=closest_distances,
            )
        return {
            'source': 'yolo_world',
            'objects': objects, 'segments': segments,
            'bboxes': bboxes, 'masks': masks, 'confs': confs,
            'world_elapsed': world_elapsed, 'vlm_elapsed': 0.0,
            'sam_elapsed': sam_elapsed,
            'error': None,
        }

    def _vlm_pipeline(self, *, rgb_img, points, valid_mask, prompt,
                      camera, header, sort_mode, return_segments,
                      target_frame='',
                      abandon_event=None, client_holder=None) -> dict:
        """Run VLM + SAM. Returns a result dict (never raises).

        If ``abandon_event`` is set at any post-HTTP checkpoint (immediately
        after `request_bboxes` returns, or before/after SAM), the
        pipeline returns an "abandoned" empty result without touching SAM
        or downstream centroid/sort work. This is what makes the abandoned
        VLM thread cheap: even if the HTTP call had to run to completion
        (no `client.close()` was issued), the thread still skips the GPU
        work and exits.
        """
        vlm_meta: dict = {}
        try:
            bboxes, raw_labels, vlm_elapsed, vlm_meta = request_bboxes(
                rgb_img, prompt,
                model=self.vlm_model,
                fallback_models=self.vlm_fallback_models,
                fallback_on_empty=self.vlm_fallback_on_empty,
                max_retries=self.vlm_max_retries,
                timeout_s=self.vlm_timeout_s,
                per_attempt_timeout_s=self.vlm_per_attempt_timeout_s,
                logger=self.get_logger(),
                abandon_event=abandon_event,
                client_holder=client_holder,
                stream=self.vlm_stream,
            )
        except VlmBboxError as exc:
            return self._empty_result(
                'vlm_sam', error=f'VLM unavailable: {exc}'
            )
        except Exception as exc:  # noqa: BLE001 — keep race partner alive
            self.get_logger().error(f'VLM call failed: {exc}')
            return self._empty_result('vlm_sam', error=f'VLM error: {exc}')

        if abandon_event is not None and abandon_event.is_set():
            # Race winner already returned. Skip SAM + downstream work.
            result = self._empty_result(
                'vlm_sam', vlm_elapsed=vlm_elapsed, error='abandoned',
            )
            result['vlm_meta'] = vlm_meta
            return result

        if not bboxes:
            result = self._empty_result(
                'vlm_sam', vlm_elapsed=vlm_elapsed,
                error=vlm_meta.get('error'),
            )
            result['vlm_meta'] = vlm_meta
            return result

        prompt_classes = self._parse_prompt_classes(prompt)
        cls_per_box = [
            lbl for lbl in raw_labels
        ]

        with self._sam_lock:
            # Re-check after acquiring the lock — caller may have abandoned
            # while we were queued behind another SAM call.
            if abandon_event is not None and abandon_event.is_set():
                return self._empty_result(
                    'vlm_sam', vlm_elapsed=vlm_elapsed, error='abandoned',
                )
            masks, sam_elapsed = self._sam.segment(rgb_img, bboxes)
        objects, segments, closest_distances = self._build_fallback_objects(
            prompt, bboxes, masks, points, valid_mask, camera,
            return_segments=return_segments,
            labels=cls_per_box,
            header=header, target_frame=target_frame,
        )
        objects, segments, closest_distances = self._apply_realsense_range_gate(
            objects, segments, camera, closest_distances,
        )
        if objects:
            objects, segments = self._sort_objects_and_segments(
                objects, segments, sort_mode,
                camera=camera, source_frame=header.frame_id, header=header,
                closest_distances=closest_distances,
            )
        return {
            'source': 'vlm_sam',
            'objects': objects, 'segments': segments,
            'bboxes': bboxes, 'masks': masks, 'confs': [],
            'world_elapsed': 0.0, 'vlm_elapsed': vlm_elapsed,
            'sam_elapsed': sam_elapsed,
            'vlm_labels': cls_per_box,
            'vlm_raw_labels': list(raw_labels),
            'vlm_meta': vlm_meta,
            'error': None,
        }

    def _race_world_vlm(self, **ctx) -> dict:
        """Race YOLO-World vs VLM. Prefer YOLO-World if it returns objects.

        Both pipelines run as daemon threads. If YOLO-World wins we cancel
        the VLM leg via two mechanisms, only one of which is fully reliable:

          1. **abandon_event (reliable).** Set before returning. The VLM
             worker checks it (a) at every retry boundary inside
             `request_bboxes`, (b) immediately after `request_bboxes`
             returns in `_vlm_pipeline`, and (c) right before acquiring
             `self._sam_lock`. So SAM, centroid math, and TF lookups
             are guaranteed to be skipped. The thread exits cleanly without
             touching any ROS state on its way out.
          2. **client.close() (best-effort).** Also issued. On the sync
             OpenAI/httpx stack a cross-thread close does NOT reliably
             interrupt a blocking socket read — the worker may stay parked
             in `recv()` until its `vlm_timeout_s` fires. We still call
             close() because some platforms / connection states (e.g. the
             request hasn't entered recv yet, or the transport pool gets
             reaped) DO unblock; it's free insurance.

        Worst-case termination latency for an abandoned VLM thread is
        therefore one in-flight HTTP attempt (≤ `vlm_timeout_s`, default
        20 s). Best-case (between retries / pre-call / post-HTTP) is
        milliseconds. In all cases, no GPU/SAM work runs after abandon, so
        the next race's `self._sam_lock` acquire is not blocked by a
        zombie tail.
        """
        results: dict[str, dict | None] = {'world': None, 'vlm': None}
        done_world = threading.Event()
        done_vlm = threading.Event()

        # Cancellation channel for the VLM leg. The holder dict is filled
        # in by `request_bboxes` once it has an OpenAI client constructed,
        # so we can close it cross-thread.
        abandon_vlm = threading.Event()
        vlm_client_holder: dict = {}

        def _world_worker():
            try:
                results['world'] = self._world_pipeline(**ctx)
            except Exception as exc:  # noqa: BLE001 — defensive
                self.get_logger().exception('world race worker crashed')
                results['world'] = self._empty_result(
                    'yolo_world', error=f'world worker crashed: {exc}'
                )
            finally:
                done_world.set()

        def _vlm_worker():
            try:
                results['vlm'] = self._vlm_pipeline(
                    **ctx,
                    abandon_event=abandon_vlm,
                    client_holder=vlm_client_holder,
                )
            except Exception as exc:  # noqa: BLE001 — defensive
                self.get_logger().exception('vlm race worker crashed')
                results['vlm'] = self._empty_result(
                    'vlm_sam', error=f'vlm worker crashed: {exc}'
                )
            finally:
                done_vlm.set()

        threading.Thread(
            target=_world_worker, daemon=True, name='gen_world_race'
        ).start()
        threading.Thread(
            target=_vlm_worker, daemon=True, name='gen_vlm_race'
        ).start()

        # Wait for YOLO-World; bound by VLM timeout so a stuck local pass
        # can't hang the whole call past the VLM ceiling.
        world_wait = max(5.0, float(self.vlm_timeout_s))
        if not done_world.wait(timeout=world_wait):
            results['world'] = self._empty_result(
                'yolo_world',
                error=f'YOLO-World did not finish within {world_wait:.1f}s',
            )
        world_res = results['world'] or self._empty_result('yolo_world')

        if world_res.get('objects'):
            self.get_logger().info(
                f'YOLO-World won race: {len(world_res["objects"])} '
                f'object(s) in {world_res.get("world_elapsed", 0.0) * 1000:.0f} '
                'ms (+ SAM); cancelling VLM call'
            )
            self._cancel_vlm(abandon_vlm, vlm_client_holder)
            return world_res

        self.get_logger().info(
            f'YOLO-World produced no objects '
            f'({world_res.get("error") or "empty"}); waiting for VLM'
        )
        # request_bboxes enforces vlm_timeout_s as a hard total VLM budget;
        # allow a small grace margin for thread scheduling and cleanup.
        vlm_wait = float(self.vlm_timeout_s) + 5.0
        if not done_vlm.wait(timeout=vlm_wait):
            # Don't leave the VLM thread running indefinitely past our wait
            # ceiling — cancel it on our way out.
            self._cancel_vlm(abandon_vlm, vlm_client_holder)
            results['vlm'] = self._empty_result(
                'vlm_sam',
                error=f'VLM did not finish within {vlm_wait:.1f}s',
            )
        vlm_res = results['vlm'] or self._empty_result('vlm_sam')

        if vlm_res.get('objects'):
            return vlm_res

        # Both empty/failed. Surface a combined error so the caller can see
        # which leg failed.
        if world_res.get('error') and vlm_res.get('error'):
            combined = self._empty_result(
                'none',
                error=(
                    f'world: {world_res["error"]}; vlm: {vlm_res["error"]}'
                ),
                world_elapsed=world_res.get('world_elapsed', 0.0),
                vlm_elapsed=vlm_res.get('vlm_elapsed', 0.0),
            )
            return combined
        # Otherwise the more recently executed (VLM) result is the cleaner
        # "no detections" answer to surface.
        return vlm_res if not vlm_res.get('error') else world_res

    def _cancel_vlm(self, abandon_event, client_holder):
        """Cancel an in-flight VLM call: set the event and close the client.

        Called from the service-callback thread when YOLO-World wins or
        when the VLM wait ceiling expires. Safe to call even before the
        VLM worker has constructed its client (holder will be empty) — the
        event alone will short-circuit the worker at the next checkpoint.
        """
        abandon_event.set()
        # request_bboxes may build more than one client (e.g. OpenRouter for
        # the gemini leg + DashScope for the qwen leg). Close all of them;
        # fall back to the single-client key for older holders.
        clients = client_holder.get('clients')
        if not clients:
            single = client_holder.get('client')
            clients = [single] if single is not None else []
        for client in clients:
            try:
                client.close()
            except Exception as exc:  # noqa: BLE001
                self.get_logger().debug(
                    f'Closing VLM client during cancel raised '
                    f'{type(exc).__name__}: {exc} (safe to ignore)'
                )

    def _log_debug(self, _t0, request, prompt, rgb_img, result):
        """Write per-call debug artifacts. Fires unconditionally — empty
        results, errors, and rejected requests all leave an audit trail
        whenever an RGB frame is available to render."""
        objects = result.get('objects') or []
        source = result.get('source') or 'none'
        used_source = source if objects else 'none'
        bboxes = result.get('bboxes') or []
        masks = result.get('masks') or []
        confs = result.get('confs') or []
        vlm_meta = result.get('vlm_meta') or {}
        request_ctx = {
            'service': 'generalist_ObjectDetection',
            'prompt': prompt,
            'camera': request.camera,
            'target_frame': request.target_frame,
            'sort_closest': bool(request.sort_closest),
            'sort_highest': bool(request.sort_highest),
            'force_vlm_sam': bool(request.force_vlm_sam),
            'use_vlm_sam_fallback': bool(request.use_vlm_sam_fallback),
            'enable_vlm': bool(self.enable_vlm),
            'vlm_model': self.vlm_model,
            'vlm_fallback_models': list(self.vlm_fallback_models),
            'vlm_fallback_on_empty': bool(self.vlm_fallback_on_empty),
            'vlm_model_used': vlm_meta.get('model_used'),
            'vlm_attempts': vlm_meta.get('attempts') or [],
            'detection_source': used_source,
            'n_objects': len(objects),
            'n_bboxes': len(bboxes),
            'error': result.get('error'),
            # Plumbed from the base class's `_acquire_depth`; only meaningful
            # on the realsense path (orbbec path doesn't touch FFS so the
            # field stays at its 'native' default).
            'depth_source': getattr(self, '_last_depth_source', 'native'),
        }

        if used_source == 'yolo':
            # YOLO path: parent class stashed the rendered detection list
            # in self._last_detection_info. Use the stashed rgb to keep the
            # exact frame YOLO ran against.
            self._write_debug_artifacts(
                self._last_rgb_img,
                self._last_detection_info,
                request_ctx=request_ctx,
                branch='yolo',
                timings={'yolo': time.perf_counter() - _t0},
            )
            return

        # vlm_sam / yolo_world / none — render from the live result dict.
        vlm_labels = result.get('vlm_labels') or []
        vlm_raw_labels = result.get('vlm_raw_labels') or []
        detections = []
        for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
            if mask is None or mask.sum() == 0:
                continue
            cls_name = (
                vlm_labels[i] if source == 'vlm_sam' and i < len(vlm_labels)
                else prompt
            )
            detections.append({
                'bbox': bbox,
                'mask': mask,
                'cls_name': cls_name,
                'conf': confs[i] if i < len(confs) else 1.0,
            })
        timings: dict[str, float] = {'total': time.perf_counter() - _t0}
        if result.get('sam_elapsed'):
            timings['sam'] = result['sam_elapsed']
        if result.get('vlm_elapsed'):
            timings['vlm'] = result['vlm_elapsed']
        if result.get('world_elapsed'):
            timings['yolo_world'] = result['world_elapsed']
        if source == 'vlm_sam' and bboxes:
            vlm_raw = [
                {
                    'box': list(bbox),
                    'raw_label': (
                        vlm_raw_labels[i] if i < len(vlm_raw_labels) else ''
                    ),
                    'cls': (
                        vlm_labels[i] if i < len(vlm_labels) else prompt
                    ),
                    'model_used': vlm_meta.get('model_used'),
                }
                for i, bbox in enumerate(bboxes)
            ]
        else:
            vlm_raw = None
        self._write_debug_artifacts(
            rgb_img,
            detections,
            request_ctx=request_ctx,
            branch=used_source,
            vlm_raw=vlm_raw,
            timings=timings,
        )

    # --- helpers ---------------------------------------------------------

    def _init_orbbec_response_depth_intake(self) -> None:
        if 'orbbec' not in self.camera_types:
            return
        depth_image_topic = self.get_parameter(
            'orbbec_depth_image_topic'
        ).value
        self._orbbec_response_depth_intake = CameraIntake(
            self,
            configure_camera_backend(
                self,
                IntakeConfig(
                    camera='orbbec_response_depth',
                    depth=StreamSpec(
                        depth_image_topic,
                        best_effort=True,
                        qos_depth=1,
                    ),
                    age_source='stamp',
                ),
                default_endpoint='/head_camera_server',
            ),
            callback_group=MutuallyExclusiveCallbackGroup(),
            bridge=self.bridge,
        )
        self._orbbec_depth_image_sub = (
            self._orbbec_response_depth_intake._subscriptions[0]
            if self._orbbec_response_depth_intake._subscriptions
            else None
        )
        if self._orbbec_response_depth_intake.cfg.backend == 'service':
            self.get_logger().info(
                'Using head camera provider for orbbec response depth'
            )
        else:
            self.get_logger().info(
                f'Subscribed to orbbec depth Image on {depth_image_topic}'
            )

    def _orbbec_depth_image_callback(self, msg: Image) -> None:
        """Compatibility callback forwarding to the response-depth intake."""
        if self._orbbec_response_depth_intake is not None:
            self._orbbec_response_depth_intake._depth_callback(msg)

    def _warn_if_depth_mismatch(self, rgb_h: int, rgb_w: int,
                                 response, camera: str) -> None:
        """Log once per call if the depth Image dims differ from rgb/segments.

        Sizes are pulled directly from the just-attached `response.depth_image`
        message header — no decode required. Realsense aligned-depth and
        orbbec depth_to_color match rgb by construction; raw orbbec depth
        does not. We only warn (not error) because the caller may genuinely
        want raw sensor-resolution depth.
        """
        depth = getattr(response, 'depth_image', None)
        if depth is None or depth.height == 0 or depth.width == 0:
            return
        if (depth.height, depth.width) != (rgb_h, rgb_w):
            self.get_logger().warn(
                f'{camera} depth_image is {(depth.height, depth.width)} '
                f'but rgb_image and segments are {(rgb_h, rgb_w)}; '
                'segments will not overlay depth pixel-for-pixel. '
                'For orbbec, launch the camera with '
                'enable_d2c_viewer:=true and use '
                '/camera/depth/image_raw, or override '
                'orbbec_depth_image_topic.'
            )

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
        return super()._wait_for_recent_frame(camera)

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

    def _load_string_list_parameter(self, name: str) -> list[str]:
        value = self.get_parameter(name).value
        if value is None:
            return []
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return [str(item).strip() for item in value if str(item).strip()]

    @staticmethod
    def _parse_prompt_classes(prompt: str) -> list[str]:
        """Split a ' . '-joined open-vocab prompt into class strings.

        ``'apple . banana . pear'`` → ``['apple', 'banana', 'pear']``. Empty /
        whitespace-only segments are dropped. A single-word prompt yields a
        single-element list, so downstream normalization collapses to the
        identity case.
        """
        parts = [p.strip() for p in prompt.split(' . ') if p.strip()]
        return parts or [prompt.strip()]

    @staticmethod
    def _normalize_vlm_label(label: str, prompt_classes: list[str],
                             full_prompt: str) -> str:
        """Map a free-form Gemini label to one of the prompt classes.

        Tokenize both. A prompt class matches when ALL of its tokens are
        present in the label tokens (case-insensitive). Pick the class with
        the most matched tokens (longest/most-specific wins). Ties broken
        by first occurrence in ``prompt_classes``. Falls back to
        ``full_prompt`` when nothing matches — preserves pre-change behavior.
        """
        if not label:
            return full_prompt
        label_tokens = set(re.findall(r'[a-z0-9]+', label.lower()))
        if not label_tokens:
            return full_prompt
        best_cls = None
        best_score = 0
        for cls in prompt_classes:
            cls_tokens = re.findall(r'[a-z0-9]+', cls.lower())
            if not cls_tokens:
                continue
            if all(t in label_tokens for t in cls_tokens):
                if len(cls_tokens) > best_score:
                    best_cls = cls
                    best_score = len(cls_tokens)
        return best_cls if best_cls is not None else full_prompt

    def _build_fallback_objects(
        self,
        prompt: str,
        bboxes,
        masks,
        points,
        valid_mask,
        camera: str,
        return_segments: bool,
        confs: list[float] | None = None,
        labels: list[str] | None = None,
        header=None,
        target_frame: str = '',
    ):
        """Convert (bbox, mask) pairs into Object[] via parent centroid logic.

        Used by both fallback paths (YOLO-World and VLM+SAM). YOLO-World
        supplies real per-box confidences via ``confs``; Gemini does not, so
        VLM-path callers leave ``confs=None`` and every object reports
        conf=1.0. Callers that want to filter on confidence should branch on
        the response's ``detection_source`` (`'yolo'` and `'yolo_world'`
        carry real scores; `'vlm_sam'` does not).

        ``labels`` is the per-box ``Object.cls`` to write. VLM-path callers
        normalize Gemini's free-form label down to one of the prompt classes
        and pass the result here. YOLO-World callers pass nothing (their
        model is configured single-class) and every object's ``cls`` falls
        back to the full ``prompt``.

        ``header`` + ``target_frame`` are forwarded to ``_transform_centroid``
        so fallback centroids land in the same frame the YOLO branch
        produces — see parent class for the contract.
        """
        import numpy as np

        objects: list[Object] = []
        segments: list = []
        closest_distances: list[float] = []
        source_frame = header.frame_id if header is not None else ''
        stamp = header.stamp if header is not None else None
        # Hoist the source -> target lookup out of the per-bbox loop.
        # On failure we drop the whole fallback batch (returning empty)
        # to avoid emitting confidently-wrong centroids.
        centroid_tf, batch_ok = self._lookup_centroid_transform(
            source_frame, target_frame, stamp, camera,
        )
        if not batch_ok:
            return [], [], []
        for i, (bbox, mask) in enumerate(zip(bboxes, masks)):
            if mask is None or mask.sum() == 0:
                self.get_logger().warn(
                    f'fallback object {i}: empty mask post-CC for bbox={bbox}; '
                    'skipping'
                )
                continue
            centroid = self._calculate_centroid(
                points, mask, valid_mask, bbox, camera
            )
            if centroid is None:
                self.get_logger().warn(
                    f'fallback object {i}: invalid centroid for bbox={bbox}; '
                    'skipping'
                )
                continue
            closest_distance = (
                centroid.x * centroid.x
                + centroid.y * centroid.y
                + centroid.z * centroid.z
            ) ** 0.5
            if centroid_tf is not None:
                centroid = self._apply_centroid_transform(
                    centroid, centroid_tf, source_frame, stamp,
                )
            obj = Object()
            obj.conf = float(confs[i]) if confs and i < len(confs) else 1.0
            obj.cls = labels[i] if labels and i < len(labels) else prompt
            obj.centroid = centroid
            obj.id = 0
            obj.object_id = -1
            obj.similarity = 0.0
            obj.being_pointed = 0
            objects.append(obj)
            closest_distances.append(closest_distance)
            if return_segments:
                segments.append(mask.astype(np.uint8) * 255)
        return objects, segments, closest_distances


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
    # MultiThreadedExecutor — required, not optional.
    #
    # The service callback now blocks (potentially for many seconds) while
    # waiting on the race threads via threading.Event. Each ROS callback
    # group (service, realsense sync, orbbec sync, camera_info x2, TF
    # listener) needs its own executor thread to keep camera streams and
    # TF lookups alive while the service is mid-race. Default thread count
    # = multiprocessing.cpu_count() which is fine on the dev host (>=8 on
    # the target workstation), but we pin a floor of 8 so embedded
    # deployments don't starve.
    #
    # The race-thread workers spawned inside the callback are plain
    # threading.Thread (daemon=True) and do NOT consume executor threads;
    # they only touch (a) pure-Python/GPU model APIs (YOLO-World, SAM,
    # OpenAI HTTP) and (b) thread-safe rclpy facilities (logger,
    # tf2_ros.Buffer reads). SAM is shared between race legs and is
    # serialized by self._sam_lock.
    import multiprocessing
    num_threads = max(8, multiprocessing.cpu_count())
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=num_threads)
    executor.add_node(node)
    node.get_logger().info(
        f'Spinning generalist node with MultiThreadedExecutor '
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
