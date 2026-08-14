"""ROS 2 service: visual-grounding object match with MobileSAM.

Subclasses ``YOLOSegmentationNode`` from ``object_detection_new`` to reuse
camera sync, intrinsics, depth-to-3D projection, the TF buffer, and the
vision logger. Replaces YOLO with a single Qwen3-VL grounding call (DashScope)
that takes the camera RGB plus a stored reference photo for the requested
item, then runs MobileSAM on the chosen bbox to obtain an exact mask.

The structural model is ``placing_location_server.py``; the differences are
the input (two images instead of one + text), the detector (Qwen3-VL via
DashScope instead of Gemini via OpenRouter), and a SAM step on the chosen
bbox.

When multiple confident candidates tie within ``confidence_tie_eps``, we
fall back to picking the closest one in 3D — the user explicitly asked for
this tiebreaker semantics.
"""

from __future__ import annotations

import copy
import os
import time

import numpy as np
import rclpy
import rclpy.duration
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.parameter import Parameter

from ament_index_python.packages import (
    get_package_share_directory,
    PackageNotFoundError,
)
import geometry_msgs.msg
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header
from tf2_ros import (
    LookupException,
    ConnectivityException,
    ExtrapolationException,
)
from tf2_geometry_msgs import do_transform_point

from tinker_vision_msgs_26.msg import Object
from tinker_vision_msgs_26.srv import ObjectMatch

from object_detection_new.object_seg_yolo import YOLOSegmentationNode
from object_detection_generalist.sam_mask import SamPredictor
from vision_util.weights_cache import resolve_weights

from .items_map_loader import ItemsMapLoader
from .qwen_match_vlm import QwenMatchError, request_match_bboxes


class ObjectMatchServer(YOLOSegmentationNode):
    """ROS 2 service node: Qwen3-VL grounding + MobileSAM segmentation."""

    def __init__(self, node_name: str = 'object_match_server',
                 parameter_overrides=None):
        super().__init__(
            node_name=node_name,
            parameter_overrides=parameter_overrides or [],
        )
        self._init_items_map()
        self._init_sam()
        self.get_logger().info(
            f'Object-match node ready: vlm_model={self.vlm_model}, '
            f'top_k={self.top_k_candidates}, '
            f'tie_eps={self.confidence_tie_eps}, '
            f'device={self.device}, items={len(self.items)}'
        )

    # The parent loads YOLO weights into GPU memory in __init__. This node
    # never invokes YOLO (Qwen3-VL + MobileSAM cover detection + masks), so
    # we override to skip the load and free ~200 MB GPU + ~1-2 s startup.
    # self.device is still set so SAM and any future GPU work pick the right
    # backend.
    def _init_model(self):
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(
            f'YOLO load skipped (this node uses Qwen3-VL + MobileSAM); '
            f'device={self.device}'
        )

    # --- parameter wiring -------------------------------------------------

    def _declare_parameters(self):
        super()._declare_parameters()
        self.declare_parameter('vlm_model', '')
        self.declare_parameter('vlm_base_url', '')
        self.declare_parameter('qwen_api_backend', 'dashscope')
        self.declare_parameter('vlm_timeout_s', 12.0)
        self.declare_parameter('vlm_max_retries', 1)
        self.declare_parameter('top_k_candidates', 3)
        self.declare_parameter('confidence_tie_eps', 0.05)
        self.declare_parameter('items_map_path', '')
        self.declare_parameter('sam_weights', 'mobile_sam.pt')
        self.declare_parameter('sam_device', '')

    def _load_parameters(self):
        super()._load_parameters()
        self.vlm_model = self.get_parameter('vlm_model').value
        self.vlm_base_url = self.get_parameter('vlm_base_url').value
        self.qwen_api_backend = self.get_parameter('qwen_api_backend').value
        self.vlm_timeout_s = float(self.get_parameter('vlm_timeout_s').value)
        self.vlm_max_retries = int(
            self.get_parameter('vlm_max_retries').value
        )
        self.top_k_candidates = int(
            self.get_parameter('top_k_candidates').value
        )
        self.confidence_tie_eps = float(
            self.get_parameter('confidence_tie_eps').value
        )
        self._items_map_path_param = (
            self.get_parameter('items_map_path').value or ''
        )
        self._sam_weights_param = self.get_parameter('sam_weights').value
        self._sam_device_param = self.get_parameter('sam_device').value or ''

    # --- service advertisement -------------------------------------------

    def _init_service(self):
        service_name = self.get_parameter('service_name').value
        self.detection_srv = self.create_service(
            ObjectMatch,
            service_name,
            self._object_match_callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(
            f'ObjectMatch service created: {service_name}'
        )

    # --- one-off init helpers --------------------------------------------

    def _resolve_items_dir(self) -> str:
        if self._items_map_path_param:
            return self._items_map_path_param
        try:
            share_dir = get_package_share_directory('tk_vision_specialized')
        except PackageNotFoundError:
            share_dir = ''
        candidate = os.path.join(share_dir, 'items') if share_dir else ''
        if candidate and os.path.isfile(
            os.path.join(candidate, 'items_map.yaml')
        ):
            return candidate
        # Fallback: walk up from this file to find src/tk26_vision/src/items.
        # Useful in dev when running before colcon install copies the data.
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(8):
            here = os.path.dirname(here)
            guess = os.path.join(here, 'src', 'items')
            if os.path.isfile(os.path.join(guess, 'items_map.yaml')):
                return guess
            guess = os.path.join(here, 'src', 'tk26_vision', 'src', 'items')
            if os.path.isfile(os.path.join(guess, 'items_map.yaml')):
                return guess
        return candidate or ''

    def _init_items_map(self) -> None:
        items_dir = self._resolve_items_dir()
        if not items_dir:
            raise RuntimeError(
                'Could not locate items_map directory; set the '
                '"items_map_path" parameter to an absolute path.'
            )
        self.items = ItemsMapLoader(items_dir, logger=self.get_logger())
        if len(self.items) == 0:
            self.get_logger().warning(
                f'ItemsMapLoader found 0 items in {items_dir}; '
                'every request will fail with "Unknown item".'
            )

    def _init_sam(self) -> None:
        device = self._sam_device_param or self.device
        weights_path = resolve_weights(self._sam_weights_param)
        self.sam = SamPredictor(
            str(weights_path), device=device, logger=self.get_logger(),
        )
        # Pre-warm: first SAM inference pays a 0.5-1.2 s JIT/CUDA cost.
        # Run once at startup on a synthetic 64x64 image so the first real
        # request doesn't see that latency.
        try:
            warmup = np.zeros((64, 64, 3), dtype=np.uint8)
            self.sam.segment(warmup, [(0, 0, 64, 64)])
            self.get_logger().info('MobileSAM warm-up complete')
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(f'MobileSAM warm-up failed: {exc}')

    # --- service callback ------------------------------------------------

    def _object_match_callback(
        self,
        request: ObjectMatch.Request,
        response: ObjectMatch.Response,
    ) -> ObjectMatch.Response:
        _t0 = time.perf_counter()
        response.header = Header(stamp=self.get_clock().now().to_msg())
        response.status = -1
        response.error_msg = ''
        response.objects = []
        response.detection_source = ''

        category = (request.category or '').strip()
        if not category:
            response.status = 1
            response.error_msg = 'category is empty'
            return response
        if category not in self.items:
            known = ', '.join(sorted(self.items.keys())) or '<none>'
            response.status = 1
            response.error_msg = (
                f'Unknown item: "{category}" (known: {known})'
            )
            return response

        camera = self._select_camera(request.camera)
        rec_msg = self._wait_for_recent_frame(camera)
        if rec_msg is None:
            response.status = 1
            response.error_msg = (
                f'No {camera} camera data within sync threshold'
            )
            return response

        intrinsic = self._get_intrinsic(camera)
        if intrinsic is None:
            response.status = 1
            response.error_msg = f'No {camera} camera intrinsics available'
            return response

        try:
            if camera == 'realsense':
                rgb_img, points, valid_mask, header = (
                    self._process_realsense_data(
                        rec_msg[0], rec_msg[1], intrinsic,
                    )
                )
            else:
                rgb_img, points, valid_mask, header = (
                    self._process_orbbec_data(
                        rec_msg[0], rec_msg[1], intrinsic,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(
                f'Error processing {camera} data: {exc}'
            )
            response.status = 1
            response.error_msg = f'camera data processing error: {exc}'
            return response

        response.header = header

        # --- Qwen3-VL grounding call -------------------------------------
        try:
            ref_url = self.items.get_data_url(category)
            boxes, confs, labels, vlm_s = request_match_bboxes(
                rgb_img,
                ref_url,
                item_name=category,
                top_k=self.top_k_candidates,
                model=self.vlm_model,
                base_url=self.vlm_base_url,
                qwen_api_backend=self.qwen_api_backend,
                max_retries=self.vlm_max_retries,
                timeout_s=self.vlm_timeout_s,
                logger=self.get_logger(),
            )
        except QwenMatchError as exc:
            response.status = 1
            response.error_msg = f'DashScope unavailable: {exc}'
            self._maybe_log(_t0, request, category, rgb_img, None,
                            None, 0.0, 0.0, error=str(exc))
            return response
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Qwen3-VL call failed: {exc}')
            response.status = 1
            response.error_msg = f'Qwen3-VL error: {exc}'
            self._maybe_log(_t0, request, category, rgb_img, None,
                            None, 0.0, 0.0, error=str(exc))
            return response

        if not boxes:
            response.status = 1
            response.error_msg = 'Qwen3-VL returned no matches'
            self._maybe_log(_t0, request, category, rgb_img, None,
                            None, vlm_s, 0.0,
                            extra_ctx={'n_candidates': 0})
            return response

        # --- candidate selection: top-conf with closest-3D tiebreak ------
        # Identify the conf-tied set first; only compute rect-mask centroids
        # for tiebreak candidates so the common single-leader case skips the
        # extra _calculate_centroid passes entirely.
        rgb_h, rgb_w = rgb_img.shape[:2]
        top_conf = max(confs)
        tied = [
            i for i, c in enumerate(confs)
            if (top_conf - c) < self.confidence_tie_eps
        ]
        used_tiebreak = len(tied) > 1
        rect_centroid_cache: dict[int, geometry_msgs.msg.Point | None] = {}

        def _rect_centroid(i: int):
            if i not in rect_centroid_cache:
                bbox = boxes[i]
                rect_mask = _rect_mask_for((rgb_h, rgb_w), bbox)
                rect_centroid_cache[i] = self._calculate_centroid(
                    points, rect_mask, valid_mask, bbox, camera,
                )
            return rect_centroid_cache[i]

        if used_tiebreak:
            tied_with_depth = [i for i in tied if _rect_centroid(i) is not None]
            if tied_with_depth:
                def _dist_sq(i):
                    p = rect_centroid_cache[i]
                    return p.x * p.x + p.y * p.y + p.z * p.z
                chosen_idx = min(tied_with_depth, key=_dist_sq)
            else:
                # Tied set has no valid depth anywhere; fall through to top
                # confidence and let SAM-mask centroid (or its rect-mask
                # fallback) fail downstream with a clearer error.
                chosen_idx = tied[0]
        else:
            chosen_idx = tied[0]

        chosen_bbox = boxes[chosen_idx]
        chosen_conf = confs[chosen_idx]
        chosen_label = labels[chosen_idx] if chosen_idx < len(labels) else ''

        # --- MobileSAM on the chosen bbox --------------------------------
        try:
            sam_masks, sam_s = self.sam.segment(rgb_img, [chosen_bbox])
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'MobileSAM call failed: {exc}')
            response.status = 1
            response.error_msg = f'SAM error: {exc}'
            self._maybe_log(
                _t0, request, category, rgb_img,
                chosen_bbox, None, vlm_s, 0.0,
                extra_ctx={'n_candidates': len(boxes),
                           'tiebreak': used_tiebreak,
                           'chosen_conf': chosen_conf},
                error=str(exc),
            )
            return response

        sam_mask = (
            sam_masks[0]
            if sam_masks and isinstance(sam_masks[0], np.ndarray)
            else np.zeros((rgb_h, rgb_w), dtype=bool)
        )

        # Recompute final centroid using the exact SAM mask. Fall back to
        # the rectangular-bbox centroid if SAM excludes all valid-depth
        # pixels (rare, but possible if the mask collapses to a sliver
        # outside the bbox's depth coverage).
        final_point = self._calculate_centroid(
            points, sam_mask, valid_mask, chosen_bbox, camera,
        )
        if final_point is None:
            self.get_logger().warning(
                'SAM mask has no valid depth pixels; falling back to '
                'rectangular-bbox centroid for the chosen candidate.'
            )
            final_point = _rect_centroid(chosen_idx)
            if final_point is None:
                response.status = 1
                response.error_msg = (
                    'No valid depth in chosen bbox; cannot compute centroid'
                )
                self._maybe_log(
                    _t0, request, category, rgb_img,
                    chosen_bbox, sam_mask, vlm_s, sam_s,
                    extra_ctx={'n_candidates': len(boxes),
                               'tiebreak': used_tiebreak,
                               'chosen_conf': chosen_conf,
                               'no_depth': True},
                    error=response.error_msg,
                )
                return response

        # --- TF transform if requested ----------------------------------
        target_frame = (request.target_frame or '').strip()
        source_frame = header.frame_id or ''
        out_frame = source_frame
        if (
            target_frame
            and target_frame != source_frame
            and self._frame_supports_tf_transform(camera)
        ):
            transformed = self._transform_point(
                final_point, target_frame, source_frame, header.stamp,
            )
            if transformed is None:
                response.status = 1
                response.error_msg = (
                    f'TF {source_frame} -> {target_frame} unavailable'
                )
                self._maybe_log(
                    _t0, request, category, rgb_img,
                    chosen_bbox, sam_mask, vlm_s, sam_s,
                    extra_ctx={'n_candidates': len(boxes),
                               'tiebreak': used_tiebreak,
                               'chosen_conf': chosen_conf,
                               'tf_failed': True},
                    error=response.error_msg,
                )
                return response
            final_point = transformed
            out_frame = target_frame

        # --- pack response -----------------------------------------------
        response.header.frame_id = out_frame

        obj = Object()
        obj.cls = category
        obj.conf = float(chosen_conf)
        obj.id = 0
        obj.object_id = -1
        obj.similarity = 0.0
        obj.being_pointed = 0
        obj.centroid = final_point
        response.objects = [obj]
        response.detection_source = 'qwen3vl_mobilesam'

        # rgb_image: forward the raw sensor msg (already encoded, no copy).
        response.rgb_image = rec_msg[0]

        # depth_image: send a uniform 32FC1 metric depth grid built from
        # points[:,:,2] so callers see the same encoding regardless of
        # camera. realsense's raw uint16 mm and orbbec's PointCloud2 both
        # become a single-channel float32 image in metres.
        depth_2d = points[:, :, 2].astype(np.float32)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_2d, encoding='32FC1')
        depth_msg.header = header
        response.depth_image = depth_msg

        # segments: one 8UC1 mask (255=foreground, 0=background) — same
        # convention as object_detection_generalist.
        seg_msg = self.bridge.cv2_to_imgmsg(
            (sam_mask.astype(np.uint8) * 255), encoding='8UC1',
        )
        seg_msg.header = header
        response.segments = [seg_msg]

        response.status = 0
        response.error_msg = ''

        self._maybe_log(
            _t0, request, category, rgb_img,
            chosen_bbox, sam_mask, vlm_s, sam_s,
            extra_ctx={'n_candidates': len(boxes),
                       'tiebreak': used_tiebreak,
                       'chosen_conf': chosen_conf,
                       'chosen_label': chosen_label,
                       'reference_image': self.items.get_filename(category)},
        )
        return response

    # --- helpers ---------------------------------------------------------

    def _select_camera(self, camera_req: str) -> str:
        if 'realsense' in (camera_req or ''):
            return 'realsense'
        if 'orbbec' in (camera_req or ''):
            return 'orbbec'
        self.get_logger().warning(
            f'Unknown camera "{camera_req}", defaulting to orbbec'
        )
        return 'orbbec'

    def _wait_for_recent_frame(self, camera: str):
        intake = self._camera_intakes.get(camera)
        if intake is not None and intake.cfg.backend == 'service':
            bundle = intake.wait_fresh(
                max_age_s=self.img_sync_thres,
                timeout_s=self.sync_wait_time_limit * 0.1,
                on_timeout='fail',
            )
            if bundle is None:
                return None
            return copy.deepcopy((bundle.color_msg, bundle.depth_msg))

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
        intake = self._camera_intakes.get(camera)
        if intake is not None and intake.cfg.backend == 'service':
            return copy.deepcopy(intake.camera_info())
        with self.lock_info:
            return copy.deepcopy(self.camera_intrinsic.get(camera))

    def _transform_point(
        self,
        point: geometry_msgs.msg.Point,
        target_frame: str,
        source_frame: str,
        stamp,
    ) -> geometry_msgs.msg.Point | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                stamp,
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except (LookupException, ConnectivityException,
                ExtrapolationException) as exc:
            self.get_logger().warning(
                f'TF {source_frame} -> {target_frame} failed: {exc}'
            )
            return None
        try:
            stamped = PointStamped()
            stamped.header = Header(stamp=stamp, frame_id=source_frame)
            stamped.point = point
            return do_transform_point(stamped, transform).point
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warning(
                f'do_transform_point failed: {exc}'
            )
            return None

    def _maybe_log(self, _t0, request, category, rgb_img,
                   chosen_bbox, sam_mask, vlm_s, sam_s,
                   *, extra_ctx=None, error=None):
        if not getattr(self, '_vision_logger', None) \
                or not self._vision_logger.enabled:
            return
        detections = []
        if chosen_bbox is not None:
            mask_for_log = (
                sam_mask
                if sam_mask is not None
                else _rect_mask_for(rgb_img.shape[:2], chosen_bbox)
            )
            detections.append({
                'bbox': chosen_bbox,
                'mask': mask_for_log,
                'cls_name': category,
                'conf': float(extra_ctx.get('chosen_conf', 0.0))
                if extra_ctx else 0.0,
            })
        request_ctx = {
            'service': 'object_match',
            'category': category,
            'camera': request.camera,
            'target_frame': request.target_frame,
            'error': error,
        }
        if extra_ctx:
            request_ctx.update(extra_ctx)
        timings = {'total': time.perf_counter() - _t0}
        if vlm_s:
            timings['vlm'] = vlm_s
        if sam_s:
            timings['sam'] = sam_s
        self._write_debug_artifacts(
            rgb_img, detections,
            request_ctx=request_ctx,
            branch='object_match',
            timings=timings,
        )


def _rect_mask_for(shape, bbox):
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = bbox
    m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return m


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMatchServer(
        node_name='object_match_server',
        parameter_overrides=[
            Parameter('service_name',
                      Parameter.Type.STRING,
                      'object_match'),
        ],
    )
    # MultiThreadedExecutor: the service callback blocks 5-12 s on the VLM
    # HTTP call, and during that window camera + camera_info + TF callbacks
    # must keep running so the next request finds fresh frames + intrinsics.
    import multiprocessing
    num_threads = max(8, multiprocessing.cpu_count())
    executor = rclpy.executors.MultiThreadedExecutor(
        num_threads=num_threads,
    )
    executor.add_node(node)
    node.get_logger().info(
        f'Spinning object_match_server with MultiThreadedExecutor '
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
