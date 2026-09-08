"""ROS 2 service node /object_match_all.

Composes CameraDataSource, MatchClient, JudgeClient, SamPredictor,
ItemsMapLoader, and MatchPipeline. Single MutuallyExclusiveCallbackGroup
on the service so concurrent callers serialise at the node boundary."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from ament_index_python.packages import (
    get_package_share_directory, PackageNotFoundError,
)
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header

from object_detection_generalist.sam_mask import SamPredictor
from tinker_vision_msgs_26.msg import Object
from tinker_vision_msgs_26.srv import ObjectMatchAll
from vision_util.weights_cache import resolve_weights

from .camera_data_source import CameraDataSource, CameraTopics
from .items_map_loader import ItemsMapLoader
from .match_pipeline import MatchPipeline, PipelineParams
from .vlm_judge_client import build_judge_client
from .vlm_match_client import build_match_client


@dataclass
class NodeParams:
    # Pipeline params (a superset of PipelineParams plus camera/io knobs).
    batch_size: int
    max_workers: int
    vlm_per_call_timeout_s: float
    vlm_max_retries: int
    stage1_timeout_s: float
    stage2_timeout_s: float
    nms_within_category_iou: float
    cluster_iou: float
    judge_crop_margin_px: int
    min_valid_centroid_pixels: int
    # Camera io
    img_sync_thres_s: float
    sync_wait_time_s: float
    min_depth_m: float
    max_depth_m: float


def _build_vlm_clients(provider, judge_provider, model, judge_model, base_url):
    return (
        build_match_client(provider, model=model, base_url=base_url),
        build_judge_client(judge_provider, model=judge_model, base_url=base_url),
    )


class ObjectMatchAllServer(Node):
    def __init__(self):
        super().__init__('object_match_all_server')

        # Service / provider params
        self.declare_parameter('service_name', 'object_match_all')
        self.declare_parameter('vlm_provider', 'qwen')
        self.declare_parameter('judge_provider', '')
        self.declare_parameter('vlm_model', '')
        self.declare_parameter('judge_model', '')
        self.declare_parameter('vlm_base_url', '')
        # Camera topics
        self.declare_parameter(
            'realsense_image_topic',
            '/camera/xarm_camera/color/image_raw',
        )
        self.declare_parameter(
            'realsense_depth_topic',
            '/camera/xarm_camera/aligned_depth_to_color/image_raw',
        )
        self.declare_parameter(
            'realsense_camera_info_topic',
            '/camera/xarm_camera/aligned_depth_to_color/camera_info',
        )
        self.declare_parameter(
            'orbbec_image_topic', '/camera/color/image_raw',
        )
        self.declare_parameter(
            'orbbec_depth_topic', '/camera/depth_registered/points',
        )
        self.declare_parameter(
            'orbbec_camera_info_topic', '/camera/color/camera_info',
        )
        # Items map + SAM
        self.declare_parameter('items_map_path', '')
        self.declare_parameter('sam_weights', 'mobile_sam.pt')
        self.declare_parameter('sam_device', '')
        # Pipeline / timeouts
        self.declare_parameter('batch_size', 3)
        self.declare_parameter('max_workers', 8)
        self.declare_parameter('vlm_per_call_timeout_s', 12.0)
        self.declare_parameter('vlm_max_retries', 1)
        self.declare_parameter('stage1_timeout_s', 15.0)
        self.declare_parameter('stage2_timeout_s', 10.0)
        self.declare_parameter('nms_within_category_iou', 0.5)
        self.declare_parameter('cluster_iou', 0.5)
        self.declare_parameter('judge_crop_margin_px', 20)
        self.declare_parameter('min_valid_centroid_pixels', 8)
        # Camera io
        self.declare_parameter('img_sync_thres_s', 0.5)
        self.declare_parameter('sync_wait_time_s', 1.5)
        self.declare_parameter('min_depth_m', 0.05)
        self.declare_parameter('max_depth_m', 8.0)
        self.declare_parameter('camera_backend', 'service')
        self.declare_parameter(
            'realsense_provider_endpoint', '/wrist_camera_server')
        self.declare_parameter(
            'orbbec_provider_endpoint', '/head_camera_server')
        self.declare_parameter(
            'transform_provider_endpoint', '/head_camera_server')
        self.declare_parameter('camera_provider_wait_timeout_s', 0.5)
        self.declare_parameter('camera_provider_response_timeout_s', 5.0)
        # Logging
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')
        self.declare_parameter('log_raw_vlm', False)

        self.params = NodeParams(
            batch_size=int(self.get_parameter('batch_size').value),
            max_workers=int(self.get_parameter('max_workers').value),
            vlm_per_call_timeout_s=float(
                self.get_parameter('vlm_per_call_timeout_s').value),
            vlm_max_retries=int(
                self.get_parameter('vlm_max_retries').value),
            stage1_timeout_s=float(
                self.get_parameter('stage1_timeout_s').value),
            stage2_timeout_s=float(
                self.get_parameter('stage2_timeout_s').value),
            nms_within_category_iou=float(
                self.get_parameter('nms_within_category_iou').value),
            cluster_iou=float(self.get_parameter('cluster_iou').value),
            judge_crop_margin_px=int(
                self.get_parameter('judge_crop_margin_px').value),
            min_valid_centroid_pixels=int(
                self.get_parameter('min_valid_centroid_pixels').value),
            img_sync_thres_s=float(
                self.get_parameter('img_sync_thres_s').value),
            sync_wait_time_s=float(
                self.get_parameter('sync_wait_time_s').value),
            min_depth_m=float(self.get_parameter('min_depth_m').value),
            max_depth_m=float(self.get_parameter('max_depth_m').value),
        )

        topics = CameraTopics(
            realsense_image=self.get_parameter(
                'realsense_image_topic').value,
            realsense_depth=self.get_parameter(
                'realsense_depth_topic').value,
            realsense_camera_info=self.get_parameter(
                'realsense_camera_info_topic').value,
            orbbec_image=self.get_parameter('orbbec_image_topic').value,
            orbbec_depth=self.get_parameter('orbbec_depth_topic').value,
            orbbec_camera_info=self.get_parameter(
                'orbbec_camera_info_topic').value,
        )

        self.bridge = CvBridge()
        self.camera = CameraDataSource(
            self, topics=topics, params=self.params,
            logger=self.get_logger(),
        )

        items_dir = self._resolve_items_dir()
        if not items_dir:
            raise RuntimeError(
                'Could not locate items_map directory; set the '
                'items_map_path parameter to an absolute path.'
            )
        self.items = ItemsMapLoader(items_dir, logger=self.get_logger())
        if len(self.items) == 0:
            self.get_logger().warning(
                f'items_map at {items_dir} is empty; '
                'every request will be 1.'
            )
        self.items_dict = {
            k: self.items.get_data_url(k) for k in self.items.keys()
        }

        provider = self.get_parameter('vlm_provider').value
        judge_provider = (
            self.get_parameter('judge_provider').value or provider
        )
        model = self.get_parameter('vlm_model').value or ''
        judge_model = self.get_parameter('judge_model').value or model
        base_url = self.get_parameter('vlm_base_url').value or ''

        self.match_client, self.judge_client = _build_vlm_clients(
            provider, judge_provider, model, judge_model, base_url,
        )

        sam_weights = resolve_weights(
            self.get_parameter('sam_weights').value,
        )
        sam_device = self.get_parameter('sam_device').value or ''
        self.sam = SamPredictor(
            str(sam_weights), device=sam_device,
            logger=self.get_logger(),
        )
        try:
            self.sam.segment(
                np.zeros((64, 64, 3), dtype=np.uint8),
                [(0, 0, 64, 64)],
            )
        except Exception as exc:    # noqa: BLE001
            self.get_logger().warning(f'SAM warm-up failed: {exc}')

        self.pipeline = MatchPipeline(
            match_client=self.match_client,
            judge_client=self.judge_client,
            sam=self.sam,
            camera=self.camera,
            items=self.items_dict,
            params=PipelineParams(
                batch_size=self.params.batch_size,
                max_workers=self.params.max_workers,
                vlm_per_call_timeout_s=self.params.vlm_per_call_timeout_s,
                vlm_max_retries=self.params.vlm_max_retries,
                stage1_timeout_s=self.params.stage1_timeout_s,
                stage2_timeout_s=self.params.stage2_timeout_s,
                nms_within_category_iou=(
                    self.params.nms_within_category_iou
                ),
                cluster_iou=self.params.cluster_iou,
                judge_crop_margin_px=self.params.judge_crop_margin_px,
                min_valid_centroid_pixels=(
                    self.params.min_valid_centroid_pixels
                ),
            ),
            logger=self.get_logger(),
        )

        service_name = self.get_parameter('service_name').value
        self.srv = self.create_service(
            ObjectMatchAll, service_name, self._callback,
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.get_logger().info(
            f'object_match_all_server ready: service={service_name}, '
            f'items={len(self.items_dict)}, provider={provider}, '
            f'judge_provider={judge_provider}, '
            f'batch_size={self.params.batch_size}'
        )

    # ---------------- callback ----------------
    def _callback(
        self,
        req: ObjectMatchAll.Request,
        resp: ObjectMatchAll.Response,
    ):
        _t0 = time.perf_counter()
        resp.header = Header(stamp=self.get_clock().now().to_msg())
        resp.status = 1
        resp.error_msg = ''
        resp.person_id = 0
        resp.objects = []
        resp.detection_source = 'vlm_match_all'

        camera = self._select_camera(req.camera)
        snap = self.camera.snapshot(camera)
        if snap is None:
            resp.error_msg = (
                f'No {camera} camera data within sync threshold'
            )
            return resp
        rgb_bgr, points_xyz, valid_mask, header, raw_color_msg = snap
        resp.header = header

        if req.category_filter:
            unknown = [
                k for k in req.category_filter
                if k not in self.items_dict
            ]
            if unknown:
                self.get_logger().warning(
                    f'category_filter dropping unknown keys: {unknown}'
                )
            known = [
                k for k in req.category_filter
                if k in self.items_dict
            ]
            if not known:
                resp.error_msg = (
                    f'Unknown items: {", ".join(unknown)}'
                )
                return resp
        else:
            known = list(self.items_dict.keys())

        target_frame = req.target_frame.strip()
        source_frame = header.frame_id

        finals, counters = self.pipeline.run(
            scene_bgr=rgb_bgr,
            points_xyz=points_xyz,
            valid_mask=valid_mask,
            camera=camera,
            category_filter=known,
            target_frame=target_frame,
            source_frame=source_frame,
            header_stamp=header.stamp,
        )

        if not finals:
            resp.error_msg = self._error_msg_for_empty(
                counters, camera, target_frame,
            )
            self._log_summary(
                counters, time.perf_counter() - _t0, resp.status,
            )
            return resp

        finals = self._sort(finals, req)

        resp.header.frame_id = target_frame or source_frame
        for fr in finals:
            obj = Object()
            obj.cls = fr.row.label
            obj.conf = float(fr.row.conf)
            obj.id = 0
            obj.object_id = -1
            obj.similarity = 0.0
            obj.being_pointed = 0
            obj.centroid = fr.point_out
            resp.objects.append(obj)

        if req.return_rgb_image:
            resp.rgb_image = raw_color_msg
        if req.return_depth_image:
            depth_msg = self.bridge.cv2_to_imgmsg(
                points_xyz[:, :, 2].astype(np.float32),
                encoding='32FC1',
            )
            depth_msg.header = resp.header
            resp.depth_image = depth_msg
        if req.return_segments:
            seg_msgs = []
            for fr in finals:
                seg_msg = self.bridge.cv2_to_imgmsg(
                    (fr.mask.astype(np.uint8) * 255), encoding='8UC1',
                )
                seg_msg.header = resp.header
                seg_msgs.append(seg_msg)
            resp.segments = seg_msgs

        resp.status = 0
        self._log_summary(
            counters, time.perf_counter() - _t0, resp.status,
        )
        return resp

    # ---------------- helpers ----------------
    def _select_camera(self, request_camera: str) -> str:
        if 'realsense' in (request_camera or ''):
            return 'realsense'
        if 'orbbec' in (request_camera or ''):
            return 'orbbec'
        self.get_logger().warning(
            f'unknown camera "{request_camera}", defaulting to orbbec'
        )
        return 'orbbec'

    def _resolve_items_dir(self) -> str:
        override = self.get_parameter('items_map_path').value or ''
        if override:
            return override
        try:
            share_dir = get_package_share_directory(
                'tk_vision_specialized',
            )
        except PackageNotFoundError:
            share_dir = ''
        candidate = (
            os.path.join(share_dir, 'items') if share_dir else ''
        )
        if candidate and os.path.isfile(
            os.path.join(candidate, 'items_map.yaml')
        ):
            return candidate
        here = os.path.dirname(os.path.abspath(__file__))
        for _ in range(8):
            here = os.path.dirname(here)
            for guess in (
                os.path.join(here, 'src', 'items'),
                os.path.join(here, 'src', 'tk26_vision', 'src', 'items'),
            ):
                if os.path.isfile(
                    os.path.join(guess, 'items_map.yaml')
                ):
                    return guess
        return candidate or ''

    def _error_msg_for_empty(
        self, counters: dict, camera: str, target_frame: str,
    ) -> str:
        if (
            counters.get('batches_ok', 0) == 0
            and counters.get('batches_fail', 0) > 0
        ):
            return (
                'all VLM match batches failed: '
                f'fail={counters["batches_fail"]}'
            )
        if counters.get('tf_failed', 0) > 0:
            return (
                f'TF -> {target_frame} unavailable for '
                f'{counters["tf_failed"]} detections'
            )
        if counters.get('detections_dropped_no_depth', 0) > 0:
            return 'no valid-depth pixels for any matched object'
        return 'no items matched'

    def _sort(self, finals, req: ObjectMatchAll.Request):
        if req.sort_closest:
            return sorted(
                finals,
                key=lambda fr: (
                    fr.point_camera.x**2
                    + fr.point_camera.y**2
                    + fr.point_camera.z**2
                ),
            )
        if req.sort_highest:
            return sorted(finals, key=lambda fr: fr.point_camera.z)
        return sorted(finals, key=lambda fr: -fr.row.conf)

    def _log_summary(
        self, counters: dict, total_s: float, status: int,
    ) -> None:
        self.get_logger().info(
            f'match_all: status={status} '
            f'batches_ok={counters.get("batches_ok", 0)} '
            f'batches_fail={counters.get("batches_fail", 0)} '
            f'rows_in={counters.get("rows_in", 0)} '
            f'after_nms={counters.get("after_nms", 0)} '
            f'clusters_conflict='
            f'{counters.get("clusters_conflict", 0)} '
            f'judge_ok={counters.get("judge_ok", 0)} '
            f'judge_abstain={counters.get("judge_abstain", 0)} '
            f'judge_fail={counters.get("judge_fail", 0)} '
            f'dropped_no_depth='
            f'{counters.get("detections_dropped_no_depth", 0)} '
            f'tf_failed={counters.get("tf_failed", 0)} '
            f'total_s={total_s:.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = ObjectMatchAllServer()
    import multiprocessing
    num_threads = max(8, multiprocessing.cpu_count())
    executor = MultiThreadedExecutor(num_threads=num_threads)
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
