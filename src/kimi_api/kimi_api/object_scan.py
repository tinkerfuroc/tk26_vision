"""Labels-only batched-VLM scene scanner — service /object_scan.

Given a candidate vocabulary, splits it into batches, runs one vision-LLM call
per batch (ALL batches in parallel; Gemini primary -> Qwen fallback), and
returns the subset of the vocabulary visible in the scene. Labels only -- no
bounding boxes, masks, depth, or centroids.

Defeats the "Gemini misses objects when asked for ~32 classes at once" problem
that the single-call generalist scan hits on the PickAndPlace table. Model
selection mirrors object_detection_generalist (gemini-2.5-flash -> qwen3-vl-plus
via DashScope). The batching + validation logic is the same as the tuning
harness at scripts/object_scan_webui/scan_core.py.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tinker_vision_msgs_26.srv import ObjectScan

from ._env import (
    default_flash_model,
    load_env,
    require_api_key,
    resolve_qwen_target,
)
from ._image_utils import encode_to_data_url
from ._scan_vlm import ScanVlmError, request_scan_labels_chain


def _batches(vocab, batch_size):
    if batch_size <= 0:
        return [list(vocab)]
    return [vocab[i:i + batch_size] for i in range(0, len(vocab), batch_size)]


class ObjectScanServer(Node):
    def __init__(self):
        super().__init__('object_scan')

        self.declare_parameter('service_name', 'object_scan')
        self.declare_parameter('llm_model', default_flash_model())
        self.declare_parameter('vlm_fallback_provider', 'qwen')  # '' to disable
        self.declare_parameter('scan_model_qwen', '')
        self.declare_parameter('qwen_api_backend', 'dashscope')
        self.declare_parameter('batch_size', 8)
        # 0 = one worker per batch (every batch VLM call in parallel);
        # >0 caps concurrency for provider rate limits.
        self.declare_parameter('max_workers', 0)
        self.declare_parameter('vlm_timeout_s', 20.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter('orbbec_image_topic', '/camera/color/image_raw')
        self.declare_parameter(
            'realsense_image_topic', '/camera/xarm_camera/color/image_raw')
        self.declare_parameter('img_sync_thres_s', 1.0)
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')

        gp = self.get_parameter
        self.llm_model = gp('llm_model').get_parameter_value().string_value
        self.vlm_fallback_provider = (
            gp('vlm_fallback_provider').get_parameter_value().string_value)
        self.scan_model_qwen = (
            gp('scan_model_qwen').get_parameter_value().string_value)
        self.qwen_api_backend = gp('qwen_api_backend').value
        self.batch_size = int(gp('batch_size').value)
        self.max_workers = int(gp('max_workers').value)
        self.vlm_timeout_s = float(gp('vlm_timeout_s').value)
        self.vlm_max_retries = int(gp('vlm_max_retries').value)
        self.img_sync_thres_s = float(gp('img_sync_thres_s').value)
        self.vision_logging = bool(gp('vision_logging_enabled').value)
        self.vision_log_folder = gp('vision_log_folder').value

        require_api_key()  # fail fast on missing OPENROUTER_API_KEY
        self._provider_chain = self._resolve_provider_chain()

        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = {'orbbec': None, 'realsense': None}  # cam -> (msg, t)

        self.create_subscription(
            Image, gp('orbbec_image_topic').value,
            lambda m: self._on_image('orbbec', m),
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup())
        self.create_subscription(
            Image, gp('realsense_image_topic').value,
            lambda m: self._on_image('realsense', m),
            qos_profile_sensor_data,
            callback_group=MutuallyExclusiveCallbackGroup())

        service_name = gp('service_name').value
        self.srv = self.create_service(
            ObjectScan, service_name, self._callback,
            callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(
            f'object_scan ready: service={service_name}, '
            f'model={self.llm_model}, batch_size={self.batch_size}, '
            f'max_workers={self.max_workers or "all"}, '
            f'providers={[p for p, _ in self._provider_chain]}')

    # ---------------- setup helpers ----------------
    def _resolve_provider_chain(self):
        """Gemini primary then, if configured + key present, a Qwen fallback
        (dropped with a warning on a missing key rather than failing init)."""
        chain = [('gemini', self.llm_model)]
        fb = self.vlm_fallback_provider
        if fb and fb != 'gemini':
            if fb != 'qwen':
                self.get_logger().warn(f'Unknown fallback provider {fb!r}; ignoring.')
            else:
                try:
                    _, _, model = resolve_qwen_target(
                        self.qwen_api_backend, self.scan_model_qwen)
                    chain.append(('qwen', model))
                except RuntimeError:
                    self.get_logger().warn(
                        'Qwen fallback key missing; fallback disabled.')
        self.get_logger().info(
            f'object_scan provider chain: {[p for p, _ in chain]}')
        return chain

    def _on_image(self, camera, msg):
        with self._lock:
            self._latest[camera] = (msg, time.time())

    def _select_camera(self, req_camera):
        if 'realsense' in (req_camera or ''):
            return 'realsense'
        if 'orbbec' in (req_camera or ''):
            return 'orbbec'
        self.get_logger().warn(
            f'unknown camera {req_camera!r}, defaulting to orbbec')
        return 'orbbec'

    def _recent_frame(self, camera):
        with self._lock:
            entry = self._latest.get(camera)
        if entry is None:
            return None
        msg, t = entry
        if time.time() - t > self.img_sync_thres_s:
            return None
        return msg

    # ---------------- callback ----------------
    def _callback(self, req: ObjectScan.Request, resp: ObjectScan.Response):
        t0 = time.perf_counter()
        resp.header = Header(stamp=self.get_clock().now().to_msg())
        resp.status = 1
        resp.error_msg = ''
        resp.found_labels = []

        camera = self._select_camera(req.camera)
        msg = self._recent_frame(camera)
        if msg is None:
            resp.error_msg = f'No {camera} frame within {self.img_sync_thres_s:.1f}s'
            self.get_logger().warn(resp.error_msg)
            return resp
        resp.header = msg.header

        vocab = [str(v).strip() for v in req.vocabulary if str(v).strip()]
        if not vocab:
            resp.error_msg = 'empty vocabulary'
            return resp

        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            image_url = encode_to_data_url(bgr)
        except Exception as exc:  # noqa: BLE001
            resp.error_msg = f'image encode failed: {exc}'
            self.get_logger().error(resp.error_msg)
            return resp

        groups = _batches(vocab, self.batch_size)
        workers = len(groups) if self.max_workers <= 0 else min(
            self.max_workers, len(groups))

        def _run(batch):
            try:
                r = request_scan_labels_chain(
                    image_url, batch, provider_models=self._provider_chain,
                    qwen_api_backend=self.qwen_api_backend,
                    timeout_s=self.vlm_timeout_s,
                    max_retries=self.vlm_max_retries, logger=self.get_logger())
                return (r.labels, None)
            except ScanVlmError as exc:
                self.get_logger().warn(f'batch {batch} failed: {exc}')
                return ([], str(exc))

        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            results = list(ex.map(_run, groups))

        found_set = set()
        n_fail = 0
        for labels, err in results:
            if err is not None:
                n_fail += 1
            found_set.update(labels)
        found = [c for c in vocab if c in found_set]

        if n_fail == len(groups):
            resp.status = 1
            resp.error_msg = f'all {len(groups)} VLM batches failed'
        else:
            resp.status = 0
            resp.found_labels = found

        dt = time.perf_counter() - t0
        self.get_logger().info(
            f'object_scan: cam={camera} vocab={len(vocab)} '
            f'batches={len(groups)} found={len(found)}/{len(vocab)} '
            f'fail={n_fail} {dt:.2f}s -> {found}')
        self._maybe_log(bgr, camera, vocab, found, results, dt)
        return resp

    # ---------------- debug artifact ----------------
    def _maybe_log(self, bgr, camera, vocab, found, results, dt):
        if not self.vision_logging:
            return
        try:
            import json
            import cv2
            folder = self._resolve_log_dir()
            ts = time.strftime('%Y%m%d_%H%M%S') + f'_{int(time.time() * 1000) % 1000:03d}'
            base = os.path.join(folder, f'object_scan_{camera}_{ts}')
            cv2.imwrite(base + '.jpg', bgr)
            with open(base + '.json', 'w') as f:
                json.dump({
                    'camera': camera, 'vocabulary': vocab, 'found': found,
                    'batches': [
                        {'labels': lbls, 'error': err} for lbls, err in results],
                    'elapsed_s': round(dt, 3),
                }, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().debug(f'debug log skipped: {exc}')

    def _resolve_log_dir(self):
        base = self.vision_log_folder
        session = os.environ.get('TINKER_VISION_SESSION_TS')
        run_dir = os.path.join(base, session) if session else base
        os.makedirs(run_dir, exist_ok=True)
        return run_dir


def main():
    load_env()
    rclpy.init()
    node = ObjectScanServer()
    executor = MultiThreadedExecutor()
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
