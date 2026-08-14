"""
Labels-only batched-VLM scene scanner action.

Given a candidate vocabulary, split it into batches, run vision-LLM calls
with bounded concurrency (Gemini primary, Qwen fallback), and return the
visible subset in deterministic vocabulary order. Labels only: no bounding
boxes, masks, depth, or centroids.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import rclpy
from cv_bridge import CvBridge
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Header
from tinker_vision_msgs_26.action import ObjectScan
from vision_util.action_queue import QueuedActionGate
from vision_util.camera_intake import (
    CameraIntake,
    IntakeConfig,
    StreamSpec,
    configure_camera_backend,
)

from ._env import (
    default_flash_model,
    load_env,
    require_api_key,
    resolve_qwen_target,
)
from ._image_utils import encode_to_data_url
from ._scan_vlm import ScanVlmError, request_scan_labels_chain


FRAME_MAX_AGE_S = 1.0
ACQUIRING_FRAME_DELAY_LIMIT_S = 2.0
CANCEL_STATE_TIMEOUT_S = 0.1
CANCEL_STATE_POLL_S = 0.005
BATCH_WAIT_POLL_S = 0.05


class _GoalCanceled(Exception):
    """Internal control flow for cooperative action cancellation."""


def _batches(vocab, batch_size):
    if batch_size <= 0:
        return [list(vocab)]
    return [
        vocab[i:i + batch_size]
        for i in range(0, len(vocab), batch_size)
    ]


class ObjectScanServer(Node):
    def __init__(self):
        super().__init__('object_scan')

        self.declare_parameter('service_name', 'object_scan')
        self.declare_parameter('llm_model', default_flash_model())
        self.declare_parameter('vlm_fallback_provider', 'qwen')
        self.declare_parameter('scan_model_qwen', '')
        self.declare_parameter('qwen_api_backend', 'dashscope')
        self.declare_parameter('batch_size', 8)
        # 0 = one worker per batch; >0 caps provider concurrency.
        self.declare_parameter('max_workers', 0)
        self.declare_parameter('vlm_timeout_s', 20.0)
        self.declare_parameter('vlm_max_retries', 3)
        self.declare_parameter(
            'orbbec_image_topic',
            '/camera/color/image_raw',
        )
        self.declare_parameter(
            'realsense_image_topic',
            '/camera/xarm_camera/color/image_raw',
        )
        self.declare_parameter('vision_logging_enabled', True)
        self.declare_parameter('vision_log_folder', 'vision_log')

        gp = self.get_parameter
        self.llm_model = gp('llm_model').value
        self.vlm_fallback_provider = gp('vlm_fallback_provider').value
        self.scan_model_qwen = gp('scan_model_qwen').value
        self.qwen_api_backend = gp('qwen_api_backend').value
        self.batch_size = int(gp('batch_size').value)
        self.max_workers = int(gp('max_workers').value)
        self.vlm_timeout_s = float(gp('vlm_timeout_s').value)
        self.vlm_max_retries = int(gp('vlm_max_retries').value)
        self.vision_logging = bool(gp('vision_logging_enabled').value)
        self.vision_log_folder = gp('vision_log_folder').value

        require_api_key()
        self._provider_chain = self._resolve_provider_chain()

        self.action_cb_group = MutuallyExclusiveCallbackGroup()
        self.intake_cb_group = MutuallyExclusiveCallbackGroup()
        self.bridge = CvBridge()
        self._camera_intakes = {
            'orbbec': self._create_color_intake(
                'orbbec',
                gp('orbbec_image_topic').value,
            ),
            'realsense': self._create_color_intake(
                'realsense',
                gp('realsense_image_topic').value,
            ),
        }

        self._action_gate = QueuedActionGate()
        self.object_scan_action = self._create_object_scan_action(
            gp('service_name').value
        )
        self.get_logger().info(
            f'object_scan ready: action={gp("service_name").value}, '
            f'model={self.llm_model}, batch_size={self.batch_size}, '
            f'max_workers={self.max_workers or "all"}, '
            f'providers={[p for p, _ in self._provider_chain]}'
        )

    def _create_color_intake(self, camera: str, image_topic: str):
        cfg = configure_camera_backend(
            self,
            IntakeConfig(
                camera=camera,
                color=StreamSpec(
                    image_topic,
                    best_effort=False,
                    qos_depth=10,
                ),
                age_source='stamp',
            ),
            default_endpoint=(
                '/wrist_camera_server'
                if camera == 'realsense'
                else '/head_camera_server'
            ),
        )
        return CameraIntake(
            self,
            cfg,
            callback_group=self.intake_cb_group,
            bridge=self.bridge,
        )

    def _create_object_scan_action(self, action_name='object_scan'):
        return ActionServer(
            self,
            ObjectScan,
            action_name,
            execute_callback=self.object_scan_execute_callback,
            cancel_callback=self.object_scan_cancel_callback,
            handle_accepted_callback=(
                self.object_scan_handle_accepted_callback
            ),
            callback_group=self.action_cb_group,
            result_timeout=0,
        )

    def _resolve_provider_chain(self):
        """Resolve Gemini primary and an optional Qwen fallback."""
        chain = [('gemini', self.llm_model)]
        fallback = self.vlm_fallback_provider
        if fallback and fallback != 'gemini':
            if fallback != 'qwen':
                self.get_logger().warn(
                    f'Unknown fallback provider {fallback!r}; ignoring.'
                )
            else:
                try:
                    _, _, model = resolve_qwen_target(
                        self.qwen_api_backend,
                        self.scan_model_qwen,
                    )
                    chain.append(('qwen', model))
                except RuntimeError:
                    self.get_logger().warn(
                        'Qwen fallback key missing; fallback disabled.'
                    )
        self.get_logger().info(
            f'object_scan provider chain: {[p for p, _ in chain]}'
        )
        return chain

    def object_scan_handle_accepted_callback(self, goal_handle) -> None:
        """Queue an accepted goal for serialized execution."""
        self._action_gate.accept(goal_handle)

    def object_scan_cancel_callback(self, goal_handle):
        """Accept cancellation and preserve intent for queued goals."""
        self._action_gate.cancel_queued(goal_handle)
        return CancelResponse.ACCEPT

    def _should_cancel(self, goal_handle) -> bool:
        return self._action_gate.should_cancel(goal_handle)

    def _raise_if_canceled(self, goal_handle) -> None:
        if self._should_cancel(goal_handle):
            raise _GoalCanceled

    def _publish_feedback(
        self,
        goal_handle,
        *,
        stage: str,
        message: str,
        delay_limit: float,
        input_frozen: bool = False,
    ) -> None:
        self._raise_if_canceled(goal_handle)
        feedback = ObjectScan.Feedback()
        feedback.status = 0
        feedback.delay_limit = float(delay_limit)
        feedback.stage = stage
        feedback.message = message
        feedback.input_frozen = bool(input_frozen)
        goal_handle.publish_feedback(feedback)

    def _vlm_delay_limit(self) -> float:
        retries = max(0, int(self.vlm_max_retries))
        retry_backoff_s = sum(
            0.5 * (2 ** i)
            for i in range(max(0, retries - 1))
        )
        per_provider_s = (
            retries * max(0.0, float(self.vlm_timeout_s))
            + retry_backoff_s
        )
        return max(
            1.0,
            len(self._provider_chain) * per_provider_s + 5.0,
        )

    def _canceled_result(self, goal_handle):
        result = ObjectScan.Result()
        result.header = Header(stamp=self.get_clock().now().to_msg())
        result.status = 1
        result.error_msg = 'Object scan canceled.'
        result.found_labels = []

        deadline = time.monotonic() + CANCEL_STATE_TIMEOUT_S
        while not bool(
            getattr(goal_handle, 'is_cancel_requested', False)
        ):
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                result.error_msg = (
                    'Object scan cancellation-state error: cancel request '
                    'did not become visible.'
                )
                self.get_logger().error(result.error_msg)
                goal_handle.abort()
                return result
            time.sleep(min(CANCEL_STATE_POLL_S, remaining_s))

        goal_handle.canceled()
        return result

    def object_scan_execute_callback(self, goal_handle):
        """Execute one queued object-scan goal."""
        try:
            self._raise_if_canceled(goal_handle)
            result = self._run_object_scan(goal_handle)
            self._raise_if_canceled(goal_handle)
        except _GoalCanceled:
            return self._canceled_result(goal_handle)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f'Unhandled object scan failure: {exc}')
            result = ObjectScan.Result()
            result.header = Header(stamp=self.get_clock().now().to_msg())
            result.status = 1
            result.error_msg = f'Object scan failed: {exc}.'
            result.found_labels = []
            goal_handle.abort()
            return result
        else:
            goal_handle.succeed()
            return result
        finally:
            self._action_gate.notify_finished(goal_handle)

    def _select_camera(self, requested_camera):
        if 'realsense' in (requested_camera or ''):
            return 'realsense'
        if 'orbbec' in (requested_camera or ''):
            return 'orbbec'
        self.get_logger().warn(
            f'unknown camera {requested_camera!r}, defaulting to orbbec'
        )
        return 'orbbec'

    def _run_object_scan(self, goal_handle):
        started = time.perf_counter()
        request = goal_handle.request
        result = ObjectScan.Result()
        result.header = Header(stamp=self.get_clock().now().to_msg())
        result.status = 1
        result.error_msg = ''
        result.found_labels = []

        camera = self._select_camera(request.camera)
        self._publish_feedback(
            goal_handle,
            stage='acquiring_frame',
            message=f'Acquiring a fresh {camera} color frame.',
            delay_limit=ACQUIRING_FRAME_DELAY_LIMIT_S,
            input_frozen=False,
        )
        frame = self._camera_intakes[camera].latest(
            max_age_s=FRAME_MAX_AGE_S
        )
        if frame is None:
            result.error_msg = (
                f'No {camera} frame within {FRAME_MAX_AGE_S:.1f}s'
            )
            self.get_logger().warn(result.error_msg)
            return result
        result.header = frame.header

        vocabulary = [
            str(value).strip()
            for value in request.vocabulary
            if str(value).strip()
        ]
        if not vocabulary:
            result.error_msg = 'empty vocabulary'
            return result

        try:
            bgr = frame.color_bgr()
            image_url = encode_to_data_url(bgr)
        except Exception as exc:  # noqa: BLE001
            result.error_msg = f'image encode failed: {exc}'
            self.get_logger().error(result.error_msg)
            return result

        self._publish_feedback(
            goal_handle,
            stage='input_frozen',
            message='Color frame captured; no further camera or TF input is required.',
            delay_limit=self._vlm_delay_limit(),
            input_frozen=True,
        )
        groups = _batches(vocabulary, self.batch_size)
        batch_results = self._run_batches(
            goal_handle,
            image_url,
            groups,
        )
        self._publish_feedback(
            goal_handle,
            stage='judging',
            message=f'Combining results from {len(groups)} batches.',
            delay_limit=2.0,
            input_frozen=True,
        )

        found_set = set()
        failed_batches = 0
        for labels, error in batch_results:
            if error is not None:
                failed_batches += 1
            found_set.update(labels)

        seen = set()
        found = []
        for candidate in vocabulary:
            if candidate in found_set and candidate not in seen:
                found.append(candidate)
                seen.add(candidate)

        if failed_batches == len(groups):
            result.error_msg = f'all {len(groups)} VLM batches failed'
        else:
            result.status = 0
            result.found_labels = found

        elapsed_s = time.perf_counter() - started
        self.get_logger().info(
            f'object_scan: cam={camera} vocab={len(vocabulary)} '
            f'batches={len(groups)} found={len(found)}/{len(vocabulary)} '
            f'fail={failed_batches} {elapsed_s:.2f}s -> {found}'
        )
        self._maybe_log(
            bgr,
            camera,
            vocabulary,
            found,
            batch_results,
            elapsed_s,
        )
        return result

    def _run_batches(self, goal_handle, image_url, groups):
        """Run batches with bounded incremental submission."""
        workers = (
            len(groups)
            if self.max_workers <= 0
            else min(self.max_workers, len(groups))
        )
        results = [None] * len(groups)
        pending = {}
        next_index = 0

        def run_batch(index):
            batch = groups[index]
            try:
                scan_result = request_scan_labels_chain(
                    image_url,
                    batch,
                    provider_models=self._provider_chain,
                    qwen_api_backend=self.qwen_api_backend,
                    timeout_s=self.vlm_timeout_s,
                    max_retries=self.vlm_max_retries,
                    logger=self.get_logger(),
                    should_abort=lambda: self._should_cancel(goal_handle),
                )
                return scan_result.labels, None
            except ScanVlmError as exc:
                self.get_logger().warn(
                    f'batch {index + 1}/{len(groups)} {batch} failed: {exc}'
                )
                return [], str(exc)

        executor = ThreadPoolExecutor(max_workers=max(1, workers))
        try:
            while pending or next_index < len(groups):
                self._raise_if_canceled(goal_handle)
                while (
                    next_index < len(groups)
                    and len(pending) < max(1, workers)
                ):
                    self._publish_feedback(
                        goal_handle,
                        stage='vlm_call',
                        message=(
                            f'Scanning batch {next_index + 1}/'
                            f'{len(groups)}.'
                        ),
                        delay_limit=self._vlm_delay_limit(),
                        input_frozen=True,
                    )
                    future = executor.submit(run_batch, next_index)
                    pending[future] = next_index
                    next_index += 1
                    self._raise_if_canceled(goal_handle)

                if not pending:
                    continue
                done, _ = wait(
                    pending,
                    timeout=BATCH_WAIT_POLL_S,
                    return_when=FIRST_COMPLETED,
                )
                self._raise_if_canceled(goal_handle)
                for future in done:
                    index = pending.pop(future)
                    results[index] = future.result()
        finally:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)

        return results

    def _maybe_log(
        self,
        bgr,
        camera,
        vocabulary,
        found,
        results,
        elapsed_s,
    ):
        if not self.vision_logging:
            return
        try:
            import json

            import cv2

            folder = self._resolve_log_dir()
            stamp = (
                time.strftime('%Y%m%d_%H%M%S')
                + f'_{int(time.time() * 1000) % 1000:03d}'
            )
            base = os.path.join(folder, f'object_scan_{camera}_{stamp}')
            cv2.imwrite(base + '.jpg', bgr)
            with open(base + '.json', 'w') as file:
                json.dump({
                    'camera': camera,
                    'vocabulary': vocabulary,
                    'found': found,
                    'batches': [
                        {'labels': labels, 'error': error}
                        for labels, error in results
                    ],
                    'elapsed_s': round(elapsed_s, 3),
                }, file, indent=2)
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
