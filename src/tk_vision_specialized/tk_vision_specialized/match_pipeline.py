"""Pure-Python orchestrator for object_match_all.

Knows nothing about ROS. Takes a captured scene + depth snapshot plus the
match/judge clients and returns the final list of FinalRow plus a
counters dict. The ROS service callback wraps this in the camera-sync,
TF, and response-packing layers."""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .nms import (
    Bbox, MatchRow, Cluster, JudgePayload,
    suppress_within_category, cluster_for_judge, build_judge_payload,
)


@dataclass(frozen=True)
class PipelineParams:
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


@dataclass
class FinalRow:
    row: MatchRow                    # final label + bbox + conf
    mask: np.ndarray                  # boolean HxW SAM mask
    point_camera: object              # geometry_msgs.Point in camera frame
    point_out: object                 # post-TF point (==camera if no TF)
    tf_failed: bool = False


def _chunks(seq: Sequence, n: int) -> Iterable[list]:
    for i in range(0, len(seq), max(1, n)):
        yield list(seq[i:i + n])


def _rect_mask(shape: tuple[int, int], bbox: Bbox) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    x1, y1, x2, y2 = bbox
    m[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = True
    return m


class MatchPipeline:
    def __init__(
        self, *,
        match_client,
        judge_client,
        sam,
        camera,
        items: dict[str, str],
        params: PipelineParams,
        logger=None,
    ):
        self.match_client = match_client
        self.judge_client = judge_client
        self.sam = sam
        self.camera = camera
        self.items = items
        self.params = params
        self.log = logger

    # ---------------- top-level entry point ----------------
    def run(
        self,
        *,
        scene_bgr: np.ndarray,
        points_xyz: np.ndarray,
        valid_mask: np.ndarray,
        camera: str,
        category_filter: Sequence[str],
        target_frame: str,
        source_frame: str = '',
        header_stamp=None,
    ) -> tuple[list[FinalRow], dict]:
        counters: dict[str, int] = {
            'batches_ok': 0, 'batches_fail': 0,
            'rows_in': 0, 'after_nms': 0,
            'clusters_total': 0, 'clusters_conflict': 0,
            'judge_ok': 0, 'judge_abstain': 0, 'judge_fail': 0,
            'detections_dropped_no_depth': 0,
            'tf_failed': 0,
        }

        # [2] Resolve category filter
        if category_filter:
            keys = [k for k in category_filter if k in self.items]
        else:
            keys = list(self.items.keys())
        if not keys:
            return [], counters

        refs = [(k, self.items[k]) for k in keys]

        # [3] Partition + [4] stage-1 concurrent VLM match
        rows: list[MatchRow] = []
        with ThreadPoolExecutor(max_workers=self.params.max_workers) as pool:
            futures = {
                pool.submit(
                    self.match_client.match_batch,
                    scene_bgr, batch,
                    timeout_s=self.params.vlm_per_call_timeout_s,
                    max_retries=self.params.vlm_max_retries,
                    logger=self.log,
                ): batch
                for batch in _chunks(refs, self.params.batch_size)
            }
            try:
                for fut in as_completed(
                    futures, timeout=self.params.stage1_timeout_s,
                ):
                    try:
                        batch_rows = fut.result()
                        rows.extend(batch_rows)
                        counters['batches_ok'] += 1
                    except Exception as exc:    # noqa: BLE001
                        counters['batches_fail'] += 1
                        if self.log is not None:
                            self.log.warning(
                                f'match batch failed: {exc}'
                            )
            except FutureTimeoutError:
                if self.log is not None:
                    self.log.warning(
                        'stage1 budget elapsed; cancelling stragglers'
                    )
                for fut in futures:
                    if not fut.done():
                        counters['batches_fail'] += 1
                        fut.cancel()

        counters['rows_in'] = len(rows)
        if not rows:
            return [], counters

        # [5] Within-category NMS
        rows = suppress_within_category(
            rows, iou_thresh=self.params.nms_within_category_iou,
        )
        counters['after_nms'] = len(rows)

        # [6] Cross-category clustering
        clusters = cluster_for_judge(
            rows, iou_thresh=self.params.cluster_iou,
        )
        counters['clusters_total'] = len(clusters)
        counters['clusters_conflict'] = sum(
            1 for c in clusters if c.is_conflict()
        )

        # [7] Stage-2 concurrent judge
        survivors = self._resolve_conflicts(clusters, scene_bgr, counters)
        if not survivors:
            return [], counters

        # [9] Batched SAM
        bboxes = [r.bbox for r in survivors]
        masks, _sam_s = self.sam.segment(scene_bgr, bboxes)
        if len(masks) != len(survivors):
            # Defensive: pad with rect masks. SamPredictor contract says
            # 1:1, but a backend swap could change that.
            h, w = scene_bgr.shape[:2]
            while len(masks) < len(survivors):
                masks.append(
                    _rect_mask((h, w), survivors[len(masks)].bbox)
                )

        # [10] Centroids
        finals: list[FinalRow] = []
        for row, mask in zip(survivors, masks):
            pt = self.camera.centroid_for(
                points_xyz, mask, valid_mask, row.bbox, camera,
            )
            if pt is None:
                rect = _rect_mask(scene_bgr.shape[:2], row.bbox)
                pt = self.camera.centroid_for(
                    points_xyz, rect, valid_mask, row.bbox, camera,
                )
            if pt is None:
                counters['detections_dropped_no_depth'] += 1
                if self.log is not None:
                    self.log.warning(
                        f'dropping {row.label}: no valid depth in '
                        f'bbox {row.bbox}'
                    )
                continue
            finals.append(
                FinalRow(
                    row=row, mask=mask,
                    point_camera=pt, point_out=pt,
                )
            )

        if not finals:
            return [], counters

        # [11] Optional TF
        if (
            target_frame
            and self.camera.frame_supports_tf_transform(camera)
        ):
            for fr in finals:
                transformed = self.camera.transform_point(
                    fr.point_camera, target_frame,
                    source_frame, header_stamp,
                )
                if transformed is None:
                    fr.tf_failed = True
                    counters['tf_failed'] += 1
                else:
                    fr.point_out = transformed
            if any(fr.tf_failed for fr in finals):
                # All-or-nothing per spec §11.
                return [], counters

        return finals, counters

    # ---------------- helpers ----------------
    def _resolve_conflicts(
        self,
        clusters: list[Cluster],
        scene_bgr: np.ndarray,
        counters: dict,
    ) -> list[MatchRow]:
        survivors: list[MatchRow] = []
        conflict_payloads: list[JudgePayload] = []

        for cluster in clusters:
            if not cluster.is_conflict():
                survivors.append(
                    max(cluster.rows, key=lambda r: r.conf)
                    if len(cluster.rows) > 1 else cluster.rows[0]
                )
                continue
            payload = build_judge_payload(
                cluster, self.items, scene_bgr,
                self.params.judge_crop_margin_px,
            )
            conflict_payloads.append(payload)

        if not conflict_payloads:
            return survivors

        with ThreadPoolExecutor(
            max_workers=self.params.max_workers,
        ) as pool:
            futures = {
                pool.submit(
                    self.judge_client.choose,
                    p.crop, p.competing,
                    timeout_s=self.params.vlm_per_call_timeout_s,
                    max_retries=self.params.vlm_max_retries,
                    logger=self.log,
                ): p
                for p in conflict_payloads
            }
            try:
                for fut in as_completed(
                    futures, timeout=self.params.stage2_timeout_s,
                ):
                    payload = futures[fut]
                    try:
                        choice = fut.result()
                    except Exception as exc:    # noqa: BLE001
                        if self.log is not None:
                            self.log.warning(
                                f'judge call failed: {exc}'
                            )
                        choice = None
                    survivors.extend(
                        self._row_from_choice(payload, choice, counters)
                    )
            except FutureTimeoutError:
                if self.log is not None:
                    self.log.warning(
                        'stage2 budget elapsed; falling back'
                    )
                for fut, payload in futures.items():
                    if fut.done():
                        continue
                    fut.cancel()
                    survivors.extend(
                        self._row_from_choice(payload, None, counters)
                    )

        return survivors

    def _row_from_choice(
        self,
        payload: JudgePayload,
        choice,
        counters: dict,
    ) -> list[MatchRow]:
        cluster = payload.cluster
        if choice is None:
            counters['judge_fail'] += 1
            return [max(cluster.rows, key=lambda r: r.conf)]

        label = getattr(choice, 'label', '')
        conf = float(getattr(choice, 'conf', 0.0))
        if not label:
            counters['judge_abstain'] += 1
            return []

        cluster_labels = {r.label for r in cluster.rows}
        if label not in cluster_labels:
            counters['judge_fail'] += 1
            return [max(cluster.rows, key=lambda r: r.conf)]

        counters['judge_ok'] += 1
        chosen = max(
            (r for r in cluster.rows if r.label == label),
            key=lambda r: r.conf,
        )
        return [MatchRow(label=chosen.label, bbox=chosen.bbox, conf=conf)]
