"""Shared per-call artifact logger for vision nodes.

Nodes that produce bounding boxes / segmentation masks / centroids instantiate
``VisionLogger`` in ``__init__`` and call ``.write(...)`` from their service /
action callback. A process-scoped timestamped subdirectory is created lazily
under ``base_folder`` on the first successful write.

Artifact layout::

    <base_folder>/<YYYYmmdd_HHMMSS>/
        orig_<YYYYmmdd_HHMMSS_mmm>.jpg        # unannotated BGR frame
        overlay_<YYYYmmdd_HHMMSS_mmm>.jpg     # bbox + mask tint + centroid dot
        req_<YYYYmmdd_HHMMSS_mmm>.json        # request context + detections

``base_folder`` is resolved relative to CWD if not absolute, matching the
convention used across the rest of the tk26_vision tree.
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterable, Mapping

import cv2
import numpy as np


class VisionLogger:
    def __init__(self, node, enabled: bool, base_folder: str):
        self._node = node
        self._enabled = bool(enabled)
        self._base = base_folder or 'vision_log'
        self._run_dir: str | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    @property
    def run_dir(self) -> str | None:
        return self._run_dir

    def _ensure_run_dir(self) -> str:
        if self._run_dir is None:
            run_ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            self._run_dir = os.path.join(self._base, run_ts)
        if not os.path.exists(self._run_dir):
            os.makedirs(self._run_dir, exist_ok=True)
        return self._run_dir

    def write(
        self,
        rgb_img: np.ndarray,
        detections: Iterable[Mapping] | None,
        request_ctx: Mapping | None = None,
        branch: str = '',
        extras: Mapping | None = None,
        timings: dict[str, float] | None = None,
    ) -> str | None:
        """Dump orig + overlay + JSON. Returns the artifact timestamp stem,
        or ``None`` when disabled / no image.

        ``detections`` entries may carry any subset of: ``bbox`` (x1,y1,x2,y2),
        ``mask`` (bool HxW ndarray), ``cls_name``, ``conf``, ``centroid``
        (x,y) or (x,y,z). Unknown keys are ignored; missing keys are fine.

        ``timings`` — dict mapping label → wall-clock seconds, e.g.
        ``{'vlm': 8.2, 'sam': 0.23}`` or ``{'yolo': 0.047}``. Each entry is
        rendered as a separate line at the bottom-left of the overlay and
        included in the JSON under ``"timings"``.
        """
        if not self._enabled:
            return None
        if rgb_img is None:
            return None

        try:
            run_dir = self._ensure_run_dir()
            ts = (
                time.strftime('%Y%m%d_%H%M%S', time.localtime())
                + f'_{int(time.time() * 1000) % 1000:03d}'
            )
            orig_path = os.path.join(run_dir, f'orig_{ts}.jpg')
            overlay_path = os.path.join(run_dir, f'overlay_{ts}.jpg')
            req_path = os.path.join(run_dir, f'req_{ts}.json')

            cv2.imwrite(orig_path, rgb_img)

            overlay = rgb_img.copy()
            dets_list = list(detections or [])
            json_dets = []
            for det in dets_list:
                bbox = det.get('bbox')
                if bbox is not None:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_bits = []
                    if det.get('cls_name'):
                        label_bits.append(str(det['cls_name']))
                    if det.get('conf') is not None:
                        label_bits.append(f"{float(det['conf']):.2f}")
                    if label_bits:
                        cv2.putText(
                            overlay, ' '.join(label_bits),
                            (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                        )

                mask = det.get('mask')
                if mask is not None and getattr(mask, 'shape', None) is not None:
                    try:
                        # Direct 50% blend on masked pixels — avoids the
                        # double-blend that produces only ~15% orange contribution.
                        overlay[mask] = (
                            overlay[mask].astype(np.float32) * 0.5
                            + np.array([0, 160, 255], dtype=np.float32) * 0.5
                        ).astype(np.uint8)
                    except Exception as _mask_exc:  # noqa: BLE001
                        if self._node is not None:
                            self._node.get_logger().warn(
                                f'vision_logging: mask render failed: {_mask_exc}'
                            )

                centroid = det.get('centroid')
                if centroid is not None and len(centroid) >= 2:
                    cx, cy = int(centroid[0]), int(centroid[1])
                    if 0 <= cx < overlay.shape[1] and 0 <= cy < overlay.shape[0]:
                        cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)

                json_dets.append({
                    k: (list(v) if hasattr(v, '__iter__') and not isinstance(v, str)
                        and k != 'mask' else v)
                    for k, v in det.items()
                    if k != 'mask'
                })

            # Timing labels — one line per entry, stacked upward from the bottom.
            if timings:
                h_img = overlay.shape[0]
                line_h = 22  # pixels per line
                for idx, (key, val) in enumerate(reversed(list(timings.items()))):
                    ms = val * 1000.0
                    label = f'{key}: {ms:.0f} ms' if ms < 1000 else f'{key}: {val:.2f} s'
                    y = h_img - 8 - idx * line_h
                    cv2.putText(
                        overlay, label, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
                    )
                    cv2.putText(
                        overlay, label, (6, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                    )

            cv2.imwrite(overlay_path, overlay)

            payload = {
                'branch': branch,
                'request': dict(request_ctx or {}),
                'n_detections': len(dets_list),
                'detections': json_dets,
            }
            if timings:
                payload['timings'] = {k: round(v, 4) for k, v in timings.items()}
            if extras:
                payload.update(dict(extras))

            with open(req_path, 'w') as fp:
                json.dump(payload, fp, indent=2, default=str)

            if self._node is not None:
                self._node.get_logger().info(
                    f'vision_logging: wrote {orig_path}, {overlay_path}, {req_path}'
                )
            return ts
        except Exception as exc:  # noqa: BLE001
            if self._node is not None:
                self._node.get_logger().warn(f'vision_logging failed: {exc}')
            return None
