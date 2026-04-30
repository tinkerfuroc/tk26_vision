"""Shared per-call artifact logger for vision nodes.

Nodes that produce bounding boxes / segmentation masks / centroids instantiate
``VisionLogger`` in ``__init__`` and call ``.write(...)`` from their service /
action callback. Every node in one robot session writes into the same
timestamped subdirectory so artifacts from sibling nodes are co-located.

Session resolution (first hit wins, evaluated lazily on first write):

1. ``$TINKER_VISION_SESSION_TS`` env var (must match ``YYYYmmdd_HHMMSS``).
   Exported by ``src/tk25_basic/src/scripts/master_*.sh`` and the tmux
   dispatchers; tmux child shells inherit it.
2. Newest existing ``<base>/<YYYYmmdd_HHMMSS>/`` subdir by mtime — lets a
   late-spawned standalone node join the active session even when no
   orchestrator stamped the env var.
3. Fresh ``time.strftime`` cold-start (first run on a new machine).

Artifact layout::

    <base>/<YYYYmmdd_HHMMSS>/
        <tag>_<branch>_orig_<YYYYmmdd_HHMMSS_mmm>.jpg
        <tag>_<branch>_overlay_<YYYYmmdd_HHMMSS_mmm>.jpg
        <tag>_<branch>_req_<YYYYmmdd_HHMMSS_mmm>.json

Where ``tag`` is ``node.get_name()`` (sanitized) and ``branch`` is the
per-call tag the caller passes to ``write()`` (``'yolo'``, ``'feature_extraction'``,
``'follow_head'``, …). When ``branch`` is empty the second underscore-segment
is dropped. Auxiliary writers (e.g. ``feature_recognition``'s person crop,
``feature_matching``'s reference image dumps) call :meth:`aux_path` to
get a path under the same run_dir with the same prefix scheme.

``base_folder`` is resolved relative to CWD when not absolute, matching
the convention used across the rest of the tk26_vision tree.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Iterable, Mapping

import cv2
import numpy as np


_SESSION_TS_RE = re.compile(r'^\d{8}_\d{6}$')


def _sanitize_tag(raw: str | None) -> str:
    """Make a node name safe for use as a filename prefix."""
    if not raw:
        return 'unknown'
    cleaned = raw.strip().lstrip('/')
    cleaned = re.sub(r'[\s/]+', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned or 'unknown'


def _json_safe(key: str, value):
    """Coerce a detection-dict value into something json.dump can handle.

    Iterables (list/tuple/ndarray) → list. ROS Point-like objects (have .x .y
    but no __len__) → {x, y, z?} dict so the JSON stays grep-able instead of
    falling through to default=str and producing repr noise.
    """
    if key == 'mask':
        return value
    if hasattr(value, '__iter__') and not isinstance(value, str):
        return list(value)
    if hasattr(value, 'x') and hasattr(value, 'y'):
        out = {'x': float(value.x), 'y': float(value.y)}
        if hasattr(value, 'z'):
            out['z'] = float(value.z)
        return out
    return value


class VisionLogger:
    def __init__(self, node, enabled: bool, base_folder: str, tag: str | None = None):
        self._node = node
        self._enabled = bool(enabled)
        self._base = base_folder or 'vision_log'
        self._run_dir: str | None = None
        derived_tag = tag
        if derived_tag is None and node is not None:
            try:
                derived_tag = node.get_name()
            except Exception:  # noqa: BLE001 — defensive; never crash logging on tag derivation
                derived_tag = None
        self._tag = _sanitize_tag(derived_tag)
        self._malformed_env_warned = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    @property
    def run_dir(self) -> str | None:
        return self._run_dir

    @property
    def tag(self) -> str:
        return self._tag

    def _resolve_run_ts(self) -> str:
        """Layered session resolution: env var → newest-existing → cold-start."""
        env_ts = os.environ.get('TINKER_VISION_SESSION_TS', '').strip()
        if env_ts:
            if _SESSION_TS_RE.match(env_ts):
                return env_ts
            if not self._malformed_env_warned and self._node is not None:
                self._node.get_logger().warn(
                    f'vision_logging: ignoring malformed '
                    f'TINKER_VISION_SESSION_TS={env_ts!r}'
                )
                self._malformed_env_warned = True

        try:
            with os.scandir(self._base) as it:
                candidates = [
                    (entry.stat().st_mtime, entry.name)
                    for entry in it
                    if entry.is_dir() and _SESSION_TS_RE.match(entry.name)
                ]
        except FileNotFoundError:
            candidates = []
        except OSError as exc:
            if self._node is not None:
                self._node.get_logger().warn(
                    f'vision_logging: scandir({self._base!r}) failed: {exc}'
                )
            candidates = []

        if candidates:
            _, newest = max(candidates, key=lambda pair: pair[0])
            if self._node is not None:
                self._node.get_logger().info(
                    f'vision_logging: joining existing session {newest}'
                )
            return newest

        run_ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if self._node is not None:
            self._node.get_logger().info(
                f'vision_logging: starting new session {run_ts}'
            )
        return run_ts

    def _ensure_run_dir(self) -> str:
        if self._run_dir is None:
            run_ts = self._resolve_run_ts()
            self._run_dir = os.path.join(self._base, run_ts)
            os.makedirs(self._run_dir, exist_ok=True)
        return self._run_dir

    def _compose_path(self, run_dir: str, ts: str, kind: str, ext: str,
                      branch: str = '') -> str:
        parts = [self._tag]
        if branch:
            parts.append(branch)
        parts.append(kind)
        stem = '_'.join(parts)
        return os.path.join(run_dir, f'{stem}_{ts}.{ext}')

    def aux_path(self, ts: str, suffix: str, ext: str, branch: str = '') -> str:
        """Compose a path under the session run_dir using the shared prefix
        scheme (``<tag>_<branch>_<suffix>_<ts>.<ext>``). Used by side-file
        writers (e.g. ``feature_recognition``'s ``crop``,
        ``feature_matching``'s ``ref<i>``). Idempotently ensures the run_dir.
        """
        run_dir = self._ensure_run_dir()
        return self._compose_path(run_dir, ts, suffix, ext, branch=branch)

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
            orig_path = self._compose_path(run_dir, ts, 'orig', 'jpg', branch=branch)
            overlay_path = self._compose_path(run_dir, ts, 'overlay', 'jpg', branch=branch)
            req_path = self._compose_path(run_dir, ts, 'req', 'json', branch=branch)

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
                # Pixel-space dot only when centroid is a 2D pixel tuple/list/
                # ndarray. geometry_msgs/Point (3D metric, no __len__) shows up
                # here on the YOLO branch — those values aren't pixel coords,
                # so skip the overlay marker and rely on JSON for the value.
                if centroid is not None and hasattr(centroid, '__len__'):
                    try:
                        if len(centroid) >= 2:
                            cx, cy = int(centroid[0]), int(centroid[1])
                            if 0 <= cx < overlay.shape[1] and 0 <= cy < overlay.shape[0]:
                                cv2.circle(overlay, (cx, cy), 5, (0, 0, 255), -1)
                    except (TypeError, ValueError):
                        pass

                json_dets.append({
                    k: _json_safe(k, v)
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
                'tag': self._tag,
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
