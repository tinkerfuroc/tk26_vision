"""SAM mask generation for the generalist detection service.

Thin wrapper around Ultralytics' ``SAM`` predictor that takes an RGB image
plus a list of bounding boxes and returns one boolean HxW mask per bbox,
ordered to match the input. Defaults to MobileSAM (``mobile_sam.pt``); the
same wrapper transparently handles ``sam_b.pt`` / ``sam_l.pt`` / ``sam_h.pt``
/ ``sam2_*.pt`` because Ultralytics' ``SAM`` class dispatches on filename.
Instantiated once at node startup so weights + CUDA kernels amortize across
requests.
"""

from __future__ import annotations

import time
from typing import Sequence

import cv2
import numpy as np

from vision_util.mask_utils import largest_connected_component_in_bbox


Bbox = tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords


class SamPredictor:
    """One-shot bbox-prompted segmentation via Ultralytics SAM (MobileSAM)."""

    def __init__(self, weights_path: str, device: str, logger=None):
        # Import lazily so the node can start even if torch/ultralytics fail.
        from ultralytics import SAM

        self._device = device
        self._logger = logger
        self._model = SAM(weights_path)
        resolved = getattr(self._model, 'ckpt_path', weights_path)
        if self._logger is not None:
            self._logger.info(
                f'SAM loaded from {resolved} on device={device}'
            )

    def segment(
        self,
        rgb_bgr: np.ndarray,
        bboxes: Sequence[Bbox],
    ) -> tuple[list[np.ndarray], float]:
        """Return ``(masks, elapsed_s)`` where masks is one bool HxW mask per
        bbox aligned 1:1 with the input, and ``elapsed_s`` is wall-clock
        seconds for the ``model.predict()`` call only."""

        if not bboxes:
            return [], 0.0

        h, w = rgb_bgr.shape[:2]
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        bbox_list = [list(b) for b in bboxes]

        _t0 = time.perf_counter()
        results = self._model.predict(
            source=rgb,
            bboxes=bbox_list,
            device=self._device,
            retina_masks=True,
            verbose=False,
        )
        _sam_elapsed = time.perf_counter() - _t0

        out: list[np.ndarray] = []
        if not results:
            return [np.zeros((h, w), dtype=bool) for _ in bboxes], _sam_elapsed

        masks = getattr(results[0], 'masks', None)
        if masks is None or masks.data is None:
            return [np.zeros((h, w), dtype=bool) for _ in bboxes], _sam_elapsed

        mask_tensor = masks.data  # (N, H', W') torch tensor
        try:
            mask_np = mask_tensor.cpu().numpy()
        except AttributeError:
            mask_np = np.asarray(mask_tensor)

        n_expected = len(bboxes)
        for i in range(n_expected):
            if i < mask_np.shape[0]:
                m = mask_np[i]
                if m.shape != (h, w):
                    m = cv2.resize(
                        m.astype(np.float32),
                        (w, h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                # Defensive post-step: keep the largest connected component
                # inside the prompt bbox so centroid ROI stays aligned with
                # the detector box. MobileSAM masks are usually already a
                # single tight blob; FastSAM frequently returned multi-blob
                # masks which is what made this safety net necessary
                # historically. Cheap insurance against backend swap surprises.
                out.append(
                    largest_connected_component_in_bbox(
                        (m > 0.5).astype(bool), bboxes[i]
                    )
                )
            else:
                # Defensive: Ultralytics' SAM predictor returns N masks for N
                # input bboxes in input order, so this branch should never
                # fire with the current backend. Kept as belt-and-braces in
                # case a future Ultralytics change breaks the contract — the
                # caller depends on len(masks) == len(bboxes).
                out.append(np.zeros((h, w), dtype=bool))
        return out, _sam_elapsed
