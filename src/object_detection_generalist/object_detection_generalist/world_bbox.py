"""YOLO-World open-vocabulary bounding-box client.

Wraps Ultralytics' `YOLOWorld` so the generalist detection service can
detect arbitrary text-prompted classes locally (no network, GPU latency
~hundreds of ms) before reaching for the slower Gemini VLM fallback.

YOLO-World accepts a class list via `set_classes([...])` which re-projects
its CLIP-derived text embeddings onto the detection head. We re-set classes
on every call because prompts are free-form per request; the embedding
recompute is cheap relative to inference.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np


Bbox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


class WorldDetectorError(RuntimeError):
    """Raised when YOLO-World cannot be loaded (e.g. weights download fail)."""


class WorldDetector:
    """Bbox-only open-vocab detector. Pair with FastSAM for masks."""

    def __init__(
        self,
        weights_path: str,
        device: str,
        conf_threshold: float = 0.05,
        iou_threshold: float = 0.5,
        logger=None,
    ):
        # Import lazily so node startup doesn't pay torch import cost twice
        # (parent class already pays it for the segmentation YOLO).
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:  # pragma: no cover - guard for stripped venv
            raise WorldDetectorError(
                f'ultralytics YOLOWorld unavailable: {exc}'
            ) from exc

        self._device = device
        self._logger = logger
        self._conf_threshold = float(conf_threshold)
        self._iou_threshold = float(iou_threshold)
        self._last_classes: list[str] | None = None

        try:
            self._model = YOLOWorld(weights_path)
        except Exception as exc:  # noqa: BLE001
            raise WorldDetectorError(
                f'failed to load YOLO-World weights "{weights_path}": {exc}'
            ) from exc

        # CRITICAL: move the model to the target device BEFORE the first
        # set_classes() runs. Ultralytics' WorldModel.get_text_pe pulls
        # `device = next(self.model.parameters()).device` and constructs
        # the cached CLIP text-encoder wrapper with that device. The
        # wrapper stores its own `self.device` attribute (used by its
        # `tokenize()` method to put token IDs on the right device), and
        # this attribute is NOT updated by subsequent nn.Module.to(device)
        # calls. So if we let set_classes run while the model is still on
        # CPU, the CLIP wrapper's device attr is permanently pinned to
        # CPU, and the next set_classes will tokenize on CPU but try to
        # use CUDA embedding weights → "Expected all tensors to be on the
        # same device" crash on every call after the first.
        self._model.to(self._device)

        resolved = getattr(self._model, 'ckpt_path', weights_path)
        if self._logger is not None:
            self._logger.info(
                f'YOLO-World loaded from {resolved} on device={device} '
                f'(conf={self._conf_threshold}, iou={self._iou_threshold})'
            )

    def warmup(self, image_hw: tuple[int, int] = (480, 640)) -> float:
        """Force a dummy inference to pay CUDA kernel-compile + CLIP text-head
        allocation costs at node startup.

        Without this, the first real call observes a ~1.7 s overhead on top
        of normal inference latency (measured on RTX 5070 Ti). Returns the
        wall-clock seconds of the warmup pass.
        """
        import time as _t
        h, w = image_hw
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        _t0 = _t.perf_counter()
        # Use a generic placeholder class to populate the CLIP text head.
        # The first real call's set_classes() will overwrite this, but the
        # underlying CUDA kernels for the detection head are now compiled.
        try:
            self._set_classes_on_device(['object'])
            self._last_classes = ['object']
            _ = self._model.predict(
                dummy,
                device=self._device,
                conf=self._conf_threshold,
                iou=self._iou_threshold,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001 — warmup failure is non-fatal
            if self._logger is not None:
                self._logger.warning(
                    f'YOLO-World warmup raised {type(exc).__name__}: {exc} '
                    '(continuing; first user call will pay the cost instead)'
                )
            return _t.perf_counter() - _t0
        elapsed = _t.perf_counter() - _t0
        if self._logger is not None:
            self._logger.info(
                f'YOLO-World warmup: {elapsed * 1000:.0f} ms on '
                f'dummy {w}x{h} input'
            )
        return elapsed

    def _set_classes_on_device(self, classes: list[str]) -> None:
        """Set YOLO-World's open-vocab class list and force everything to
        ``self._device``.

        Why this exists: ``YOLOWorld.set_classes()`` lazily loads CLIP and
        re-encodes the class list into ``model.txt_feats``. ``txt_feats``
        is stored as a *raw tensor attribute* on the inner module — not a
        registered ``Parameter`` or ``Buffer`` — so ``self._model.to(device)``
        does NOT move it. Same story for the cached ``clip_model``: it's
        an attribute on the inner module, loaded onto whatever device CLIP
        defaulted to (usually CPU). The result of leaving them un-pinned
        is the well-known runtime crash at the next forward pass:

            RuntimeError: Expected all tensors to be on the same device,
            but got index is on cpu, different from other tensors on cuda:0
            (when checking argument in method wrapper_CUDA__index_select)

        We re-pin both attributes after every ``set_classes`` so all the
        tensors used by the detection head are guaranteed to live on the
        same device as the model weights.
        """
        self._model.set_classes(classes)
        # Walk both the public and the inner-model wrapper — the attribute
        # may live on either depending on Ultralytics version.
        import torch as _torch
        target_device = _torch.device(self._device)
        for module in (self._model, getattr(self._model, 'model', None)):
            if module is None:
                continue
            txt = getattr(module, 'txt_feats', None)
            if txt is not None and hasattr(txt, 'to'):
                module.txt_feats = txt.to(self._device)
            clip_model = getattr(module, 'clip_model', None)
            if clip_model is not None and hasattr(clip_model, 'to'):
                module.clip_model = clip_model.to(self._device)
                # Re-pin the wrapper's own device attribute (used by its
                # tokenize() method) — nn.Module.to does not touch it.
                if hasattr(module.clip_model, 'device'):
                    module.clip_model.device = target_device
        # Also call the standard Module.to() so any properly registered
        # buffers picked up via set_classes are migrated.
        self._model.to(self._device)

    def detect(
        self,
        rgb_bgr: np.ndarray,
        prompt: str,
    ) -> tuple[List[Bbox], List[float], float]:
        """Return ``(boxes, confs, elapsed_s)`` for instances of ``prompt``.

        ``boxes`` is xyxy pixel coords clipped to the image; ``confs`` is the
        per-box confidence (parallel to ``boxes``). Empty lists on no matches.
        """

        h, w = rgb_bgr.shape[:2]
        classes = [prompt]
        if classes != self._last_classes:
            self._set_classes_on_device(classes)
            self._last_classes = classes

        _t0 = time.perf_counter()
        results = self._model.predict(
            rgb_bgr,
            device=self._device,
            conf=self._conf_threshold,
            iou=self._iou_threshold,
            verbose=False,
        )
        elapsed = time.perf_counter() - _t0

        boxes_out: List[Bbox] = []
        confs_out: List[float] = []
        if not results:
            return boxes_out, confs_out, elapsed

        for result in results:
            boxes = getattr(result, 'boxes', None)
            if boxes is None or boxes.xyxy is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
            for i in range(xyxy.shape[0]):
                x1, y1, x2, y2 = xyxy[i].tolist()
                px1 = max(0, min(int(round(x1)), w - 1))
                py1 = max(0, min(int(round(y1)), h - 1))
                px2 = max(0, min(int(round(x2)), w - 1))
                py2 = max(0, min(int(round(y2)), h - 1))
                if px2 <= px1 or py2 <= py1:
                    continue
                boxes_out.append((px1, py1, px2, py2))
                confs_out.append(float(confs[i]) if confs is not None else 1.0)

        if self._logger is not None:
            self._logger.info(
                f'YOLO-World "{prompt}": {len(boxes_out)} box(es) in '
                f'{elapsed * 1000:.1f} ms'
            )
        return boxes_out, confs_out, elapsed
