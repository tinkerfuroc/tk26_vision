"""Vendored slim YOLO-World seat detector for S3 candidate generation.

Logic (esp. the CUDA device-pinning of txt_feats/clip_model) is copied from
object_detection_generalist/world_bbox.py — see that file for the rationale
behind re-pinning after every set_classes(). Kept here so seat_bench has no
cross-package import.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

Bbox = Tuple[int, int, int, int]

SEAT_TERMS = ["chair", "sofa", "couch", "stool", "bench", "armchair"]
_WEIGHTS = "yolov8s-worldv2.pt"   # same default as the generalist node
_CONF = 0.05
_IOU = 0.5

_model = None
_device = None


def _get_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    import torch
    from ultralytics import YOLOWorld
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _model = YOLOWorld(_WEIGHTS)
    _model.to(_device)
    return _model, _device


def _set_classes(model, device, classes):
    import torch
    model.set_classes(classes)
    target = torch.device(device)
    for module in (model, getattr(model, "model", None)):
        if module is None:
            continue
        txt = getattr(module, "txt_feats", None)
        if txt is not None and hasattr(txt, "to"):
            module.txt_feats = txt.to(device)
        clip = getattr(module, "clip_model", None)
        if clip is not None and hasattr(clip, "to"):
            module.clip_model = clip.to(device)
            if hasattr(clip, "device"):
                clip.device = target
    model.to(device)


def _nms(boxes: List[Bbox], scores: List[float], iou_thr: float = 0.6) -> List[Bbox]:
    from .geometry import iou
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: List[Bbox] = []
    used = [False] * len(boxes)
    for idx in order:
        if used[idx]:
            continue
        keep.append(boxes[idx])
        for jdx in order:
            if not used[jdx] and jdx != idx and iou(boxes[idx], boxes[jdx]) > iou_thr:
                used[jdx] = True
        used[idx] = True
    return keep


def detect_seat_candidates(img_bgr: np.ndarray, logger=None) -> tuple[List[Bbox], float]:
    """Return (boxes, elapsed_s): seat-like boxes across SEAT_TERMS, NMS-merged."""
    h, w = img_bgr.shape[:2]
    t0 = time.perf_counter()
    try:
        model, device = _get_model()
        _set_classes(model, device, SEAT_TERMS)
        results = model.predict(img_bgr, device=device, conf=_CONF, iou=_IOU,
                                verbose=False)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger(f"[yolo-world] detect failed: {exc}")
        return [], time.perf_counter() - t0

    boxes: List[Bbox] = []
    scores: List[float] = []
    for r in results or []:
        b = getattr(r, "boxes", None)
        if b is None or b.xyxy is None:
            continue
        xyxy = b.xyxy.cpu().numpy()
        confs = b.conf.cpu().numpy() if b.conf is not None else None
        for i in range(xyxy.shape[0]):
            x1, y1, x2, y2 = xyxy[i].tolist()
            px1 = max(0, min(int(round(x1)), w - 1))
            py1 = max(0, min(int(round(y1)), h - 1))
            px2 = max(0, min(int(round(x2)), w - 1))
            py2 = max(0, min(int(round(y2)), h - 1))
            if px2 <= px1 or py2 <= py1:
                continue
            boxes.append((px1, py1, px2, py2))
            scores.append(float(confs[i]) if confs is not None else 1.0)

    merged = _nms(boxes, scores)
    # Stable left-to-right ordering so the drawn numbers read naturally.
    merged.sort(key=lambda bx: bx[0])
    if logger:
        logger(f"[yolo-world] {len(merged)} seat candidate(s) in "
               f"{(time.perf_counter()-t0)*1000:.0f} ms")
    return merged, time.perf_counter() - t0
