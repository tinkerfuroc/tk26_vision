"""Coordinate decode + 2D scoring geometry, shared by all strategies.

Point decode mirrors kimi_api._seat_vlm._decode_point ([y,x] 0-1000).
Box decode mirrors tk_vision_specialized.qwen_match_vlm._decode_bbox
([x1,y1,x2,y2] 0-1000). Both clamp to image bounds.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]            # (x, y) pixels
Box = Tuple[int, int, int, int]    # (x1, y1, x2, y2) pixels


def decode_point_yx(point_yx, w: int, h: int) -> Optional[Point]:
    if not isinstance(point_yx, (list, tuple)) or len(point_yx) < 2:
        return None
    try:
        y0, x0 = float(point_yx[0]), float(point_yx[1])
    except (TypeError, ValueError):
        return None
    if y0 == 0.0 and x0 == 0.0:
        return None
    px = max(0, min(int(round(x0 * w / 1000.0)), w - 1))
    py = max(0, min(int(round(y0 * h / 1000.0)), h - 1))
    return (px, py)


def decode_box_xyxy(box_2d, w: int, h: int) -> Optional[Box]:
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) < 4:
        return None
    try:
        x1, y1, x2, y2 = (float(box_2d[i]) for i in range(4))
    except (TypeError, ValueError):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    px1 = max(0, min(int(round(x1 * w / 1000.0)), w - 1))
    py1 = max(0, min(int(round(y1 * h / 1000.0)), h - 1))
    px2 = max(0, min(int(round(x2 * w / 1000.0)), w - 1))
    py2 = max(0, min(int(round(y2 * h / 1000.0)), h - 1))
    if px2 <= px1 or py2 <= py1:
        return None
    return (px1, py1, px2, py2)


def point_in_box(pt: Point, box: Box) -> bool:
    x, y = pt
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def box_center(box: Box) -> Point:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


def draw_overlay(
    img_bgr: np.ndarray,
    *,
    point: Optional[Point] = None,
    box: Optional[Box] = None,
    gt_boxes: Optional[list[tuple[Box, bool]]] = None,
    label: str = "",
    hit: Optional[bool] = None,
) -> np.ndarray:
    """Render a result overlay: GT cushions (green=empty, red=occupied),
    predicted box (cyan), predicted point (magenta dot), and a hit/miss tag.
    """
    out = img_bgr.copy()
    if gt_boxes:
        for gbox, occ in gt_boxes:
            color = (0, 0, 200) if occ else (0, 200, 0)
            cv2.rectangle(out, gbox[:2], gbox[2:], color, 2)
    if box is not None:
        cv2.rectangle(out, box[:2], box[2:], (255, 255, 0), 2)
    if point is not None:
        cv2.circle(out, point, 8, (255, 0, 255), -1)
    tag = label
    if hit is not None:
        tag = f"[{'HIT' if hit else 'MISS'}] {label}"
    if tag:
        cv2.putText(out, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return out
