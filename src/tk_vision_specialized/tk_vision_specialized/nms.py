"""Pure-function NMS and clustering helpers for object_match_all.

No ROS imports here on purpose: this module is unit-testable from a plain
pytest run without sourcing the workspace. The shapes (`MatchRow`,
`Cluster`, `JudgePayload`) are reused by `match_pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


Bbox = tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


@dataclass(frozen=True)
class MatchRow:
    label: str
    bbox: Bbox
    conf: float


def iou(a: Bbox, b: Bbox) -> float:
    """Standard intersection-over-union on xyxy boxes. 0.0 on zero-area inputs."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    if a_area == 0 or b_area == 0:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(a_area + b_area - inter)


def suppress_within_category(
    rows: Sequence[MatchRow],
    iou_thresh: float,
) -> list[MatchRow]:
    """Greedy NMS, applied within each label independently.

    Same-label boxes that overlap above `iou_thresh` collapse to the higher
    confidence one. Different-label overlaps are preserved (resolved
    elsewhere by the cross-category clusterer + judge)."""

    by_label: dict[str, list[MatchRow]] = {}
    for r in rows:
        by_label.setdefault(r.label, []).append(r)

    kept: list[MatchRow] = []
    for _label, group in by_label.items():
        group.sort(key=lambda r: r.conf, reverse=True)
        survivors: list[MatchRow] = []
        for cand in group:
            if all(iou(cand.bbox, s.bbox) < iou_thresh for s in survivors):
                survivors.append(cand)
        kept.extend(survivors)
    return kept
