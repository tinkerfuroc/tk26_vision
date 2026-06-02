"""Align a prediction stream to GT frames by timestamp.

The tracker emits one ``PredFrame`` per loop tick. Scoring is done per GT frame:
each GT frame is matched to the nearest prediction within a tolerance, or None
when no prediction falls inside the window. Pure function, fully unit-testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .schema import GtFrame


@dataclass
class PredFrame:
    t_ns: int
    target_lost: bool
    target_track_id: int
    point_xyz: Optional[Tuple[float, float, float]]


def align_pred_to_gt(
    preds: List[PredFrame],
    gt_frames: List[GtFrame],
    tol_ms: float = 50.0,
) -> List[Tuple[GtFrame, Optional[PredFrame]]]:
    """Match each GT frame to its nearest prediction within ``tol_ms``.

    For each GT frame (in order), find the prediction with the smallest
    ``|t_ns|`` delta; if that delta is <= ``tol_ms`` (inclusive) the prediction
    is paired, otherwise None. Predictions may be out of order, reused across GT
    frames, and either list may be empty.
    """
    tol_ns = tol_ms * 1e6  # ms -> ns (t_ns is genuine nanoseconds, e.g. rosbag2 stamps)
    result: List[Tuple[GtFrame, Optional[PredFrame]]] = []

    for g in gt_frames:
        best: Optional[PredFrame] = None
        best_delta: Optional[float] = None
        for p in preds:
            delta = abs(p.t_ns - g.t_ns)
            if best_delta is None or delta < best_delta:
                best_delta = delta
                best = p
        if best is not None and best_delta is not None and best_delta <= tol_ns:
            result.append((g, best))
        else:
            result.append((g, None))

    return result
