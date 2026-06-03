"""ROS-free operator-selection heuristic for tracker initialization.

At goal start, among class-person detections pick the candidate maximizing a
combined centeredness (bbox-center proximity to image center, normalized) and
nearness (smaller median depth), tie-broken by detection confidence. Replaces
the nondeterministic ``results[0]`` init (yolo_tracker.initialize_tracking).

Assumes the operator starts roughly centered/near — true for "follow me"
framing. Depth is supplied via a callable so callers without a depth image
(e.g. the offline benchmark init) can pass ``lambda bbox: None``; when depth is
unavailable for all candidates the score reduces to centeredness + confidence.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

# Score weights: centeredness dominates, nearness assists, confidence is the
# tie-break (small).
W_CENTER = 1.0
W_NEAR = 0.7
W_CONF = 0.05
# Depth (m) used to normalize nearness; anything >= this scores ~0 nearness.
NEAR_NORM_M = 6.0


def _centeredness(bbox, image_wh) -> float:
    """1.0 at image center, → 0 at the far corner. Normalized by half-diagonal."""
    w, h = image_wh
    cx, cy = w / 2.0, h / 2.0
    bx = (bbox[0] + bbox[2]) / 2.0
    by = (bbox[1] + bbox[3]) / 2.0
    dist = ((bx - cx) ** 2 + (by - cy) ** 2) ** 0.5
    half_diag = ((cx) ** 2 + (cy) ** 2) ** 0.5
    if half_diag <= 0:
        return 0.0
    return max(0.0, 1.0 - dist / half_diag)


def _nearness(depth_m: Optional[float]) -> float:
    """1.0 at 0 m, → 0 at NEAR_NORM_M. None ⇒ 0 (neutral, no depth signal)."""
    if depth_m is None or depth_m <= 0:
        return 0.0
    return max(0.0, 1.0 - depth_m / NEAR_NORM_M)


def select_operator_detection(
    detections: List,
    *,
    image_wh: Tuple[int, int],
    depth_lookup: Callable[[Tuple[int, int, int, int]], Optional[float]],
    target_class: str = "person",
):
    """Pick the best operator candidate, or ``None`` if there are no persons.

    Args:
        detections: objects with ``.bbox`` (x1,y1,x2,y2), ``.confidence``,
            and ``.class_name``.
        image_wh: (width, height) of the color image in pixels.
        depth_lookup: maps a bbox to its median depth in meters, or ``None``.
        target_class: class name to filter to (case-insensitive).

    Returns:
        The chosen detection object, or ``None``.
    """
    persons = [
        d for d in detections
        if getattr(d, "class_name", "").lower() == target_class.lower()
    ]
    if not persons:
        return None

    def score(d) -> float:
        center = _centeredness(d.bbox, image_wh)
        near = _nearness(depth_lookup(d.bbox))
        conf = float(getattr(d, "confidence", 0.0) or 0.0)
        return W_CENTER * center + W_NEAR * near + W_CONF * conf

    return max(persons, key=score)
