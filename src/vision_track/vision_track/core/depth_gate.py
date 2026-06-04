"""Pure depth-consistency predicate for crosser rejection.

ROS-free + numpy-only. A candidate whose median depth jumps toward the camera
beyond a threshold (relative to the operator's last known depth) is geometrically
a crosser passing between robot and operator — a cue appearance cannot spoof.
"""
import math
from typing import Optional, Tuple

import numpy as np


def is_depth_consistent(
    candidate_depth: float,
    operator_depth: Optional[float],
    jump_threshold: float,
) -> bool:
    """Return True if the candidate is NOT a toward-camera crosser.

    Args:
        candidate_depth: candidate's median depth (m). 0/NaN ⇒ permissive (True).
        operator_depth: operator's last known depth (m). None ⇒ permissive.
        jump_threshold: max allowed toward-camera jump (m). A candidate nearer
            than ``operator_depth - jump_threshold`` is rejected.

    Moving farther than the operator is never a crosser cue (always consistent).
    """
    if operator_depth is None:
        return True
    if candidate_depth is None or candidate_depth <= 0.0 or math.isnan(candidate_depth):
        return True
    # Nearer to the camera by more than the threshold ⇒ crosser ⇒ inconsistent.
    return candidate_depth >= (operator_depth - jump_threshold)


def roi_median_depth(
    depth_mm,
    bbox: Tuple[int, int, int, int],
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Optional[float]:
    """Median valid depth (m) over a bbox ROI of a uint16/float mm depth image.

    Returns None if no pixel is in (min_depth, max_depth).
    """
    depth = np.asarray(depth_mm).astype(np.float32) * 0.001
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth[y1:y2, x1:x2]
    valid = roi[(roi > min_depth) & (roi < max_depth)]
    if valid.size == 0:
        return None
    return float(np.median(valid))
