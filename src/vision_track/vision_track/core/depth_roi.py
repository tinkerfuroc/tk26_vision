"""ROS-free helper: compute a padded, clamped depth-unproject sub-window.

Today the node unprojects the entire HxW depth frame every tick; only the
target bbox is ever sampled. This computes the (x0,y0,x1,y1) sub-window to
unproject — the bbox padded by ``pad`` px and clamped to the image. A missing
or degenerate bbox falls back to the full frame so the caller never crashes.
"""
from __future__ import annotations

from typing import Optional, Tuple


def roi_window(
    bbox: Optional[Tuple[int, int, int, int]],
    *,
    w: int,
    h: int,
    pad: int = 16,
) -> Tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) of the padded, clamped unproject window."""
    if bbox is None:
        return (0, 0, w, h)
    bx1, by1, bx2, by2 = bbox
    x0 = max(0, int(bx1) - pad)
    y0 = max(0, int(by1) - pad)
    x1 = min(w, int(bx2) + pad)
    y1 = min(h, int(by2) + pad)
    if x1 <= x0 or y1 <= y0:
        return (0, 0, w, h)
    return (x0, y0, x1, y1)
