"""Pure geometry helpers: torso-band row selection + EMA point smoothing.

ROS-free + numpy-only. Layers on TOP of Phase 0's robust median-x/y + z-outlier
reduction in PersonTrackNode._calculate_centroid: the band restricts which mask
rows feed that reduction, and PointEMA smooths the resulting 3D point across
frames (reset on loss).
"""
from typing import Optional, Tuple


def torso_band_mask(
    bbox: Tuple[int, int, int, int],
    lo: float = 0.15,
    hi: float = 0.55,
    min_rows: int = 4,
) -> Tuple[int, int]:
    """Return absolute (y1_band, y2_band) image rows for the chest band of a bbox.

    Args:
        bbox: (x1, y1, x2, y2). Only the y-range is used.
        lo, hi: band fractions of the bbox height (chest ≈ 0.15..0.55 from top).
        min_rows: if the band is thinner than this, fall back to the full bbox
            y-range (avoids starving the centroid on small/far people).
    """
    _, y1, _, y2 = bbox
    h = y2 - y1
    if h <= 0:
        return y1, y2
    band_y1 = y1 + int(lo * h)
    band_y2 = y1 + int(hi * h)
    if band_y2 - band_y1 < min_rows:
        return y1, y2
    return band_y1, band_y2


class PointEMA:
    """Exponential-moving-average smoother for a 3D point, with reset-on-loss."""

    def __init__(self, alpha: float = 0.5) -> None:
        """alpha in (0,1]: 1.0 = passthrough, lower = smoother/laggier."""
        self.alpha = alpha
        self._state: Optional[Tuple[float, float, float]] = None

    def reset(self) -> None:
        """Drop the smoothed state (call on target loss)."""
        self._state = None

    def update(
        self, point: Optional[Tuple[float, float, float]]
    ) -> Optional[Tuple[float, float, float]]:
        """Blend a new sample; first sample (or first after reset) passes through.

        A None sample returns None and leaves the stored state untouched.
        """
        if point is None:
            return None
        if self._state is None:
            self._state = (float(point[0]), float(point[1]), float(point[2]))
            return self._state
        a = self.alpha
        sx, sy, sz = self._state
        self._state = (
            a * point[0] + (1 - a) * sx,
            a * point[1] + (1 - a) * sy,
            a * point[2] + (1 - a) * sz,
        )
        return self._state
