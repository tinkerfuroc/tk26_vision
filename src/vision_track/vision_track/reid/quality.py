"""Quality gate for appearance-history inserts (gallery hygiene).

Pure functions so they unit-test with synthetic primitives (no model/crop I/O).
Rejects crops that would poison the ReID gallery: too small, poorly segmented,
blurry, or degenerate aspect (a cheap back-view / non-standing proxy).
"""
from typing import Optional

# Default thresholds; overridable from the node param surface later if needed.
DEFAULT_GATE = dict(
    min_crop_h=80,          # px; reject far/tiny detections
    min_mask_coverage=0.4,  # spec: mask_coverage > 0.4
    min_blur_var=50.0,      # Laplacian variance floor (sharpness)
    max_aspect_ratio=0.9,   # w/h; standing person is tall (<~0.6); >0.9 is degenerate
)


def crop_quality_ok(
    crop_h: int,
    crop_w: int,
    mask_coverage: Optional[float],
    blur_var: float,
    *,
    aspect_ratio: float,
    min_crop_h: int,
    min_mask_coverage: float,
    min_blur_var: float,
    max_aspect_ratio: float,
) -> bool:
    if crop_h < min_crop_h or crop_w < 2:
        return False
    # mask_coverage is None when no seg mask is available — don't reject on it then.
    if mask_coverage is not None and mask_coverage <= min_mask_coverage:
        return False
    if blur_var < min_blur_var:
        return False
    if aspect_ratio > max_aspect_ratio:
        return False
    return True
