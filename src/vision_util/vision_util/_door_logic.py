"""Pure (numpy-only) door open/closed decision from a depth image.

No ROS / cv_bridge imports, so it can be unit-tested with synthetic arrays.
door_detection.py (the ROS node) decodes the depth Image into a metres array
and calls these functions.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class DoorResult:
    is_open: int          # 1 = open, 0 = closed
    valid_count: int      # valid pixels in the center patch
    median_m: float       # median depth of valid pixels (0.0 if none)


def depth_to_meters(arr: np.ndarray, encoding: str) -> np.ndarray:
    """Convert a decoded depth image to a float32 metres array.

    16UC1 / mono16 are millimetres; 32FC1 is already metres. Raises
    ValueError on any other encoding so the caller can report it.
    """
    enc = encoding.lower()
    if enc in ('16uc1', 'mono16'):
        return arr.astype(np.float32) / 1000.0
    if enc == '32fc1':
        return arr.astype(np.float32)
    raise ValueError(f'Unsupported depth encoding: {encoding}')


def evaluate_door(depth_m: np.ndarray, *, open_threshold_m: float,
                  center_patch_px: int, min_valid_px: int) -> DoorResult:
    """Decide door open/closed from a metres depth array.

    Extract the centered center_patch_px x center_patch_px patch (clamped to
    the image bounds), keep finite pixels > 1e-3 m as valid, and take their
    median. Closed (is_open=0) iff valid_count >= min_valid_px AND
    median < open_threshold_m; otherwise open (is_open=1).
    """
    h, w = depth_m.shape[:2]
    half_lo = center_patch_px // 2
    half_hi = center_patch_px - half_lo
    r0 = max(0, h // 2 - half_lo)
    r1 = min(h, h // 2 + half_hi)
    c0 = max(0, w // 2 - half_lo)
    c1 = min(w, w // 2 + half_hi)
    patch = depth_m[r0:r1, c0:c1]

    mask = np.isfinite(patch) & (patch > 1e-3)
    valid_count = int(mask.sum())

    if valid_count < min_valid_px:
        return DoorResult(is_open=1, valid_count=valid_count, median_m=0.0)

    median_m = float(np.median(patch[mask]))
    is_open = 0 if median_m < open_threshold_m else 1
    return DoorResult(is_open=is_open, valid_count=valid_count, median_m=median_m)
