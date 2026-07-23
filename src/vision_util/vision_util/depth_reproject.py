"""Depth-image reprojection helpers.

Standard pinhole back-projection in the camera's own optical frame
(x=right, y=down, z=forward -- the convention the intrinsics describe, and
the same one `vision_util/_pc_utils.py:build_xy_table_cuda` uses on GPU).
Dimensions are always read from the depth image's own shape, never assumed,
so this works at whatever resolution the camera driver is launched with.

Shared by object_detection_new/object_seg_yolo.py and
vision_util/door_detection.py, which used to carry independent copies of
this reprojection hardcoded to 720x1280 (the Orbbec's old default color
resolution) -- see
docs/superpowers/specs/2026-07-03-orbbec-hri-resolution-bump-design.md.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def decode_depth_metres(depth_arr: np.ndarray) -> np.ndarray:
    """
    Coerce a decoded depth image to float32 metres.

    Orbbec Y16 depth decodes to uint16 millimetres via cv_bridge
    ``passthrough``; FoundationStereo-style depth is already float32 metres.
    """
    if depth_arr.dtype == np.uint16:
        return depth_arr.astype(np.float32) * 0.001
    if depth_arr.dtype == np.float32:
        return depth_arr
    raise ValueError(
        f'Unsupported depth dtype {depth_arr.dtype}; expected uint16 mm or '
        'float32 m.'
    )


def _intrinsics(k) -> Tuple[float, float, float, float]:
    """Return ``fx, fy, cx, cy`` from a CameraInfo K or 3x3 matrix."""
    arr = np.asarray(k)
    if arr.shape == (3, 3):
        return (
            float(arr[0, 0]),
            float(arr[1, 1]),
            float(arr[0, 2]),
            float(arr[1, 2]),
        )
    flat = arr.reshape(-1)
    if flat.size != 9:
        raise ValueError(
            'k must be a 9-element CameraInfo matrix or 3x3 array'
        )
    return float(flat[0]), float(flat[4]), float(flat[2]), float(flat[5])


def _roi_window(
    roi,
    width: int,
    height: int,
    pad: int = 0,
) -> Tuple[int, int, int, int]:
    """Normalize an xyxy tuple or RegionOfInterest-like object."""
    if roi is None:
        return 0, 0, width, height
    if hasattr(roi, 'x_offset'):
        x0 = int(roi.x_offset) - pad
        y0 = int(roi.y_offset) - pad
        x1 = int(roi.x_offset) + int(roi.width) + pad
        y1 = int(roi.y_offset) + int(roi.height) + pad
    else:
        if len(roi) != 4:
            raise ValueError('roi must contain (x0, y0, x1, y1)')
        x0, y0, x1, y1 = (int(value) for value in roi)
        x0 -= pad
        y0 -= pad
        x1 += pad
        y1 += pad
    window = (
        max(0, min(width, x0)),
        max(0, min(height, y0)),
        max(0, min(width, x1)),
        max(0, min(height, y1)),
    )
    if window[2] <= window[0] or window[3] <= window[1]:
        return 0, 0, width, height
    return window


def _clip_depth(
    depth_m: np.ndarray,
    clip: Optional[Sequence[float] | float],
) -> np.ndarray:
    if clip is None:
        return depth_m.astype(np.float32, copy=False)
    if np.isscalar(clip):
        low, high = 0.0, float(clip)
    else:
        if len(clip) != 2:
            raise ValueError(
                'clip must be a maximum or a (minimum, maximum) pair'
            )
        low, high = float(clip[0]), float(clip[1])
    return np.clip(depth_m, low, high).astype(np.float32, copy=False)


def depth_image_to_points(
    depth_m: np.ndarray,
    k,
    *,
    valid_band: Optional[Tuple[float, float]] = None,
    clip: Optional[Sequence[float] | float] = None,
    roi=None,
    roi_pad: int = 0,
    return_valid_mask: bool = False,
):
    """
    Back-project a metres depth image to an (H, W, 3) points array.

    Parameters
    ----------
    depth_m : np.ndarray
        (H, W) depth in metres.
    k : indexable
        9-element row-major camera intrinsic matrix (CameraInfo.k):
        fx = k[0], fy = k[4], cx = k[2], cy = k[5].
    valid_band : (float, float), optional
        Strict lower/upper depth bounds. Used only for the returned mask.
    clip : float or (float, float), optional
        Clip z before reprojection. The validity mask is always computed from
        the original, unclipped depth.
    roi : tuple or RegionOfInterest-like, optional
        Restrict reprojection and validity to an xyxy window. Points outside
        the window are zero.
    roi_pad : int
        Expand the ROI by this many pixels before clamping it to the image.
    return_valid_mask : bool
        Return ``(points, valid_mask)``. The default remains points-only for
        compatibility with the original shared helper.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        The points array, optionally paired with a bool validity mask.

    """
    depth_arr = np.asarray(depth_m)
    if depth_arr.ndim != 2:
        raise ValueError('depth_m must be a two-dimensional array')
    h, w = depth_arr.shape
    fx, fy, cx, cy = _intrinsics(k)
    if fx == 0.0 or fy == 0.0:
        raise ValueError('camera focal lengths must be non-zero')
    us, vs = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    z = _clip_depth(depth_arr, clip)
    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)

    x0, y0, x1, y1 = _roi_window(roi, w, h, pad=int(roi_pad))
    if roi is not None:
        restricted = np.zeros_like(points)
        restricted[y0:y1, x0:x1] = points[y0:y1, x0:x1]
        points = restricted

    if not return_valid_mask:
        return points

    if valid_band is None:
        valid_mask = np.isfinite(depth_arr)
    else:
        low, high = float(valid_band[0]), float(valid_band[1])
        valid_mask = (
            np.isfinite(depth_arr) & (depth_arr > low) & (depth_arr < high)
        )
    if roi is not None:
        roi_mask = np.zeros((h, w), dtype=bool)
        roi_mask[y0:y1, x0:x1] = True
        valid_mask &= roi_mask
    return points, valid_mask


def realsense_body_axes_points(
    depth_m: np.ndarray,
    k,
    *,
    valid_band: Tuple[float, float] = (1e-6, 10.0),
    clip: Optional[Sequence[float] | float] = (0.0, 10.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Preserve YOLO's bug-compatible RealSense body-axis convention.

    This deliberately pairs image rows with ``(cx, fx)`` and columns with
    ``(cy, fy)``. Its centroids feed a matching hand-written body-axis path;
    replacing it with optical-frame pinhole math would change grasp behavior.
    Do not "fix" this function by swapping the axes.
    """
    depth_arr = np.asarray(depth_m)
    if depth_arr.ndim != 2:
        raise ValueError('depth_m must be a two-dimensional array')
    h, w = depth_arr.shape
    fx, fy, cx, cy = _intrinsics(k)
    if fx == 0.0 or fy == 0.0:
        raise ValueError('camera focal lengths must be non-zero')

    valid = np.ones_like(depth_arr)
    valid[depth_arr > float(valid_band[1])] = 0
    valid[depth_arr < float(valid_band[0])] = 0
    z = _clip_depth(depth_arr, clip)
    rows = np.repeat(np.arange(h)[:, None], w, axis=1)
    cols = np.repeat(np.arange(w)[None, :], h, axis=0)
    x = (rows - cx) * depth_arr / fx
    y = (cols - cy) * depth_arr / fy
    points = np.stack([x, y, z], axis=-1)
    return points, valid


def waving_optical_points(
    depth_m: np.ndarray,
    k,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce waving's optical-frame valid-band and clipping policy."""
    return depth_image_to_points(
        depth_m,
        k,
        valid_band=(1e-6, 10.0),
        clip=(0.0, 10.0),
        return_valid_mask=True,
    )


def tracking_optical_points(
    depth_m: np.ndarray,
    k,
    *,
    roi=None,
    roi_pad: int = 16,
    valid_band: Tuple[float, float] = (0.1, 10.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce person-track optical reprojection, optionally in an ROI."""
    return depth_image_to_points(
        depth_m,
        k,
        valid_band=valid_band,
        roi=roi,
        roi_pad=roi_pad,
        return_valid_mask=True,
    )


def follow_head_optical_points(
    depth_m: np.ndarray,
    k,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce follow-head's optical-frame valid-depth policy."""
    return depth_image_to_points(
        depth_m,
        k,
        valid_band=(1e-3, 10.0),
        return_valid_mask=True,
    )
