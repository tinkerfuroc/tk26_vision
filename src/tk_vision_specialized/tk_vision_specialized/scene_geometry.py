"""ROS-free conservative bbox/depth position estimates (no identity or heading)."""
import numpy as np


def matched_depth_index(stamps_ns, received_times, rgb_stamp_ns, now, max_age, slop):
    """Select only fresh depth with a positive capture stamp close to this RGB."""
    candidates = [i for i, stamp in enumerate(stamps_ns)
                  if stamp > 0 and 0 <= now-received_times[i] <= max_age
                  and abs(stamp-rgb_stamp_ns) <= slop*1e9]
    if not candidates:
        raise ValueError('No fresh depth matched to RGB capture stamp')
    return min(candidates, key=lambda i: abs(stamps_ns[i]-rgb_stamp_ns))


def estimate_position(depth, k, bbox, *, min_depth=0.2, max_depth=8.0,
                      min_pixels=25):
    """Use the central half of the person box and a median-depth inlier band.

    Input depth must be registered to RGB and expressed in metres. A bbox
    cannot guarantee foreground membership; this is an approximate position.
    """
    depth = np.asarray(depth)
    k = np.asarray(k, dtype=float).reshape(3, 3)
    if depth.ndim != 2 or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid depth dimensions or camera intrinsics')
    if not 0 < min_depth < max_depth or min_pixels < 1:
        raise ValueError('Invalid depth range or minimum pixel count')
    h, w = depth.shape
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('Bounding box outside registered depth')
    dx, dy = (x2-x1)//4, (y2-y1)//4
    x1, y1, x2, y2 = x1+dx, y1+dy, x2-dx, y2-dy
    roi = depth[y1:y2, x1:x2]
    valid = np.isfinite(roi) & (roi >= min_depth) & (roi <= max_depth)
    if valid.sum() < min_pixels:
        raise ValueError('Insufficient valid depth in central person ROI')
    median = float(np.median(roi[valid]))
    valid &= np.abs(roi-median) <= max(0.15, 0.1*median)
    if valid.sum() < min_pixels:
        raise ValueError('Insufficient consistent person depth')
    v, u = np.nonzero(valid)
    z = roi[valid]
    xyz = np.array([np.median((u+x1-k[0, 2])*z/k[0, 0]),
                    np.median((v+y1-k[1, 2])*z/k[1, 1]), np.median(z)])
    if not np.isfinite(xyz).all():
        raise ValueError('Non-finite person position')
    return xyz
