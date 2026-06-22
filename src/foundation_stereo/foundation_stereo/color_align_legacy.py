"""LEGACY forward-warp IR1→color reprojection. Kept for non-RealSense
sources or as a fallback — the FS node now uses
`color_align_rs2.RealsenseAligner` which wraps librealsense's
`rs.align` via a `software_device` and produces dense sub-pixel-
splatted output without the sparse hole pattern this implementation
leaves behind.

Single pure-numpy entry point: `reproject_ir_to_color`. No GPU, no
extra deps beyond numpy.

Algorithm:
  1. Backproject every valid IR1 pixel to a 3-D point in IR1 frame.
  2. Transform to color frame: P_c = R · P_ir + T.
  3. Project through K_color: (u_c, v_c) = (fx X_c / Z_c + cx, fy Y_c / Z_c + cy).
  4. Round to color pixel grid; np.minimum.at handles occlusion
     (nearer Z wins on collision).
  5. Pixels not hit by any valid projection stay zero (holes).

Known artifact: forward-projection from a lower-resolution IR1 grid into
a higher-resolution color grid leaves ~89% of color pixels as holes.
Downstream consumers must dilate/median-blur to fill them; the sparse
holes adjacent to valid projected pixels also make Sobel edge detection
on the raw output produce spurious gradients everywhere.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def reproject_ir_to_color(
    depth_ir: np.ndarray,
    K_ir: np.ndarray,
    K_color: np.ndarray,
    R_ir_to_color: np.ndarray,
    T_ir_to_color: np.ndarray,
    out_hw: Tuple[int, int],
) -> np.ndarray:
    """Reproject `depth_ir` (m, IR1 grid) into the color camera grid.

    Args:
      depth_ir: (H_ir, W_ir) float32, metres. Zero = invalid.
      K_ir: (3, 3) intrinsics for the IR1 image.
      K_color: (3, 3) intrinsics for the color image.
      R_ir_to_color: (3, 3) rotation IR1 -> color, in the ROS optical
        convention (x right, y down, z forward).
      T_ir_to_color: (3,) translation IR1 -> color, in metres.
      out_hw: (H_color, W_color) — output image shape.

    Returns:
      (H_color, W_color) float32 depth, metres. Zero where nothing
      projected.
    """
    H_ir, W_ir = depth_ir.shape
    H_out, W_out = out_hw

    # 1. Backproject valid pixels.
    fx_ir = float(K_ir[0, 0])
    fy_ir = float(K_ir[1, 1])
    cx_ir = float(K_ir[0, 2])
    cy_ir = float(K_ir[1, 2])

    vv, uu = np.indices((H_ir, W_ir), dtype=np.float32)
    Z = depth_ir
    valid = Z > 0.0
    if not np.any(valid):
        return np.zeros(out_hw, dtype=np.float32)

    X = (uu - cx_ir) * Z / fx_ir
    Y = (vv - cy_ir) * Z / fy_ir
    pts_ir = np.stack([X, Y, Z], axis=-1)  # (H_ir, W_ir, 3)

    # 2. Transform to color frame.
    R = R_ir_to_color.astype(np.float32)
    T = T_ir_to_color.astype(np.float32).reshape(3)
    pts_c = pts_ir @ R.T + T  # (H_ir, W_ir, 3)

    Xc = pts_c[..., 0]
    Yc = pts_c[..., 1]
    Zc = pts_c[..., 2]

    # 3. Project through K_color.
    fx_c = float(K_color[0, 0])
    fy_c = float(K_color[1, 1])
    cx_c = float(K_color[0, 2])
    cy_c = float(K_color[1, 2])

    good = valid & (Zc > 1e-6)
    if not np.any(good):
        return np.zeros(out_hw, dtype=np.float32)

    u_c = fx_c * Xc / np.where(good, Zc, 1.0) + cx_c
    v_c = fy_c * Yc / np.where(good, Zc, 1.0) + cy_c

    ui = np.round(u_c).astype(np.int32)
    vi = np.round(v_c).astype(np.int32)

    in_bounds = good & (ui >= 0) & (ui < W_out) & (vi >= 0) & (vi < H_out)
    if not np.any(in_bounds):
        return np.zeros(out_hw, dtype=np.float32)

    flat_idx = (vi[in_bounds] * W_out + ui[in_bounds]).astype(np.intp)
    z_values = Zc[in_bounds].astype(np.float32)

    # 4. Occlusion: nearer Z wins on collision. Seed with +inf and take min.
    depth_out = np.full(H_out * W_out, np.inf, dtype=np.float32)
    np.minimum.at(depth_out, flat_idx, z_values)

    # 5. Holes left as 0 (rather than inf).
    depth_out[np.isinf(depth_out)] = 0.0
    return depth_out.reshape(H_out, W_out)
