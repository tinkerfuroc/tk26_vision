#!/usr/bin/env python3
"""Compose alignment-verification visuals:

  - side_by_side.png : [color | depth_colormap]
  - overlay.png      : color image with depth pixels alpha-blended (50%)
                       in turbo colormap on top, holes left as pure color
  - overlay_solid.png: color where depth is missing, pure depth colormap
                       where depth is valid (high-contrast)

Reads cloud.ply (in color-optical frame) + color.jpg, reprojects the
cloud points through K_color (hardcoded from the validation run) into a
848x480 depth image, then composites.
"""
import os

import cv2
import numpy as np
import open3d as o3d

OUT = "/home/tinker/tk25_ws/src/tk26_vision/debug_renders/2026-05-24-fs-align-validation"
PLY = f"{OUT}/cloud.ply"
COLOR = f"{OUT}/color.jpg"

# K_color from the FS service-call output during the validation run.
FX, FY = 606.665, 606.776
CX, CY = 429.609, 235.532
W, H = 848, 480

color = cv2.imread(COLOR, cv2.IMREAD_COLOR)
assert color.shape == (H, W, 3), color.shape
color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

pcd = o3d.io.read_point_cloud(PLY)
pts = np.asarray(pcd.points, dtype=np.float32)
print(f"loaded {pts.shape[0]} points from {PLY}")

X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
valid = Z > 1e-6
u = (FX * X / Z + CX).round().astype(np.int32)
v = (FY * Y / Z + CY).round().astype(np.int32)
in_bounds = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
u_ok = u[in_bounds]; v_ok = v[in_bounds]; Z_ok = Z[in_bounds]
print(f"projected to image: {in_bounds.sum()} pixels in-bounds of {pts.shape[0]}")

# Rebuild the depth image with nearer-Z wins on collision.
depth = np.full(H * W, np.inf, dtype=np.float32)
flat_idx = v_ok * W + u_ok
np.minimum.at(depth, flat_idx, Z_ok)
depth[np.isinf(depth)] = 0.0
depth = depth.reshape(H, W)
mask = depth > 0
print(f"depth: valid={mask.sum()}/{mask.size}  "
      f"range={depth[mask].min():.3f}..{depth[mask].max():.3f} m")

# Turbo colormap for the depth (clamped at Z<=1.5 to match Z_MAX from earlier).
Z_MAX = 1.5
Z_MIN = 0.05
norm = np.clip((depth - Z_MIN) / (Z_MAX - Z_MIN), 0, 1)
norm_u8 = (norm * 255).astype(np.uint8)
norm_u8[~mask] = 0
depth_cmap = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)  # BGR
depth_cmap[~mask] = 0  # mute holes to black

# --- Side-by-side ---
gap = np.full((H, 8, 3), 30, dtype=np.uint8)
side_by_side = np.concatenate([color, gap, depth_cmap], axis=1)
# Add a thin caption strip on top
caption = np.full((24, side_by_side.shape[1], 3), 20, dtype=np.uint8)
cv2.putText(caption, "color", (12, 17), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (240, 240, 240), 1, cv2.LINE_AA)
cv2.putText(caption, "FS aligned depth (turbo, 0.05..1.5 m)",
            (W + 20, 17), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (240, 240, 240), 1, cv2.LINE_AA)
side_by_side = np.concatenate([caption, side_by_side], axis=0)
cv2.imwrite(f"{OUT}/side_by_side.png", side_by_side)
print(f"wrote {OUT}/side_by_side.png")

# Dilate the mask + depth colormap by 3x3 so each sparse projected
# pixel becomes a 3x3 block. Makes the overlay visually readable at
# ~11% sparsity. The dilated depth keeps each pixel's original value
# (cv2.dilate on a colormap propagates the brightest neighbor — close
# enough for visualization).
kernel = np.ones((3, 3), np.uint8)
mask_d = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
depth_cmap_d = cv2.dilate(depth_cmap, kernel)

# --- Overlay (alpha-blend) ---
alpha = 0.6
overlay = color.copy()
blended = (alpha * depth_cmap_d.astype(np.float32)
           + (1 - alpha) * color.astype(np.float32)).astype(np.uint8)
overlay[mask_d] = blended[mask_d]
cv2.imwrite(f"{OUT}/overlay.png", overlay)
print(f"wrote {OUT}/overlay.png")

# --- Overlay (solid, high-contrast) ---
overlay_solid = color.copy()
overlay_solid[mask_d] = depth_cmap_d[mask_d]
cv2.imwrite(f"{OUT}/overlay_solid.png", overlay_solid)
print(f"wrote {OUT}/overlay_solid.png")

# --- One-page summary: 2x2 of color, depth, overlay, overlay_solid ---
# Add captions.
def with_caption(img_bgr, text):
    cap = np.full((22, img_bgr.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(cap, text, (10, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([cap, img_bgr], axis=0)

tl = with_caption(color,         "color (848x480)")
tr = with_caption(depth_cmap_d,  "depth (turbo, 0.05..1.5 m, 3x3 dilated)")
bl = with_caption(overlay,       "alpha-blend (depth over color, alpha=0.60)")
br = with_caption(overlay_solid, "solid: color where hole, depth where valid")
sep_v = np.full((tl.shape[0], 8, 3), 30, dtype=np.uint8)
top = np.concatenate([tl, sep_v, tr], axis=1)
bot = np.concatenate([bl, sep_v, br], axis=1)
sep_h = np.full((8, top.shape[1], 3), 30, dtype=np.uint8)
quad = np.concatenate([top, sep_h, bot], axis=0)
cv2.imwrite(f"{OUT}/alignment_quad.png", quad)
print(f"wrote {OUT}/alignment_quad.png")
