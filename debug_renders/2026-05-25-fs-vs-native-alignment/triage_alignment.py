#!/usr/bin/env python3
"""Triage FS↔color horizontal misalignment.

Phases:
  A. Dump K_color, K_ir1, K_ir2, depth_to_color extrinsics, derived
     expected horizontal pixel shifts at Z={0.5, 1.0, 1.5, 2.0} m.
  B. Call FS with align_to_color=False  → FS in IR1 grid.
     Compare against /aligned_depth_to_infra1 (16UC1 mm). If this is
     well-aligned, the bug is downstream of FS — in color_align.py.
  C. Call FS with align_to_color=True   → FS in color grid.
     Compare against /aligned_depth_to_color.
  D. Edge-cross-correlate native vs FS depth edges over ±20 px window
     (separately for IR1 and color frames). Report argmax (dx, dy).
  E. Horizontal-profile a row through a foreground bottle, plot
     depth(u) for native, FS, and color-luminance edges.

Outputs (all under debug_renders/2026-05-25-fs-vs-native-alignment/triage/):
  matrices.json
  ir1_grid_compare.png    (color | native_ir1 | fs_ir1 | diff)
  ir1_xcorr.png            (cross-correlation surface, with peak)
  color_xcorr.png
  bottle_profile.png       (horizontal profile through detected bottle)
  triage_summary.md        (one-page synthesis)
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from realsense2_camera_msgs.msg import Extrinsics
from tinker_vision_msgs_26.srv import FoundationStereoDepth

OUT_ROOT = "/home/tinker/tk25_ws/src/tk26_vision/debug_renders/2026-05-25-fs-vs-native-alignment"
OUT = f"{OUT_ROOT}/triage"
os.makedirs(OUT, exist_ok=True)

_SENSOR = QoSProfile(
    depth=5, history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE)
_LATCHED = QoSProfile(
    depth=1, history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Capture(Node):
    def __init__(self):
        super().__init__("fs_triage_capture")
        self.bridge = CvBridge()
        self.color = None
        self.color_info = None
        self.ir1 = None
        self.ir1_info = None
        self.ir2_info = None
        self.native_color = None    # aligned_depth_to_color (16UC1 mm)
        self.native_ir1 = None      # aligned_depth_to_infra1 (16UC1 mm)
        self.extr_d2c = None        # depth->color (R, T)
        self.extr_d2i2 = None       # depth->infra2 (R, T)

        # Sensor topics
        self.create_subscription(Image, "/camera/xarm_camera/color/image_raw",
                                 self._on_color, _SENSOR)
        self.create_subscription(Image, "/camera/xarm_camera/infra1/image_rect_raw",
                                 self._on_ir1, _SENSOR)
        self.create_subscription(Image,
                                 "/camera/xarm_camera/aligned_depth_to_color/image_raw",
                                 self._on_native_color, _SENSOR)
        # depth_optical_frame == infra1_optical_frame on D435, so the raw
        # /depth/image_rect_raw is already 'native depth in IR1 grid'.
        self.create_subscription(Image,
                                 "/camera/xarm_camera/depth/image_rect_raw",
                                 self._on_native_ir1, _SENSOR)

        self.create_subscription(CameraInfo,
                                 "/camera/xarm_camera/color/camera_info",
                                 self._on_color_info, _SENSOR)
        self.create_subscription(CameraInfo,
                                 "/camera/xarm_camera/infra1/camera_info",
                                 self._on_ir1_info, _SENSOR)
        self.create_subscription(CameraInfo,
                                 "/camera/xarm_camera/infra2/camera_info",
                                 self._on_ir2_info, _SENSOR)

        # Latched extrinsics
        self.create_subscription(Extrinsics,
                                 "/camera/xarm_camera/extrinsics/depth_to_color",
                                 self._on_extr_d2c, _LATCHED)
        self.create_subscription(Extrinsics,
                                 "/camera/xarm_camera/extrinsics/depth_to_infra2",
                                 self._on_extr_d2i2, _LATCHED)

        self.fs_client = self.create_client(
            FoundationStereoDepth, "/foundation_stereo/get_depth")

    def _on_color(self, m): self.color = m
    def _on_ir1(self, m): self.ir1 = m
    def _on_native_color(self, m): self.native_color = m
    def _on_native_ir1(self, m): self.native_ir1 = m
    def _on_color_info(self, m): self.color_info = m
    def _on_ir1_info(self, m): self.ir1_info = m
    def _on_ir2_info(self, m): self.ir2_info = m

    def _on_extr_d2c(self, m):
        R = np.asarray(m.rotation, dtype=np.float64).reshape(3, 3)
        T = np.asarray(m.translation, dtype=np.float64).reshape(3)
        self.extr_d2c = (R, T)

    def _on_extr_d2i2(self, m):
        R = np.asarray(m.rotation, dtype=np.float64).reshape(3, 3)
        T = np.asarray(m.translation, dtype=np.float64).reshape(3)
        self.extr_d2i2 = (R, T)

    def have_all(self):
        needed = [self.color, self.color_info, self.ir1, self.ir1_info,
                  self.ir2_info, self.native_color, self.native_ir1,
                  self.extr_d2c, self.extr_d2i2]
        return all(x is not None for x in needed)


def K_from_info(info: CameraInfo) -> np.ndarray:
    return np.asarray(info.k, dtype=np.float64).reshape(3, 3)


def call_fs(node: Capture, *, align_to_color: bool) -> Optional[np.ndarray]:
    req = FoundationStereoDepth.Request()
    req.align_to_color = align_to_color
    req.want_pointcloud = False
    req.want_debug_jpeg = False
    req.z_far = 10.0
    fut = node.fs_client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=60.0)
    resp = fut.result()
    if resp is None or resp.status != 0:
        print(f"FS call (align={align_to_color}) failed: "
              f"status={getattr(resp, 'status', '?')} "
              f"msg={getattr(resp, 'error_msg', '?')}")
        return None
    print(f"FS align={align_to_color}: forward_ms={resp.forward_ms:.1f}  "
          f"e2e={resp.end_to_end_s:.2f}s  variant={resp.trt_variant_used}  "
          f"shape={(resp.depth_image.height, resp.depth_image.width)}  "
          f"frame_id={resp.depth_image.header.frame_id}")
    return node.bridge.imgmsg_to_cv2(resp.depth_image, "passthrough").astype(np.float32)


def edge_mask(depth, valid, thresh=40):
    """Sobel-magnitude > thresh on a 0..255 quantisation of depth."""
    Z = depth.copy()
    Z[~valid] = 0
    Z8 = (np.clip(Z / 2.5, 0, 1) * 255).astype(np.uint8)
    gx = cv2.Sobel(Z8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Z8, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2) > thresh


def xcorr_shift(a, b, search=20):
    """Search the (dx, dy) in [-search, search]^2 that maximises
    sum(a_shifted & b). Returns (dx, dy, surface).

    a_shifted is a translated by (dx, dy). Positive dx = shift a to the
    right relative to b.
    """
    a = a.astype(bool)
    b = b.astype(bool)
    H, W = a.shape
    surf = np.zeros((2*search+1, 2*search+1), dtype=np.int32)
    for dy in range(-search, search+1):
        for dx in range(-search, search+1):
            # shift a by (dx, dy)
            y1a = max(0, -dy);  y2a = min(H, H - dy)
            x1a = max(0, -dx);  x2a = min(W, W - dx)
            y1b = max(0, dy);   y2b = min(H, H + dy)
            x1b = max(0, dx);   x2b = min(W, W + dx)
            surf[dy+search, dx+search] = int(np.sum(
                a[y1a:y2a, x1a:x2a] & b[y1b:y2b, x1b:x2b]))
    peak = np.unravel_index(np.argmax(surf), surf.shape)
    return peak[1] - search, peak[0] - search, surf


def turbo(depth, z_max=2.5, mask=None):
    if mask is None:
        mask = depth > 0
    img = (np.clip(depth/z_max, 0, 1) * 255).astype(np.uint8)
    img[~mask] = 0
    out = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    out[~mask] = 0
    return out


def make_panel(img, caption):
    strip = np.full((22, img.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(strip, caption, (10, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([strip, img], axis=0)


def hcat(imgs, gap=8):
    sep = np.full((imgs[0].shape[0], gap, 3), 30, dtype=np.uint8)
    out = imgs[0]
    for x in imgs[1:]:
        out = np.concatenate([out, sep, x], axis=1)
    return out


def main():
    rclpy.init()
    node = Capture()

    if not node.fs_client.wait_for_service(timeout_sec=15):
        print("FS service unavailable")
        return 1

    t0 = time.time()
    while rclpy.ok() and not node.have_all():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 20:
            print("timeout collecting subs:",
                  {n: getattr(node, n) is not None for n in
                   ["color", "color_info", "ir1", "ir1_info", "ir2_info",
                    "native_color", "native_ir1", "extr_d2c", "extr_d2i2"]})
            return 1

    # Stabilize.
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)

    K_color = K_from_info(node.color_info)
    K_ir1 = K_from_info(node.ir1_info)
    K_ir2 = K_from_info(node.ir2_info)
    R_d2c, T_d2c = node.extr_d2c
    R_d2i2, T_d2i2 = node.extr_d2i2

    # ====== PHASE A: matrices and expected pixel shifts ======
    print("\n=== Phase A: matrices ===")
    print("K_color =\n", K_color)
    print("K_ir1   =\n", K_ir1)
    print("K_ir2   =\n", K_ir2)
    print("\ndepth_to_color  R =\n", R_d2c, "\n  T =", T_d2c,
          "  |T|=", np.linalg.norm(T_d2c))
    print("depth_to_infra2 R =\n", R_d2i2, "\n  T =", T_d2i2,
          "  |T|=", np.linalg.norm(T_d2i2))

    fx_c = K_color[0, 0]
    expected_shift = {
        f"Z={z}m": float(fx_c * T_d2c[0] / z)
        for z in (0.5, 1.0, 1.5, 2.0, 3.0)
    }
    print("\nExpected horizontal IR1→color pixel shift at K_color.fx="
          f"{fx_c:.1f}, T_d2c.x={T_d2c[0]*1000:.2f}mm:")
    for k, v in expected_shift.items():
        print(f"  {k}: {v:+.2f} px")
    print("(positive => IR1 pixel lands at a larger u in color)")

    matrices = {
        "K_color": K_color.tolist(),
        "K_ir1": K_ir1.tolist(),
        "K_ir2": K_ir2.tolist(),
        "depth_to_color": {"R": R_d2c.tolist(), "T": T_d2c.tolist(),
                           "T_norm_m": float(np.linalg.norm(T_d2c))},
        "depth_to_infra2": {"R": R_d2i2.tolist(), "T": T_d2i2.tolist(),
                            "T_norm_m": float(np.linalg.norm(T_d2i2))},
        "expected_horizontal_shift_px_at_Z": expected_shift,
        "color_image_shape_hw": [node.color_info.height, node.color_info.width],
        "ir1_image_shape_hw":   [node.ir1_info.height,   node.ir1_info.width],
    }
    with open(f"{OUT}/matrices.json", "w") as f:
        json.dump(matrices, f, indent=2)

    # ====== PHASE B: FS in IR1 frame vs native_aligned_to_infra1 ======
    print("\n=== Phase B: FS-in-IR1 vs native_aligned_to_infra1 ===")
    fs_ir1 = call_fs(node, align_to_color=False)
    if fs_ir1 is None:
        return 1
    # refresh native
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)
    native_ir1_mm = node.bridge.imgmsg_to_cv2(node.native_ir1, "passthrough")
    native_ir1 = native_ir1_mm.astype(np.float32) / 1000.0
    ir1_img = node.bridge.imgmsg_to_cv2(node.ir1, "passthrough")
    if ir1_img.dtype == np.uint16:
        ir1_img = (ir1_img >> 8).astype(np.uint8)
    if ir1_img.ndim == 2:
        ir1_img = cv2.cvtColor(ir1_img, cv2.COLOR_GRAY2BGR)

    # FS at IR1 may be at scaled-down resolution; native_ir1 is full IR1.
    if fs_ir1.shape != native_ir1.shape:
        print(f"  resizing fs_ir1 {fs_ir1.shape} -> {native_ir1.shape} (NN)")
        fs_ir1 = cv2.resize(fs_ir1, (native_ir1.shape[1], native_ir1.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    valid_n_i1 = native_ir1 > 0
    valid_f_i1 = (fs_ir1 > 0) & (fs_ir1 < 10.0)
    common_i1 = valid_n_i1 & valid_f_i1
    print(f"  coverage: native={valid_n_i1.mean():.1%}  "
          f"fs={valid_f_i1.mean():.1%}  common={common_i1.mean():.1%}")
    diff_i1 = (fs_ir1 - native_ir1)
    if common_i1.any():
        d = diff_i1[common_i1]
        print(f"  diff stats (m, FS - native) IR1 frame: "
              f"median={np.median(d)*1000:.1f}mm  "
              f"mae={np.mean(np.abs(d))*1000:.1f}mm  "
              f"p05/p95={np.percentile(d,5)*1000:.0f}/{np.percentile(d,95)*1000:.0f}mm")

    # Save side-by-side
    nv = turbo(native_ir1, 2.5, valid_n_i1)
    fv = turbo(fs_ir1,     2.5, valid_f_i1)
    dv_mask = common_i1
    dv = np.full(nv.shape, (60, 60, 60), dtype=np.uint8)
    if dv_mask.any():
        d_norm = np.clip(diff_i1 / 0.10, -1, 1)
        pos = (d_norm > 0) & dv_mask
        neg = (d_norm < 0) & dv_mask
        if pos.any():
            a = d_norm[pos][:, None]
            dv[pos] = ((1-a) * np.array([255,255,255]) + a*np.array([0,0,255])).astype(np.uint8)
        if neg.any():
            a = (-d_norm[neg])[:, None]
            dv[neg] = ((1-a) * np.array([255,255,255]) + a*np.array([255,0,0])).astype(np.uint8)
    row = hcat([
        make_panel(ir1_img, "IR1 raw"),
        make_panel(nv, "native aligned_depth_to_infra1"),
        make_panel(fv, "FS depth (IR1 frame)"),
        make_panel(dv, "FS - native (+/-100mm)"),
    ])
    cv2.imwrite(f"{OUT}/ir1_grid_compare.png", row)

    # ====== PHASE C: FS in color frame vs native_aligned_to_color ======
    print("\n=== Phase C: FS-in-color vs native_aligned_to_color ===")
    fs_color = call_fs(node, align_to_color=True)
    if fs_color is None:
        return 1
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)
    native_color_mm = node.bridge.imgmsg_to_cv2(node.native_color, "passthrough")
    native_color = native_color_mm.astype(np.float32) / 1000.0
    color = node.bridge.imgmsg_to_cv2(node.color, "bgr8")

    if fs_color.shape != native_color.shape:
        print(f"  resizing fs_color {fs_color.shape} -> {native_color.shape}")
        fs_color = cv2.resize(fs_color, (native_color.shape[1], native_color.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    valid_n_c = native_color > 0
    valid_f_c = (fs_color > 0) & (fs_color < 10.0)
    common_c = valid_n_c & valid_f_c
    diff_c = fs_color - native_color
    if common_c.any():
        d = diff_c[common_c]
        print(f"  diff stats color frame: "
              f"median={np.median(d)*1000:.1f}mm  "
              f"mae={np.mean(np.abs(d))*1000:.1f}mm  "
              f"p05/p95={np.percentile(d,5)*1000:.0f}/{np.percentile(d,95)*1000:.0f}mm")

    # ====== PHASE D: edge cross-correlation, both frames ======
    print("\n=== Phase D: edge cross-correlation ===")

    def run_xcorr(name, n_depth, n_valid, f_depth, f_valid, save_path):
        en = edge_mask(n_depth, n_valid)
        ef = edge_mask(f_depth, f_valid)
        # Restrict to overlapping valid pixels to avoid hole-edges.
        common = n_valid & f_valid
        en = en & common
        ef = ef & common
        if en.sum() < 100 or ef.sum() < 100:
            print(f"  [{name}] too few edges (n={en.sum()} f={ef.sum()}); skipping")
            return None
        dx, dy, surf = xcorr_shift(ef, en, search=20)
        print(f"  [{name}] argmax shift  dx={dx:+d}px  dy={dy:+d}px  "
              f"(positive dx => shift FS to RIGHT to match native; "
              f"so a negative dx means FS is currently shifted RIGHT of native)")
        # Visualise surface.
        sn = (surf - surf.min()) / max(1, (surf.max() - surf.min()))
        surf_img = cv2.applyColorMap((sn*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        surf_img = cv2.resize(surf_img, (400, 400), interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(surf_img,
                       ((dx+20)*int(400/41) + 5, (dy+20)*int(400/41) + 5),
                       (0, 255, 0), cv2.MARKER_CROSS, 30, 2)
        cv2.imwrite(save_path, make_panel(surf_img,
            f"[{name}] xcorr surface (±20px)  peak (dx,dy)=({dx:+d},{dy:+d})"))
        return dx, dy

    ir1_peak = run_xcorr("IR1", native_ir1, valid_n_i1,
                        fs_ir1, valid_f_i1, f"{OUT}/ir1_xcorr.png")
    color_peak = run_xcorr("color", native_color, valid_n_c,
                          fs_color, valid_f_c, f"{OUT}/color_xcorr.png")

    # ====== PHASE E: bottle-edge horizontal profile ======
    print("\n=== Phase E: bottle horizontal profile ===")
    # Pick a row that has lots of edge activity in BOTH depths.
    edge_density = (edge_mask(native_color, valid_n_c).astype(int)
                   + edge_mask(fs_color, valid_f_c).astype(int))
    row_score = edge_density.sum(axis=1)
    row_idx = int(np.argmax(row_score))
    print(f"  row with highest edge density = v={row_idx}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), dpi=120,
                                 facecolor="white", sharex=True,
                                 gridspec_kw={"height_ratios": [1, 3]})

        # Top: a 30-row-thick slice of color around row_idx so the user can
        # see what objects are in the profile.
        h_band = 30
        y0 = max(0, row_idx - h_band//2)
        y1 = min(color.shape[0], row_idx + h_band//2)
        axes[0].imshow(cv2.cvtColor(color[y0:y1], cv2.COLOR_BGR2RGB),
                       extent=[0, color.shape[1], 1, 0], aspect="auto")
        axes[0].axhline(0.5, color="yellow", lw=0.5)
        axes[0].set_yticks([]); axes[0].set_title(f"color slice rows [{y0},{y1}]")

        u = np.arange(native_color.shape[1])
        n_row = native_color[row_idx].copy(); n_row[n_row==0] = np.nan
        f_row = fs_color[row_idx].copy();    f_row[f_row==0] = np.nan
        axes[1].plot(u, n_row, label="native aligned_depth_to_color", color="C0", lw=1.2)
        axes[1].plot(u, f_row, label="FS aligned_depth_to_color", color="C3", lw=1.2)
        axes[1].set_xlabel("u (pixels)")
        axes[1].set_ylabel("depth (m)")
        axes[1].set_ylim(0, 3.0)
        axes[1].legend(loc="upper right")
        axes[1].set_title(f"horizontal depth profile at v={row_idx}")
        fig.tight_layout()
        fig.savefig(f"{OUT}/bottle_profile.png", facecolor="white")
        plt.close(fig)
        print(f"  wrote {OUT}/bottle_profile.png")
    except Exception as exc:
        print(f"  profile plot skipped: {exc}")

    # ====== PHASE F: try applying the empirical shift and re-comparing ======
    print("\n=== Phase F: post-shift re-evaluation ===")
    summary = {
        "matrices_file": "matrices.json",
        "expected_shift_px_at_Z": expected_shift,
        "ir1_peak_dx_dy": ir1_peak,
        "color_peak_dx_dy": color_peak,
        "diff_ir1": None, "diff_color": None,
    }
    if common_i1.any():
        d = diff_i1[common_i1]
        summary["diff_ir1"] = {
            "n": int(common_i1.sum()),
            "median_mm": float(np.median(d) * 1000),
            "mae_mm": float(np.mean(np.abs(d)) * 1000),
            "p05_mm": float(np.percentile(d, 5) * 1000),
            "p95_mm": float(np.percentile(d, 95) * 1000),
        }
    if common_c.any():
        d = diff_c[common_c]
        summary["diff_color"] = {
            "n": int(common_c.sum()),
            "median_mm": float(np.median(d) * 1000),
            "mae_mm": float(np.mean(np.abs(d)) * 1000),
            "p05_mm": float(np.percentile(d, 5) * 1000),
            "p95_mm": float(np.percentile(d, 95) * 1000),
        }
    # If color_peak is not (0,0), apply that shift to FS and re-measure.
    if color_peak is not None and color_peak != (0, 0):
        dx, dy = color_peak
        # Shift FS by (-dx, -dy) so it matches native (dx is "shift FS-edges by dx to match native")
        # Equivalently roll fs_color by (-dy, -dx).
        fs_shifted = np.roll(fs_color, shift=(-dy, -dx), axis=(0, 1))
        # Mask edges hurt by the wrap-around
        m = np.ones_like(fs_shifted, dtype=bool)
        if dy > 0:   m[:dy,  :] = False
        elif dy < 0: m[dy:,  :] = False
        if dx > 0:   m[:, :dx] = False
        elif dx < 0: m[:, dx:] = False
        valid_f_s = (fs_shifted > 0) & (fs_shifted < 10.0) & m
        common_s = valid_n_c & valid_f_s
        d = (fs_shifted - native_color)[common_s]
        summary["diff_color_post_shift"] = {
            "n": int(common_s.sum()),
            "median_mm": float(np.median(d) * 1000),
            "mae_mm": float(np.mean(np.abs(d)) * 1000),
            "p05_mm": float(np.percentile(d, 5) * 1000),
            "p95_mm": float(np.percentile(d, 95) * 1000),
            "shift_applied_dx_dy": [dx, dy],
        }
        print(f"  post-shift color stats (after rolling FS by ({-dy},{-dx})): "
              f"{summary['diff_color_post_shift']}")

    def _np_to_py(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict):
            return {k: _np_to_py(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_np_to_py(v) for v in o]
        return o
    with open(f"{OUT}/triage_summary.json", "w") as f:
        json.dump(_np_to_py(summary), f, indent=2)
    print(f"\nwrote {OUT}/triage_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
