#!/usr/bin/env python3
"""Phase G: focused horizontal-shift estimation.

Three complementary methods (all on the same captured frame):

  G1. MSE-vs-dx sweep. For each dx in [-15, +15], shift FS_color by dx
      pixels horizontally, compute MAE over common-valid pixels, plot
      the curve. Argmin gives the integer best-fit shift; parabolic fit
      around the argmin gives sub-pixel resolution.

  G2. Vertical-edge-only xcorr. Sobel-x only (no Sobel-y), so the score
      only depends on horizontal placement. This isolates dx from dy and
      eliminates the table/floor-edge horizontal bias.

  G3. Per-ROI xcorr. Crop a ~50x50 patch around a vertical depth edge
      in the scene (auto-detected), and run xcorr there. Reports per-ROI
      argmax — if the bug is "small shift, drowned by other content",
      ROI tests will show it.

  G4. FS-color vs color-image cross-modal edge xcorr. Sobel of color
      image vs FS depth gradient. If FS_color is shifted N pixels right
      of where it should be, this xcorr peaks at dx = -N.
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Tuple

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
OUT = f"{OUT_ROOT}/triage_g"
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
        super().__init__("fs_triage_g")
        self.bridge = CvBridge()
        self.color = None
        self.color_info = None
        self.native_color = None
        self.native_ir1 = None

        self.create_subscription(Image, "/camera/xarm_camera/color/image_raw",
                                 self._set_color, _SENSOR)
        self.create_subscription(Image,
                                 "/camera/xarm_camera/aligned_depth_to_color/image_raw",
                                 self._set_native_color, _SENSOR)
        self.create_subscription(Image,
                                 "/camera/xarm_camera/depth/image_rect_raw",
                                 self._set_native_ir1, _SENSOR)
        self.create_subscription(CameraInfo,
                                 "/camera/xarm_camera/color/camera_info",
                                 self._set_color_info, _SENSOR)
        self.fs_client = self.create_client(
            FoundationStereoDepth, "/foundation_stereo/get_depth")

    def _set_color(self, m): self.color = m
    def _set_native_color(self, m): self.native_color = m
    def _set_native_ir1(self, m): self.native_ir1 = m
    def _set_color_info(self, m): self.color_info = m

    def have_all(self):
        return all(x is not None for x in (
            self.color, self.color_info, self.native_color, self.native_ir1))


def fetch_fs(node, *, align_to_color, scale=0.0):
    req = FoundationStereoDepth.Request()
    req.align_to_color = align_to_color
    req.want_pointcloud = False
    req.want_debug_jpeg = False
    req.z_far = 10.0
    req.scale = scale
    fut = node.fs_client.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=60.0)
    resp = fut.result()
    if resp is None or resp.status != 0:
        print(f"  FS call (align={align_to_color}, scale={scale}) failed: "
              f"{getattr(resp, 'error_msg', '?')}")
        return None, None
    return node.bridge.imgmsg_to_cv2(resp.depth_image,
                                     "passthrough").astype(np.float32), resp


def shift_image(img, dx, dy):
    """Shift `img` by (dx, dy) using cv2.warpAffine (zeros for out-of-frame)."""
    H, W = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def mae_at_shift(fs, native, dx, dy):
    """MAE of (FS shifted by (dx, dy)) - native, on common-valid pixels."""
    fs_s = shift_image(fs, dx, dy)
    valid_n = native > 0
    valid_f = (fs_s > 0) & (fs_s < 10.0)
    common = valid_n & valid_f
    if not common.any():
        return float("nan"), 0
    d = (fs_s - native)[common]
    return float(np.mean(np.abs(d))), int(common.sum())


def sweep_dx_mae(fs, native, dxs=range(-15, 16)):
    """Return (dx_array, mae_array, n_array) sweep over dxs at dy=0."""
    dx_a = list(dxs)
    mae_a = []
    n_a = []
    for dx in dx_a:
        m, n = mae_at_shift(fs, native, dx, 0)
        mae_a.append(m)
        n_a.append(n)
    return np.array(dx_a), np.array(mae_a), np.array(n_a)


def parabola_min(x, y):
    """Parabolic interpolation around discrete argmin to get sub-pixel."""
    i = int(np.argmin(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i])
    a = y[i-1]; b = y[i]; c = y[i+1]
    denom = (a - 2*b + c)
    if abs(denom) < 1e-9:
        return float(x[i])
    return float(x[i]) + 0.5 * (a - c) / denom


def vertical_edge_mask(depth, valid, thresh=15):
    Z = depth.copy(); Z[~valid] = 0
    Z8 = (np.clip(Z / 2.5, 0, 1) * 255).astype(np.uint8)
    gx = cv2.Sobel(Z8, cv2.CV_32F, 1, 0, ksize=3)
    return np.abs(gx) > thresh


def xcorr1d_dx(a_mask, b_mask, search=20):
    """1-D dx-only cross-correlation (sum of pixel agreements as a function of dx)."""
    H, W = a_mask.shape
    out = np.zeros(2*search+1, dtype=np.int64)
    for dx in range(-search, search+1):
        x1a = max(0, -dx); x2a = min(W, W - dx)
        x1b = max(0, dx);  x2b = min(W, W + dx)
        out[dx+search] = int(np.sum(a_mask[:, x1a:x2a] & b_mask[:, x1b:x2b]))
    return np.arange(-search, search+1), out


def color_grad_mag(color_bgr, thresh=30):
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    return np.abs(gx) > thresh


def main():
    rclpy.init()
    node = Capture()
    if not node.fs_client.wait_for_service(timeout_sec=15):
        print("FS service not available"); return 1
    t0 = time.time()
    while rclpy.ok() and not node.have_all():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 20:
            print("timeout"); return 1
    for _ in range(15):
        rclpy.spin_once(node, timeout_sec=0.05)

    color = node.bridge.imgmsg_to_cv2(node.color, "bgr8")
    native_color_mm = node.bridge.imgmsg_to_cv2(node.native_color, "passthrough")
    native_color = native_color_mm.astype(np.float32) / 1000.0
    native_ir1_mm = node.bridge.imgmsg_to_cv2(node.native_ir1, "passthrough")
    native_ir1 = native_ir1_mm.astype(np.float32) / 1000.0

    H, W = color.shape[:2]
    print(f"color={color.shape} native_color={native_color.shape} native_ir1={native_ir1.shape}")

    # ---------- FS in color frame (default scale) ----------
    print("\n--- FS align=color, scale=default ---")
    fs_color, resp_c = fetch_fs(node, align_to_color=True, scale=0.0)
    if fs_color is None: return 1
    if fs_color.shape != (H, W):
        fs_color = cv2.resize(fs_color, (W, H), interpolation=cv2.INTER_NEAREST)
    print(f"  fs_color shape={fs_color.shape}")

    # ---------- FS in IR1 frame (default scale) ----------
    print("\n--- FS align=ir1, scale=default ---")
    fs_ir1, resp_i = fetch_fs(node, align_to_color=False, scale=0.0)
    if fs_ir1 is None: return 1
    if fs_ir1.shape != native_ir1.shape:
        fs_ir1_up = cv2.resize(fs_ir1, (native_ir1.shape[1], native_ir1.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    else:
        fs_ir1_up = fs_ir1
    print(f"  fs_ir1 shape={fs_ir1.shape} upsampled={fs_ir1_up.shape}")

    # ---------- FS in IR1 frame at SCALE=1.0 (no internal downsample) ----------
    print("\n--- FS align=ir1, scale=1.0 ---")
    fs_ir1_full, resp_i1 = fetch_fs(node, align_to_color=False, scale=1.0)
    if fs_ir1_full is not None:
        print(f"  fs_ir1_full shape={fs_ir1_full.shape}")
        if fs_ir1_full.shape != native_ir1.shape:
            fs_ir1_full = cv2.resize(fs_ir1_full,
                                     (native_ir1.shape[1], native_ir1.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
    # refresh native after the long call
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)

    # ====== G1: MAE-vs-dx sweep ======
    print("\n=== G1: MAE-vs-dx sweep ===")
    plots = {}
    for name, (fs_arr, native_arr) in [
            ("color",            (fs_color, native_color)),
            ("ir1 (scale=def)",  (fs_ir1_up, native_ir1)),
    ] + ([("ir1 (scale=1.0)", (fs_ir1_full, native_ir1))]
         if fs_ir1_full is not None else []):
        dx, mae, n = sweep_dx_mae(fs_arr, native_arr,
                                   dxs=range(-15, 16))
        dx_argmin = int(dx[np.argmin(mae)])
        dx_sub = parabola_min(dx, mae)
        print(f"  [{name}] argmin dx_int={dx_argmin:+d}  "
              f"sub-pixel dx={dx_sub:+.2f}px  "
              f"min_mae={np.min(mae)*1000:.1f}mm  "
              f"mae@0={mae[15]*1000:.1f}mm")
        plots[name] = {"dx": dx.tolist(),
                       "mae_m": mae.tolist(),
                       "n": n.tolist(),
                       "dx_argmin_int": dx_argmin,
                       "dx_argmin_sub": dx_sub}

    # Combined plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120, facecolor="white")
        for name, p in plots.items():
            mae_mm = np.array(p["mae_m"]) * 1000
            ax.plot(p["dx"], mae_mm,
                    marker="o", lw=1.2, label=f"{name} (argmin {p['dx_argmin_int']:+d}, sub {p['dx_argmin_sub']:+.2f})")
            ax.axvline(p["dx_argmin_sub"], lw=0.5, alpha=0.6)
        ax.set_xlabel("horizontal shift of FS (px) — positive = shift FS to the RIGHT")
        ax.set_ylabel("MAE on common-valid pixels (mm)")
        ax.set_title("G1: how much do we need to shift FS to best-fit native?")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.axvline(0, color="k", lw=0.8)
        fig.tight_layout()
        fig.savefig(f"{OUT}/g1_dx_sweep.png", facecolor="white")
        plt.close(fig)
        print(f"  wrote {OUT}/g1_dx_sweep.png")
    except Exception as exc:
        print(f"  plot skipped: {exc}")

    # ====== G2: vertical-edge-only xcorr ======
    print("\n=== G2: vertical-edge-only xcorr ===")
    for name, fs_arr, n_arr in [
            ("color",           fs_color,   native_color),
            ("ir1 (scale=def)", fs_ir1_up,  native_ir1),
    ] + ([("ir1 (scale=1.0)", fs_ir1_full, native_ir1)]
         if fs_ir1_full is not None else []):
        vn = (n_arr > 0)
        vf = (fs_arr > 0) & (fs_arr < 10.0)
        en = vertical_edge_mask(n_arr, vn) & vn & vf
        ef = vertical_edge_mask(fs_arr, vf) & vn & vf
        if en.sum() < 100 or ef.sum() < 100:
            print(f"  [{name}] too few edges, skipping"); continue
        dx, xc = xcorr1d_dx(ef, en, search=20)
        peak = int(dx[np.argmax(xc)])
        # parabolic sub-pixel
        i = int(np.argmax(xc))
        if 0 < i < len(xc) - 1:
            a, b, c = float(xc[i-1]), float(xc[i]), float(xc[i+1])
            sub = float(dx[i]) + 0.5*(a - c) / max((a - 2*b + c), 1e-9)
        else:
            sub = float(peak)
        print(f"  [{name}] vertical-edge xcorr peak dx={peak:+d}  sub={sub:+.2f}px  "
              f"(peak={xc[i]}, ratio_to_dx=0: {xc[i]/max(xc[20],1):.2f})")

    # ====== G3: per-ROI bottle xcorr ======
    print("\n=== G3: per-ROI bottle xcorr ===")
    # Auto-detect ROI: cluster vertical edges in fs_color and pick the
    # most prominent ones.
    vn = native_color > 0
    vf = (fs_color > 0) & (fs_color < 10.0)
    common = vn & vf
    ev = vertical_edge_mask(fs_color, vf) & common
    en = vertical_edge_mask(native_color, vn) & common
    # Use a 200-row band in the middle of the image so we don't include
    # the floor edge.
    mid_band = np.zeros_like(common)
    mid_band[max(0, H//2 - 100): H//2 + 100, :] = True
    ev = ev & mid_band
    en = en & mid_band
    # Find columns with high edge counts (sum over rows).
    cols = ev.sum(axis=0)
    # Pick the top-3 isolated peaks at least 50 px apart.
    rois = []
    cols_smooth = cv2.GaussianBlur(cols.astype(np.float32)[None, :],
                                    (1, 21), 0).ravel()
    used = np.zeros(W, dtype=bool)
    for _ in range(4):
        c = int(np.argmax(cols_smooth * (~used)))
        if cols_smooth[c] < 4: break
        # ROI: 60 px wide around c, full vertical extent of mid_band
        rois.append((max(0, c-30), min(W, c+30), H//2-100, H//2+100, int(c)))
        used[max(0, c-50): min(W, c+50)] = True
        cols_smooth[max(0, c-50): min(W, c+50)] = 0
    for (x0, x1, y0, y1, c_peak) in rois:
        en_roi = en[y0:y1, x0:x1]
        ev_roi = ev[y0:y1, x0:x1]
        if en_roi.sum() < 20 or ev_roi.sum() < 20:
            print(f"  ROI c={c_peak}: too few edges (n={en_roi.sum()} f={ev_roi.sum()}); skipping")
            continue
        dxs, xc = xcorr1d_dx(ev_roi, en_roi, search=10)
        peak = int(dxs[np.argmax(xc)])
        i = int(np.argmax(xc))
        if 0 < i < len(xc) - 1:
            a, b, c2 = float(xc[i-1]), float(xc[i]), float(xc[i+1])
            sub = float(dxs[i]) + 0.5*(a - c2) / max((a - 2*b + c2), 1e-9)
        else:
            sub = float(peak)
        print(f"  ROI c={c_peak} (x={x0}..{x1}): "
              f"peak dx={peak:+d}  sub={sub:+.2f}px  "
              f"(n_native={int(en_roi.sum())} n_fs={int(ev_roi.sum())})")

    # ====== G4: FS depth vs color image (cross-modal) ======
    print("\n=== G4: FS-color vs color-image grad ===")
    color_edges = color_grad_mag(color)
    for name, depth_arr in [
            ("native_color", native_color),
            ("fs_color",     fs_color),
    ]:
        vd = depth_arr > 0
        ed = vertical_edge_mask(depth_arr, vd) & vd
        # Cross-modal: how much does shifting depth edges right by dx
        # increase overlap with color image edges?
        dxs, xc = xcorr1d_dx(ed, color_edges & vd, search=15)
        peak = int(dxs[np.argmax(xc)])
        i = int(np.argmax(xc))
        if 0 < i < len(xc) - 1:
            a, b, c2 = float(xc[i-1]), float(xc[i]), float(xc[i+1])
            sub = float(dxs[i]) + 0.5*(a - c2) / max((a - 2*b + c2), 1e-9)
        else:
            sub = float(peak)
        print(f"  [{name}] cross-modal peak dx={peak:+d}  sub={sub:+.2f}px  "
              f"(positive = shift depth right to match color)")

    # Save a quick visualisation of the ROI selections + per-ROI label.
    roi_viz = color.copy()
    for (x0, x1, y0, y1, c_peak) in rois:
        cv2.rectangle(roi_viz, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(roi_viz, f"c={c_peak}", (x0, y0-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUT}/g3_rois.png", roi_viz)

    # Save G1 plots data
    with open(f"{OUT}/g1_data.json", "w") as f:
        json.dump(plots, f, indent=2)
    print(f"wrote {OUT}/g1_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
