#!/usr/bin/env python3
"""Phase H: split where the 3-pixel color-frame offset comes from.

Test: feed native /depth/image_rect_raw (already in IR1/depth frame, no FS
in the loop) through the SAME color_align.reproject_ir_to_color() that the
FS node uses, with the published K's and extrinsics. Then compare against
the ASIC's /aligned_depth_to_color via the same dx sweep.

If THIS still shows a +3 px shift → the bug is in color_align.py (or in
the published extrinsics not matching realsense ASIC firmware's
alignment math).

If THIS shows 0 px shift → color_align.py is correct against the
realsense ASIC, and the +3 px must be coming from FS itself (e.g. FS's
IR1-frame depth has a half-pixel grid offset that didn't show up in
G1's IR1-frame sweep because that sweep compares against the same
shifted reference).
"""
from __future__ import annotations

import json
import os
import sys
import time

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

sys.path.insert(0, "/home/tinker/tk25_ws/src/tk26_vision/src/foundation_stereo")
from foundation_stereo.color_align import reproject_ir_to_color  # noqa: E402

OUT = "/home/tinker/tk25_ws/src/tk26_vision/debug_renders/2026-05-25-fs-vs-native-alignment/triage_h"
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
        super().__init__("fs_triage_h")
        self.bridge = CvBridge()
        self.color_info = None
        self.ir1_info = None
        self.native_color = None
        self.native_ir1 = None
        self.extr = None

        self.create_subscription(Image,
                                 "/camera/xarm_camera/aligned_depth_to_color/image_raw",
                                 self._set_native_color, _SENSOR)
        self.create_subscription(Image,
                                 "/camera/xarm_camera/depth/image_rect_raw",
                                 self._set_native_ir1, _SENSOR)
        self.create_subscription(CameraInfo,
                                 "/camera/xarm_camera/color/camera_info",
                                 self._set_color_info, _SENSOR)
        self.create_subscription(CameraInfo,
                                 "/camera/xarm_camera/infra1/camera_info",
                                 self._set_ir1_info, _SENSOR)
        self.create_subscription(Extrinsics,
                                 "/camera/xarm_camera/extrinsics/depth_to_color",
                                 self._set_extr, _LATCHED)

    def _set_native_color(self, m): self.native_color = m
    def _set_native_ir1(self, m): self.native_ir1 = m
    def _set_color_info(self, m): self.color_info = m
    def _set_ir1_info(self, m): self.ir1_info = m
    def _set_extr(self, m):
        R = np.asarray(m.rotation, dtype=np.float64).reshape(3, 3)
        T = np.asarray(m.translation, dtype=np.float64).reshape(3)
        self.extr = (R, T)

    def have_all(self):
        return all(x is not None for x in (
            self.color_info, self.ir1_info,
            self.native_color, self.native_ir1, self.extr))


def shift_image(img, dx, dy):
    H, W = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def sweep_dx_mae(fs, native, dxs):
    out = []
    for dx in dxs:
        s = shift_image(fs, dx, 0)
        vn = native > 0
        vf = (s > 0) & (s < 10.0)
        common = vn & vf
        if not common.any():
            out.append((dx, float("nan"), 0))
            continue
        d = (s - native)[common]
        out.append((dx, float(np.mean(np.abs(d))), int(common.sum())))
    return out


def parabola_min(pts):
    """pts = list of (x, y); find sub-pixel argmin via parabolic fit
    around the integer argmin."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    i = int(np.argmin(ys))
    if i == 0 or i == len(ys) - 1:
        return float(xs[i])
    a, b, c = ys[i-1], ys[i], ys[i+1]
    denom = a - 2*b + c
    if abs(denom) < 1e-9:
        return float(xs[i])
    return float(xs[i]) + 0.5 * (a - c) / denom


def main():
    rclpy.init()
    node = Capture()
    t0 = time.time()
    while rclpy.ok() and not node.have_all():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 20:
            print("timeout"); return 1
    for _ in range(15):
        rclpy.spin_once(node, timeout_sec=0.05)

    K_color = np.asarray(node.color_info.k, dtype=np.float64).reshape(3, 3)
    K_ir1 = np.asarray(node.ir1_info.k, dtype=np.float64).reshape(3, 3)
    R, T = node.extr
    print("K_color:\n", K_color)
    print("K_ir1:\n", K_ir1)
    print("R:\n", R)
    print("T (m):", T)

    native_color_mm = node.bridge.imgmsg_to_cv2(node.native_color, "passthrough")
    native_color = native_color_mm.astype(np.float32) / 1000.0
    native_ir1_mm = node.bridge.imgmsg_to_cv2(node.native_ir1, "passthrough")
    native_ir1 = native_ir1_mm.astype(np.float32) / 1000.0
    H_c, W_c = native_color.shape
    print(f"native_color shape: {native_color.shape}  "
          f"native_ir1 shape: {native_ir1.shape}")

    # Run native_ir1 through the SAME color_align.reproject_ir_to_color
    # that the FS node uses.
    my_color = reproject_ir_to_color(
        depth_ir=native_ir1.astype(np.float32),
        K_ir=K_ir1.astype(np.float32),
        K_color=K_color.astype(np.float32),
        R_ir_to_color=R.astype(np.float32),
        T_ir_to_color=T.astype(np.float32),
        out_hw=(H_c, W_c))
    print(f"my_color shape: {my_color.shape}  "
          f"coverage: {(my_color > 0).mean():.1%}  "
          f"asic coverage: {(native_color > 0).mean():.1%}")

    # Phase H1: sweep dx on (my_color  vs  native_color_asic)
    dxs = list(range(-15, 16))
    h1 = sweep_dx_mae(my_color, native_color, dxs)
    int_min = min(h1, key=lambda x: x[1])
    sub = parabola_min(h1)
    print(f"\n[H1] my_color vs ASIC native_color:  int argmin dx={int_min[0]:+d}  "
          f"sub={sub:+.2f}px  min_mae={int_min[1]*1000:.1f}mm  "
          f"mae@0={[m for d,m,n in h1 if d==0][0]*1000:.1f}mm")

    # For comparison: native_ir1 reproject naively (round, no occlusion).
    # ... actually reproject_ir_to_color already does this. Above is the
    # same algorithm FS uses, so this isolates color_align quality.

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120, facecolor="white")
        xs = [p[0] for p in h1]; ys = [p[1]*1000 for p in h1]
        ax.plot(xs, ys, marker="o", lw=1.2,
                label=f"my_color (ours) vs ASIC native_color  argmin={int_min[0]:+d}")
        ax.axvline(0, color="k", lw=0.6)
        ax.axvline(sub, color="r", lw=0.8, label=f"sub-pixel min={sub:+.2f}px")
        ax.set_xlabel("dx (px) — positive = shift our reprojection right")
        ax.set_ylabel("MAE vs ASIC native_color (mm)")
        ax.set_title("Phase H: same-input alignment check\n"
                     "(native IR1 depth → our color_align  vs  ASIC aligned_depth_to_color)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{OUT}/h_sweep.png", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT}/h_sweep.png")
    except Exception as exc:
        print(f"plot skipped: {exc}")

    # Side-by-side panel.
    def turbo(d, vmax=2.5):
        v = d > 0
        i = (np.clip(d/vmax, 0, 1) * 255).astype(np.uint8)
        i[~v] = 0
        out = cv2.applyColorMap(i, cv2.COLORMAP_TURBO); out[~v] = 0
        return out

    def panel(img, cap):
        s = np.full((22, img.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(s, cap, (10, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (240, 240, 240), 1, cv2.LINE_AA)
        return np.concatenate([s, img], axis=0)

    # ------- Phase H2: Z-binned sweep -------
    # Constant-pixel offset → cx/principal-point bug.
    # Z-dependent offset (more shift at small Z) → Tx baseline error.
    print("\n=== H2: Z-binned dx sweep ===")
    h2 = {}
    for lo, hi in [(0.30, 0.60), (0.60, 1.00), (1.00, 1.50),
                    (1.50, 2.50), (2.50, 4.00)]:
        zmask = (native_color > lo) & (native_color < hi) & (my_color > 0) & (my_color < 10)
        if zmask.sum() < 2000:
            print(f"  Z=[{lo:.2f},{hi:.2f})m  n={zmask.sum()}  too few; skip")
            continue
        # Sweep dx on the zmask region.
        per_dx = []
        for dx in range(-15, 16):
            s = shift_image(my_color, dx, 0)
            sm = zmask & (s > 0)
            if not sm.any():
                per_dx.append((dx, float("nan"))); continue
            d = (s - native_color)[sm]
            per_dx.append((dx, float(np.mean(np.abs(d)))))
        finite = [p for p in per_dx if not np.isnan(p[1])]
        bm = min(finite, key=lambda x: x[1])
        sub = parabola_min(finite)
        h2[f"Z_{lo:.2f}_{hi:.2f}"] = {
            "n": int(zmask.sum()),
            "dx_int": bm[0], "dx_sub": sub,
            "min_mae_mm": bm[1] * 1000,
        }
        print(f"  Z=[{lo:.2f},{hi:.2f})m  n={zmask.sum():>7}  "
              f"int_argmin={bm[0]:+d}  sub={sub:+.2f}px  "
              f"min_mae={bm[1]*1000:.1f}mm")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # H2 viz: bar of sub-pixel best dx per Z bin
        fig, ax = plt.subplots(figsize=(7, 4), dpi=120, facecolor="white")
        labels = list(h2.keys())
        dx_subs = [h2[k]["dx_sub"] for k in labels]
        ax.bar(labels, dx_subs)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel("best dx (sub-pixel)")
        ax.set_title("H2: Z-binned argmin dx for our color_align vs ASIC.\n"
                     "Flat across Z → cx bug. Rising at small Z → Tx baseline bug.")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(f"{OUT}/h2_z_binned.png", facecolor="white")
        plt.close(fig)
        print(f"  wrote {OUT}/h2_z_binned.png")
    except Exception as exc:
        print(f"  plot skipped: {exc}")

    # ------- Phase H3: solve for effective T_x and cx_c that match the ASIC ----
    # Treat our forward-projection u_c as a function of (T_x, cx_c). The
    # 2-parameter linear model:
    #   u_c_required = u_c_ours + dx_observed(Z)
    # Express dx_observed(Z) as a + b/Z, where a is the constant cx_offset
    # and b is the baseline_offset_in_pixels-times-something. Solve via
    # linear regression over Z bins.
    print("\n=== H3: solve constant + Z-dependent components ===")
    if len(h2) >= 2:
        zs = []
        dxs_sub = []
        for k in h2:
            lo_s, hi_s = k.replace("Z_", "").split("_")
            z_mid = 0.5 * (float(lo_s) + float(hi_s))
            zs.append(z_mid)
            dxs_sub.append(h2[k]["dx_sub"])
        zs = np.array(zs); dxs_sub = np.array(dxs_sub)
        # dx_obs ≈ a + b / Z
        A = np.stack([np.ones_like(zs), 1.0/zs], axis=1)
        coef, *_ = np.linalg.lstsq(A, dxs_sub, rcond=None)
        a, b = coef
        fx_c = K_color[0, 0]
        Tx_extra_m = b / fx_c
        print(f"  fit dx ≈ {a:+.3f} + {b:+.3f}/Z  (Z in m)")
        print(f"  → constant pixel offset a={a:+.2f}px  "
              f"(would need cx_color += {a:+.2f}px)")
        print(f"  → Z-coupled term  b={b:+.2f}px·m  "
              f"≡  extra Tx of {Tx_extra_m*1000:+.2f}mm  "
              f"(published Tx={T[0]*1000:.2f}mm; effective should be "
              f"{(T[0]+Tx_extra_m)*1000:.2f}mm)")

    diff = my_color - native_color
    common = (my_color > 0) & (native_color > 0)
    diff_viz = np.full((H_c, W_c, 3), (60, 60, 60), dtype=np.uint8)
    if common.any():
        t = np.clip(diff/0.10, -1, 1)
        pos = (t > 0) & common
        neg = (t < 0) & common
        if pos.any():
            a = t[pos][:, None]
            diff_viz[pos] = ((1-a)*np.array([255,255,255]) + a*np.array([0,0,255])).astype(np.uint8)
        if neg.any():
            a = (-t[neg])[:, None]
            diff_viz[neg] = ((1-a)*np.array([255,255,255]) + a*np.array([255,0,0])).astype(np.uint8)

    panels = [
        panel(turbo(native_color), "ASIC aligned_depth_to_color"),
        panel(turbo(my_color),     "our color_align(native IR1) [same algo as FS]"),
        panel(diff_viz,            "diff: ours - ASIC  (+/-100mm)"),
    ]
    h_panel = panels[0].shape[0]
    sep = np.full((h_panel, 8, 3), 30, dtype=np.uint8)
    row = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
    cv2.imwrite(f"{OUT}/h_panel.png", row)
    print(f"wrote {OUT}/h_panel.png")

    with open(f"{OUT}/h_data.json", "w") as f:
        json.dump({
            "h1_dx_int_argmin": int_min[0],
            "h1_dx_sub": sub,
            "h1_min_mae_mm": int_min[1] * 1000,
            "h1_mae_at_0_mm": [m*1000 for d,m,n in h1 if d==0][0],
            "h1_sweep": [{"dx": d, "mae_m": m, "n": n} for d,m,n in h1],
            "T_ir_to_color_m": T.tolist(),
            "K_color": K_color.tolist(),
            "K_ir1": K_ir1.tolist(),
        }, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
