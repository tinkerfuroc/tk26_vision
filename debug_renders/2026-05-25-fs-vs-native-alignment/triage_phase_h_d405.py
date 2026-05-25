#!/usr/bin/env python3
"""Phase H on D405. Same test as triage_phase_h.py but pointed at the
D405 topics under /camera/head_camera/. If the same Z-dependent offset
pattern appears, the bug is in color_align.py math; if D405 shows zero
offset, the issue is D435 ASIC firmware vs published-extrinsics
calibration mismatch.
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

CAM = os.environ.get("RS_CAM", "head_camera")
OUT = ("/home/tinker/tk25_ws/src/tk26_vision/debug_renders/"
       f"2026-05-25-fs-vs-native-alignment/triage_h_{CAM}")
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
    def __init__(self, cam: str):
        super().__init__(f"fs_triage_h_{cam}")
        self.bridge = CvBridge()
        self.color_info = None
        self.ir1_info = None
        self.native_color = None
        self.native_ir1 = None
        self.extr = None

        self.create_subscription(Image, f"/camera/{cam}/aligned_depth_to_color/image_raw",
                                 self._sNc, _SENSOR)
        self.create_subscription(Image, f"/camera/{cam}/depth/image_rect_raw",
                                 self._sNi, _SENSOR)
        self.create_subscription(CameraInfo, f"/camera/{cam}/color/camera_info",
                                 self._sCi, _SENSOR)
        self.create_subscription(CameraInfo, f"/camera/{cam}/infra1/camera_info",
                                 self._sIi, _SENSOR)
        self.create_subscription(Extrinsics, f"/camera/{cam}/extrinsics/depth_to_color",
                                 self._sEx, _LATCHED)

    def _sNc(self, m): self.native_color = m
    def _sNi(self, m): self.native_ir1 = m
    def _sCi(self, m): self.color_info = m
    def _sIi(self, m): self.ir1_info = m
    def _sEx(self, m):
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
        c = vn & vf
        if not c.any():
            out.append((dx, float("nan"), 0)); continue
        d = (s - native)[c]
        out.append((dx, float(np.mean(np.abs(d))), int(c.sum())))
    return out


def parabola_min(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    i = int(np.argmin(ys))
    if i in (0, len(ys) - 1):
        return float(xs[i])
    a, b, c = ys[i-1], ys[i], ys[i+1]
    denom = a - 2*b + c
    return float(xs[i]) + (0.5*(a-c)/denom if abs(denom) > 1e-9 else 0)


def main():
    rclpy.init()
    node = Capture(CAM)
    t0 = time.time()
    while rclpy.ok() and not node.have_all():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 25:
            print(f"timeout subs={node.have_all()}")
            for n in ["color_info", "ir1_info", "native_color", "native_ir1", "extr"]:
                print(f"  {n}: {'OK' if getattr(node, n) is not None else 'MISSING'}")
            return 1
    for _ in range(15):
        rclpy.spin_once(node, timeout_sec=0.05)

    K_color = np.asarray(node.color_info.k, dtype=np.float64).reshape(3, 3)
    K_ir1 = np.asarray(node.ir1_info.k, dtype=np.float64).reshape(3, 3)
    R, T = node.extr
    print(f"=== {CAM} ===")
    print("K_color:\n", K_color)
    print("K_ir1:\n", K_ir1)
    print("R:\n", R)
    print("T (m):", T)

    native_color_mm = node.bridge.imgmsg_to_cv2(node.native_color, "passthrough")
    native_color = native_color_mm.astype(np.float32) / 1000.0
    native_ir1_mm = node.bridge.imgmsg_to_cv2(node.native_ir1, "passthrough")
    native_ir1 = native_ir1_mm.astype(np.float32) / 1000.0
    H_c, W_c = native_color.shape
    print(f"shapes: native_color={native_color.shape}  native_ir1={native_ir1.shape}")

    my_color = reproject_ir_to_color(
        depth_ir=native_ir1.astype(np.float32),
        K_ir=K_ir1.astype(np.float32),
        K_color=K_color.astype(np.float32),
        R_ir_to_color=R.astype(np.float32),
        T_ir_to_color=T.astype(np.float32),
        out_hw=(H_c, W_c))
    print(f"my_color coverage: {(my_color > 0).mean():.1%}  "
          f"ASIC coverage: {(native_color > 0).mean():.1%}")

    # ===== H1: global sweep =====
    dxs = list(range(-15, 16))
    h1 = sweep_dx_mae(my_color, native_color, dxs)
    bm = min(h1, key=lambda x: x[1])
    sub = parabola_min(h1)
    print(f"\n[H1] global: int_argmin={bm[0]:+d}  sub={sub:+.2f}px  "
          f"min_mae={bm[1]*1000:.1f}mm")

    # ===== H2: Z-binned sweep =====
    print("\n=== H2: Z-binned dx sweep ===")
    h2 = {}
    for lo, hi in [(0.10, 0.30), (0.30, 0.50), (0.50, 0.80),
                    (0.80, 1.20), (1.20, 2.00), (2.00, 3.50)]:
        zmask = (native_color > lo) & (native_color < hi) & (my_color > 0) & (my_color < 10)
        if zmask.sum() < 1500:
            print(f"  Z=[{lo:.2f},{hi:.2f})m  n={zmask.sum()}  too few; skip")
            continue
        per_dx = []
        for dx in dxs:
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
            "Z_mid": 0.5*(lo+hi),
        }
        print(f"  Z=[{lo:.2f},{hi:.2f})m  n={zmask.sum():>7}  "
              f"int_argmin={bm[0]:+d}  sub={sub:+.2f}px  "
              f"min_mae={bm[1]*1000:.1f}mm")

    # ===== H3: solve constant + 1/Z components =====
    print("\n=== H3: solve a + b/Z fit ===")
    if len(h2) >= 2:
        zs = np.array([h2[k]["Z_mid"] for k in h2])
        ws = np.array([np.log10(max(1, h2[k]["n"])) for k in h2])  # weight by point count log
        dx = np.array([h2[k]["dx_sub"] for k in h2])
        A = np.stack([np.ones_like(zs), 1.0/zs], axis=1)
        W = np.diag(ws)
        coef, *_ = np.linalg.lstsq(W @ A, W @ dx, rcond=None)
        a, b = coef
        fx_c = K_color[0, 0]
        Tx_extra_m = b / fx_c
        Tx_published_m = T[0]
        print(f"  weighted fit  dx ≈ {a:+.3f} + {b:+.3f}/Z  (Z in m, w=log10 n)")
        print(f"  → const a={a:+.2f}px  (cx_color delta)")
        print(f"  → 1/Z   b={b:+.2f}px·m  ≡  Tx_extra={Tx_extra_m*1000:+.2f}mm")
        print(f"  published Tx={Tx_published_m*1000:.2f}mm; "
              f"effective should be {(Tx_published_m + Tx_extra_m)*1000:.2f}mm "
              f"(ratio={(Tx_published_m+Tx_extra_m)/Tx_published_m:.2f}x)")

    # ===== H4: viz =====
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Z-binned bar plot
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120, facecolor="white")
        if h2:
            labels = list(h2.keys())
            subs = [h2[k]["dx_sub"] for k in labels]
            ns = [h2[k]["n"] for k in labels]
            zs_mid = [h2[k]["Z_mid"] for k in labels]
            bars = ax.bar(labels, subs, color="C0", alpha=0.85)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylabel("best sub-pixel dx (px)")
            ax.set_xlabel(f"Z bin  (sensor: {CAM},  Tx={T[0]*1000:+.2f}mm)")
            ax.set_title(
                f"{CAM}  Phase H Z-binned: our color_align vs ASIC")
            ax.tick_params(axis="x", rotation=30)
            for b, n in zip(bars, ns):
                ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                         f"n={n//1000}k", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{OUT}/h2_z_binned.png", facecolor="white")
        plt.close(fig)
    except Exception as exc:
        print(f"plot skipped: {exc}")

    with open(f"{OUT}/h_data.json", "w") as f:
        json.dump({
            "camera": CAM,
            "T_ir_to_color_m": T.tolist(),
            "K_color": K_color.tolist(),
            "K_ir1": K_ir1.tolist(),
            "h2": h2,
        }, f, indent=2)
    print(f"\nwrote {OUT}/h_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
