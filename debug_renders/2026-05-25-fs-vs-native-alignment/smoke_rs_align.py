#!/usr/bin/env python3
"""Smoke test: align via pyrealsense2 software_device + rs.align.

Goal: gate the integration step. Pull native depth from
/camera/<cam>/depth/image_rect_raw, push it through a software_device
pipeline, apply rs.align(rs.stream.color), and compare the result
against the ASIC's /aligned_depth_to_color using the same Z-binned
MAE-vs-dx sweep that exposed the original bug.

Gate to proceed with integration: every Z bin's sub-pixel argmin |dx| ≤ 0.5.

Run:
  RS_CAM=xarm_camera  ./smoke_rs_align.py        # D435
  RS_CAM=head_camera  ./smoke_rs_align.py        # D405
"""
from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs
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

CAM = os.environ.get("RS_CAM", "xarm_camera")
OUT = ("/home/tinker/tk25_ws/src/tk26_vision/debug_renders/"
       f"2026-05-25-fs-vs-native-alignment/smoke_rs_align_{CAM}")
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
        super().__init__(f"smoke_rs_align_{cam}")
        self.bridge = CvBridge()
        self.color_info = None; self.ir1_info = None
        self.native_color = None; self.native_ir1 = None
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
            self.color_info, self.ir1_info, self.native_color,
            self.native_ir1, self.extr))


def rs_intrinsics_from_info(K, W, H, D=None, distortion_kind="color"):
    """Build rs.intrinsics from K, image size, optional distortion coeffs.

    RealSense firmware uses `inverse_brown_conrady` for color streams (even
    when coeffs are zero) and `brown_conrady`/none for IR/depth. Mirror
    that here so rs.align matches the firmware path bit-for-bit.
    """
    intr = rs.intrinsics()
    intr.width = int(W); intr.height = int(H)
    intr.fx = float(K[0, 0]); intr.fy = float(K[1, 1])
    intr.ppx = float(K[0, 2]); intr.ppy = float(K[1, 2])
    if distortion_kind == "color":
        intr.model = rs.distortion.inverse_brown_conrady
    else:
        intr.model = rs.distortion.brown_conrady
    if D is None:
        intr.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        coeffs = [float(D[i]) if i < len(D) else 0.0 for i in range(5)]
        intr.coeffs = coeffs
    return intr


def rs_extrinsics(R, T):
    """librealsense stores rotation column-major; numpy R is row-major.
    Transpose when copying."""
    ex = rs.extrinsics()
    ex.rotation = list(np.asarray(R, dtype=np.float64).T.ravel())  # column-major
    ex.translation = [float(T[0]), float(T[1]), float(T[2])]
    return ex


class RealsenseAligner:
    """One-call wrapper around rs.software_device + rs.align(color).

    Holds the sw_device + streams + syncer + align block alive between
    calls so we pay setup cost once.
    """
    def __init__(self, K_ir, K_color, R, T, ir_hw, color_hw, *,
                 D_color=None, D_ir=None, depth_units_m=0.001):
        H_ir, W_ir = ir_hw
        H_c, W_c = color_hw
        self._H_ir = H_ir; self._W_ir = W_ir
        self._H_c = H_c;   self._W_c = W_c
        self._depth_units_m = float(depth_units_m)

        self._dev = rs.software_device()
        # NB: the canonical librealsense align-with-software-device example
        # does NOT call create_matcher; the default matcher kicks in
        # implicitly. Calling it explicitly here was silently dropping the
        # depth frame.

        d_sensor = self._dev.add_sensor("Depth")
        c_sensor = self._dev.add_sensor("Color")

        # Depth stream (uid=1, mirrors the canonical example).
        d_vs = rs.video_stream()
        d_vs.type = rs.stream.depth
        d_vs.fmt = rs.format.z16
        d_vs.uid = 1
        d_vs.index = 0
        d_vs.width = W_ir; d_vs.height = H_ir
        d_vs.fps = 30; d_vs.bpp = 2
        d_vs.intrinsics = rs_intrinsics_from_info(K_ir, W_ir, H_ir, D=D_ir,
                                                  distortion_kind="ir")
        d_prof_raw = d_sensor.add_video_stream(d_vs)
        # Keep raw stream_profile for open()/start(); use video downcast for frames.
        self._d_prof = d_prof_raw.as_video_stream_profile()

        # Color stream (uid=2). Pixel content is unused by rs.align.
        c_vs = rs.video_stream()
        c_vs.type = rs.stream.color
        c_vs.fmt = rs.format.rgb8
        c_vs.uid = 2
        c_vs.index = 0
        c_vs.width = W_c; c_vs.height = H_c
        c_vs.fps = 30; c_vs.bpp = 3
        c_vs.intrinsics = rs_intrinsics_from_info(K_color, W_c, H_c, D=D_color,
                                                  distortion_kind="color")
        c_prof_raw = c_sensor.add_video_stream(c_vs)
        self._c_prof = c_prof_raw.as_video_stream_profile()

        # Wire extrinsics  depth -> color.  Use the video downcasts so the
        # transform is stored against the same profile object both ends see.
        self._d_prof.register_extrinsics_to(self._c_prof, rs_extrinsics(R, T))

        # Start both sensors into a shared syncer so align can frameset
        # them. open() / start() take the raw stream_profile.
        self._sync = rs.syncer()
        d_sensor.open(d_prof_raw)
        c_sensor.open(c_prof_raw)
        d_sensor.start(self._sync)
        c_sensor.start(self._sync)
        self._d_sensor = d_sensor
        self._c_sensor = c_sensor

        self._align = rs.align(rs.stream.color)

        # Reusable empty color buffer
        self._color_buf = np.zeros((H_c, W_c, 3), dtype=np.uint8)
        self._frame_n = 0

    def align(self, depth_ir_m: np.ndarray, timestamp_ms: float = 0.0) -> np.ndarray:
        """Align float32-m depth at IR1 grid -> float32-m depth at color grid."""
        assert depth_ir_m.shape == (self._H_ir, self._W_ir), depth_ir_m.shape
        # Convert to Z16-mm (depth_units_m = 0.001 means lsb = 1 mm)
        z16 = np.clip(np.round(depth_ir_m / self._depth_units_m), 0, 65535).astype(np.uint16)

        # rs.syncer in this software_device setup emits a paired (D+C)
        # frameset only once it sees a second pair in its window — a
        # single push is held back. Workaround: push twice with widely-
        # spaced timestamps so the second push flushes the first paired
        # frameset through. We discard the second pair's output.
        for _ in range(2):
            self._frame_n += 1
            ts_one = 1000.0 * self._frame_n  # 1 s apart per push, matches canonical test
            d_frame = rs.software_video_frame()
            d_frame.pixels = z16
            d_frame.bpp = 2
            d_frame.stride = self._W_ir * 2
            d_frame.timestamp = ts_one
            d_frame.domain = rs.timestamp_domain.hardware_clock
            d_frame.frame_number = self._frame_n
            d_frame.profile = self._d_prof
            d_frame.depth_units = self._depth_units_m
            self._d_sensor.on_video_frame(d_frame)

            c_frame = rs.software_video_frame()
            c_frame.pixels = self._color_buf
            c_frame.bpp = 3
            c_frame.stride = self._W_c * 3
            c_frame.timestamp = ts_one
            c_frame.domain = rs.timestamp_domain.hardware_clock
            c_frame.frame_number = self._frame_n
            c_frame.profile = self._c_prof
            self._c_sensor.on_video_frame(c_frame)

        # The syncer may emit each frame separately if it can't pair them
        # in one tick. Pull until we see both depth and color, then merge.
        depth_in = None; color_in = None
        for tick in range(8):
            try:
                fs = self._sync.wait_for_frames(500)
            except Exception:
                continue
            for i in range(fs.size()):
                f = fs[i]
                st = f.profile.stream_type()
                if st == rs.stream.depth and depth_in is None:
                    depth_in = f
                elif st == rs.stream.color and color_in is None:
                    color_in = f
            if self._frame_n <= 2:
                print(f"  [diag tick {tick}] fs size={fs.size()} "
                      f"streams={[fs[i].profile.stream_type() for i in range(fs.size())]} "
                      f"have_depth={depth_in is not None} have_color={color_in is not None}")
            if depth_in is not None and color_in is not None:
                break
        if depth_in is None:
            raise RuntimeError("syncer never produced a depth frame")
        if color_in is None:
            raise RuntimeError("syncer never produced a color frame")

        # Now feed both into align. Construct an explicit frameset by
        # calling align.process(frame) — but align.process needs a
        # frameset. So we need to construct one. The trick: rs.align
        # actually accepts a composite frameset OR we use the underlying
        # processing block to push our two frames into the same frameset.
        # Easiest: use rs.frame_queue + push frames, but align operates
        # on framesets so this is still odd.
        # Workaround: push depth+color via aligner.invoke directly.
        aligned = self._align.process(fs).as_frameset()
        if self._frame_n <= 2:
            print(f"  [diag] post-align size={aligned.size()}  "
                  f"streams={[aligned[i].profile.stream_type() for i in range(aligned.size())]}")
        depth = aligned.get_depth_frame()
        if not depth:
            # Fall back: iterate frames manually
            for i in range(aligned.size()):
                f = aligned[i]
                if f.profile.stream_type() == rs.stream.depth:
                    depth = f.as_depth_frame()
                    break
            if not depth:
                raise RuntimeError("align returned no depth frame")
        # depth.get_data() returns Z16 bytes
        arr = np.frombuffer(depth.get_data(), dtype=np.uint16)
        arr = arr.reshape(self._H_c, self._W_c)
        return arr.astype(np.float32) * self._depth_units_m


def shift_image(img, dx, dy):
    H, W = img.shape[:2]
    M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def sweep_dx_mae(my, asic, dxs):
    out = []
    for dx in dxs:
        s = shift_image(my, dx, 0)
        v = (asic > 0) & (s > 0) & (s < 10.0)
        if not v.any():
            out.append((dx, float("nan"), 0)); continue
        d = (s - asic)[v]
        out.append((dx, float(np.mean(np.abs(d))), int(v.sum())))
    return out


def parabola_min(pts):
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    i = int(np.argmin(ys))
    if i in (0, len(ys) - 1):
        return float(xs[i])
    a, b, c = ys[i-1], ys[i], ys[i+1]
    denom = a - 2*b + c
    return float(xs[i]) + (0.5 * (a - c) / denom if abs(denom) > 1e-9 else 0)


def main():
    rclpy.init()
    node = Capture(CAM)
    t0 = time.time()
    while rclpy.ok() and not node.have_all():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 20:
            print(f"timeout"); return 1
    for _ in range(15):
        rclpy.spin_once(node, timeout_sec=0.05)

    K_color = np.asarray(node.color_info.k, dtype=np.float64).reshape(3, 3)
    K_ir1 = np.asarray(node.ir1_info.k, dtype=np.float64).reshape(3, 3)
    R, T = node.extr
    native_color = node.bridge.imgmsg_to_cv2(node.native_color, "passthrough").astype(np.float32) / 1000.0
    native_ir1 = node.bridge.imgmsg_to_cv2(node.native_ir1, "passthrough").astype(np.float32) / 1000.0
    H_c, W_c = native_color.shape
    H_i, W_i = native_ir1.shape
    print(f"=== {CAM} ===")
    print(f"native_color {native_color.shape}  native_ir1 {native_ir1.shape}")
    print(f"K_color cx,cy,fx,fy = {K_color[0,2]:.2f},{K_color[1,2]:.2f},"
          f"{K_color[0,0]:.2f},{K_color[1,1]:.2f}")
    print(f"K_ir1   cx,cy,fx,fy = {K_ir1[0,2]:.2f},{K_ir1[1,2]:.2f},"
          f"{K_ir1[0,0]:.2f},{K_ir1[1,1]:.2f}")
    print(f"T_ir_to_color = {T*1000} mm  |T|={np.linalg.norm(T)*1000:.2f}mm")

    D_color = list(node.color_info.d)
    D_ir1 = list(node.ir1_info.d)
    print(f"D_color = {D_color}")
    print(f"D_ir1   = {D_ir1}")

    aligner = RealsenseAligner(
        K_ir=K_ir1, K_color=K_color, R=R, T=T,
        ir_hw=(H_i, W_i), color_hw=(H_c, W_c),
        D_color=D_color, D_ir=D_ir1)

    # Warm-up call (first align inside librealsense compiles internal state).
    t_w0 = time.time()
    out0 = aligner.align(native_ir1)
    print(f"warmup align: {(time.time()-t_w0)*1000:.1f}ms  shape={out0.shape}  "
          f"coverage={(out0 > 0).mean():.1%}")

    # Timed calls
    times = []
    for _ in range(20):
        t = time.time(); _ = aligner.align(native_ir1); times.append(time.time()-t)
    t_med_ms = float(np.median(times)) * 1000
    t_p95_ms = float(np.percentile(times, 95)) * 1000
    print(f"per-call median={t_med_ms:.2f}ms  p95={t_p95_ms:.2f}ms")

    my = aligner.align(native_ir1)
    print(f"my coverage  ={(my>0).mean():.1%}  "
          f"ASIC coverage={(native_color>0).mean():.1%}")

    # ==== Sweep ====
    dxs = list(range(-15, 16))
    global_sweep = sweep_dx_mae(my, native_color, dxs)
    bm = min(global_sweep, key=lambda x: x[1])
    sub = parabola_min(global_sweep)
    print(f"\n[global] int_argmin={bm[0]:+d}  sub={sub:+.2f}px  "
          f"min_mae={bm[1]*1000:.1f}mm  mae@0="
          f"{[m for d,m,n in global_sweep if d==0][0]*1000:.1f}mm")

    # ==== Z-binned ====
    print("\n=== Z-binned ===")
    h2 = {}
    gate_pass = True
    bins = [(0.10, 0.30), (0.30, 0.50), (0.50, 0.80),
            (0.80, 1.20), (1.20, 2.00), (2.00, 3.50)]
    for lo, hi in bins:
        zmask = ((native_color > lo) & (native_color < hi) &
                 (my > 0) & (my < 10))
        if zmask.sum() < 1500:
            print(f"  Z=[{lo:.2f},{hi:.2f})m  n={zmask.sum()}  too few; skip")
            continue
        per = []
        for dx in dxs:
            s = shift_image(my, dx, 0)
            sm = zmask & (s > 0)
            if not sm.any():
                per.append((dx, float("nan"))); continue
            d = (s - native_color)[sm]
            per.append((dx, float(np.mean(np.abs(d)))))
        finite = [p for p in per if not np.isnan(p[1])]
        bmz = min(finite, key=lambda x: x[1])
        subz = parabola_min(finite)
        h2[f"Z_{lo:.2f}_{hi:.2f}"] = {
            "n": int(zmask.sum()),
            "dx_int": bmz[0], "dx_sub": subz,
            "min_mae_mm": bmz[1] * 1000,
            "Z_mid": 0.5 * (lo + hi),
        }
        status = "PASS" if abs(subz) <= 0.5 else "FAIL"
        if status == "FAIL": gate_pass = False
        print(f"  Z=[{lo:.2f},{hi:.2f})m  n={zmask.sum():>7}  "
              f"int={bmz[0]:+d}  sub={subz:+.2f}px  "
              f"min_mae={bmz[1]*1000:.1f}mm  [{status}]")

    print(f"\n===== Gate (|sub|≤0.5 in every Z bin): "
          f"{'PASS' if gate_pass else 'FAIL'} =====")

    # Plot
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120, facecolor="white")
        if h2:
            labels = list(h2.keys())
            subs = [h2[k]["dx_sub"] for k in labels]
            ns = [h2[k]["n"] for k in labels]
            bars = ax.bar(labels, subs, color=["C2" if abs(s) <= 0.5 else "C3" for s in subs])
            ax.axhline(0, color="k", lw=0.6)
            ax.axhline(0.5, color="g", lw=0.5, linestyle="--", alpha=0.5)
            ax.axhline(-0.5, color="g", lw=0.5, linestyle="--", alpha=0.5)
            ax.set_ylabel("best sub-pixel dx (px)")
            ax.set_xlabel(f"Z bin  ({CAM}, Tx={T[0]*1000:+.2f}mm)")
            ax.set_title(f"{CAM}: rs.align via software_device — Z-binned argmin dx\n"
                         f"(green band = ±0.5 px gate)")
            ax.tick_params(axis="x", rotation=30)
            for b, n in zip(bars, ns):
                ax.text(b.get_x() + b.get_width()/2, b.get_height(),
                         f"n={n//1000}k", ha="center", va="bottom", fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{OUT}/smoke_z_binned.png", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT}/smoke_z_binned.png")
    except Exception as exc:
        print(f"plot skipped: {exc}")

    # Side-by-side render
    def turbo(d, vmax=2.5):
        v = d > 0
        i = (np.clip(d/vmax, 0, 1) * 255).astype(np.uint8); i[~v] = 0
        out = cv2.applyColorMap(i, cv2.COLORMAP_TURBO); out[~v] = 0
        return out

    def panel(img, cap):
        s = np.full((22, img.shape[1], 3), 20, dtype=np.uint8)
        cv2.putText(s, cap, (10, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (240, 240, 240), 1, cv2.LINE_AA)
        return np.concatenate([s, img], axis=0)

    diff = my - native_color
    common = (my > 0) & (native_color > 0)
    diff_viz = np.full((H_c, W_c, 3), (60, 60, 60), dtype=np.uint8)
    if common.any():
        t = np.clip(diff/0.10, -1, 1)
        pos = (t > 0) & common; neg = (t < 0) & common
        if pos.any():
            a = t[pos][:, None]
            diff_viz[pos] = ((1-a)*np.array([255,255,255]) + a*np.array([0,0,255])).astype(np.uint8)
        if neg.any():
            a = (-t[neg])[:, None]
            diff_viz[neg] = ((1-a)*np.array([255,255,255]) + a*np.array([255,0,0])).astype(np.uint8)

    panels = [
        panel(turbo(native_color), "ASIC aligned_depth_to_color"),
        panel(turbo(my),           "rs.align(software_device) ours"),
        panel(diff_viz,            "diff: ours - ASIC  (+/-100mm)"),
    ]
    h = panels[0].shape[0]
    sep = np.full((h, 8, 3), 30, dtype=np.uint8)
    row = np.concatenate([panels[0], sep, panels[1], sep, panels[2]], axis=1)
    cv2.imwrite(f"{OUT}/smoke_panel.png", row)

    with open(f"{OUT}/smoke.json", "w") as f:
        json.dump({
            "camera": CAM,
            "T_ir_to_color_m": T.tolist(),
            "global_int_argmin": bm[0], "global_sub_argmin": sub,
            "global_min_mae_mm": bm[1] * 1000,
            "z_bins": h2,
            "gate_pass": gate_pass,
            "per_call_median_ms": t_med_ms,
            "per_call_p95_ms": t_p95_ms,
        }, f, indent=2)

    return 0 if gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
