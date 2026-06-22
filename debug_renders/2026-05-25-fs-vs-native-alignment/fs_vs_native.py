#!/usr/bin/env python3
"""Compare FS aligned depth against D435 native aligned_depth_to_color.

The FS service is parameters-only (the node grabs camera frames itself),
so the workflow is:
  1. Subscribe to live D435 (color + native aligned depth + color_info).
  2. Call /foundation_stereo/get_depth with align_to_color=true.
  3. Grab the freshest native depth at call-completion time.
  4. Compute per-pixel signed diff on the intersection of valid masks.
  5. Write panels + stats to this dir.

The native aligned_depth_to_color is the ground-truth reference for
alignment: it is the same K_color grid the FS output claims to live in.
If FS alignment is correct, the bottle silhouettes / table edges in the
diff panel will follow the same outlines as the color image (and the
median error will be a small global offset, not a structured halo).
"""
from __future__ import annotations

import json
import os
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
from tinker_vision_msgs_26.srv import FoundationStereoDepth

OUT = "/home/tinker/tk25_ws/src/tk26_vision/debug_renders/2026-05-25-fs-vs-native-alignment"

_SENSOR_QOS = QoSProfile(
    depth=5,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)


class Capture(Node):
    def __init__(self):
        super().__init__("fs_vs_native_capture")
        self.bridge = CvBridge()
        self.color = None
        self.native_depth = None
        self.color_info = None

        self.create_subscription(
            Image, "/camera/xarm_camera/color/image_raw",
            self._on_color, _SENSOR_QOS)
        self.create_subscription(
            Image, "/camera/xarm_camera/aligned_depth_to_color/image_raw",
            self._on_native, _SENSOR_QOS)
        self.create_subscription(
            CameraInfo, "/camera/xarm_camera/color/camera_info",
            self._on_color_info, _SENSOR_QOS)

        self.fs_client = self.create_client(
            FoundationStereoDepth, "/foundation_stereo/get_depth")

    def _on_color(self, msg): self.color = msg
    def _on_native(self, msg): self.native_depth = msg
    def _on_color_info(self, msg): self.color_info = msg

    def have_all(self):
        return all(x is not None for x in (
            self.color, self.native_depth, self.color_info))


def turbo_depth(depth_m, z_min=0.05, z_max=2.5, mask=None):
    if mask is None:
        mask = depth_m > 0
    norm = np.clip((depth_m - z_min) / (z_max - z_min), 0, 1)
    img = (norm * 255).astype(np.uint8)
    img[~mask] = 0
    out = cv2.applyColorMap(img, cv2.COLORMAP_TURBO)
    out[~mask] = 0
    return out


def signed_diff_colormap(diff_m, mask, vmax=0.1):
    """Diverging blue->white->red: positive (FS deeper) is red, negative
    (FS shallower) is blue, zero is white, hole is gray. Built by hand
    because cv2.COLORMAP_COOLWARM is not in this OpenCV build."""
    H, W = diff_m.shape
    out = np.full((H, W, 3), (60, 60, 60), dtype=np.uint8)  # gray bg
    # Symmetric clamp
    t = np.clip(diff_m / vmax, -1.0, 1.0)
    # Blend from blue (-1) to white (0) to red (+1) — BGR order
    # blue  = (255, 0, 0), white = (255, 255, 255), red = (0, 0, 255)
    pos = (t > 0) & mask
    neg = (t < 0) & mask
    zero = (t == 0) & mask
    if pos.any():
        a = t[pos][:, None]                    # [0,1]
        col = (1 - a) * np.array([255, 255, 255]) + a * np.array([0, 0, 255])
        out[pos] = col.astype(np.uint8)
    if neg.any():
        a = (-t[neg])[:, None]                 # [0,1]
        col = (1 - a) * np.array([255, 255, 255]) + a * np.array([255, 0, 0])
        out[neg] = col.astype(np.uint8)
    if zero.any():
        out[zero] = (255, 255, 255)
    return out


def make_panel(img, caption, h_strip=24):
    strip = np.full((h_strip, img.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(strip, caption, (10, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([strip, img], axis=0)


def hcat(images, gap=8):
    h = images[0].shape[0]
    sep = np.full((h, gap, 3), 30, dtype=np.uint8)
    out = images[0]
    for img in images[1:]:
        out = np.concatenate([out, sep, img], axis=1)
    return out


def vcat(images, gap=8):
    w = images[0].shape[1]
    sep = np.full((gap, w, 3), 30, dtype=np.uint8)
    out = images[0]
    for img in images[1:]:
        out = np.concatenate([out, sep, img], axis=0)
    return out


def main():
    rclpy.init()
    node = Capture()

    print("waiting for /foundation_stereo/get_depth ...")
    if not node.fs_client.wait_for_service(timeout_sec=15.0):
        print("ERROR: FS service unavailable")
        return 1

    print("waiting for color + native_depth + color_info ...")
    t0 = time.time()
    while rclpy.ok() and not node.have_all():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 15:
            print("ERROR: timed out collecting topics")
            return 1
    print(f"collected in {time.time()-t0:.1f}s")

    # Spin a bit more so all three converge to the same era.
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.05)
    color_stamp_before = node.color.header.stamp
    native_stamp_before = node.native_depth.header.stamp
    print(f"color stamp before  = {color_stamp_before.sec}.{color_stamp_before.nanosec}")
    print(f"native stamp before = {native_stamp_before.sec}.{native_stamp_before.nanosec}")

    # Call FS.  Aligned to color → 32FC1 meters at color resolution.
    req = FoundationStereoDepth.Request()
    req.align_to_color = True
    req.want_pointcloud = False
    req.want_debug_jpeg = False
    req.z_far = 10.0

    print("calling FS service ...")
    t_call = time.time()
    future = node.fs_client.call_async(req)
    rclpy.spin_until_future_complete(node, future, timeout_sec=60.0)
    resp = future.result()
    e2e = time.time() - t_call
    print(f"  status={resp.status}  forward_ms={resp.forward_ms:.1f}  "
          f"e2e={e2e:.2f}s  model={resp.model_used}/{resp.trt_variant_used}  "
          f"msg='{resp.error_msg}'")
    if resp.status != 0:
        print("ERROR from FS service")
        return 1

    # Grab fresh color + native after the call.
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.05)

    # Decode.
    color = node.bridge.imgmsg_to_cv2(node.color, "bgr8")
    native_mm = node.bridge.imgmsg_to_cv2(node.native_depth, "passthrough")
    native = native_mm.astype(np.float32) / 1000.0
    fs = node.bridge.imgmsg_to_cv2(resp.depth_image, "passthrough").astype(np.float32)
    # FS returns 32FC1 in meters when align_to_color=True.

    H, W = color.shape[:2]
    assert native.shape == (H, W), (native.shape, (H, W))
    if fs.shape != (H, W):
        print(f"NOTE: fs shape {fs.shape} != color {(H, W)}; resizing")
        fs = cv2.resize(fs, (W, H), interpolation=cv2.INTER_NEAREST)
    print(f"shapes: color={color.shape}  native={native.shape}  fs={fs.shape}")

    Z_MAX_PANEL = 2.5
    fs_mask = (fs > 0) & (fs < 10.0)
    native_mask = native > 0
    fs[~fs_mask] = 0.0
    common = native_mask & fs_mask
    print(f"coverage:  native={native_mask.mean():.1%}  "
          f"fs={fs_mask.mean():.1%}  common={common.mean():.1%}")

    diff = fs - native
    diff_common = diff[common]
    if diff_common.size > 0:
        stats = {
            "n": int(common.sum()),
            "mean_m": float(diff_common.mean()),
            "median_m": float(np.median(diff_common)),
            "mae_m": float(np.mean(np.abs(diff_common))),
            "rmse_m": float(np.sqrt(np.mean(diff_common**2))),
            "p05_m": float(np.percentile(diff_common, 5)),
            "p95_m": float(np.percentile(diff_common, 95)),
            "abs_lt_2cm_pct": float(np.mean(np.abs(diff_common) < 0.02) * 100),
            "abs_lt_5cm_pct": float(np.mean(np.abs(diff_common) < 0.05) * 100),
            "abs_lt_10cm_pct": float(np.mean(np.abs(diff_common) < 0.10) * 100),
        }
    else:
        stats = {"n": 0}
    print("diff stats (m, fs - native):")
    for k, v in stats.items():
        print(f"  {k:>18} = {v}")

    native_viz = turbo_depth(native, 0.05, Z_MAX_PANEL, mask=native_mask)
    fs_viz = turbo_depth(fs,     0.05, Z_MAX_PANEL, mask=fs_mask)
    diff_viz = signed_diff_colormap(diff, common, vmax=0.10)

    cv2.imwrite(f"{OUT}/color.jpg", color)
    cv2.imwrite(f"{OUT}/native_depth_viz.jpg", native_viz)
    cv2.imwrite(f"{OUT}/fs_depth_viz.jpg", fs_viz)
    cv2.imwrite(f"{OUT}/diff_viz.jpg", diff_viz)

    # Row composite: color | native | fs | diff.
    row = hcat([
        make_panel(color,      f"color  {W}x{H}"),
        make_panel(native_viz, f"native aligned depth (turbo 0.05..{Z_MAX_PANEL} m)"),
        make_panel(fs_viz,     f"FS aligned depth   (turbo 0.05..{Z_MAX_PANEL} m)"),
        make_panel(diff_viz,   f"FS-native (coolwarm +/-0.10 m, gray=hole)"),
    ])
    cv2.imwrite(f"{OUT}/comparison_row.png", row)

    # 2x2 grid (compact for screenshots).
    top = hcat([
        make_panel(color,      f"color  {W}x{H}"),
        make_panel(native_viz, "native aligned depth"),
    ])
    bot = hcat([
        make_panel(fs_viz,     "FS aligned depth"),
        make_panel(diff_viz,   "FS-native +/-0.10 m"),
    ])
    cv2.imwrite(f"{OUT}/comparison_grid.png", vcat([top, bot]))

    # Histogram.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 4), dpi=120, facecolor="white")
        ax = fig.add_subplot(111)
        ax.hist(diff_common, bins=200, range=(-0.3, 0.3),
                color="#1f77b4", edgecolor="none")
        ax.axvline(0, color="k", lw=0.8)
        ax.axvline(stats["median_m"], color="r", lw=1.0,
                   label=f"median={stats['median_m']*1000:.0f} mm")
        ax.set_xlabel("FS - native depth (m)")
        ax.set_ylabel("pixel count")
        ax.set_title(f"FS vs native aligned depth — "
                     f"n={stats['n']:,}  mae={stats['mae_m']*1000:.0f} mm  "
                     f"<5 cm: {stats['abs_lt_5cm_pct']:.1f}%")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUT}/diff_histogram.png", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT}/diff_histogram.png")
    except Exception as exc:
        print(f"hist skipped: {exc}")

    # Edge overlay: lay native depth Sobel edges (white) and FS depth
    # Sobel edges (red) on top of color so we can visually check if the
    # *structure* of the two depths agrees, independent of absolute Z.
    def edges(depth, valid):
        d = depth.copy()
        d[~valid] = 0
        d8 = (np.clip(d / Z_MAX_PANEL, 0, 1) * 255).astype(np.uint8)
        gx = cv2.Sobel(d8, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(d8, cv2.CV_32F, 0, 1, ksize=3)
        g = np.sqrt(gx**2 + gy**2)
        return g > 40  # threshold tuned for ~2.5 m range

    edge_color = color.copy()
    en = edges(native, native_mask)
    ef = edges(fs, fs_mask)
    edge_color[en] = (255, 255, 255)   # native = white
    edge_color[ef] = (60, 60, 255)     # FS     = red (BGR)
    edge_color[en & ef] = (0, 255, 255)  # both = yellow
    cv2.imwrite(f"{OUT}/edge_overlay.png", make_panel(
        edge_color, "edges: native=white  FS=red  both=yellow"))

    with open(f"{OUT}/stats.json", "w") as f:
        json.dump({
            "stats_diff_m": stats,
            "fs_response": {
                "forward_ms": float(resp.forward_ms),
                "load_s": float(resp.load_s),
                "end_to_end_s": float(resp.end_to_end_s),
                "model_used": resp.model_used,
                "trt_variant_used": resp.trt_variant_used,
            },
            "image_shape": list(color.shape),
            "K_color_fx_fy_cx_cy": [
                float(node.color_info.k[0]), float(node.color_info.k[4]),
                float(node.color_info.k[2]), float(node.color_info.k[5])],
        }, f, indent=2)
    print(f"wrote {OUT}/stats.json")
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
