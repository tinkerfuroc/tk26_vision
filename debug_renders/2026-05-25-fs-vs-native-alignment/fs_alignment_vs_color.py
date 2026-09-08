#!/usr/bin/env python3
"""The right test for the user's actual concern.

Question: when FS depth is aligned to color, do the depth-edge silhouettes
(bottle outlines, table-edge arc, etc.) land on the matching color-image
edges? Does the "leftward bleed of background into objects" go away?

Test:
  1. Call FS with align_to_color=False → FS depth on IR1 grid.
  2. Align it two ways:
       (a) The current path: foundation_stereo.color_align.reproject_ir_to_color
           — this is what the user originally saw the bleed in.
       (b) Option 1 path: rs.align via software_device, our new aligner.
  3. For each: extract depth edges (Sobel), overlay on the color image
     in distinguishable colors. Compare side-by-side.

If (b) tracks the color edges noticeably better than (a) — option 1
fixes the bleed. If (a) and (b) look essentially the same — option 1
doesn't help with the user's actual concern.

Run for both D435 (xarm_camera, RS_CAM=xarm_camera) and D405
(head_camera, RS_CAM=head_camera).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

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
from tinker_vision_msgs_26.srv import FoundationStereoDepth

# Reuse the RealsenseAligner from the smoke test (already validated to work
# mechanically, even if it doesn't bit-match the ASIC).
sys.path.insert(0, str(Path(__file__).parent))
from smoke_rs_align import RealsenseAligner

# And the legacy forward-warp from the FS package.
sys.path.insert(0, "/home/tinker/tk25_ws/src/tk26_vision/src/foundation_stereo")
from foundation_stereo.color_align_legacy import reproject_ir_to_color

CAM = os.environ.get("RS_CAM", "xarm_camera")
OUT = ("/home/tinker/tk25_ws/src/tk26_vision/debug_renders/"
       f"2026-05-25-fs-vs-native-alignment/fs_align_vs_color_{CAM}")
os.makedirs(OUT, exist_ok=True)

_SENSOR = QoSProfile(
    depth=5, history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE)
_LATCHED = QoSProfile(
    depth=1, history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)


class Cap(Node):
    def __init__(self, cam):
        super().__init__(f"fs_align_vs_color_{cam}")
        self.bridge = CvBridge()
        self.color = None; self.color_info = None
        self.ir1_info = None; self.extr = None
        self.create_subscription(Image, f"/camera/{cam}/color/image_raw",
                                 self._sc, _SENSOR)
        self.create_subscription(CameraInfo, f"/camera/{cam}/color/camera_info",
                                 self._sci, _SENSOR)
        self.create_subscription(CameraInfo, f"/camera/{cam}/infra1/camera_info",
                                 self._sii, _SENSOR)
        self.create_subscription(Extrinsics, f"/camera/{cam}/extrinsics/depth_to_color",
                                 self._sex, _LATCHED)
        self.fs = self.create_client(FoundationStereoDepth,
                                     "/foundation_stereo/get_depth")

    def _sc(self, m): self.color = m
    def _sci(self, m): self.color_info = m
    def _sii(self, m): self.ir1_info = m
    def _sex(self, m):
        R = np.asarray(m.rotation, dtype=np.float64).reshape(3, 3)
        T = np.asarray(m.translation, dtype=np.float64).reshape(3)
        self.extr = (R, T)

    def have_all(self):
        return all(x is not None for x in (
            self.color, self.color_info, self.ir1_info, self.extr))


def call_fs(node, *, align_to_color=False):
    req = FoundationStereoDepth.Request()
    req.align_to_color = bool(align_to_color)
    req.want_pointcloud = False
    req.want_debug_jpeg = False
    req.z_far = 10.0
    fut = node.fs.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=60.0)
    resp = fut.result()
    if resp is None or resp.status != 0:
        print(f"FS failed: {getattr(resp, 'error_msg', '?')}"); return None
    print(f"FS (align={align_to_color}) forward={resp.forward_ms:.1f}ms shape="
          f"{(resp.depth_image.height, resp.depth_image.width)}")
    return node.bridge.imgmsg_to_cv2(resp.depth_image, "passthrough").astype(np.float32)


def overlay_depth_edges(color_bgr, depth_m, *, edge_color_bgr,
                        z_max=3.0, edge_thresh=15):
    """Draw depth edges in `edge_color` over `color_bgr`."""
    Z = depth_m.copy()
    Z[Z <= 0] = 0
    Z8 = (np.clip(Z / z_max, 0, 1) * 255).astype(np.uint8)
    gx = cv2.Sobel(Z8, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(Z8, cv2.CV_32F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    mask = g > edge_thresh
    # Dilate by 1 to make edges visible
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)
    out = color_bgr.copy()
    out[mask] = edge_color_bgr
    return out, mask


def make_panel(img, caption, h=24):
    s = np.full((h, img.shape[1], 3), 20, dtype=np.uint8)
    cv2.putText(s, caption, (10, 17), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return np.concatenate([s, img], axis=0)


def hcat(imgs, gap=8):
    sep = np.full((imgs[0].shape[0], gap, 3), 30, dtype=np.uint8)
    out = imgs[0]
    for x in imgs[1:]:
        out = np.concatenate([out, sep, x], axis=1)
    return out


def vcat(imgs, gap=8):
    sep = np.full((gap, imgs[0].shape[1], 3), 30, dtype=np.uint8)
    out = imgs[0]
    for x in imgs[1:]:
        out = np.concatenate([out, sep, x], axis=0)
    return out


def main():
    rclpy.init()
    node = Cap(CAM)
    if not node.fs.wait_for_service(timeout_sec=15):
        print("FS service unavailable"); return 1

    t0 = time.time()
    while rclpy.ok() and not node.have_all():
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - t0 > 20:
            print("timeout"); return 1
    for _ in range(15):
        rclpy.spin_once(node, timeout_sec=0.05)

    color = node.bridge.imgmsg_to_cv2(node.color, "bgr8")
    K_color = np.asarray(node.color_info.k, dtype=np.float64).reshape(3, 3)
    K_ir1 = np.asarray(node.ir1_info.k, dtype=np.float64).reshape(3, 3)
    R, T = node.extr
    D_color = list(node.color_info.d); D_ir = list(node.ir1_info.d)
    H_c, W_c = color.shape[:2]
    H_i, W_i = (node.ir1_info.height, node.ir1_info.width)
    print(f"color {color.shape}  IR {H_i}x{W_i}")

    # Pull FS depth on IR1 grid.
    fs_ir = call_fs(node)
    if fs_ir is None: return 1
    print(f"fs_ir {fs_ir.shape}  coverage={(fs_ir > 0).mean():.1%}")
    if fs_ir.shape != (H_i, W_i):
        # FS used scaled-down internal grid.  Use native IR size as ir_hw.
        H_i_fs, W_i_fs = fs_ir.shape
    else:
        H_i_fs, W_i_fs = H_i, W_i

    # ----- LEGACY path: reproject_ir_to_color -----
    # Scale K_ir to the FS-output IR resolution.
    K_ir_scaled = K_ir1.copy()
    K_ir_scaled[:2] *= (W_i_fs / W_i)
    print(f"K_ir scaled by {W_i_fs / W_i:.4f}")
    legacy_color_raw = reproject_ir_to_color(
        depth_ir=fs_ir.astype(np.float32),
        K_ir=K_ir_scaled.astype(np.float32),
        K_color=K_color.astype(np.float32),
        R_ir_to_color=R.astype(np.float32),
        T_ir_to_color=T.astype(np.float32),
        out_hw=(H_c, W_c))
    # Hole-fill via dilation of valid depth into Z=0 neighbours so the
    # Sobel edges reflect *real* depth structure, not the sparsity
    # pattern. This is what FS dev notes recommend for downstream use.
    valid = (legacy_color_raw > 0).astype(np.uint8)
    legacy_color = legacy_color_raw.copy()
    # Iterative dilation with nearest valid value — 3 passes of (3x3
    # max-of-valid) fills the ~89% holes between scattered IR projections.
    for _ in range(3):
        # cv2.dilate on the depth itself: each invalid pixel gets the
        # max of its 3x3 neighbours' depths. For mostly-isolated valid
        # pixels in a sea of zeros, this propagates the depth outwards.
        dilated_depth = cv2.dilate(legacy_color, np.ones((3, 3), np.uint8))
        dilated_valid = cv2.dilate(valid, np.ones((3, 3), np.uint8))
        # Only fill where invalid; keep original valid pixels.
        fill_mask = (valid == 0) & (dilated_valid > 0)
        legacy_color = np.where(fill_mask, dilated_depth, legacy_color)
        valid = (legacy_color > 0).astype(np.uint8)
    print(f"legacy_color_raw coverage={(legacy_color_raw > 0).mean():.1%}  "
          f"after hole-fill={(legacy_color > 0).mean():.1%}")

    # ----- OPTION 1 path: rs.align via software_device, end-to-end via FS service -----
    # Ask the FS node to do the alignment itself (uses RealsenseAligner inside).
    rsalign_color = call_fs(node, align_to_color=True)
    if rsalign_color is None: return 1
    if rsalign_color.shape != (H_c, W_c):
        rsalign_color = cv2.resize(rsalign_color, (W_c, H_c),
                                    interpolation=cv2.INTER_NEAREST)
    print(f"rsalign_color (FS svc) shape={rsalign_color.shape}  "
          f"coverage={(rsalign_color > 0).mean():.1%}")

    # ----- visualisation -----
    # Depth-edges over color (legacy vs option1):
    legacy_overlay, m_legacy = overlay_depth_edges(
        color, legacy_color, edge_color_bgr=(0, 100, 255), z_max=3.0)
    rsalign_overlay, m_rs = overlay_depth_edges(
        color, rsalign_color, edge_color_bgr=(0, 255, 255), z_max=3.0)

    # Side-by-side
    panels = [
        make_panel(color, "color"),
        make_panel(legacy_overlay, "legacy reproject_ir_to_color: depth edges (orange) on color"),
        make_panel(rsalign_overlay, "Option 1: rs.align(software_device): depth edges (yellow) on color"),
    ]
    cv2.imwrite(f"{OUT}/edges_over_color_row.png", hcat(panels))

    # Highlight crops where the bleed was visible. Pick the top-3
    # columns with the widest legacy bleeds — auto-detect by looking
    # at where legacy depth edges miss color edges by the largest gap.
    color_gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    color_edges = cv2.Sobel(color_gray, cv2.CV_32F, 1, 0, ksize=3)
    color_edges = (np.abs(color_edges) > 30)

    # 2x2 zoom comparison — pick the central region around middle of image.
    crop_y0, crop_y1 = max(0, H_c//2 - 120), min(H_c, H_c//2 + 120)
    crop_x0, crop_x1 = W_c//4, 3*W_c//4
    sub_color = color[crop_y0:crop_y1, crop_x0:crop_x1]
    sub_legacy = legacy_overlay[crop_y0:crop_y1, crop_x0:crop_x1]
    sub_rsalign = rsalign_overlay[crop_y0:crop_y1, crop_x0:crop_x1]
    # Zoom 2x for visibility
    zf = 2
    sub_color = cv2.resize(sub_color, (sub_color.shape[1]*zf, sub_color.shape[0]*zf),
                           interpolation=cv2.INTER_NEAREST)
    sub_legacy = cv2.resize(sub_legacy, (sub_legacy.shape[1]*zf, sub_legacy.shape[0]*zf),
                            interpolation=cv2.INTER_NEAREST)
    sub_rsalign = cv2.resize(sub_rsalign, (sub_rsalign.shape[1]*zf, sub_rsalign.shape[0]*zf),
                             interpolation=cv2.INTER_NEAREST)
    zoom_row = hcat([
        make_panel(sub_color, "color (zoomed 2x)"),
        make_panel(sub_legacy, "legacy depth edges (orange)"),
        make_panel(sub_rsalign, "Option 1 depth edges (yellow)"),
    ])
    cv2.imwrite(f"{OUT}/edges_over_color_zoom.png", zoom_row)

    # ----- quantitative: how many edge-pixels of the depth land on a
    # color-image edge?  Higher = depth edges agree with color edges.
    # We dilate the color-edge mask slightly to allow ±1 px slack.
    color_edges_d = cv2.dilate(color_edges.astype(np.uint8),
                                np.ones((3, 3), np.uint8)).astype(bool)
    # Per-shift sweep: how many of the depth edges fall on the color
    # edges when the depth-edge mask is shifted by dx?
    def shift_bool(m, dx):
        out = np.zeros_like(m)
        if dx >= 0:
            out[:, dx:] = m[:, :m.shape[1]-dx]
        else:
            out[:, :m.shape[1]+dx] = m[:, -dx:]
        return out

    def sweep_overlap(depth_edge_mask):
        n = depth_edge_mask.sum()
        sweep = []
        for dx in range(-15, 16):
            shifted = shift_bool(depth_edge_mask, dx)
            overlap = (shifted & color_edges_d).sum()
            sweep.append((dx, int(overlap),
                          float(overlap / max(int(shifted.sum()), 1))))
        return n, sweep

    n_legacy, sweep_legacy = sweep_overlap(m_legacy)
    n_rs, sweep_rs = sweep_overlap(m_rs)

    print()
    def best_of(sweep):
        return max(sweep, key=lambda x: x[2])
    print(f"legacy: n_edges={n_legacy}  best_overlap={best_of(sweep_legacy)}  "
          f"@dx=0: {[s for s in sweep_legacy if s[0]==0][0]}")
    print(f"rsalign: n_edges={n_rs}  best_overlap={best_of(sweep_rs)}  "
          f"@dx=0: {[s for s in sweep_rs if s[0]==0][0]}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4), dpi=120, facecolor="white")
        ax.plot([d for d,_,_ in sweep_legacy], [p for _,_,p in sweep_legacy],
                marker="o", lw=1.2, label=f"legacy ({CAM})")
        ax.plot([d for d,_,_ in sweep_rs], [p for _,_,p in sweep_rs],
                marker="s", lw=1.2, label=f"Option 1 rs.align ({CAM})")
        ax.set_xlabel("horizontal shift of depth edges (px)")
        ax.set_ylabel("fraction of depth edges lying on color-image edges (±1 px)")
        ax.set_title(f"{CAM}: do depth edges hit COLOR edges?  "
                     "higher = better aligned to color")
        ax.axvline(0, color="k", lw=0.5); ax.grid(alpha=0.3); ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUT}/edges_overlap_sweep.png", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT}/edges_overlap_sweep.png")
    except Exception as exc:
        print(f"plot skipped: {exc}")

    print(f"\nOutputs in {OUT}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
