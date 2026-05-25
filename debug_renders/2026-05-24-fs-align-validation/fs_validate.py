#!/usr/bin/env python3
"""Foundation_stereo color-alignment validation.

1. Calls /foundation_stereo/get_depth with align_to_color=true + want_pointcloud=true.
2. Subscribes to /camera/xarm_camera/color/image_raw to grab a color frame.
3. Builds a colored PLY by sampling color image at each (u,v) the depth lives at.
4. Renders 5 views via open3d (headless) to /tmp/fs_views/*.png.
5. Also dumps the raw depth as a colormap-overlaid JPEG for sanity.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from tinker_vision_msgs_26.srv import FoundationStereoDepth

OUT_DIR = "/tmp/fs_views"
os.makedirs(OUT_DIR, exist_ok=True)


def img_to_np(msg) -> np.ndarray:
    """Decode sensor_msgs/Image into numpy. No cv_bridge."""
    H, W = msg.height, msg.width
    if msg.encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(H, W, 3)
        if msg.encoding == "bgr8":
            arr = arr[..., ::-1]
        return arr.copy()
    if msg.encoding == "32FC1":
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape(H, W)
        return arr.copy()
    if msg.encoding in ("mono8", "8UC1"):
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(H, W)
        return arr.copy()
    raise ValueError(f"unsupported encoding: {msg.encoding}")


class Validator(Node):
    def __init__(self):
        super().__init__("fs_validate")
        self._color = None
        self.create_subscription(
            Image,
            "/camera/xarm_camera/color/image_raw",
            self._on_color,
            qos_profile_sensor_data,
        )
        self._cli = self.create_client(FoundationStereoDepth, "/foundation_stereo/get_depth")

    def _on_color(self, msg):
        self._color = msg


def main():
    rclpy.init()
    node = Validator()

    print("waiting for service + color frames...")
    t0 = time.time()
    while time.time() - t0 < 15.0:
        rclpy.spin_once(node, timeout_sec=0.1)
        if node._color is not None and node._cli.service_is_ready():
            break
    if node._color is None:
        print("ERROR: no color frame in 15s")
        rclpy.shutdown()
        sys.exit(1)
    if not node._cli.service_is_ready():
        print("ERROR: service /foundation_stereo/get_depth not ready in 15s")
        rclpy.shutdown()
        sys.exit(1)

    color_img = img_to_np(node._color)
    print(f"got color frame: {color_img.shape} dtype={color_img.dtype}")

    # Call the service.
    req = FoundationStereoDepth.Request()
    req.align_to_color = True
    req.want_pointcloud = True
    req.want_debug_jpeg = False
    print("calling /foundation_stereo/get_depth ...")
    fut = node._cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=60.0)
    if not fut.done():
        print("ERROR: service call timed out")
        rclpy.shutdown()
        sys.exit(1)
    resp = fut.result()
    print(f"status={resp.status} error_msg={resp.error_msg!r}")
    print(f"forward_ms={resp.forward_ms:.1f} load_s={resp.load_s:.2f} "
          f"end_to_end_s={resp.end_to_end_s:.2f}")
    print(f"model_used={resp.model_used} trt_variant_used={resp.trt_variant_used}")
    if resp.status != 0:
        rclpy.shutdown()
        sys.exit(1)

    depth = img_to_np(resp.depth_image)
    H, W = depth.shape
    print(f"depth: {depth.shape} dtype={depth.dtype} "
          f"valid={int((depth > 0).sum())}/{depth.size} "
          f"range={depth[depth > 0].min():.3f}..{depth[depth > 0].max():.3f} m"
          if (depth > 0).any() else "")

    # Color image is in the same camera frame as the depth (color_optical),
    # at the same resolution. Sample colors at each depth pixel.
    assert color_img.shape[:2] == (H, W), \
        f"color shape {color_img.shape[:2]} != depth shape {(H, W)}"

    # Build (N, 3) points using K_color (from response.camera_info).
    K = np.asarray(resp.camera_info.p, dtype=np.float32).reshape(3, 4)[:3, :3]
    if not np.any(K):
        K = np.asarray(resp.camera_info.k, dtype=np.float32).reshape(3, 3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    print(f"K: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    vv, uu = np.indices((H, W), dtype=np.float32)
    Z = depth
    valid = Z > 0
    X = (uu - cx) * Z / fx
    Y = (vv - cy) * Z / fy
    pts = np.stack([X, Y, Z], axis=-1)[valid]
    rgb = color_img[valid] / 255.0

    print(f"building cloud: N={pts.shape[0]} points")

    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))

    ply_path = f"{OUT_DIR}/cloud.ply"
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"wrote {ply_path}")

    # Also save the color frame (jpg) and a depth-viz (turbo colormap).
    import cv2
    cv2.imwrite(f"{OUT_DIR}/color.jpg",
                cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
    z_for_viz = np.where(Z > 0, Z, np.nan)
    z_min, z_max = (np.nanmin(z_for_viz), np.nanmax(z_for_viz)) if np.any(valid) else (0, 1)
    norm = (np.clip((Z - z_min) / max(z_max - z_min, 1e-3), 0, 1) * 255).astype(np.uint8)
    norm[~valid] = 0
    depth_viz = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    depth_viz[~valid] = 0
    cv2.imwrite(f"{OUT_DIR}/depth_viz.jpg", depth_viz)
    print(f"wrote {OUT_DIR}/color.jpg and {OUT_DIR}/depth_viz.jpg")

    # Headless render via matplotlib — multiple views around the cloud's centroid.
    # Color optical frame: x right, y down, z forward. matplotlib 3D defaults to
    # x right, y forward, z up — so we remap axes for an intuitive top/side view.
    centroid = pts.mean(axis=0)
    print(f"centroid={centroid}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Subsample for plotting speed (matplotlib chokes above ~50k points).
    N_render = min(pts.shape[0], 40000)
    if pts.shape[0] > N_render:
        idx = np.random.default_rng(0).choice(pts.shape[0], N_render, replace=False)
        pts_r = pts[idx]
        rgb_r = rgb[idx]
    else:
        pts_r = pts
        rgb_r = rgb

    # Remap from camera-optical → world-friendly (right, up, forward):
    # x stays x, y_world = -y_camera (flip down→up), z stays z (forward).
    Xc, Yc, Zc = pts_r[:, 0], -pts_r[:, 1], pts_r[:, 2]

    views = [
        # (name, elev_deg, azim_deg)
        ("front",       0,    -90),  # camera looking forward (down +z)
        ("left_30",    10,   -120),
        ("right_30",   10,    -60),
        ("top_45",     45,    -90),
        ("bottom_30", -30,    -90),
    ]
    for name, elev, azim in views:
        fig = plt.figure(figsize=(12, 9), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(Xc, Zc, Yc, c=rgb_r, s=0.5, marker=".", linewidths=0)
        ax.set_xlabel("X (right, m)")
        ax.set_ylabel("Z (forward, m)")
        ax.set_zlabel("Y (up, m)")
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"FoundationStereo aligned depth — view: {name} "
                     f"(elev={elev}°, azim={azim}°)")
        ax.set_box_aspect((1, 1, 0.6))
        # Equalize axis spans so things don't look squished.
        for axis_setter, vals in (
                (ax.set_xlim, Xc), (ax.set_ylim, Zc), (ax.set_zlim, Yc)):
            mid = (vals.min() + vals.max()) / 2
            r = max(vals.max() - vals.min(), 0.1)
            axis_setter(mid - r/2, mid + r/2)
        out_png = f"{OUT_DIR}/view_{name}.png"
        fig.savefig(out_png, bbox_inches="tight", facecolor="black")
        plt.close(fig)
        print(f"wrote {out_png}")

    node.destroy_node()
    rclpy.shutdown()
    print("done")


if __name__ == "__main__":
    main()
