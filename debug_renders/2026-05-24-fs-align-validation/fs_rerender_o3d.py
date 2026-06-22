#!/usr/bin/env python3
"""Re-render cloud.ply with open3d's filament offscreen renderer.

Filters to Z <= 1.5 m and writes 5 view PNGs at 1600x1200.
"""
import os

import numpy as np
import open3d as o3d

Z_MAX = 1.5  # meters
PLY = "/home/tinker/tk25_ws/src/tk26_vision/debug_renders/2026-05-24-fs-align-validation/cloud.ply"
OUT = "/home/tinker/tk25_ws/src/tk26_vision/debug_renders/2026-05-24-fs-align-validation"
W, H = 1600, 1200

pcd = o3d.io.read_point_cloud(PLY)
pts = np.asarray(pcd.points)
rgb = np.asarray(pcd.colors)

near = pts[:, 2] <= Z_MAX
pcd_near = o3d.geometry.PointCloud()
pcd_near.points = o3d.utility.Vector3dVector(pts[near])
pcd_near.colors = o3d.utility.Vector3dVector(rgb[near])
print(f"after filter Z <= {Z_MAX} m: {near.sum()} points "
      f"(of {near.size}, kept {near.mean():.1%})")

centroid = np.asarray(pcd_near.points).mean(axis=0)
bbox = pcd_near.get_axis_aligned_bounding_box()
extent = bbox.get_extent()
radius = float(np.linalg.norm(extent)) / 2.0
print(f"centroid={centroid}  extent={extent}  radius={radius:.3f}")

renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
scene = renderer.scene
scene.set_background(np.array([0.04, 0.04, 0.04, 1.0]))
mat = o3d.visualization.rendering.MaterialRecord()
mat.shader = "defaultUnlit"
mat.point_size = 3.0
scene.add_geometry("cloud", pcd_near, mat)

# Camera: orbit around the centroid at radius * 1.6, looking back at centroid.
# OpenGL camera convention (open3d): looks down -Z by default; "up" is +Y.
# Our cloud is in the color-optical frame (x right, y down, z forward).
# Pick `up` = (0, -1, 0) so "up in image" stays "up in scene" (= -y).

def render_view(name, cam_offset):
    """cam_offset is from centroid, in scene coords."""
    eye = centroid + np.asarray(cam_offset)
    center = centroid
    # Color-optical frame: Y points DOWN, so real-world up is -Y.
    up = np.array([0.0, -1.0, 0.0])
    scene.camera.look_at(center, eye, up)
    # Use a narrower FOV to fill more of the frame with the cloud.
    aspect = W / H
    scene.camera.set_projection(
        35.0,                    # vertical FOV in degrees
        aspect,
        0.05, 100.0,             # near, far planes (m)
        o3d.visualization.rendering.Camera.FovType.Vertical,
    )
    img = renderer.render_to_image()
    o3d.io.write_image(f"{OUT}/view_{name}_close.png", img, 9)
    print(f"wrote {OUT}/view_{name}_close.png")

d = radius * 1.4  # camera distance from centroid

# Orbit camera around centroid. The original FS camera POV is at world
# origin, looking +Z. centroid lies at ~(0, 0, 0.6). So the "front" view
# is eye = centroid + (0, 0, -d) (i.e., camera at z = -d relative to
# centroid, looking forward into the cloud) — same direction as the
# original D435 capture but pulled back to fit the cloud.
import math

def rot_y(deg):
    r = math.radians(deg)
    return np.array([
        [math.cos(r), 0, math.sin(r)],
        [0, 1, 0],
        [-math.sin(r), 0, math.cos(r)],
    ])

def rot_x(deg):
    r = math.radians(deg)
    return np.array([
        [1, 0, 0],
        [0, math.cos(r), -math.sin(r)],
        [0, math.sin(r), math.cos(r)],
    ])

# Eye is at centroid + R @ (0, 0, -d). For R=I, eye is at z = centroid.z - d,
# i.e., between the original camera (origin) and the centroid — we look from
# that vantage toward centroid, replicating the FS source POV.
base = np.array([0.0, 0.0, -d])

# (name, R)
views = [
    ("front",      np.eye(3)),
    ("left_30",    rot_y(-30)),
    ("right_30",   rot_y(30)),
    ("top_45",     rot_x(-45)),
    ("bottom_30",  rot_x(20)),
]

for name, R in views:
    offset = R @ base
    render_view(name, offset)

renderer.scene.clear_geometry()
del renderer
print("done")
