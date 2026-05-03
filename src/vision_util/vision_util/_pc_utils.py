"""Shared CUDA point-cloud helpers for vision_util nodes.

Extracted from ``get_orbbec_pc.py`` so ``monocular_depth_pc.py`` can reuse
the deprojection + PointCloud2 packing path. Pure helpers — no ROS state,
no logging.
"""
from __future__ import annotations

import numpy as np
import torch
from sensor_msgs.msg import PointCloud2, PointField


def make_pc2_xyz(header, xyz_np: np.ndarray) -> PointCloud2:
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = xyz_np.shape[0]
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = np.ascontiguousarray(xyz_np, dtype=np.float32).tobytes()
    return msg


def make_pc2_xyzrgb(
    header,
    xyz_np: np.ndarray,
    rgb_packed_f32_np: np.ndarray,
) -> PointCloud2:
    n = xyz_np.shape[0]
    arr = np.empty((n, 4), dtype=np.float32)
    arr[:, :3] = xyz_np
    arr[:, 3] = rgb_packed_f32_np
    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = n
    msg.fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = msg.point_step * n
    msg.is_dense = True
    msg.data = arr.tobytes()
    return msg


def build_xy_table_cuda(
    h: int, w: int, fx: float, fy: float, cx: float, cy: float,
    device: torch.device,
) -> torch.Tensor:
    """Pinhole (u-cx)/fx, (v-cy)/fy table, shape (H, W, 2), float32 on device."""
    us = torch.arange(w, device=device, dtype=torch.float32)
    vs = torch.arange(h, device=device, dtype=torch.float32)
    uu, vv = torch.meshgrid(us, vs, indexing='xy')
    x_table = (uu - cx) / fx
    y_table = (vv - cy) / fy
    return torch.stack([x_table, y_table], dim=-1)


def pack_rgb_u8_to_float32_cuda(rgb_u8: torch.Tensor) -> torch.Tensor:
    """(H, W, 3) uint8 RGB -> (H, W) float32 with packed uint32 bits."""
    r = rgb_u8[..., 0].to(torch.int32)
    g = rgb_u8[..., 1].to(torch.int32)
    b = rgb_u8[..., 2].to(torch.int32)
    return (r << 16) | (g << 8) | b
