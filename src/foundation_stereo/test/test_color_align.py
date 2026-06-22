"""Synthetic-data tests for color_align.reproject_ir_to_color.

These tests exercise the reprojection math without any GPU / model
dependency, so they run in plain pytest under any environment with
numpy installed.
"""

import numpy as np
import pytest

from foundation_stereo.color_align import reproject_ir_to_color


def _intrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)


def test_identity_extrinsics_preserves_depth_at_same_pixels():
    """With identity rotation, zero translation, and matching intrinsics,
    every IR1 pixel should map to its own coordinate in the color grid
    and carry the same depth value."""
    H, W = 60, 80
    depth_ir = np.full((H, W), 2.0, dtype=np.float32)  # uniform 2 m
    K_ir = _intrinsics()
    K_color = _intrinsics()
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)

    depth_color = reproject_ir_to_color(
        depth_ir, K_ir, K_color, R, T,
        out_hw=(H, W),
    )

    assert depth_color.shape == (H, W)
    assert depth_color.dtype == np.float32
    # All valid IR pixels project to themselves; allow zero holes only at
    # the borders (forward projection can leave one-pixel-wide gaps).
    interior = depth_color[2:-2, 2:-2]
    assert np.all(interior > 0), "interior should be fully filled"
    np.testing.assert_allclose(interior, 2.0, atol=1e-3)


def test_translation_shifts_projected_pixels():
    """With a +5 cm translation in X (in the IR1 optical frame), a planar
    surface at known depth should land at predictable shifted color pixels.

    For a point (X_ir, Y_ir, Z_ir) in IR1 coordinates, after applying
    P_c = R·P_ir + T with R=I and T=(0.05, 0, 0)^T:
      u_color = fx · (X_ir + 0.05) / Z_ir + cx
    With Z_ir = 1.0 m, fx = 500, a 5 cm X-shift translates to a 25-pixel
    column shift in the color grid.
    """
    H, W = 60, 80
    Z = 1.0
    depth_ir = np.full((H, W), Z, dtype=np.float32)
    K_ir = _intrinsics()
    K_color = _intrinsics()
    R = np.eye(3, dtype=np.float32)
    T = np.array([0.05, 0.0, 0.0], dtype=np.float32)  # 5 cm in X

    depth_color = reproject_ir_to_color(
        depth_ir, K_ir, K_color, R, T,
        out_hw=(H, W),
    )

    # Pixel u=10 in IR1 (X = (10 - 320) / 500 = -0.62 m) should project to
    # u_color = 500 * (-0.62 + 0.05) / 1.0 + 320 = 285.
    # Use the centre row to pick a representative column-shift signal.
    row = depth_color[H // 2, :]
    filled = np.where(row > 0)[0]
    assert filled.size > 0
    # Left edge of the filled band should be ~25 columns to the right of
    # IR1's left edge (col 0 → col 25 ± a couple pixels of rounding).
    assert 22 <= filled[0] <= 27, f"left edge shifted to {filled[0]} (expected ~25)"


def test_zero_depth_pixels_produce_holes():
    """Invalid (zero) depth in IR1 should not contribute; output cells
    not hit by any valid projection stay zero."""
    H, W = 40, 60
    depth_ir = np.zeros((H, W), dtype=np.float32)
    K_ir = _intrinsics()
    K_color = _intrinsics()
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)

    depth_color = reproject_ir_to_color(
        depth_ir, K_ir, K_color, R, T,
        out_hw=(H, W),
    )

    assert np.all(depth_color == 0.0)


def test_collision_keeps_nearer_z():
    """When two IR pixels project to the same color pixel, the nearer
    Z wins. Construct a 2-row scene where row 0 has Z=1 m and row 1
    has Z=2 m. K_color is given a near-zero fy so both rows' projected
    v_c rounds to 0 — both land at color (u, 0) and collide."""
    H, W = 2, 4
    depth_ir = np.array([
        [1.0, 1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0, 2.0],
    ], dtype=np.float32)
    K_ir = _intrinsics(fx=10.0, fy=10.0, cx=2.0, cy=0.0)
    # Tiny fy_color collapses both rows to v_c≈0 → forced collision.
    K_color = _intrinsics(fx=10.0, fy=0.01, cx=2.0, cy=0.0)
    R = np.eye(3, dtype=np.float32)
    T = np.zeros(3, dtype=np.float32)

    depth_color = reproject_ir_to_color(
        depth_ir, K_ir, K_color, R, T,
        out_hw=(1, W),
    )

    # All cells that received any projection should equal 1.0 (the nearer
    # value), never 2.0.
    filled = depth_color[depth_color > 0]
    assert filled.size > 0
    assert np.all(filled <= 1.0 + 1e-6)
