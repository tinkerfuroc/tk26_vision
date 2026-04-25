"""Shared safety envelope for xArm motion during calibration.

Used by both `calibrate_collect` (autonomous collector) and `calib_web` (the
waypoint-authoring UI) to keep rejection rules in one place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SafetyEnvelope:
    z_floor_m: float = 0.25
    mast_xy_center: tuple = (-0.275, -0.013)
    mast_radius_m: float = 0.12
    mast_z_max: float = 1.70

    def validate(self, t_base_ee: np.ndarray) -> Optional[str]:
        """Return None if the proposed EE pose is safe; else a rejection reason."""
        z = float(t_base_ee[2, 3])
        if z < self.z_floor_m:
            return f"z={z:.3f} below floor {self.z_floor_m:.3f}"
        dx = t_base_ee[0, 3] - self.mast_xy_center[0]
        dy = t_base_ee[1, 3] - self.mast_xy_center[1]
        r = math.hypot(dx, dy)
        if r < self.mast_radius_m and z < self.mast_z_max:
            return (
                f"xy=({t_base_ee[0,3]:.3f},{t_base_ee[1,3]:.3f}) inside mast "
                f"exclusion (r={r:.3f} < {self.mast_radius_m})"
            )
        return None

    def to_dict(self) -> dict:
        return {
            "z_floor_m": self.z_floor_m,
            "mast_xy_center": list(self.mast_xy_center),
            "mast_radius_m": self.mast_radius_m,
            "mast_z_max": self.mast_z_max,
        }
