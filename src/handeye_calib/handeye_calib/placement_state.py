"""Per-board-placement state container for HandeyeWebNode.

Pure Python — no ROS, no numpy imports at module level (numpy used in type
hints only, imported locally where needed). Unit-testable in isolation.
"""
import re
from dataclasses import dataclass, field
from handeye_calib.handeye_collect import CaptureSession


@dataclass
class PlacementState:
    label: str
    session: CaptureSession
    thumbs: dict = field(default_factory=dict)           # {int idx: bytes jpeg}
    sample_joints: dict = field(default_factory=dict)    # {int idx: list[float]|None}
    sample_ts: dict = field(default_factory=dict)        # {int idx: float monotonic}
    sample_reproj_px: dict = field(default_factory=dict) # {int idx: float}
    sample_area_frac: dict = field(default_factory=dict) # {int idx: float}
    sample_depth_source: dict = field(default_factory=dict) # {int idx: str}
    anchor_obs: list = field(default_factory=list)       # [4x4 T_base_board]
    tbb_head: object = None                              # np.ndarray|None
    anchor_scatter: object = None                        # dict|None


def make_placement(label, **session_kwargs) -> PlacementState:
    return PlacementState(label=label, session=CaptureSession(**session_kwargs))


def slug_id(label: str, existing_ids) -> str:
    base = re.sub(r'[^a-z0-9_-]+', '_', label.lower()).strip('_') or "placement"
    if base not in existing_ids:
        return base
    k = 2
    while f"{base}_{k}" in existing_ids:
        k += 1
    return f"{base}_{k}"
