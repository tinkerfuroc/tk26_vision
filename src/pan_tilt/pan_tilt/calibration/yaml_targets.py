"""Resolve the YAML files that should receive runtime-offset patches.

Mirrors :mod:`urdf_targets` but for `pan_tilt.yaml`. The calibration's
`theta_p_offset_rad` / `theta_t_offset_rad` are runtime params consumed by
:class:`pan_tilt_state_publisher.PanTiltStatePublisherNode`; without them
the URDF chain mis-represents the camera pose at any non-zero firmware
tilt. `apply_to_urdf` patches both the URDF and this YAML in lockstep.

Currently a single-entry list (the `pan_tilt` package's installed config),
kept as a list for symmetry with `list_targets()` and to leave room for
future multi-config setups (e.g. per-robot YAML overlays).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class YamlTarget:
    label: str
    path: str
    exists: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _share(pkg: str) -> Optional[Path]:
    """Return the share directory for ``pkg`` or None if not installed."""
    try:
        from ament_index_python.packages import get_package_share_directory
        from ament_index_python.packages import PackageNotFoundError
    except ImportError:
        return None
    try:
        return Path(get_package_share_directory(pkg))
    except PackageNotFoundError:
        return None


def list_yaml_targets() -> list[YamlTarget]:
    """Single-entry list pointing at the installed `pan_tilt/config/pan_tilt.yaml`.

    With colcon `--symlink-install` the installed file is a symlink back to
    `src/tk26_vision/src/pan_tilt/config/pan_tilt.yaml`, so writing through
    the install path also updates source. Without symlink-install the
    operator should re-run `colcon build --packages-select pan_tilt` after
    apply_to_urdf so the source change propagates.
    """
    share = _share("pan_tilt")
    if share is None:
        return [YamlTarget(
            label="pan_tilt config (not installed)",
            path="<pan_tilt not installed>",
            exists=False,
        )]
    path = share / "config" / "pan_tilt.yaml"
    return [YamlTarget(
        label="pan_tilt runtime offsets",
        path=str(path),
        exists=path.is_file(),
    )]
