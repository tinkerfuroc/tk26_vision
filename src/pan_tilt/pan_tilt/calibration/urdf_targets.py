"""Resolve the xacro files that should receive the calibration patch.

Two distinct URDFs define the pan-tilt geometry in this workspace:

* ``tk25_basic`` exposes a *macro* form (``<xacro:macro name="pan_tilt_macro" …>``)
  at ``<share>/tinker_urdf/src/pan_tilt.urdf.xacro`` -- the macro is parameterised
  on ``attach_xyz`` / ``attach_rpy`` and is the geometry that the main robot
  bringup loads (combined mobile-manipulator URDF).
* The tk26 ``pan_tilt`` package ships a *standalone* form at
  ``<share>/pan_tilt/urdf/pan_tilt.urdf.xacro`` for dev bringup
  (``pan_tilt.launch.py``).

Both need to stay in sync so RViz and the live robot agree. The Calibrate tab
lets the operator diff against either target. ``apply_to_urdf.py`` itself
auto-detects which form it was given (see ``MACRO_DECL_RE`` in that module)
so we just need to feed it the right file.

This helper resolves both share directories via ``ament_index_python`` and
returns a uniform ``[{label, path, exists, form}, ...]`` descriptor the web UI
can render directly. It never raises on missing packages -- a missing share
dir just yields ``exists=False`` so the UI can grey that option out.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class UrdfTarget:
    label: str
    path: str
    exists: bool
    form: str          # "macro" or "standalone" -- documents the xacro shape
    build_package: str # which colcon package to rebuild after applying the patch

    def to_dict(self) -> dict:
        return asdict(self)


def _share(pkg: str) -> Optional[Path]:
    """Return the share directory for ``pkg`` or None if the package is not
    installed in this overlay. Tolerates ament_index_python being missing
    (e.g. during headless unit tests)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        from ament_index_python.packages import PackageNotFoundError
    except ImportError:
        return None
    try:
        return Path(get_package_share_directory(pkg))
    except PackageNotFoundError:
        return None


def _target(label: str, pkg: str, rel: str, form: str, build_package: str) -> UrdfTarget:
    share = _share(pkg)
    if share is None:
        return UrdfTarget(label=label, path=f"<{pkg} not installed>",
                          exists=False, form=form, build_package=build_package)
    path = share / rel
    return UrdfTarget(label=label, path=str(path), exists=path.is_file(),
                      form=form, build_package=build_package)


def list_targets() -> list[UrdfTarget]:
    """Two-entry list, macro first (authoritative runtime URDF).

    Both should be patched so dev and production stay in lockstep; we surface
    both and let the operator apply whichever is relevant.
    """
    return [
        _target(
            label="tk25_basic macro (authoritative)",
            pkg="tinker_urdf",
            rel="src/pan_tilt.urdf.xacro",
            form="macro",
            build_package="tinker_urdf",
        ),
        _target(
            label="tk26_vision standalone (dev bringup)",
            pkg="pan_tilt",
            rel="urdf/pan_tilt.urdf.xacro",
            form="standalone",
            build_package="pan_tilt",
        ),
    ]
