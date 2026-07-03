"""Resolve the calibration apply target for the Calibrate tab.

Since the per-robot split (tk25_basic ``db1524a`` + Task 3 / Phase 1c), the
apply target is exactly ONE location, keyed by ``$ROBOT_NAME``::

    src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/pan_tilt/
        pan_tilt_overrides.xacro + offsets.yaml

The shared xacros under ``tinker_urdf/`` are no longer patchable targets —
``pan_tilt.urdf.xacro`` auto-includes the per-robot overrides file at
xacro-parse time, so writing the per-robot pair IS the complete deployment.

``list_targets()`` returns a single-entry descriptor list (same
``[{label, path, exists, ...}]`` shape as before so the web UI renders it
unchanged). When ``ROBOT_NAME`` is unset or the robot has no profile, the
entry carries ``exists=False`` and a label explaining the fix; the actual
refusal is enforced server-side by ``apply_to_urdf`` (HTTP 400 in calib_web).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

from .apply_to_urdf import (
    BUILD_COMMAND,
    OVERRIDES_XACRO_NAME,
    WORKSPACE_HINT,
    CalibrationApplyError,
    resolve_per_robot_dir,
)


@dataclass
class UrdfTarget:
    label: str
    path: str
    exists: bool
    form: str           # "per-robot" — the only apply-target form left
    build_package: str   # which colcon package to rebuild after applying
    build_command: str   # exact shell command the operator should run
    workspace_hint: str  # cwd hint for the rebuild (which workspace root)

    def to_dict(self) -> dict:
        return asdict(self)


def list_targets() -> list[UrdfTarget]:
    """Single-entry list: ``robots/<ROBOT_NAME>/pan_tilt/``.

    ``exists`` is True only when the robot's ``pan_tilt_overrides.xacro`` is
    present in the tk25_basic source tree (every onboarded robot ships one).
    """
    common = dict(
        form="per-robot",
        build_package="tinker_robot_config",
        build_command=BUILD_COMMAND,
        workspace_hint=WORKSPACE_HINT,
    )
    robot = os.environ.get("ROBOT_NAME", "").strip()
    if not robot:
        return [UrdfTarget(
            label=("ROBOT_NAME not set — export ROBOT_NAME=tinker1|tinker2 "
                   "and restart calib_web"),
            path="",
            exists=False,
            **common,
        )]
    label = f"robots/{robot}/pan_tilt/ (per-robot apply target)"
    try:
        per_robot_dir = resolve_per_robot_dir(robot)
    except CalibrationApplyError as exc:
        return [UrdfTarget(
            label=label, path=f"<unresolved: {exc}>", exists=False, **common,
        )]
    return [UrdfTarget(
        label=label,
        path=str(per_robot_dir),
        exists=(per_robot_dir / OVERRIDES_XACRO_NAME).is_file(),
        **common,
    )]
