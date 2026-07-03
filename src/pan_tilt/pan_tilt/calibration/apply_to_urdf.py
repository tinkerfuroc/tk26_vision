"""Apply pan-tilt calibration results to the PER-ROBOT config files.

The apply target is exactly two files in the **tk25_basic SOURCE tree**,
keyed by ``$ROBOT_NAME``::

    src/tk25_basic/src/tinker_robot_config/robots/<ROBOT_NAME>/pan_tilt/
        pan_tilt_overrides.xacro   <- mount geometry (4 xacro properties)
        offsets.yaml               <- runtime joint offsets (theta_p/theta_t)

``tinker_urdf/src/pan_tilt.urdf.xacro`` auto-includes the per-robot overrides
file at xacro-parse time whenever ``ROBOT_NAME`` is set (tk25_basic
``db1524a``), so writing these two files is the complete deployment — the
shared xacros under ``tinker_urdf/`` are NEVER touched, and tinker1/tinker2
calibrations can no longer overwrite each other.

Written properties (complete-file render, no regex patching):

- ``pan_tilt_attach_xyz``      <- ``t_a`` (base_link -> pan-axis translation)
- ``pan_tilt_attach_rpy``      <- ``t_a_rotvec`` when non-trivial (>1e-6 norm),
  otherwise COPIED from the current per-robot overrides file — a calibration
  that didn't fit the pan-axis tilt never zeroes one a prior run wrote.
- ``pan_tilt_camera_mount_xyz`` <- ``t_b_trans``
- ``pan_tilt_camera_mount_rpy`` <- ``t_b_rotvec`` when non-trivial, otherwise
  copied from the current file (same preserve rule). Non-trivial values go
  through the forward-camera invariant (|yaw| < pi/2 unless
  ``allow_flipped_camera``).
- ``offsets.yaml`` ``pan_offset_rad``/``tilt_offset_rad`` <- the solve's
  ``theta_p_offset_rad``/``theta_t_offset_rad``, wrapped to (-pi, pi].

Both files travel together (atomic pair with ``.old-<ts>`` backups, rollback
on partial failure): the ``(theta_t_offset, T_B)`` pair is degenerate and a
half-applied calibration models the camera ~180 deg wrong (the 2026-04-30 /
2026-06-27 incidents).

CLI::

    python -m pan_tilt.calibration.apply_to_urdf --results chain.json|polish.json \
        [--basic-root PATH] [--allow-flipped-camera]

Refuses when ``ROBOT_NAME`` is unset. After a successful apply, rebuild
``tinker_robot_config`` and relaunch robot_state_publisher + the pan_tilt
state_publisher.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .utils import rotvec_to_xyz_euler, wrap_to_pi


# ---- forward-camera invariant ----------------------------------------------
#
# camera_mount rpy yaw must be within ±π/2 of zero for a forward-facing
# head camera (the only configuration we ship). A value near ±π is the
# 2026-04-30 backward-camera bug — refuse to write it unless the operator
# explicitly opts in via `allow_flipped_camera=True` (or the equivalent CLI
# flag), which widens the bound to ±π for genuinely flipped hardware.

_FORWARD_YAW_LIMIT_RAD = math.pi / 2.0


class CalibrationApplyError(RuntimeError):
    """Refusal to write a calibration (ROBOT_NAME unset, missing per-robot
    profile, or forward-camera invariant violation).

    Distinct from :class:`ValueError` so callers can catch it surgically and
    surface a focused operator message (calib_web maps it to HTTP 400).
    """


# Back-compat alias for callers that prefer the short name.
ApplyError = CalibrationApplyError


# The two per-robot files this module owns, and the rebuild that deploys them.
OVERRIDES_XACRO_NAME = "pan_tilt_overrides.xacro"
OFFSETS_YAML_NAME = "offsets.yaml"
BUILD_COMMAND = "tkbuild tk25_basic --packages-select tinker_robot_config"
WORKSPACE_HINT = (
    "run from the workspace root (e.g. ~/tk25_ws); after the rebuild, "
    "relaunch robot_state_publisher + pan_tilt state_publisher so the new "
    "geometry and offsets load"
)


def _fmt_triplet(v) -> str:
    return f"{v[0]:.6g} {v[1]:.6g} {v[2]:.6g}"


def _load_params(results_path: Path) -> dict:
    blob = json.loads(results_path.read_text())
    if "params" in blob:
        return blob["params"]
    raise ValueError(f"{results_path} has no 'params' key")


def _rotvec_to_rpy_str(t_b_rotvec, allow_flipped_camera: bool) -> str | None:
    """Convert t_b_rotvec → URDF rpy triplet, enforcing the forward-camera invariant.

    Returns ``None`` when the rotvec is sub-threshold (the caller preserves
    the existing rpy), otherwise returns the formatted "roll pitch yaw" string.

    Raises :class:`CalibrationApplyError` when the resulting yaw is outside
    ±π/2 and `allow_flipped_camera` is False. This is the universal failsafe —
    it catches every code path that produces a flipped rotvec, regardless of
    which solver, warm-start, or manual-edit fed it.
    """
    rotvec = np.asarray(t_b_rotvec)
    if np.linalg.norm(rotvec) <= 1e-6:
        return None
    euler = rotvec_to_xyz_euler(rotvec)
    yaw = float(euler[2])
    if abs(yaw) > _FORWARD_YAW_LIMIT_RAD and not allow_flipped_camera:
        raise CalibrationApplyError(
            f"Refusing to write calibration: camera_mount fitted yaw = "
            f"{yaw:.4f} rad ({math.degrees(yaw):.1f}°). Forward-facing "
            f"cameras must have |yaw| < π/2 (90°). This usually means the "
            f"optical→body convention was missed in the Phase-1 hand-eye "
            f"warm start. To override (e.g. you genuinely mounted the "
            f"camera backward), pass --allow-flipped-camera. The existing "
            f"per-robot files are untouched."
        )
    return _fmt_triplet(euler)


def _pan_axis_rpy_str(t_a_rotvec) -> str | None:
    """Convert the base→pan-axis rotvec (`t_a_rotvec`) to a URDF rpy triplet for
    the pan-axis attach. Returns ``None`` when sub-threshold (>1e-6 norm), so a
    zero/absent tilt preserves the existing rpy. No forward-camera yaw guard —
    this is the pan axis, not the camera (and the fit holds its Z/yaw at 0). The
    rotvec→xyz-euler→URDF-rpy path is the same one validated for the camera
    mount, so the emitted rpy reproduces transform_matrix(t_a_rotvec, ·)."""
    rotvec = np.asarray(t_a_rotvec, dtype=float)
    if np.linalg.norm(rotvec) <= 1e-6:
        return None
    return _fmt_triplet(rotvec_to_xyz_euler(rotvec))


# ---- per-robot target resolution -------------------------------------------


def _require_robot_name() -> str:
    """Return the target robot from $ROBOT_NAME, refusing when unset."""
    robot = os.environ.get("ROBOT_NAME", "").strip()
    if not robot:
        raise CalibrationApplyError(
            "ROBOT_NAME not set — refusing to apply calibration. Export "
            "ROBOT_NAME=tinker1|tinker2 (or source "
            "src/tk25_basic/tools/robot-env.sh) so the calibration lands in "
            "that robot's per-robot files and cannot overwrite another "
            "robot's."
        )
    return robot


def _workspace_robots_root() -> Path:
    """Locate ``src/tk25_basic/src/tinker_robot_config/robots`` on disk.

    Ported from handeye_calib.handeye_web._tk25_basic_repo_root. Resolution
    order (first hit wins):

    1. Walk parents of THIS file (resolved), checking both
       ``<parent>/tk25_basic/...`` and ``<parent>/src/tk25_basic/...``.
       Covers source-tree runs (this module at
       ``src/tk26_vision/src/pan_tilt/pan_tilt/calibration/``).
    2. Walk parents of CWD with the same two prefixes — covers invocation
       from the workspace root.
    3. ``ament_index_python`` share of ``tinker_robot_config`` → walk up to
       ``install/``'s parent (the workspace root) — covers install-tree runs.

    Raises :class:`CalibrationApplyError` when every resolver fails. NEVER
    globs by filename (the per-robot layout has one identically-named file
    per robot, so a ``**/<name>`` glob is inherently ambiguous).
    """

    def _check(parent: Path) -> Optional[Path]:
        for prefix in ("", "src"):
            base = parent / prefix if prefix else parent
            cand = base / "tk25_basic" / "src" / "tinker_robot_config" / "robots"
            if cand.is_dir():
                return cand.resolve()
        return None

    here = Path(__file__).resolve()
    for parent in here.parents:
        hit = _check(parent)
        if hit is not None:
            return hit
    cwd = Path.cwd().resolve()
    for parent in (cwd, *cwd.parents):
        hit = _check(parent)
        if hit is not None:
            return hit
    try:
        from ament_index_python.packages import get_package_share_directory
        share = Path(get_package_share_directory("tinker_robot_config")).resolve()
        for parent in share.parents:
            if parent.name == "install":
                hit = _check(parent.parent)
                if hit is not None:
                    return hit
                break
    except Exception:
        pass
    raise CalibrationApplyError(
        "could not locate the tk25_basic SOURCE tree "
        "(src/tk25_basic/src/tinker_robot_config/robots) from this module, "
        "the current directory, or the install index — pass "
        "basic_root/--basic-root pointing at the tk25_basic package root."
    )


def resolve_per_robot_dir(robot: str, basic_root=None) -> Path:
    """Resolve ``robots/<robot>/pan_tilt/`` in the tk25_basic SOURCE tree.

    ``basic_root`` (when given) is the tk25_basic package root — the directory
    containing ``src/tinker_robot_config``. When None, the workspace is
    located by walking up from this module's resolved path (see
    :func:`_workspace_robots_root`). Raises :class:`CalibrationApplyError`
    when the tree or the robot's profile directory is missing.
    """
    if basic_root is not None:
        robots = Path(basic_root) / "src" / "tinker_robot_config" / "robots"
        if not robots.is_dir():
            raise CalibrationApplyError(
                f"{robots} is not a directory — basic_root must point at the "
                f"tk25_basic package root (the directory containing "
                f"src/tinker_robot_config)."
            )
    else:
        robots = _workspace_robots_root()
    robot_dir = robots / robot
    if not robot_dir.is_dir():
        raise CalibrationApplyError(
            f"no per-robot profile for ROBOT_NAME={robot!r}: {robot_dir} does "
            f"not exist. Onboard the robot first (copy an existing profile "
            f"under {robots} and run lint-profiles.sh)."
        )
    return robot_dir / "pan_tilt"


# ---- renderers (complete file contents — no regex patching of shared xacros)


_XACRO_PROP_RE = r'<xacro:property\s+name="{name}"\s+value="([^"]*)"'

_OVERRIDES_XACRO_TEMPLATE = """\
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="pan_tilt_overrides_{robot}">
  <!-- Per-robot pan-tilt calibration for {robot}. Auto-included by
       tinker_urdf/src/pan_tilt.urdf.xacro when ROBOT_NAME={robot}.
       Written by `python -m pan_tilt.calibration.apply_to_urdf` (calib_web
       Apply). Re-run the calibration + Apply to refresh; hand-edits are
       overwritten by the next Apply. -->
  <xacro:property name="pan_tilt_attach_xyz" value="{attach_xyz}"/>
  <xacro:property name="pan_tilt_attach_rpy" value="{attach_rpy}"/>
  <xacro:property name="pan_tilt_camera_mount_xyz" value="{camera_mount_xyz}"/>
  <xacro:property name="pan_tilt_camera_mount_rpy" value="{camera_mount_rpy}"/>
</robot>
"""

_OFFSETS_YAML_TEMPLATE = """\
# Per-robot pan-tilt runtime joint offsets for {robot}.
# Written by `python -m pan_tilt.calibration.apply_to_urdf` (calib_web Apply)
# from the SAME solve as pan_tilt_overrides.xacro — the (theta_t_offset, T_B)
# pair is degenerate and must always travel together.
pan_tilt:
  offsets:
    pan_offset_rad: {pan:.10f}
    tilt_offset_rad: {tilt:.10f}
"""


def _current_property(xacro_text: str, name: str, path: Path) -> str:
    """Extract an ``<xacro:property name=... value=.../>`` value from the
    CURRENT per-robot overrides file (used to preserve an rpy when the solve
    carries no rotation for it). Missing file content / property is a hard
    error — the per-robot layout seeds every robot with a complete file."""
    m = re.search(_XACRO_PROP_RE.format(name=re.escape(name)), xacro_text)
    if m is None:
        raise CalibrationApplyError(
            f"{path} has no <xacro:property name=\"{name}\"> — cannot "
            f"preserve the existing rpy for a trivial/absent rotvec. Re-seed "
            f"the per-robot overrides file (all four pan_tilt_* properties) "
            f"before applying."
        )
    return m.group(1)


def render_overrides_xacro(robot: str, values: dict) -> str:
    """Render the complete ``pan_tilt_overrides.xacro`` for ``robot``.

    ``values`` must carry the four property strings: ``attach_xyz``,
    ``attach_rpy``, ``camera_mount_xyz``, ``camera_mount_rpy``.
    """
    missing = [k for k in ("attach_xyz", "attach_rpy",
                           "camera_mount_xyz", "camera_mount_rpy")
               if not values.get(k)]
    if missing:
        raise CalibrationApplyError(
            f"render_overrides_xacro: missing value(s) {missing} for {robot}"
        )
    return _OVERRIDES_XACRO_TEMPLATE.format(robot=robot, **values)


def render_offsets_yaml(robot: str, pan_offset_rad: float,
                        tilt_offset_rad: float) -> str:
    """Render the complete per-robot ``offsets.yaml``. Offsets are wrapped to
    (-pi, pi] here (defense-in-depth for OLD result JSONs that predate the
    solver-side normalization) — rotation-equivalent, keeps the deployed
    joint value in range."""
    return _OFFSETS_YAML_TEMPLATE.format(
        robot=robot,
        pan=wrap_to_pi(float(pan_offset_rad)),
        tilt=wrap_to_pi(float(tilt_offset_rad)),
    )


def render_calibration(params: dict, robot: str, per_robot_dir: Path, *,
                       allow_flipped_camera: bool = False) -> dict:
    """Render the prospective contents of BOTH per-robot files from a solve's
    ``params`` dict (``t_a``, optional ``t_a_rotvec``, ``t_b_trans``, optional
    ``t_b_rotvec``, ``theta_p_offset_rad``, ``theta_t_offset_rad``).

    Reads the CURRENT overrides file to preserve rpy values for
    trivial/absent rotvecs (never writes ``0 0 0`` for an unfitted rotation).
    Pure render — no file writes. Returns a dict with ``xacro_path`` /
    ``xacro_text`` / ``offsets_path`` / ``offsets_text`` plus the normalized
    ``pan_offset_rad`` / ``tilt_offset_rad``.
    """
    xacro_path = per_robot_dir / OVERRIDES_XACRO_NAME
    offsets_path = per_robot_dir / OFFSETS_YAML_NAME
    if not xacro_path.is_file():
        raise CalibrationApplyError(
            f"per-robot overrides file {xacro_path} is missing — seed the "
            f"robot profile (all three robots ship one) before applying "
            f"calibration."
        )
    current_xacro = xacro_path.read_text()

    t_a = np.asarray(params["t_a"], dtype=float)
    t_a_rotvec = np.asarray(params.get("t_a_rotvec") or [0, 0, 0], dtype=float)
    t_b_trans = np.asarray(params["t_b_trans"], dtype=float)
    t_b_rotvec = np.asarray(params.get("t_b_rotvec") or [0, 0, 0], dtype=float)

    attach_rpy = _pan_axis_rpy_str(t_a_rotvec)
    if attach_rpy is None:
        attach_rpy = _current_property(
            current_xacro, "pan_tilt_attach_rpy", xacro_path)
    camera_mount_rpy = _rotvec_to_rpy_str(t_b_rotvec, allow_flipped_camera)
    if camera_mount_rpy is None:
        camera_mount_rpy = _current_property(
            current_xacro, "pan_tilt_camera_mount_rpy", xacro_path)

    pan_offset = wrap_to_pi(float(params.get("theta_p_offset_rad", 0.0)))
    tilt_offset = wrap_to_pi(float(params.get("theta_t_offset_rad", 0.0)))

    return {
        "robot": robot,
        "xacro_path": xacro_path,
        "xacro_text": render_overrides_xacro(robot, {
            "attach_xyz": _fmt_triplet(t_a),
            "attach_rpy": attach_rpy,
            "camera_mount_xyz": _fmt_triplet(t_b_trans),
            "camera_mount_rpy": camera_mount_rpy,
        }),
        "offsets_path": offsets_path,
        "offsets_text": render_offsets_yaml(robot, pan_offset, tilt_offset),
        "pan_offset_rad": pan_offset,
        "tilt_offset_rad": tilt_offset,
    }


# ---- atomic write machinery -------------------------------------------------


def _write_target(path: Path) -> Path:
    """Where os.replace should land so a symlink is preserved: write to the
    symlink's real target, not the link itself."""
    return Path(os.path.realpath(path)) if path.is_symlink() else path


def _atomic_write_single(
    path: Path, new_text: str, *, timestamp: Optional[str] = None,
) -> dict:
    """Atomically replace `path` with `new_text`, saving a `.old-<ts>` backup.

    Idempotent: when the new content matches, no write/backup happens and
    `applied` is False. When `path` is a symlink, writes through to the real
    target so the link is preserved."""
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    original = path.read_bytes()
    new_bytes = new_text.encode("utf-8")
    if new_bytes == original:
        return {"path": str(path), "applied": False, "backup_path": None}
    target = _write_target(path)
    tmp = target.with_name(target.name + f".tmp-{run_id}")
    tmp.write_bytes(new_bytes)
    bak = target.with_name(target.name + f".old-{ts}")
    try:
        bak.write_bytes(original)
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return {"path": str(path), "applied": True, "backup_path": str(bak)}


def _atomic_write_pair(
    pairs: Sequence[tuple[Path, str]], *, timestamp: Optional[str] = None,
) -> list[dict]:
    """Atomically replace a set of files with pre-rendered contents, as a
    single all-or-nothing transaction.

    For each ``(path, rendered_text)`` pair: stage the new content as a
    ``.tmp-<8hex>`` sibling, back up the existing file as ``.old-<ts>``, then
    ``os.replace`` every staged tmp into place — rolling back ALL completed
    replacements if any one fails, so the pair never lands half-applied.

    Idempotent per file: when the new content matches the original, no
    write/backup happens and that file's ``applied`` flag is False. A file
    that does not exist yet is created (``backup_path`` None). Symlinks are
    written through to their real target (link preserved).

    Returns one ``{"path", "applied", "backup_path"}`` dict per input pair,
    in input order.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]

    staged: list[dict] = []
    for path, text in pairs:
        path = Path(path)
        target = _write_target(path)
        original = target.read_bytes() if target.exists() else None
        new_bytes = text.encode("utf-8")
        staged.append({
            "path": path,
            "target": target,
            "original": original,
            "new": new_bytes,
            "changed": new_bytes != original,
            "tmp": None,
            "backup": None,
        })

    # Stage phase: all fallible writes happen BEFORE any replace.
    try:
        for st in staged:
            if not st["changed"]:
                continue
            tmp = st["target"].with_name(st["target"].name + f".tmp-{run_id}")
            tmp.write_bytes(st["new"])
            st["tmp"] = tmp
            if st["original"] is not None:
                bak = st["target"].with_name(st["target"].name + f".old-{ts}")
                bak.write_bytes(st["original"])
                st["backup"] = bak
    except Exception:
        for st in staged:
            if st["tmp"] is not None:
                Path(st["tmp"]).unlink(missing_ok=True)
            if st["backup"] is not None:
                Path(st["backup"]).unlink(missing_ok=True)
        raise

    # Replace phase: roll back every completed replace on failure.
    replaced: list[dict] = []
    try:
        for st in staged:
            if st["tmp"] is None:
                continue
            os.replace(st["tmp"], st["target"])
            replaced.append(st)
    except Exception:
        for st in replaced:
            if st["original"] is not None:
                st["target"].write_bytes(st["original"])
            else:
                st["target"].unlink(missing_ok=True)
        for st in staged:
            if st["tmp"] is not None:
                Path(st["tmp"]).unlink(missing_ok=True)
            if st["backup"] is not None:
                Path(st["backup"]).unlink(missing_ok=True)
        raise

    return [
        {
            "path": str(st["path"]),
            "applied": st["changed"],
            "backup_path": str(st["backup"]) if st["backup"] is not None else None,
        }
        for st in staged
    ]


# ---- public apply entry ------------------------------------------------------


def apply_calibration_detail(params: dict, basic_root=None,
                             allow_flipped_camera: bool = False) -> dict:
    """Apply a solve to the two per-robot files; return the full result dict
    (paths, per-file applied flags + backups, normalized offsets, rebuild
    command). See :func:`apply_calibration` for the simple entry point."""
    robot = _require_robot_name()
    per_robot_dir = resolve_per_robot_dir(robot, basic_root)
    rendered = render_calibration(
        params, robot, per_robot_dir,
        allow_flipped_camera=allow_flipped_camera,
    )
    xacro_res, offsets_res = _atomic_write_pair([
        (rendered["xacro_path"], rendered["xacro_text"]),
        (rendered["offsets_path"], rendered["offsets_text"]),
    ])
    return {
        "robot": robot,
        "written": [str(rendered["xacro_path"]), str(rendered["offsets_path"])],
        "xacro_path": str(rendered["xacro_path"]),
        "xacro_applied": xacro_res["applied"],
        "xacro_backup_path": xacro_res["backup_path"],
        "offsets_path": str(rendered["offsets_path"]),
        "offsets_applied": offsets_res["applied"],
        "offsets_backup_path": offsets_res["backup_path"],
        "pan_offset_rad": rendered["pan_offset_rad"],
        "tilt_offset_rad": rendered["tilt_offset_rad"],
        "build_package": "tinker_robot_config",
        "build_command": BUILD_COMMAND,
        "workspace_hint": WORKSPACE_HINT,
    }


def apply_calibration(params: dict, basic_root=None,
                      allow_flipped_camera: bool = False) -> list[Path]:
    """Single public apply entry: write ``$ROBOT_NAME``'s two per-robot files
    from a solve's ``params`` dict, atomically and in lockstep.

    Returns the list of target paths (``pan_tilt_overrides.xacro``,
    ``offsets.yaml``). Raises :class:`CalibrationApplyError` when ROBOT_NAME
    is unset, the per-robot profile is missing, or the forward-camera
    invariant is violated (without ``allow_flipped_camera``)."""
    detail = apply_calibration_detail(
        params, basic_root=basic_root,
        allow_flipped_camera=allow_flipped_camera,
    )
    return [Path(detail["xacro_path"]), Path(detail["offsets_path"])]


# ---- CLI ---------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply pan-tilt calibration results to $ROBOT_NAME's two "
                    "per-robot files (pan_tilt_overrides.xacro + offsets.yaml "
                    "under tk25_basic's tinker_robot_config/robots/) — "
                    "atomically, in lockstep, with .old-<ts> backups. Never "
                    "touches the shared tinker_urdf xacros.",
    )
    parser.add_argument("--results", required=True, type=Path,
                        help="chain.json or polish.json produced by "
                             "run_calibration")
    parser.add_argument("--basic-root", type=Path, default=None,
                        help="path to the tk25_basic package root (the "
                             "directory containing src/tinker_robot_config). "
                             "Default: auto-discover by walking up from this "
                             "module to the workspace root.")
    parser.add_argument("--allow-flipped-camera", action="store_true",
                        help="Override the forward-camera invariant (|yaw| < π/2). "
                             "Use ONLY when the head camera is genuinely mounted "
                             "backward on the head. Without this flag, calibration "
                             "results with |yaw| ≥ π/2 are refused — that condition "
                             "is the smoking-gun signature of an upstream "
                             "optical→body convention bug, not a legitimate "
                             "calibration outcome.")
    args = parser.parse_args(argv)

    params = _load_params(args.results)
    try:
        detail = apply_calibration_detail(
            params, basic_root=args.basic_root,
            allow_flipped_camera=args.allow_flipped_camera,
        )
    except CalibrationApplyError as exc:
        # Operator-facing one-shot: the message itself names the fix, so
        # don't bury it under a stack trace.
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if not detail["xacro_applied"] and not detail["offsets_applied"]:
        print(
            f"No change — {detail['robot']}'s per-robot overrides + offsets "
            f"already match the calibration."
        )
        return

    print(f"Robot: {detail['robot']}")
    print(f"Overrides xacro: {detail['xacro_path']}")
    if detail["xacro_backup_path"]:
        print(f"  backup:        {detail['xacro_backup_path']}")
    elif not detail["xacro_applied"]:
        print("  (no change — already matches calibration)")
    print(f"Offsets yaml:    {detail['offsets_path']}")
    if detail["offsets_backup_path"]:
        print(f"  backup:        {detail['offsets_backup_path']}")
    elif not detail["offsets_applied"]:
        print("  (no change — already matches calibration)")
    print(
        f"  pan_offset_rad:  {detail['pan_offset_rad']:.10f}\n"
        f"  tilt_offset_rad: {detail['tilt_offset_rad']:.10f}"
    )
    print(
        f"\nRebuild + relaunch to deploy:\n"
        f"  {detail['build_command']}\n"
        f"  ({detail['workspace_hint']})\n"
        f"Both files are sourced from the same calibration "
        f"({args.results.name})."
    )


if __name__ == "__main__":
    main()
