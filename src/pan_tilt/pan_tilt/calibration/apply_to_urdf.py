"""Emit a unified diff patching a pan-tilt xacro file from calibration results.

This tool **does not apply** the patch. It prints the diff to stdout (and
optionally writes the full patched file to `--out`) so the operator can review
and apply it manually with e.g. `patch -p0 < calib.patch`.

Two xacro layouts are supported, because the stack ships both:

1. **Standalone** form (``.worktrees/tk26_vision/src/pan_tilt/urdf/pan_tilt.urdf.xacro``):
   literal ``<joint name="pan_joint"><origin xyz=... rpy=.../></joint>`` etc.
   Used by the standalone pan_tilt launch (`pan_tilt.launch.py`).

2. **Macro** form (``src/tk25_basic/src/tinker_urdf/src/pan_tilt.urdf.xacro``):
   a ``<xacro:macro name="pan_tilt_macro">`` with parameterized
   ``attach_xyz`` / ``attach_rpy`` defaults that the main robot URDF
   (``tracer_mini_manipulator.urdf.xacro``) consumes. This is what the live
   robot actually loads, so it's the one the operator normally wants to patch.

The patcher writes:

- ``t_a`` (translation) ⇒ pan_joint origin xyz **or** ``attach_xyz`` default.
- ``t_b_trans`` ⇒ camera_mount_joint origin xyz.
- ``t_b_rotvec`` ⇒ camera_mount_joint origin rpy, **only if** the rotvec is
  non-trivial (>1e-6 norm). A zero rotvec preserves the existing rpy, which
  matters because the chain-phase fit locks T_B rotation and we don't want
  to silently zero-out a rotation that a previous polish run had written.
- ``t_a_rotvec`` ⇒ pan_joint origin rpy (standalone) / ``attach_rpy`` default
  (macro), **only if** non-trivial. This is the base→pan-axis tilt fit by the
  optional ``--fit-pan-axis-tilt`` calibration path (a physically non-vertical
  pan axis). A zero/absent rotvec preserves the existing rpy, so calibration
  results from before that fit existed patch exactly as they always did.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import math
import os
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .utils import body_yaw_from_rotvec, rotvec_to_xyz_euler, wrap_to_pi
from .yaml_targets import list_yaml_targets


# ---- forward-camera invariant ----------------------------------------------
#
# camera_mount_joint rpy yaw must be within ±π/2 of zero for a forward-facing
# head camera (the only configuration we ship). A value near ±π is the
# 2026-04-30 backward-camera bug — refuse to write it unless the operator
# explicitly opts in via `allow_flipped_camera=True` (or the equivalent CLI
# flag), which widens the bound to ±π for genuinely flipped hardware.

_FORWARD_YAW_LIMIT_RAD = math.pi / 2.0


class CalibrationApplyError(RuntimeError):
    """Refusal to write a URDF that violates the forward-camera invariant.

    Distinct from :class:`ValueError` so callers can catch it surgically and
    surface a focused operator message (the patcher's main() does this).
    """


# ---- regexes ---------------------------------------------------------------
# Handle both bare names and xacro-prefixed names like `${prefix}pan_joint`.
_PREFIX_RE = re.compile(r'^\$\{[^}]+\}')

JOINT_BLOCK_RE = re.compile(
    r'<joint\s+name="(?P<name>[^"]+)"[^>]*>(?P<body>.*?)</joint>',
    re.DOTALL,
)

ORIGIN_RE = re.compile(
    r'<origin\s+xyz="(?P<xyz>[^"]+)"\s+rpy="(?P<rpy>[^"]+)"\s*/>'
)

MACRO_DECL_RE = re.compile(r'<xacro:macro\s+name="pan_tilt_macro"', re.DOTALL)

ATTACH_XYZ_DEFAULT_RE = re.compile(
    r"(?P<key>attach_xyz:=')(?P<val>[^']*)(?P<close>')"
)
ATTACH_RPY_DEFAULT_RE = re.compile(
    r"(?P<key>attach_rpy:=')(?P<val>[^']*)(?P<close>')"
)


# YAML patchers — surgical regex (preserves comments/whitespace; no ruamel
# dependency). The trailing `(.*)$` capture preserves any inline comment.
_YAML_PAN_RE = re.compile(
    r"^(?P<lead>\s*pan_offset_rad:\s*)\S+(?P<trail>.*)$",
    re.MULTILINE,
)
_YAML_TILT_RE = re.compile(
    r"^(?P<lead>\s*tilt_offset_rad:\s*)\S+(?P<trail>.*)$",
    re.MULTILINE,
)


def _fmt_triplet(v) -> str:
    return f"{v[0]:.6g} {v[1]:.6g} {v[2]:.6g}"


def _bare_name(n: str) -> str:
    """Strip a leading `${prefix}` from a joint name so macros match too."""
    return _PREFIX_RE.sub("", n)


def _load_params(results_path: Path) -> dict:
    blob = json.loads(results_path.read_text())
    if "params" in blob:
        return blob["params"]
    raise ValueError(f"{results_path} has no 'params' key")


def _replace_origin(body: str, new_xyz: str, preserve_rpy: bool,
                    new_rpy: str | None) -> str:
    """Replace the first `<origin ...>` in `body`. If `preserve_rpy`, keep the
    existing rpy string; otherwise use `new_rpy`."""
    def repl(m):
        rpy_out = m.group("rpy") if preserve_rpy else (new_rpy or "0 0 0")
        return f'<origin xyz="{new_xyz}" rpy="{rpy_out}"/>'
    return ORIGIN_RE.sub(repl, body, count=1)


def _rotvec_to_rpy_str(t_b_rotvec, allow_flipped_camera: bool) -> str | None:
    """Convert t_b_rotvec → URDF rpy triplet, enforcing the forward-camera invariant.

    Returns ``None`` when the rotvec is sub-threshold (preserves the existing
    rpy in the xacro), otherwise returns the formatted "roll pitch yaw" string.

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
            f"Refusing to patch URDF: camera_mount_joint fitted yaw = "
            f"{yaw:.4f} rad ({math.degrees(yaw):.1f}°). Forward-facing "
            f"cameras must have |yaw| < π/2 (90°). This usually means the "
            f"optical→body convention was missed in the Phase-1 hand-eye "
            f"warm start. To override (e.g. you genuinely mounted the "
            f"camera backward), pass --allow-flipped-camera. The existing "
            f"URDF rpy is preserved."
        )
    return _fmt_triplet(euler)


def _pan_axis_rpy_str(t_a_rotvec) -> str | None:
    """Convert the base→pan-axis rotvec (`t_a_rotvec`) to a URDF rpy triplet for
    the pan_joint origin. Returns ``None`` when sub-threshold (>1e-6 norm), so a
    zero/absent tilt preserves the existing rpy. No forward-camera yaw guard —
    this is the pan axis, not the camera (and the fit holds its Z/yaw at 0). The
    rotvec→xyz-euler→URDF-rpy path is the same one validated for the camera
    mount, so the emitted rpy reproduces transform_matrix(t_a_rotvec, ·)."""
    rotvec = np.asarray(t_a_rotvec, dtype=float)
    if np.linalg.norm(rotvec) <= 1e-6:
        return None
    return _fmt_triplet(rotvec_to_xyz_euler(rotvec))


def _patched_standalone(
    xacro_text: str, t_a, t_b_trans, t_b_rotvec, t_a_rotvec=None, *,
    allow_flipped_camera: bool = False,
) -> str:
    """Patch the tk26_vision standalone form: literal pan_joint + camera_mount_joint."""
    rpy_str = _rotvec_to_rpy_str(t_b_rotvec, allow_flipped_camera)
    have_rot = rpy_str is not None
    pan_rpy_str = _pan_axis_rpy_str(t_a_rotvec if t_a_rotvec is not None else np.zeros(3))
    have_pan_rot = pan_rpy_str is not None

    def repl(match):
        name = _bare_name(match.group("name"))
        body = match.group("body")
        if name == "pan_joint":
            # T_A rotation: write the fitted pan-axis tilt when present,
            # otherwise preserve the existing rpy (legacy behavior).
            new_body = _replace_origin(
                body, _fmt_triplet(t_a),
                preserve_rpy=not have_pan_rot, new_rpy=pan_rpy_str,
            )
        elif name == "camera_mount_joint":
            new_body = _replace_origin(
                body, _fmt_triplet(t_b_trans),
                preserve_rpy=not have_rot, new_rpy=rpy_str,
            )
        else:
            return match.group(0)
        return match.group(0).replace(body, new_body, 1)

    return JOINT_BLOCK_RE.sub(repl, xacro_text)


def _patched_macro(
    xacro_text: str, t_a, t_b_trans, t_b_rotvec, t_a_rotvec=None, *,
    allow_flipped_camera: bool = False,
) -> str:
    """Patch the tk25_basic macro form: `attach_xyz` default + camera_mount_joint."""
    rpy_str = _rotvec_to_rpy_str(t_b_rotvec, allow_flipped_camera)
    have_rot = rpy_str is not None
    pan_rpy_str = _pan_axis_rpy_str(t_a_rotvec if t_a_rotvec is not None else np.zeros(3))

    out = ATTACH_XYZ_DEFAULT_RE.sub(
        lambda m: f"{m.group('key')}{_fmt_triplet(t_a)}{m.group('close')}",
        xacro_text, count=1,
    )
    # attach_rpy is the T_A (base->pan-axis) rotation. Write the fitted pan-axis
    # tilt when present; otherwise leave whatever was there (legacy behavior).
    if pan_rpy_str is not None:
        out = ATTACH_RPY_DEFAULT_RE.sub(
            lambda m: f"{m.group('key')}{pan_rpy_str}{m.group('close')}",
            out, count=1,
        )

    def repl(match):
        name = _bare_name(match.group("name"))
        body = match.group("body")
        if name == "camera_mount_joint":
            new_body = _replace_origin(
                body, _fmt_triplet(t_b_trans),
                preserve_rpy=not have_rot, new_rpy=rpy_str,
            )
            return match.group(0).replace(body, new_body, 1)
        return match.group(0)

    return JOINT_BLOCK_RE.sub(repl, out)


def _patched_xacro(
    xacro_text: str, t_a, t_b_trans, t_b_rotvec=None, t_a_rotvec=None, *,
    allow_flipped_camera: bool = False,
) -> str:
    t_b_rotvec = np.zeros(3) if t_b_rotvec is None else np.asarray(t_b_rotvec, dtype=float)
    t_a_rotvec = np.zeros(3) if t_a_rotvec is None else np.asarray(t_a_rotvec, dtype=float)
    if MACRO_DECL_RE.search(xacro_text):
        return _patched_macro(
            xacro_text, t_a, t_b_trans, t_b_rotvec, t_a_rotvec,
            allow_flipped_camera=allow_flipped_camera,
        )
    return _patched_standalone(
        xacro_text, t_a, t_b_trans, t_b_rotvec, t_a_rotvec,
        allow_flipped_camera=allow_flipped_camera,
    )


# ---- pan_tilt.yaml runtime-offset patcher ----------------------------------
#
# The URDF chain is geometrically incomplete on its own: the calibration's
# theta_p_offset / theta_t_offset live in pan_tilt.yaml and the state
# publisher applies them to firmware feedback. URDF + YAML must come from
# the same calibration JSON or the TF chain is wrong. The 2026-04-30
# below-ground-projection bug came from exactly this drift.
#
# We surgically substitute the two values in-place, preserving every
# surrounding character (indentation, comments, blank lines, key order).

def _patch_yaml_offsets(
    yaml_text: str, pan_offset_rad: float, tilt_offset_rad: float,
) -> str:
    """Surgical in-place replacement of `pan_offset_rad` and `tilt_offset_rad`.

    Raises :class:`CalibrationApplyError` if either key is missing — the
    operator must add them once to the source YAML (see calibration/readme.md)
    before running the patcher. Failing loudly here beats silently leaving
    the calibration half-applied.
    """
    # Defense-in-depth: normalize even when applying an OLD json that predates
    # the solver-side wrap (Task 1). Rotation-equivalent; keeps the deployed
    # joint value in range. See utils.wrap_to_pi.
    pan_offset_rad = wrap_to_pi(pan_offset_rad)
    tilt_offset_rad = wrap_to_pi(tilt_offset_rad)
    pan_text, n_pan = _YAML_PAN_RE.subn(
        lambda m: f"{m.group('lead')}{pan_offset_rad:.10f}{m.group('trail')}",
        yaml_text, count=1,
    )
    if n_pan == 0:
        raise CalibrationApplyError(
            "pan_tilt.yaml has no `pan_offset_rad:` key under "
            "pan_tilt_state_publisher.ros__parameters. Add the key once "
            "(see calibration/readme.md → Runtime joint offsets) before "
            "running apply_to_urdf, or pass --no-yaml to skip the YAML "
            "patch entirely."
        )
    out_text, n_tilt = _YAML_TILT_RE.subn(
        lambda m: f"{m.group('lead')}{tilt_offset_rad:.10f}{m.group('trail')}",
        pan_text, count=1,
    )
    if n_tilt == 0:
        raise CalibrationApplyError(
            "pan_tilt.yaml has no `tilt_offset_rad:` key under "
            "pan_tilt_state_publisher.ros__parameters. Add the key once "
            "(see calibration/readme.md → Runtime joint offsets) before "
            "running apply_to_urdf, or pass --no-yaml to skip the YAML "
            "patch entirely."
        )
    return out_text


def _resolve_yaml_path(cli_yaml: Optional[Path], no_yaml: bool) -> Optional[Path]:
    """Resolve which (if any) pan_tilt.yaml to patch.

    Precedence:
      1. `--no-yaml` → None.
      2. `--yaml <path>` → that path (must exist).
      3. Auto-discovery via `list_yaml_targets()` — install share dir.

    Returns ``None`` only when the operator explicitly opted out via
    `--no-yaml`. Auto-discovery failure raises so the operator can't
    accidentally ship a half-applied calibration.
    """
    if no_yaml:
        return None
    if cli_yaml is not None:
        if not cli_yaml.is_file():
            raise CalibrationApplyError(
                f"--yaml path {cli_yaml} does not exist."
            )
        return cli_yaml
    targets = [t for t in list_yaml_targets() if t.exists]
    if not targets:
        raise CalibrationApplyError(
            "Could not auto-discover pan_tilt.yaml (pan_tilt package not "
            "installed? rebuild it). Pass --yaml <path> or --no-yaml to "
            "skip the YAML patch."
        )
    return Path(targets[0].path)


# ---- per-robot urdf_overrides.yaml patcher --------------------------------
#
# The pan-tilt geometry actually consumed at launch lives in
# tinker_robot_config/robots/<ROBOT_NAME>/pan_tilt/urdf_overrides.yaml. Its
# attach_xyz / attach_rpy / camera_mount_xyz / camera_mount_rpy OVERRIDE the
# xacro defaults (pan_tilt.launch.py / robot_description.launch.py flatten them
# in), so patching only the xacro has no runtime effect on robots that use
# overrides. We keep this file in sync too. Surgical regex per key preserves
# comments, quoting, indentation, and key order.

_OVERRIDE_KEYS = ("attach_xyz", "attach_rpy", "camera_mount_xyz", "camera_mount_rpy")


def _override_key_re(key: str) -> re.Pattern:
    # Matches `   key: "v0 v1 v2"   # comment`, preserving indent/quotes/comment.
    return re.compile(rf'^(?P<lead>\s*{key}:\s*")[^"]*(?P<close>".*)$', re.MULTILINE)


def _patch_urdf_overrides(
    yaml_text: str, t_a, t_b_trans, t_b_rotvec, t_a_rotvec=None, *,
    allow_flipped_camera: bool = False,
) -> tuple[str, list[str]]:
    """Surgically patch the pan-tilt urdf_overrides.yaml.

    attach_xyz <- t_a and camera_mount_xyz <- t_b_trans are written whenever the
    key is present. attach_rpy <- t_a_rotvec and camera_mount_rpy <- t_b_rotvec
    are written only when the rotvec is non-trivial — a zero rotvec preserves the
    existing rpy (same rule as the xacro patcher), so a calibration that didn't
    fit a rotation never zeroes one a prior run wrote. camera_mount_rpy goes
    through the forward-camera yaw guard. Missing keys are skipped silently (the
    file may legitimately omit some). Returns (patched_text, changed_keys)."""
    t_a_rotvec = np.zeros(3) if t_a_rotvec is None else np.asarray(t_a_rotvec, dtype=float)
    new_vals = {
        "attach_xyz": _fmt_triplet(t_a),
        "attach_rpy": _pan_axis_rpy_str(t_a_rotvec),                     # None -> preserve
        "camera_mount_xyz": _fmt_triplet(t_b_trans),
        "camera_mount_rpy": _rotvec_to_rpy_str(t_b_rotvec, allow_flipped_camera),  # None -> preserve
    }
    out = yaml_text
    changed: list[str] = []
    for key in _OVERRIDE_KEYS:
        val = new_vals[key]
        if val is None:
            continue
        new_out, n = _override_key_re(key).subn(
            lambda m: f"{m.group('lead')}{val}{m.group('close')}", out, count=1,
        )
        if n:
            out = new_out
            changed.append(key)
    return out, changed


def _resolve_overrides_yaml(
    cli_path: Optional[Path], no_overrides: bool,
) -> Optional[Path]:
    """Resolve which per-robot urdf_overrides.yaml to patch.

    Precedence: --no-overrides -> None; --overrides-yaml <path> -> that path
    (must exist); else auto-discover the installed
    `tinker_robot_config/robots/$ROBOT_NAME/pan_tilt/urdf_overrides.yaml`.
    Auto-discovery returns None (not an error) when $ROBOT_NAME is unset,
    tinker_robot_config isn't installed, or the file is absent — overrides are
    an optional target, so a missing one is skipped with a note rather than
    aborting the whole apply."""
    if no_overrides:
        return None
    if cli_path is not None:
        if not cli_path.is_file():
            raise CalibrationApplyError(
                f"--overrides-yaml path {cli_path} does not exist."
            )
        return cli_path
    robot = os.environ.get("ROBOT_NAME", "").strip()
    if not robot:
        return None
    try:
        from ament_index_python.packages import (
            get_package_share_directory, PackageNotFoundError,
        )
    except ImportError:
        return None
    try:
        share = Path(get_package_share_directory("tinker_robot_config"))
    except (PackageNotFoundError, Exception):
        return None
    cand = share / "robots" / robot / "pan_tilt" / "urdf_overrides.yaml"
    return cand if cand.is_file() else None


def _write_target(path: Path) -> Path:
    """Where os.replace should land so a --symlink-install symlink is
    preserved: write to the symlink's real target, not the link itself."""
    return Path(os.path.realpath(path)) if path.is_symlink() else path


# Path segments that are never part of the true colcon source tree but may
# contain copies of source files (build artifacts, install overlays, log dirs).
# Virtualenvs and git worktrees are caught by the hidden-segment (startswith
# ".") check in the filter below.
_SRC_EXCLUDE = {"build", "install", "log"}


def resolve_source_path(install_path: Path) -> Path:
    """Map an install-tree file to its colcon source-tree path.

    (1) symlink -> follow it when it lands under a 'src/' segment;
    (2) else glob '<ws>/src/**/<name>' (ws = parts before 'install') for a
        unique match that shares the install file's package directory name,
        excluding paths that contain build/install/log segments or hidden
        directories (e.g. .claude/worktrees, .venv-*) that may contain
        spurious copies of source files;
    (3) else log a warning and return install_path unchanged — writing the
        install tree, which a colcon rebuild will silently revert."""
    install_path = Path(install_path)
    if install_path.is_symlink():
        resolved = install_path.resolve()
        if "src" in resolved.parts:
            return resolved
    parts = install_path.parts
    if "install" in parts:
        ws = Path(*parts[: parts.index("install")])
        src_root = ws / "src"
        if src_root.is_dir():
            # package dir name appears right after .../share/<pkg>/ or is the
            # install package dir; use the file name + nearest parent dir name.
            parent_name = install_path.parent.name
            matches = [
                m for m in src_root.glob(f"**/{install_path.name}")
                if m.parent.name == parent_name and m.is_file()
                and not any(
                    part.startswith(".") or part in _SRC_EXCLUDE
                    for part in m.parts
                )
            ]
            if len(matches) == 1:
                return matches[0]
        # Fell through an install path without a unique source match: the write
        # will land in the install tree and a colcon rebuild will REVERT it.
        # Warn loudly so the operator doesn't think the calibration stuck.
        logging.getLogger(__name__).warning(
            "resolve_source_path: could not map install file %s to a unique "
            "source-tree path (0 or >1 candidates); writing the install tree, "
            "which a colcon rebuild will REVERT. Re-apply from a "
            "--symlink-install workspace, or patch the source file directly.",
            install_path,
        )
    return install_path


def _atomic_write_single(
    path: Path, new_text: str, *, timestamp: Optional[str] = None,
) -> dict:
    """Atomically replace `path` with `new_text`, saving a `.old-<ts>` backup.

    Idempotent: when the new content matches, no write/backup happens and
    `applied` is False. Mirrors the per-file half of `_atomic_write_pair` for
    standalone targets (the override yaml) that aren't part of the URDF+offset
    lockstep pair.

    When `path` is a --symlink-install symlink, writes through to the real
    source target so the symlink is preserved and a subsequent rebuild does not
    silently revert the calibration."""
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
    xacro: Path, patched_xacro: str,
    yaml_path: Optional[Path], pan_offset_rad: float, tilt_offset_rad: float,
    *,
    timestamp: Optional[str] = None,
) -> dict:
    """Atomically replace `xacro` (and optionally `yaml_path`) with patched
    content. Both originals are saved as `.old-<ts>` siblings on success.

    Returns a dict describing what landed and the backup paths, suitable
    for callers (CLI / calib_web) to print or render.

    On YAML write failure, the URDF is rolled back from its backup so the
    operator never sees a half-applied calibration. Idempotent: when the
    new content matches the original, no backup is written and the
    corresponding `*_applied` flag is False.
    """
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]

    original_xacro_bytes = xacro.read_bytes()
    new_xacro_bytes = patched_xacro.encode("utf-8")
    xacro_changed = new_xacro_bytes != original_xacro_bytes

    yaml_changed = False
    new_yaml_bytes: Optional[bytes] = None
    original_yaml_bytes: Optional[bytes] = None
    if yaml_path is not None:
        original_yaml_text = yaml_path.read_text()
        original_yaml_bytes = original_yaml_text.encode("utf-8")
        patched_yaml = _patch_yaml_offsets(
            original_yaml_text, pan_offset_rad, tilt_offset_rad,
        )
        new_yaml_bytes = patched_yaml.encode("utf-8")
        yaml_changed = new_yaml_bytes != original_yaml_bytes

    result = {
        "xacro_path": str(xacro),
        "xacro_applied": xacro_changed,
        "xacro_backup_path": None,
        "yaml_path": str(yaml_path) if yaml_path is not None else None,
        "yaml_applied": yaml_changed,
        "yaml_backup_path": None,
    }
    if not xacro_changed and not yaml_changed:
        return result

    # Resolve real targets so writes go through --symlink-install symlinks to
    # the source tree (preserving the link so rebuilds don't revert calibration).
    xacro_target = _write_target(xacro)
    yaml_target = _write_target(yaml_path) if yaml_path is not None else None

    xacro_tmp = xacro_target.with_name(xacro_target.name + f".tmp-{run_id}") if xacro_changed else None
    if xacro_tmp is not None:
        xacro_tmp.write_bytes(new_xacro_bytes)

    yaml_tmp = (
        yaml_target.with_name(yaml_target.name + f".tmp-{run_id}")
        if yaml_changed and yaml_target is not None
        else None
    )
    if yaml_tmp is not None:
        yaml_tmp.write_bytes(new_yaml_bytes)  # type: ignore[arg-type]

    xacro_bak: Optional[Path] = None
    yaml_bak: Optional[Path] = None
    try:
        if xacro_tmp is not None:
            xacro_bak = xacro_target.with_name(xacro_target.name + f".old-{ts}")
            xacro_bak.write_bytes(original_xacro_bytes)
            os.replace(xacro_tmp, xacro_target)
            result["xacro_backup_path"] = str(xacro_bak)
        if yaml_tmp is not None and yaml_target is not None:
            yaml_bak = yaml_target.with_name(yaml_target.name + f".old-{ts}")
            yaml_bak.write_bytes(original_yaml_bytes)  # type: ignore[arg-type]
            os.replace(yaml_tmp, yaml_target)
            result["yaml_backup_path"] = str(yaml_bak)
    except Exception:
        # Roll back URDF if the YAML side failed mid-replace; clean up tmps.
        if xacro_bak is not None and xacro.read_bytes() != original_xacro_bytes:
            os.replace(xacro_bak, xacro_target)
        if xacro_tmp is not None:
            xacro_tmp.unlink(missing_ok=True)
        if yaml_tmp is not None:
            yaml_tmp.unlink(missing_ok=True)
        raise

    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Patch a pan-tilt xacro AND pan_tilt.yaml from calibration "
                    "results in lockstep — single command leaves the deployment "
                    "completely ready (no manual YAML edit).",
    )
    parser.add_argument("--results", required=True, type=Path,
                        help="chain.json or polish.json produced by run_calibration")
    parser.add_argument("--xacro", required=True, type=Path,
                        help="path to the pan-tilt xacro to update "
                             "(standalone form under tk26_vision/, or the "
                             "tk25_basic macro form under tinker_urdf/src/)")
    parser.add_argument("--out", type=Path, default=None,
                        help="If set, write the full patched xacro here AND "
                             "skip the in-place URDF replacement / YAML patch "
                             "(diff/dry-run mode).")
    parser.add_argument("--yaml", type=Path, default=None,
                        help="Path to pan_tilt.yaml (default: auto-discover the "
                             "pan_tilt package's installed config). The file's "
                             "pan_offset_rad / tilt_offset_rad keys are updated "
                             "in place with a `.old-<ts>` backup, atomically "
                             "alongside the URDF patch.")
    parser.add_argument("--no-yaml", action="store_true",
                        help="Skip the YAML patch entirely. Only use for "
                             "dry runs or when the runtime offsets live "
                             "elsewhere — without the YAML, the URDF chain "
                             "mis-represents the camera pose at any non-zero "
                             "firmware tilt.")
    parser.add_argument("--allow-partial", action="store_true",
                        help="permit patching the URDF or YAML alone (normally "
                             "refused — they must stay in lockstep from the same "
                             "solve to avoid tilting the camera)")
    parser.add_argument("--overrides-yaml", type=Path, default=None,
                        help="Path to the per-robot pan-tilt urdf_overrides.yaml "
                             "(default: auto-discover the installed "
                             "tinker_robot_config/robots/$ROBOT_NAME/pan_tilt/"
                             "urdf_overrides.yaml). This file's attach_xyz/"
                             "attach_rpy/camera_mount_xyz/camera_mount_rpy keys "
                             "OVERRIDE the xacro defaults at launch, so it must "
                             "be patched too or the calibration has no runtime "
                             "effect on robots that use overrides.")
    parser.add_argument("--no-overrides", action="store_true",
                        help="Skip the urdf_overrides.yaml patch. Use for dry "
                             "runs or robots that don't use the override system.")
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
    t_a = np.asarray(params["t_a"], dtype=float)
    # Optional: absent in pre-pan-axis-tilt results -> zero -> existing rpy preserved.
    t_a_rotvec = np.asarray(params.get("t_a_rotvec", [0, 0, 0]), dtype=float)
    t_b_trans = np.asarray(params["t_b_trans"], dtype=float)
    t_b_rotvec = np.asarray(params.get("t_b_rotvec", [0, 0, 0]), dtype=float)
    pan_offset_rad = float(params.get("theta_p_offset_rad", 0.0))
    tilt_offset_rad = float(params.get("theta_t_offset_rad", 0.0))

    original = args.xacro.read_text()
    try:
        patched = _patched_xacro(
            original, t_a, t_b_trans, t_b_rotvec, t_a_rotvec,
            allow_flipped_camera=args.allow_flipped_camera,
        )
    except CalibrationApplyError as exc:
        # Operator-facing one-shot: the message itself names the override
        # flag, so don't bury it under a stack trace.
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    # Dry-run / diff mode: --out detaches the URDF write from the YAML
    # patch. Print the diff/file as before and a YAML-paste hint, but
    # don't touch any installed file.
    if args.out:
        args.out.write_text(patched)
        print(f"Wrote patched xacro to {args.out}")
        print("Review, then replace the original if the content is correct.")
        print(
            f"\n→ For the matching runtime offsets (paste into "
            f"src/tk26_vision/src/pan_tilt/config/pan_tilt.yaml under "
            f"pan_tilt_state_publisher.ros__parameters):\n"
            f"    pan_offset_rad:  {pan_offset_rad:.10f}\n"
            f"    tilt_offset_rad: {tilt_offset_rad:.10f}\n"
            f"\n(Re-run without --out to apply URDF + YAML + per-robot "
            f"urdf_overrides.yaml atomically in place.)"
        )
        # Show what the per-robot override patch would write (the file that
        # actually wins at launch), without touching it.
        try:
            ovr_path = _resolve_overrides_yaml(args.overrides_yaml, args.no_overrides)
        except CalibrationApplyError as exc:
            print(f"\n(overrides: {exc})")
            ovr_path = None
        if ovr_path is not None:
            _, changed = _patch_urdf_overrides(
                ovr_path.read_text(), t_a, t_b_trans, t_b_rotvec, t_a_rotvec,
                allow_flipped_camera=args.allow_flipped_camera,
            )
            print(
                f"\n→ Would also patch per-robot overrides "
                f"({', '.join(changed) or 'no matching keys'}):\n"
                f"    {ovr_path}"
            )
        elif not args.no_overrides:
            print(
                "\n(no urdf_overrides.yaml patched: $ROBOT_NAME unset, "
                "tinker_robot_config not installed, or file absent — pass "
                "--overrides-yaml to point at one.)"
            )
        return

    # Lockstep guard: refuse a URDF-only patch unless the operator explicitly
    # acknowledges they know what they're doing. T_b (written to the URDF) and
    # theta_*_offset (written to pan_tilt.yaml) MUST come from the same solve
    # or the TF chain misrepresents the camera pose. The 2026-04-30 bug came
    # from exactly this drift. Pass --allow-partial to override.
    if args.no_yaml and not args.allow_partial:
        raise CalibrationApplyError(
            "Refusing to patch the URDF without the matching pan_tilt.yaml "
            "offsets: T_b and theta_*_offset MUST come from the same solve "
            "(mixing them tilts the camera). Re-run without --no-yaml, or pass "
            "--allow-partial if you really intend a partial apply."
        )

    # In-place lockstep apply: URDF + YAML, atomic with backups.
    try:
        yaml_path = _resolve_yaml_path(args.yaml, args.no_yaml)
    except CalibrationApplyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        result = _atomic_write_pair(
            args.xacro, patched,
            yaml_path, pan_offset_rad, tilt_offset_rad,
        )
    except CalibrationApplyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    # Per-robot override patch (the file that actually wins at launch). Done as
    # a separate atomic write after the URDF+YAML pair: if it fails the pair
    # stays applied (idempotent re-run recovers) rather than rolling back a
    # good URDF/YAML write over an optional, secondary target.
    try:
        overrides_path = _resolve_overrides_yaml(args.overrides_yaml, args.no_overrides)
    except CalibrationApplyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
    override_result = None
    override_changed: list[str] = []
    if overrides_path is not None:
        try:
            patched_ovr, override_changed = _patch_urdf_overrides(
                overrides_path.read_text(), t_a, t_b_trans, t_b_rotvec, t_a_rotvec,
                allow_flipped_camera=args.allow_flipped_camera,
            )
            override_result = _atomic_write_single(overrides_path, patched_ovr)
        except CalibrationApplyError as exc:
            print(f"ERROR (urdf_overrides.yaml): {exc}", file=sys.stderr)
            sys.exit(2)

    if (not result["xacro_applied"] and not result["yaml_applied"]
            and (override_result is None or not override_result["applied"])):
        print("No change — URDF, YAML, and overrides already match the calibration.")
        return

    print(f"Patched URDF: {result['xacro_path']}")
    if result["xacro_backup_path"]:
        print(f"  backup:    {result['xacro_backup_path']}")
    elif not result["xacro_applied"]:
        print(f"  (no change — URDF already matches calibration)")

    if yaml_path is not None:
        print(f"Patched YAML: {result['yaml_path']}")
        if result["yaml_backup_path"]:
            print(f"  backup:    {result['yaml_backup_path']}")
        elif not result["yaml_applied"]:
            print(f"  (no change — YAML already matches calibration)")
        print(
            f"  pan_offset_rad:  {pan_offset_rad:.10f}\n"
            f"  tilt_offset_rad: {tilt_offset_rad:.10f}"
        )
    else:
        print(
            "YAML patch SKIPPED (--no-yaml). The URDF chain will misrepresent "
            f"the camera pose by ~{abs(math.degrees(tilt_offset_rad)):.0f}° "
            f"of tilt at firmware-zero until you set pan_offset_rad / "
            f"tilt_offset_rad in pan_tilt.yaml manually."
        )

    if override_result is not None:
        if override_result["applied"]:
            print(f"Patched overrides: {override_result['path']}")
            print(f"  keys:      {', '.join(override_changed) or '(none matched)'}")
            print(f"  backup:    {override_result['backup_path']}")
        else:
            print(f"Overrides: {overrides_path} (no change — already matches)")
    elif not args.no_overrides:
        print(
            "urdf_overrides.yaml NOT patched ($ROBOT_NAME unset, "
            "tinker_robot_config not installed, or file absent). If this robot "
            "uses the override system, the xacro patch alone will NOT take "
            "effect — pass --overrides-yaml <path>."
        )

    print(
        "\nRebuild the affected package(s) (pan_tilt / tinker_urdf / "
        "tinker_robot_config), then restart robot_state_publisher + "
        "pan_tilt state_publisher. URDF, YAML, and per-robot overrides are now "
        f"sourced from the same calibration ({args.results.name})."
    )


if __name__ == "__main__":
    main()
