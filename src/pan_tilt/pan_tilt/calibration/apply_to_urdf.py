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

T_A rotation is never written — our plan locks it at identity and only the
optional polish phase would touch it, which we haven't wired through.
"""

from __future__ import annotations

import argparse
import difflib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .utils import body_yaw_from_rotvec, rotvec_to_xyz_euler


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


def _patched_standalone(
    xacro_text: str, t_a, t_b_trans, t_b_rotvec, *,
    allow_flipped_camera: bool = False,
) -> str:
    """Patch the tk26_vision standalone form: literal pan_joint + camera_mount_joint."""
    rpy_str = _rotvec_to_rpy_str(t_b_rotvec, allow_flipped_camera)
    have_rot = rpy_str is not None

    def repl(match):
        name = _bare_name(match.group("name"))
        body = match.group("body")
        if name == "pan_joint":
            # T_A rotation locked identity in our model — don't overwrite rpy.
            new_body = _replace_origin(
                body, _fmt_triplet(t_a), preserve_rpy=True, new_rpy=None,
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
    xacro_text: str, t_a, t_b_trans, t_b_rotvec, *,
    allow_flipped_camera: bool = False,
) -> str:
    """Patch the tk25_basic macro form: `attach_xyz` default + camera_mount_joint."""
    rpy_str = _rotvec_to_rpy_str(t_b_rotvec, allow_flipped_camera)
    have_rot = rpy_str is not None

    out = ATTACH_XYZ_DEFAULT_RE.sub(
        lambda m: f"{m.group('key')}{_fmt_triplet(t_a)}{m.group('close')}",
        xacro_text, count=1,
    )
    # attach_rpy is T_A rotation; we don't fit it, so preserve whatever was there.

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
    xacro_text: str, t_a, t_b_trans, t_b_rotvec=None, *,
    allow_flipped_camera: bool = False,
) -> str:
    t_b_rotvec = np.zeros(3) if t_b_rotvec is None else np.asarray(t_b_rotvec, dtype=float)
    if MACRO_DECL_RE.search(xacro_text):
        return _patched_macro(
            xacro_text, t_a, t_b_trans, t_b_rotvec,
            allow_flipped_camera=allow_flipped_camera,
        )
    return _patched_standalone(
        xacro_text, t_a, t_b_trans, t_b_rotvec,
        allow_flipped_camera=allow_flipped_camera,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Emit a diff patching a pan-tilt xacro from calibration results.",
    )
    parser.add_argument("--results", required=True, type=Path,
                        help="chain.json or polish.json produced by run_calibration")
    parser.add_argument("--xacro", required=True, type=Path,
                        help="path to the pan-tilt xacro to update "
                             "(standalone form under tk26_vision/, or the "
                             "tk25_basic macro form under tinker_urdf/src/)")
    parser.add_argument("--out", type=Path, default=None,
                        help="if set, write the full patched xacro here (instead of stdout diff)")
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
    t_b_trans = np.asarray(params["t_b_trans"], dtype=float)
    t_b_rotvec = np.asarray(params.get("t_b_rotvec", [0, 0, 0]), dtype=float)
    pan_offset_rad = float(params.get("theta_p_offset_rad", 0.0))
    tilt_offset_rad = float(params.get("theta_t_offset_rad", 0.0))

    original = args.xacro.read_text()
    try:
        patched = _patched_xacro(
            original, t_a, t_b_trans, t_b_rotvec,
            allow_flipped_camera=args.allow_flipped_camera,
        )
    except CalibrationApplyError as exc:
        # Operator-facing one-shot: the message itself names the override
        # flag, so don't bury it under a stack trace.
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.out:
        args.out.write_text(patched)
        print(f"Wrote patched xacro to {args.out}")
        print("Review, then replace the original if the content is correct.")
    else:
        diff = difflib.unified_diff(
            original.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=str(args.xacro),
            tofile=str(args.xacro) + " (calibrated)",
        )
        print("".join(diff), end="")

    # Operator hint: the URDF patch alone isn't enough — the calibration's
    # joint offsets must also land in pan_tilt.yaml so the state publisher
    # adds them to firmware feedback before publishing /joint_states. Print
    # the values straight from the JSON we already loaded; the operator
    # pastes them into pan_tilt_state_publisher.ros__parameters.
    print(
        f"\n→ Calibration runtime offsets (paste into "
        f"src/tk26_vision/src/pan_tilt/config/pan_tilt.yaml under "
        f"pan_tilt_state_publisher.ros__parameters):\n"
        f"    pan_offset_rad:  {pan_offset_rad:.6f}\n"
        f"    tilt_offset_rad: {tilt_offset_rad:.6f}\n"
        f"\nRestart the pan_tilt launch after BOTH the URDF rebuild AND "
        f"the YAML edit; without the offsets the URDF chain mis-represents "
        f"the camera's pose by ~{abs(math.degrees(tilt_offset_rad)):.0f}° "
        f"of tilt at firmware-zero."
    )


if __name__ == "__main__":
    main()
