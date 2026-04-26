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
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


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


def _patched_standalone(xacro_text: str, t_a, t_b_trans, t_b_rotvec) -> str:
    """Patch the tk26_vision standalone form: literal pan_joint + camera_mount_joint."""
    have_rot = np.linalg.norm(t_b_rotvec) > 1e-6
    rpy_str = (
        _fmt_triplet(Rotation.from_rotvec(np.asarray(t_b_rotvec)).as_euler("xyz"))
        if have_rot else None
    )

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


def _patched_macro(xacro_text: str, t_a, t_b_trans, t_b_rotvec) -> str:
    """Patch the tk25_basic macro form: `attach_xyz` default + camera_mount_joint."""
    have_rot = np.linalg.norm(t_b_rotvec) > 1e-6
    rpy_str = (
        _fmt_triplet(Rotation.from_rotvec(np.asarray(t_b_rotvec)).as_euler("xyz"))
        if have_rot else None
    )

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


def _patched_xacro(xacro_text: str, t_a, t_b_trans, t_b_rotvec=None) -> str:
    t_b_rotvec = np.zeros(3) if t_b_rotvec is None else np.asarray(t_b_rotvec, dtype=float)
    if MACRO_DECL_RE.search(xacro_text):
        return _patched_macro(xacro_text, t_a, t_b_trans, t_b_rotvec)
    return _patched_standalone(xacro_text, t_a, t_b_trans, t_b_rotvec)


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
    args = parser.parse_args(argv)

    params = _load_params(args.results)
    t_a = np.asarray(params["t_a"], dtype=float)
    t_b_trans = np.asarray(params["t_b_trans"], dtype=float)
    t_b_rotvec = np.asarray(params.get("t_b_rotvec", [0, 0, 0]), dtype=float)

    original = args.xacro.read_text()
    patched = _patched_xacro(original, t_a, t_b_trans, t_b_rotvec)

    if args.out:
        args.out.write_text(patched)
        print(f"Wrote patched xacro to {args.out}")
        print("Review, then replace the original if the content is correct.")
        return

    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        patched.splitlines(keepends=True),
        fromfile=str(args.xacro),
        tofile=str(args.xacro) + " (calibrated)",
    )
    print("".join(diff), end="")


if __name__ == "__main__":
    main()
