"""Emit a unified diff patching `urdf/pan_tilt.urdf.xacro` from calibration results.

This tool **does not apply** the patch. It prints the diff to stdout (and
optionally writes to a file) so the operator can review and apply it
manually with e.g. `patch -p0 < calib.patch`.

We update only two origin entries:

  - `pan_joint` origin xyz: fitted `t_a`; rpy remains "0 0 0".
  - `camera_mount_joint` origin xyz: fitted `t_b_trans`; rpy remains "0 0 0"
    (or the fitted rotvec converted to RPY if `fit_tb_rotation` was used and
    the value is non-trivial).

If the `--results` file is a `polish.json`, its `t_b_rotvec` is written out to
the `camera_mount_joint` rpy slot; otherwise rpy is forced to zero per plan.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


JOINT_BLOCK_RE = re.compile(
    r'<joint\s+name="(?P<name>[^"]+)"[^>]*>(?P<body>.*?)</joint>',
    re.DOTALL,
)

ORIGIN_RE = re.compile(
    r'<origin\s+xyz="[^"]+"\s+rpy="[^"]+"\s*/>'
)


def _fmt_triplet(v) -> str:
    return f"{v[0]:.6g} {v[1]:.6g} {v[2]:.6g}"


def _load_params(results_path: Path) -> dict:
    """Return the `params` block from a chain.json / polish.json."""
    blob = json.loads(results_path.read_text())
    if "params" in blob:
        return blob["params"]
    raise ValueError(f"{results_path} has no 'params' key")


def _patched_xacro(xacro_text: str, t_a, t_b_trans, t_b_rotvec=None) -> str:
    """Return xacro text with `pan_joint` and `camera_mount_joint` origins updated."""

    def repl(match):
        name = match.group("name")
        body = match.group("body")
        if name == "pan_joint":
            new_xyz = _fmt_triplet(t_a)
            new_rpy = "0 0 0"
        elif name == "camera_mount_joint":
            new_xyz = _fmt_triplet(t_b_trans)
            if t_b_rotvec is not None and np.linalg.norm(t_b_rotvec) > 1e-6:
                rpy = Rotation.from_rotvec(np.asarray(t_b_rotvec)).as_euler("xyz")
                new_rpy = _fmt_triplet(rpy)
            else:
                new_rpy = "0 0 0"
        else:
            return match.group(0)

        new_body = ORIGIN_RE.sub(
            f'<origin xyz="{new_xyz}" rpy="{new_rpy}"/>', body, count=1,
        )
        return match.group(0).replace(body, new_body, 1)

    return JOINT_BLOCK_RE.sub(repl, xacro_text)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Emit a diff patching pan_tilt.urdf.xacro from calibration results.",
    )
    parser.add_argument("--results", required=True, type=Path,
                        help="chain.json or polish.json produced by run_calibration")
    parser.add_argument("--xacro", required=True, type=Path,
                        help="path to pan_tilt.urdf.xacro to update")
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
