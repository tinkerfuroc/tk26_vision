#!/usr/bin/env python3
"""Fetch the Fast-FoundationStereo `23-36-37` checkpoint into the weights cache.

    python -m foundation_stereo.scripts.fetch_fast_fs_weights [--weights-root DIR]

Downloads the upstream Google-Drive folder with gdown (only the 23-36-37
subfolder is kept), records SHA256SUMS after the first successful download
and verifies against it on later runs. Idempotent. Fails loudly if Drive is
unreachable or the folder layout changed — no workaround is attempted.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_WEIGHTS_ROOT = "~/.cache/tk26_vision/weights/foundation_stereo"
DRIVE_FOLDER_ID = "1HuTt7UIp7gQsMiDvJwVuWmKpvFzIIMap"   # readme "Weights and Trade-off"
CKPT_NAME = "23-36-37"
PICKLE_NAME = "model_best_bp2_serialize.pth"

# Validated digest for the 23-36-37 checkpoint (task-6 report). Passed as the
# default `expected` to verify_or_write_sums / --expected-sha256 below, so a
# corrupted download or a Drive-side swap is caught before SHA256SUMS is
# (re)written, rather than only being trust-on-first-use.
EXPECTED_SHA256 = "af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990"


def checkpoint_path(weights_root: str) -> Path:
    root = Path(os.path.expanduser(weights_root))
    return root / "Fast-FoundationStereo" / "weights" / CKPT_NAME / PICKLE_NAME


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_or_write_sums(ckpt_dir: Path, expected: str | None = EXPECTED_SHA256) -> str:
    pickle = ckpt_dir / PICKLE_NAME
    digest = _sha256(pickle)
    if expected and digest != expected:
        raise RuntimeError(
            f"{pickle} sha256 {digest} != expected {expected}; "
            "refusing to trust this checkpoint (pass --expected-sha256 '' "
            "to disable this check for a future checkpoint)")
    sums = ckpt_dir / "SHA256SUMS"
    if sums.exists():
        recorded = sums.read_text().split()[0]
        if recorded != digest:
            raise RuntimeError(
                f"{pickle} sha256 {digest} != recorded {recorded} in {sums}; "
                "delete the directory to re-download")
    else:
        sums.write_text(f"{digest}  {PICKLE_NAME}\n")
    return digest


def download(folder_id: str, dest_ckpt_dir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="fast_fs_") as tmp:
        cmd = [sys.executable, "-m", "gdown", "--folder",
               f"https://drive.google.com/drive/folders/{folder_id}", "-O", tmp]
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True)
        hits = list(Path(tmp).rglob(f"{CKPT_NAME}/{PICKLE_NAME}"))
        if not hits:
            found = sorted(str(p.relative_to(tmp)) for p in Path(tmp).rglob("*"))[:40]
            raise RuntimeError(
                f"{CKPT_NAME}/{PICKLE_NAME} not found in Drive folder {folder_id}; "
                f"layout seen: {found}")
        src_dir = hits[0].parent
        dest_ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, dest_ckpt_dir, dirs_exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights-root", default=DEFAULT_WEIGHTS_ROOT)
    ap.add_argument("--folder-id", default=DRIVE_FOLDER_ID)
    ap.add_argument(
        "--expected-sha256", default=EXPECTED_SHA256,
        help="Expected sha256 of the checkpoint pickle; raises if it "
             "doesn't match. Pass '' to disable (e.g. for a future "
             "checkpoint whose digest isn't validated yet).")
    args = ap.parse_args()
    pickle = checkpoint_path(args.weights_root)
    if not pickle.exists():
        download(args.folder_id, pickle.parent)
    if not pickle.exists():
        print(f"ERROR: {pickle} still missing after download", file=sys.stderr)
        return 1
    digest = verify_or_write_sums(pickle.parent, expected=args.expected_sha256 or None)
    print(f"ok {pickle} ({pickle.stat().st_size / 1e6:.1f} MB) sha256 {digest[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
