#!/usr/bin/env python3
"""Pre-warm every checkpoint the tk26_vision stack loads at runtime.

The nodes already download on-demand through ``vision_util.weights_cache``,
so this script is optional — run it once before a match / demo to avoid
paying the download cost at node-start time. It populates the *same* cache
the nodes use, so any future cold-start resolves to a local file.

Coverage:
  * Ultralytics YOLO / YOLO-seg / YOLO-World / SAM (MobileSAM) .pt weights
  * torchvision ResNet50 + ResNet18 ImageNet weights (custom ReID)
  * MediaPipe Tasks pose landmarker bundle (waving detection)

Not covered (API-only, no local weights):
  * Gemini 2.5 Flash via OpenRouter (object_detection_generalist, enable_vlm)
  * OpenRouter LLMs used by kimi_api
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
VISION_DIR = SCRIPT_DIR.parent
# vision_util source isn't on sys.path until colcon-installed, but running
# this straight from the repo is convenient — splice it in ahead of time.
sys.path.insert(0, str(VISION_DIR / "src" / "vision_util"))
sys.path.insert(0, str(VISION_DIR / "src" / "tk_vision_specialized"))

from vision_util.weights_cache import resolve_weights  # noqa: E402
from tk_vision_specialized._pose_backend import (  # noqa: E402
    POSE_MODEL_FILENAME,
    POSE_MODEL_SHA256,
    POSE_MODEL_URL,
)

# Default manifest — what the nodes declare out of the box. Extra sizes are
# opt-in; nobody wants ~1 GB of l/x variants they never load.
DEFAULT_MANIFEST = [
    "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
    "yolov8s-seg.pt", "yolov8s.pt",
    "yolov8s-worldv2.pt",
    "mobile_sam.pt",
]
EXTRA_SIZES = [
    "yolo11l-seg.pt", "yolo11x-seg.pt",
    "yolov8n-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
    "yolov8m-worldv2.pt", "yolov8l-worldv2.pt", "yolov8x-worldv2.pt",
]


def human_size(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f} MB"


def warm_ultralytics(names: list[str]) -> None:
    for name in names:
        path = resolve_weights(name)
        print(f"  ✓ {name:<26} {human_size(path)}  ({path})")


def warm_torchvision() -> None:
    print("warming torchvision (~/.cache/torch/hub/checkpoints)…")
    try:
        import torchvision.models as M
    except ImportError as exc:
        print(f"  ! torchvision unavailable ({exc}); skipping")
        return
    M.resnet50(weights=M.ResNet50_Weights.IMAGENET1K_V2)
    M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    print("  ✓ resnet50 + resnet18")


def fetch_pose_landmarker() -> None:
    """Stage the MediaPipe Tasks pose bundle used by waving_person_server."""
    import hashlib
    from urllib.request import urlopen
    from vision_util.weights_cache import _writable_cache, find_cached
    print("staging mediapipe pose landmarker (.task)…")
    existing = find_cached(POSE_MODEL_FILENAME)
    if existing is not None:
        print(f"  ✓ {POSE_MODEL_FILENAME:<26} {human_size(existing)}  ({existing})")
        return
    target = _writable_cache() / POSE_MODEL_FILENAME
    part = target.with_suffix(target.suffix + ".part")
    try:
        with urlopen(POSE_MODEL_URL, timeout=60) as resp, open(part, "wb") as fp:
            while chunk := resp.read(1 << 20):
                fp.write(chunk)
    except BaseException:
        part.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(part.read_bytes()).hexdigest()
    if digest != POSE_MODEL_SHA256:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"pose_landmarker_full.task digest mismatch: expected {POSE_MODEL_SHA256}, "
            f"got {digest} (downloaded from {POSE_MODEL_URL})"
        )
    part.replace(target)   # atomic: a partial file never satisfies find_cached
    print(f"  ✓ {POSE_MODEL_FILENAME:<26} {human_size(target)}  ({target})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--extra-sizes", action="store_true",
                    help="also warm l/x variants (~1 GB extra)")
    ap.add_argument("--skip-ultralytics", action="store_true")
    ap.add_argument("--skip-torchvision", action="store_true")
    ap.add_argument("--skip-pose", "--skip-mediapipe", dest="skip_pose", action="store_true")
    args = ap.parse_args()

    names = list(DEFAULT_MANIFEST)
    if args.extra_sizes:
        names += EXTRA_SIZES

    if not args.skip_ultralytics:
        warm_ultralytics(names)
    if not args.skip_torchvision:
        warm_torchvision()
    if not args.skip_pose:
        fetch_pose_landmarker()

    print("all vision checkpoints staged in the shared weights cache.")


if __name__ == "__main__":
    main()
