"""Thin cv2 UI to label the operator (target person) in a recorded rosbag.

    python -m ptbench.labeler.label_cli --bag DIR [--out GT.json] \
        [--scenario NAME] [--color-topic ...] [--depth-topic ...] \
        [--camera-info-topic ...]

Steps the color frames of a bag, lets the operator draw/adjust a bounding box
around the target person, toggle presence, and step back and forth. The box is
copied forward as the default for each new frame. On save, depth + CameraInfo
are read from the bag, per-frame 3D centroids are sampled, and a schema-valid
``GtClip`` is written to ``--out`` (default ``<bag_dir>/gt.json``).

All real logic lives in :mod:`ptbench.labeler.label_io`. cv2 is imported lazily
inside :func:`main` so that ``import ptbench.labeler.label_cli`` succeeds in a
headless environment (e.g. unit-test collection).

Keys:
    n / →          next frame
    p / ←          previous frame
    space          toggle present / absent for this frame
    b / r          (re)draw the bounding box for this frame
    c              clear the box (marks absent)
    s              save GT json
    q / Esc        quit (prompts nothing; save explicitly with 's' first)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

from ptbench.common.schema import GtClip, save_gt
from ptbench.labeler.label_io import (
    FrameAnnotation,
    build_gt_clip,
    propagate_default,
    read_color_frames,
    read_depth_and_info,
)

# Schema defaults — keep in sync with GtClip's field defaults.
DEFAULT_COLOR_TOPIC = GtClip.__dataclass_fields__["color_topic"].default
DEFAULT_DEPTH_TOPIC = GtClip.__dataclass_fields__["depth_topic"].default
DEFAULT_INFO_TOPIC = GtClip.__dataclass_fields__["camera_info_topic"].default


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ptbench.labeler.label_cli",
        description="Label the operator (target person) in a rosbag → GT json.",
    )
    p.add_argument("--bag", required=True, help="rosbag2 directory")
    p.add_argument(
        "--out",
        default=None,
        help="output GT json path (default: <bag_dir>/gt.json)",
    )
    p.add_argument(
        "--scenario",
        default="unlabeled",
        help="scenario name recorded in the GT (e.g. cml_crossing)",
    )
    p.add_argument(
        "--clip-id",
        default=None,
        help="clip id recorded in the GT (default: bag dir basename)",
    )
    p.add_argument("--notes", default="", help="free-text notes for the GT")
    p.add_argument("--color-topic", default=DEFAULT_COLOR_TOPIC)
    p.add_argument("--depth-topic", default=DEFAULT_DEPTH_TOPIC)
    p.add_argument("--camera-info-topic", default=DEFAULT_INFO_TOPIC)
    p.add_argument("--fps-hint", type=float, default=30.0)
    return p.parse_args(argv)


def _save(annotations, args, bag_dir: Path, out_path: Path) -> None:
    """Read depth+info, build the GtClip, and write it."""
    print(f"[labeler] reading depth + camera_info from {bag_dir} ...")
    depth_list, K = read_depth_and_info(
        str(bag_dir), args.depth_topic, args.camera_info_topic
    )
    if K is None:
        print(
            f"[labeler] WARNING: no CameraInfo on {args.camera_info_topic!r}; "
            "centroids will be None."
        )
    clip_id = args.clip_id or bag_dir.name
    clip = build_gt_clip(
        annotations,
        depth_list,
        K if K is not None else [1, 0, 0, 0, 1, 0, 0, 0, 1],
        clip_id=clip_id,
        bag_path=str(bag_dir),
        scenario=args.scenario,
        color_topic=args.color_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
        fps_hint=args.fps_hint,
        notes=args.notes,
    )
    # When K is unknown, force centroids to None rather than garbage.
    if K is None:
        for f in clip.frames:
            f.centroid_3d = None
    save_gt(clip, out_path)
    n_present = sum(1 for f in clip.frames if f.present)
    print(
        f"[labeler] wrote {out_path} "
        f"({len(clip.frames)} frames, {n_present} present)"
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    # Lazy cv2 import so module import is headless-safe.
    import cv2
    import numpy as np

    bag_dir = Path(args.bag)
    out_path = Path(args.out) if args.out else bag_dir / "gt.json"

    print(f"[labeler] reading color frames from {bag_dir} ...")
    color_frames = read_color_frames(str(bag_dir), args.color_topic)
    if not color_frames:
        print(f"[labeler] no color frames on {args.color_topic!r}; nothing to do.")
        return 1
    print(f"[labeler] {len(color_frames)} color frames.")

    # One annotation slot per color frame, seeded absent.
    annotations: List[FrameAnnotation] = [
        FrameAnnotation(t_ns=t_ns, present=False, bbox=None)
        for (t_ns, _img) in color_frames
    ]

    win = "ptbench labeler"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    idx = 0

    def _apply_default(i: int) -> None:
        """Seed an untouched frame's box from the previous frame."""
        prev = annotations[i - 1] if i > 0 else None
        box = propagate_default(prev)
        if box is not None:
            annotations[i].bbox = box
            annotations[i].present = True

    def _draw(i: int):
        _t_ns, img = color_frames[i]
        disp = img.copy()
        ann = annotations[i]
        color = (0, 255, 0) if ann.present else (0, 0, 255)
        if ann.bbox is not None:
            x1, y1, x2, y2 = (int(round(v)) for v in ann.bbox)
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
        status = "PRESENT" if ann.present else "absent"
        label = f"[{i + 1}/{len(color_frames)}] {status}"
        cv2.putText(
            disp, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
        cv2.putText(
            disp,
            "n/p step  space=toggle  b=box  c=clear  s=save  q=quit",
            (10, disp.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.imshow(win, disp)

    def _select_box(i: int) -> None:
        _t_ns, img = color_frames[i]
        roi = cv2.selectROI(win, img, showCrosshair=True, fromCenter=False)
        cv2.setWindowTitle(win, win)
        x, y, w, h = roi
        if w > 0 and h > 0:
            annotations[i].bbox = (float(x), float(y), float(x + w), float(y + h))
            annotations[i].present = True
        # if cancelled (zero size), leave the existing box untouched

    # Seed the first frame's default (no-op since there is no prev).
    _apply_default(idx)
    _draw(idx)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 255:
            continue
        if key in (ord("q"), 27):  # q / Esc
            break
        elif key in (ord("n"), 83):  # n / right-arrow
            if idx < len(color_frames) - 1:
                idx += 1
                # Seed default only for a frame the operator hasn't touched.
                if annotations[idx].bbox is None and not annotations[idx].present:
                    _apply_default(idx)
                _draw(idx)
        elif key in (ord("p"), 81):  # p / left-arrow
            if idx > 0:
                idx -= 1
                _draw(idx)
        elif key == ord(" "):  # toggle present/absent
            ann = annotations[idx]
            if ann.present:
                ann.present = False
            else:
                ann.present = ann.bbox is not None
            _draw(idx)
        elif key in (ord("b"), ord("r")):  # (re)draw box
            _select_box(idx)
            _draw(idx)
        elif key == ord("c"):  # clear box -> absent
            annotations[idx].bbox = None
            annotations[idx].present = False
            _draw(idx)
        elif key == ord("s"):  # save
            _save(annotations, args, bag_dir, out_path)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
