"""TPT-Bench sequence loader (pure python).

TPT-Bench (https://medlartea.github.io/tpt-bench/, arXiv 2505.07446) is a
large-scale, long-term, robot-egocentric dataset for benchmarking *target
person tracking*. Each sequence follows a single target person and is
annotated LaSOT-style:

* The ground-truth bounding box is the tightest up-right box around any visible
  part of the target. The on-disk convention (confirmed in the paper, Sec. 4)
  is ``[u, v, w, h]`` = upper-left corner ``(u, v)`` plus ``(width, height)``,
  one box per frame.
* When the target is not present in a frame it receives an *absent* label.
  The paper writes an absent box as ``0,0,0,0``; LaSOT additionally ships
  separate per-frame absence flag files (``out_of_view.txt`` and
  ``full_occlusion.txt``, ``1`` = absent). In the real LaSOT release each flag
  file is a **single comma-separated line** of 0/1 values (NOT one per line),
  and both files are present — a frame is absent if EITHER is set. This loader
  supports all of: the ``0,0,0,0`` box convention, single-line-comma flag
  files, one-flag-per-line files, and unions every flag file that exists.

Expected per-sequence directory layout (the loader is tolerant of the common
variants seen across LaSOT-style releases)::

    <seq_dir>/
      img/                 # frames: 00000001.jpg, ...  (or *.png)
        00000001.jpg
        ...
      groundtruth.txt      # one bbox per line: "x,y,w,h" (comma OR whitespace)
      absent.txt           # OPTIONAL: one 0/1 flag per line (1 = target absent)

Variants handled:

* Frames may live directly in ``<seq_dir>`` if no ``img/`` subdir exists.
* The ground-truth file may be named ``groundtruth.txt`` or
  ``groundtruth_rect.txt``.
* The absence flag files may be ``absent.txt``, ``out_of_view.txt`` and/or
  ``full_occlusion.txt``; all are optional and every one that exists is
  unioned. Each file's flag count (after flattening on commas/whitespace) must
  match the ground-truth line count.
* Delimiters in the text files may be commas or arbitrary whitespace, and the
  flag files may be a single comma-joined line or one value per line.

NOTE / ASSUMPTION: the project page and arXiv abstract confirm the LaSOT-style
single-target annotation and the ``[u,v,w,h]`` box convention with ``0,0,0,0``
for absent frames, but do not publish the exact on-disk *file names* for every
release. The variant handling above is therefore implemented defensively; if a
real download uses different names, extend ``_GT_NAMES`` / ``_ABSENT_NAMES``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Recognised image extensions (lower-cased), in no particular priority.
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Candidate ground-truth file names, first match wins.
_GT_NAMES = ("groundtruth.txt", "groundtruth_rect.txt")

# Candidate absence-flag file names, first match wins. All optional.
_ABSENT_NAMES = ("absent.txt", "out_of_view.txt", "full_occlusion.txt")


class TptDatasetError(Exception):
    """Raised when a TPT-Bench sequence directory is malformed.

    Covers: missing image/ground-truth files, unparseable bbox lines, and
    line-count mismatches between frames / ground-truth / absence flags.
    """


@dataclass
class TptFrame:
    """One annotated frame of a TPT-Bench sequence.

    Attributes:
        index: zero-based frame index within the sequence.
        image_path: absolute path to the frame image on disk.
        gt_bbox: ground-truth box as ``(x1, y1, x2, y2)`` in pixels, or
            ``None`` when the target is absent in this frame.
    """

    index: int
    image_path: str
    gt_bbox: Optional[Tuple[float, float, float, float]]


def _split_fields(line: str) -> List[str]:
    """Split a text line on commas and/or whitespace, dropping empties."""
    # Normalise commas to spaces, then split on any whitespace run.
    return [tok for tok in line.replace(",", " ").split() if tok]


def _xywh_to_xyxy(
    x: float, y: float, w: float, h: float
) -> Tuple[float, float, float, float]:
    """Convert a top-left ``(x, y, w, h)`` box to ``(x1, y1, x2, y2)``."""
    return (x, y, x + w, y + h)


def _parse_gt_line(line: str, line_no: int) -> Optional[Tuple[float, float, float, float]]:
    """Parse one ground-truth line into an xyxy box, or ``None`` if absent.

    A line of ``0,0,0,0`` (or an empty line) denotes an absent target.
    """
    fields = _split_fields(line)
    if not fields:
        return None
    if len(fields) != 4:
        raise TptDatasetError(
            f"ground-truth line {line_no}: expected 4 values (x,y,w,h), "
            f"got {len(fields)}: {line.strip()!r}"
        )
    try:
        x, y, w, h = (float(v) for v in fields)
    except ValueError as exc:
        raise TptDatasetError(
            f"ground-truth line {line_no}: non-numeric value in {line.strip()!r}"
        ) from exc
    # All-zero box (and degenerate zero-area boxes) => absent.
    if w <= 0 or h <= 0:
        return None
    return _xywh_to_xyxy(x, y, w, h)


def _read_nonempty_lines(path: str) -> List[str]:
    """Read all lines of a file, preserving order, dropping a trailing blank.

    Blank interior lines are kept (they parse as absent in the gt file). Only a
    single trailing newline-induced empty entry is trimmed.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # drop the artifact of a trailing newline
    return lines


def _read_flags(path: str) -> List[int]:
    """Read a binary absence-flag file into a flat list of 0/1 ints.

    Handles both on-disk conventions seen across LaSOT-style releases: one flag
    per line, OR a single line of comma-separated flags (the real form of
    LaSOT's ``out_of_view.txt`` / ``full_occlusion.txt``). Tokens are split on
    commas and/or any whitespace (newlines included); empty tokens dropped.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    tokens = [t for t in text.replace(",", " ").split() if t]
    flags: List[int] = []
    for i, tok in enumerate(tokens):
        try:
            flags.append(int(float(tok)))
        except ValueError as exc:
            raise TptDatasetError(
                f"absence file {path!r}: non-numeric flag #{i + 1}: {tok!r}"
            ) from exc
    return flags


def _find_first(seq_dir: str, names: Tuple[str, ...]) -> Optional[str]:
    """Return the first existing path among ``names`` under ``seq_dir``."""
    for name in names:
        cand = os.path.join(seq_dir, name)
        if os.path.isfile(cand):
            return cand
    return None


def _enumerate_frames(seq_dir: str) -> List[str]:
    """Return sorted absolute frame image paths for a sequence.

    Prefers an ``img/`` subdirectory; falls back to image files in the
    sequence directory itself.
    """
    img_subdir = os.path.join(seq_dir, "img")
    search_dir = img_subdir if os.path.isdir(img_subdir) else seq_dir
    try:
        entries = os.listdir(search_dir)
    except OSError as exc:
        raise TptDatasetError(f"cannot list frames in {search_dir!r}: {exc}") from exc
    images = [
        os.path.join(search_dir, name)
        for name in entries
        if os.path.splitext(name)[1].lower() in _IMAGE_EXTS
    ]
    images.sort()  # lexical sort works for zero-padded frame names
    return images


def load_sequence(seq_dir: str) -> List[TptFrame]:
    """Load one TPT-Bench sequence into a list of :class:`TptFrame`.

    Parses the ground-truth boxes, optional per-frame absence flags, and the
    enumerated (sorted) frame images. ``gt_bbox`` is ``None`` for frames where
    the absence flag is set OR the ground-truth box is ``0,0,0,0`` / empty.

    Args:
        seq_dir: path to a single sequence directory.

    Returns:
        Frames in sequence order, each with its xyxy ground-truth box or None.

    Raises:
        TptDatasetError: missing ground-truth/image files, unparseable lines,
            or mismatched line/frame counts.
    """
    if not os.path.isdir(seq_dir):
        raise TptDatasetError(f"sequence dir not found: {seq_dir!r}")

    gt_path = _find_first(seq_dir, _GT_NAMES)
    if gt_path is None:
        raise TptDatasetError(
            f"no ground-truth file ({' / '.join(_GT_NAMES)}) in {seq_dir!r}"
        )

    gt_lines = _read_nonempty_lines(gt_path)
    if not gt_lines:
        raise TptDatasetError(f"ground-truth file is empty: {gt_path!r}")

    gt_boxes = [_parse_gt_line(line, i + 1) for i, line in enumerate(gt_lines)]

    # Optional per-frame absence flags. LaSOT ships BOTH out_of_view.txt and
    # full_occlusion.txt (each a single comma-separated line of 0/1); a frame
    # is "absent" if ANY present flag file marks it, so we union all that exist.
    absent_flags: Optional[List[bool]] = None
    for name in _ABSENT_NAMES:
        cand = os.path.join(seq_dir, name)
        if not os.path.isfile(cand):
            continue
        flags = _read_flags(cand)
        if len(flags) != len(gt_lines):
            raise TptDatasetError(
                f"absence flag count ({len(flags)}) != ground-truth "
                f"count ({len(gt_lines)}) for {cand!r}"
            )
        if absent_flags is None:
            absent_flags = [False] * len(gt_lines)
        for i, fl in enumerate(flags):
            if fl != 0:
                absent_flags[i] = True

    frame_paths = _enumerate_frames(seq_dir)
    if not frame_paths:
        raise TptDatasetError(f"no frame images found under {seq_dir!r}")
    if len(frame_paths) != len(gt_lines):
        raise TptDatasetError(
            f"frame count ({len(frame_paths)}) != ground-truth count "
            f"({len(gt_lines)}) for {seq_dir!r}"
        )

    frames: List[TptFrame] = []
    for i, (path, box) in enumerate(zip(frame_paths, gt_boxes)):
        if absent_flags is not None and absent_flags[i]:
            box = None
        frames.append(TptFrame(index=i, image_path=path, gt_bbox=box))
    return frames
