"""Few-shot example loader for seat-pointing VLM calls.

Examples are produced by `kimi_api.fewshot_annotator` and committed under
`<package>/fewshot/<slug>/{image.jpg,answer.json,meta.json}`. After
`colcon build --packages-select kimi_api`, they are visible at
`share/kimi_api/fewshot/<slug>/...` (symlinked under `--symlink-install`).

This module is the single place that resolves the share path, validates
each `answer.json` against the seat-pointing schema, decodes the sibling
image, and returns a list ordered deterministically by slug. The result
is mtime-cached on the resolved share dir so a `--symlink-install` edit
to an existing example reloads on the next call without restarting the
node. Adding a *new* slug dir requires a rebuild (data_files runs at
build time) — by design.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ._seat_vlm import _RESPONSE_SCHEMA  # reuse — single source of truth


_PKG = 'kimi_api'
_FEWSHOT_DIRNAME = 'fewshot'
_IMAGE_NAMES = ('image.jpg', 'image.jpeg', 'image.png')
_ANSWER_NAME = 'answer.json'


@dataclass
class FewshotExample:
    slug: str
    image_bgr: np.ndarray
    answer: dict


def _resolve_fewshot_dir() -> str | None:
    """Return absolute path of share/kimi_api/fewshot, or None if unavailable."""
    try:
        from ament_index_python.packages import (
            PackageNotFoundError,
            get_package_share_directory,
        )
    except ImportError:
        return None
    try:
        share = get_package_share_directory(_PKG)
    except PackageNotFoundError:
        return None
    path = os.path.join(share, _FEWSHOT_DIRNAME)
    return path if os.path.isdir(path) else None


def _validate_answer(payload: Any) -> bool:
    """Schema check + cross-field constraints from the seat-pointing prompt.

    Mirrors `_seat_vlm._RESPONSE_SCHEMA` and the `_SYSTEM_PROMPT` rules:
      - top-level keys: visible_seats[], label, point
      - each visible_seat: {label:str, occupied:bool, reason:str}
      - if label == "none" then point == [0, 0]
      - else label must equal one of visible_seats[].label and point is
        two ints in [0, 1000]
    """
    if not isinstance(payload, dict):
        return False
    required = set(_RESPONSE_SCHEMA['required'])
    if not required.issubset(payload.keys()):
        return False
    seats = payload['visible_seats']
    if not isinstance(seats, list):
        return False
    seat_labels: list[str] = []
    for s in seats:
        if not isinstance(s, dict):
            return False
        if not all(k in s for k in ('label', 'occupied', 'reason')):
            return False
        if not isinstance(s['label'], str) or not isinstance(s['reason'], str):
            return False
        if not isinstance(s['occupied'], bool):
            return False
        if not s['label'].strip():
            return False
        seat_labels.append(s['label'])
    label = payload['label']
    if not isinstance(label, str):
        return False
    point = payload['point']
    if not (isinstance(point, list) and len(point) == 2):
        return False
    try:
        py, px = int(point[0]), int(point[1])
    except (TypeError, ValueError):
        return False
    if label.strip().lower() == 'none':
        return py == 0 and px == 0
    if label not in seat_labels:
        return False
    if not (0 <= py <= 1000 and 0 <= px <= 1000):
        return False
    return True


def _find_image(slug_dir: str) -> str | None:
    for name in _IMAGE_NAMES:
        p = os.path.join(slug_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _dir_signature(path: str) -> tuple:
    """Cheap mtime-based fingerprint of the dir tree.

    Walks one level deep — enough since each slug is a single dir of small
    files. Cheap to recompute (microseconds), so we do it on every call.
    """
    sig: list = [os.path.getmtime(path)]
    try:
        for entry in sorted(os.listdir(path)):
            sub = os.path.join(path, entry)
            try:
                sig.append((entry, os.path.getmtime(sub)))
                if os.path.isdir(sub):
                    for f in sorted(os.listdir(sub)):
                        fp = os.path.join(sub, f)
                        sig.append((entry, f, os.path.getmtime(fp)))
            except OSError:
                continue
    except OSError:
        pass
    return tuple(sig)


_CACHE: dict = {}  # {dir_path: (signature, [FewshotExample])}


def load_fewshots(max_n: int, *, logger=None) -> list[FewshotExample]:
    """Load up to `max_n` validated few-shot examples.

    Returns [] (never raises) when:
      - share/kimi_api/fewshot/ is absent
      - no slug subdirs exist
      - all slugs failed validation

    Bad slugs are skipped with a warning via `logger` (if provided).
    """
    if max_n <= 0:
        return []
    fewshot_dir = _resolve_fewshot_dir()
    if fewshot_dir is None:
        return []

    signature = _dir_signature(fewshot_dir)
    cached = _CACHE.get(fewshot_dir)
    if cached is not None and cached[0] == signature:
        return cached[1][:max_n]

    examples: list[FewshotExample] = []
    try:
        slugs = sorted(
            d for d in os.listdir(fewshot_dir)
            if os.path.isdir(os.path.join(fewshot_dir, d))
        )
    except OSError:
        slugs = []

    for slug in slugs:
        slug_dir = os.path.join(fewshot_dir, slug)
        answer_path = os.path.join(slug_dir, _ANSWER_NAME)
        if not os.path.isfile(answer_path):
            continue
        try:
            with open(answer_path, 'r', encoding='utf-8') as f:
                answer = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            if logger is not None:
                logger.warn(f'fewshot {slug}: failed to read answer.json ({exc})')
            continue
        if not _validate_answer(answer):
            if logger is not None:
                logger.warn(f'fewshot {slug}: answer.json failed schema check; skipping')
            continue
        image_path = _find_image(slug_dir)
        if image_path is None:
            if logger is not None:
                logger.warn(f'fewshot {slug}: no image.{{jpg,jpeg,png}}; skipping')
            continue
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            if logger is not None:
                logger.warn(f'fewshot {slug}: cv2.imread returned None; skipping')
            continue
        examples.append(FewshotExample(slug=slug, image_bgr=image, answer=answer))

    _CACHE[fewshot_dir] = (signature, examples)
    return examples[:max_n]
