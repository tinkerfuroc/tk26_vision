"""Relocatable weight cache for tk26_vision nodes.

Every vision node that loads a YOLO / FastSAM / YOLO-World `.pt` file should
pipe the bare filename through ``resolve_weights`` before handing it to
Ultralytics. The helper returns an absolute path so the result is independent
of the CWD that ``ros2 run`` was launched from.

Lookup order:
    1. ``name`` is an absolute path and exists on disk.
    2. ``name`` contains a path separator but is not absolute → ``ValueError``
       (catches accidental CWD-relative bugs early).
    3. ``$TK26_MODEL_CACHE/<name>`` if the env var is set and file exists.
    4. ``~/.cache/tk26_vision/weights/<name>`` if file exists.
    5. Auto-download via Ultralytics into the writable cache (#4 if env var is
       unset, otherwise into ``$TK26_MODEL_CACHE``).

The download step ``os.chdir``'s to the cache because Ultralytics always
writes to CWD regardless of the filename it was asked to load. A ``FileLock``
serialises concurrent downloads if several nodes start at once.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_DEFAULT_CACHE = Path.home() / ".cache" / "tk26_vision" / "weights"
_ENV_VAR = "TK26_MODEL_CACHE"


def _writable_cache() -> Path:
    override = os.environ.get(_ENV_VAR)
    cache = Path(override).expanduser() if override else _DEFAULT_CACHE
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _search_paths(name: str) -> list[Path]:
    hits: list[Path] = []
    override = os.environ.get(_ENV_VAR)
    if override:
        hits.append(Path(override).expanduser() / name)
    hits.append(_DEFAULT_CACHE / name)
    return hits


@contextmanager
def _download_lock(path: Path) -> Iterator[None]:
    """Serialise concurrent downloads of the same weight.

    Prefer `filelock` (pulled in transitively by torch / ultralytics); fall
    back to `fcntl.flock` on a sidecar file if it isn't importable.
    """
    try:
        from filelock import FileLock
    except ImportError:
        import fcntl
        fp = open(path, "w")
        try:
            fcntl.flock(fp, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)
            fp.close()
        return

    with FileLock(str(path)):
        yield


def _pick_ultralytics_cls(name: str):
    lower = name.lower()
    if lower.startswith("fastsam"):
        from ultralytics import FastSAM
        return FastSAM
    if "world" in lower:
        from ultralytics import YOLOWorld
        return YOLOWorld
    from ultralytics import YOLO
    return YOLO


def _download(name: str, cache: Path) -> Path:
    cls = _pick_ultralytics_cls(name)
    prev_cwd = Path.cwd()
    os.chdir(cache)
    try:
        cls(name)  # Ultralytics writes <name> into CWD if missing.
    finally:
        os.chdir(prev_cwd)
    dst = cache / name
    if not dst.exists():
        raise FileNotFoundError(
            f"Ultralytics reported success but {dst} is missing"
        )
    return dst


def resolve_weights(name: str) -> Path:
    """Return an absolute path to the requested weight file.

    Downloads into the user cache on miss. See module docstring for lookup
    order.
    """
    if not name:
        raise ValueError("resolve_weights: name must be non-empty")

    candidate = Path(name)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            f"resolve_weights: absolute path {candidate} does not exist"
        )

    if os.sep in name or (os.altsep and os.altsep in name):
        raise ValueError(
            f"resolve_weights: relative paths with separators are rejected "
            f"(got {name!r}) — pass a bare filename or an absolute path"
        )

    for path in _search_paths(name):
        if path.exists():
            return path

    cache = _writable_cache()
    lock_path = cache / f"{name}.lock"
    with _download_lock(lock_path):
        # Re-check inside the lock — another process may have downloaded
        # while we waited.
        target = cache / name
        if target.exists():
            return target
        return _download(name, cache)
