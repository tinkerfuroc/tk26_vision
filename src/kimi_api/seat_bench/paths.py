"""Filesystem locations + sys.path bootstrap so `import kimi_api.*` works
when running `python -m seat_bench.<mod>` from `src/kimi_api/`."""

from __future__ import annotations

import sys
from pathlib import Path

# seat_bench/ -> kimi_api/ (parent of the seat_bench package dir)
PKG_DIR = Path(__file__).resolve().parent          # .../src/kimi_api/seat_bench
KIMI_API_SRC = PKG_DIR.parent                       # .../src/kimi_api

DATASET_DIR = PKG_DIR / "dataset"
RESULTS_DIR = PKG_DIR / "results"
SHEETS_DIR = PKG_DIR / "sheets"
REPORT_PATH = PKG_DIR / "report.md"


def find_vision_log() -> Path:
    """Walk up from this package to find the tk25_ws/vision_log directory.

    Starts the search from outside the git repository root so that a
    vision_log/ symlink or directory inside the repo tree (e.g.
    tk26_vision/vision_log/) is not mistakenly returned instead of the
    canonical workspace-level tk25_ws/vision_log/.
    """
    import subprocess

    # Determine the git repo root so we can start the search above it.
    try:
        result = subprocess.run(
            ["git", "-C", str(PKG_DIR), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        repo_root = Path(result.stdout.strip())
        search_start = repo_root.parent
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Not in a git repo or git not available — fall back to walking from PKG_DIR.
        search_start = PKG_DIR

    for parent in [search_start, *search_start.parents]:
        candidate = parent / "vision_log"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate a vision_log/ directory above seat_bench")


def ensure_kimi_api_importable() -> None:
    """Put src/kimi_api on sys.path so `import kimi_api._seat_vlm` resolves."""
    p = str(KIMI_API_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)
