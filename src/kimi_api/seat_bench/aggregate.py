"""Aggregate results/<cell>/*.json into report.md + sheets/<cell>.jpg.

No API calls. Reads every cell dir, recomputes per-cell aggregates, writes
a scoreboard sorted by hit_rate, and tiles each cell's overlays into a
contact sheet for eyeballing misses.
"""

from __future__ import annotations

import glob
import json
import math
import os

import cv2
import numpy as np

from .paths import REPORT_PATH, RESULTS_DIR, SHEETS_DIR
from .score import aggregate


def _cells() -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(str(RESULTS_DIR / "*"))
                  if os.path.isdir(p))


def _cell_rows(cell: str) -> list[dict]:
    rows = []
    for jf in sorted(glob.glob(str(RESULTS_DIR / cell / "*.json"))):
        rec = json.load(open(jf))
        sc = rec.get("scoring")
        if sc:
            rows.append({**sc, "elapsed_s": rec.get("elapsed_s", 0.0),
                         "n_calls": rec.get("n_calls", 0)})
    return rows


def _contact_sheet(cell: str, cols: int = 5) -> None:
    imgs = [cv2.imread(p) for p in sorted(glob.glob(str(RESULTS_DIR / cell / "*.jpg")))]
    imgs = [im for im in imgs if im is not None]
    if not imgs:
        return
    th, tw = 240, 360
    tiles = [cv2.resize(im, (tw, th)) for im in imgs]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    SHEETS_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SHEETS_DIR / f"{cell}.jpg"), sheet)


def main():
    lines = ["# Seat-Recommendation Strategy Benchmark — Results", ""]
    lines += ["| cell | n | hit_rate | hits | wrong_seat | miss | false_none | "
              "correct_reject | mean_s | mean_calls |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    table = []
    for cell in _cells():
        rows = _cell_rows(cell)
        if not rows:
            continue
        agg = aggregate(rows)
        mean_s = sum(r["elapsed_s"] for r in rows) / len(rows)
        mean_calls = sum(r["n_calls"] for r in rows) / len(rows)
        table.append((agg["hit_rate"], cell, agg, mean_s, mean_calls))
        _contact_sheet(cell)
    for hit_rate, cell, agg, mean_s, mean_calls in sorted(table, reverse=True):
        lines.append(
            f"| {cell} | {agg['n']} | {hit_rate:.0%} | {agg['hits']} | "
            f"{agg['wrong_seat']} | {agg['miss']} | {agg['false_none']} | "
            f"{agg['correct_reject']} | {mean_s:.1f} | {mean_calls:.1f} |")
    lines += ["", "Contact sheets per cell under `sheets/`. Green box = empty "
              "GT cushion, red = occupied, cyan = predicted box, magenta dot = "
              "predicted point.", ""]
    REPORT_PATH.write_text("\n".join(lines))
    print(f"wrote {REPORT_PATH} and {SHEETS_DIR}/*.jpg")


if __name__ == "__main__":
    main()
