"""Score a strategy result against hand-annotated ground truth (2D only).

Outcome taxonomy (per scene):
  hit           : recommendation point lands inside an EMPTY cushion bbox.
  wrong_seat    : point lands inside an OCCUPIED cushion bbox.
  miss          : point lands outside every cushion bbox (but a seat existed).
  false_none    : strategy said "none" though >=1 empty seat exists.
  correct_reject: strategy said "none" and no empty seat exists.
The headline metric is hit_rate = hits / scenes_with_empty_seat.
"""

from __future__ import annotations

from typing import Optional

from .geometry import point_in_box


def _has_empty(gt: dict) -> bool:
    return any(not s["occupied"] for s in gt["seats"])


def classify(result: dict, gt: dict) -> dict:
    point = result.get("point_xy")
    chose_none = (
        point is None
        or str(result.get("chosen_label", "")).strip().lower() == "none"
    )
    empty_exists = _has_empty(gt)

    if chose_none:
        outcome = "false_none" if empty_exists else "correct_reject"
        return {"outcome": outcome, "in_box": None}

    pt = (int(point[0]), int(point[1]))
    for s in gt["seats"]:
        if point_in_box(pt, tuple(s["cushion_bbox"])):
            return {
                "outcome": "hit" if not s["occupied"] else "wrong_seat",
                "in_box": s["label"],
            }
    return {"outcome": "miss", "in_box": None}


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    by = {}
    for r in rows:
        by[r["outcome"]] = by.get(r["outcome"], 0) + 1
    hits = by.get("hit", 0)
    # denominator excludes correct_reject (no empty seat to find)
    scored = n - by.get("correct_reject", 0)
    return {
        "n": n,
        "hits": hits,
        "hit_rate": hits / scored if scored else 0.0,
        "wrong_seat": by.get("wrong_seat", 0),
        "miss": by.get("miss", 0),
        "false_none": by.get("false_none", 0),
        "correct_reject": by.get("correct_reject", 0),
    }
