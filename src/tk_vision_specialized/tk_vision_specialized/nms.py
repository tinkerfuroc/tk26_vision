"""Pure-function NMS, clustering, and judge-payload helpers for
object_match_all.

No ROS imports here on purpose: this module is unit-testable from a plain
pytest run without sourcing the workspace. The shapes defined here are
reused by `match_pipeline.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


Bbox = tuple[int, int, int, int]  # (x1, y1, x2, y2) in pixel coords


@dataclass(frozen=True)
class MatchRow:
    label: str
    bbox: Bbox
    conf: float


def iou(a: Bbox, b: Bbox) -> float:
    """Standard intersection-over-union on xyxy boxes. 0.0 on zero-area inputs."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    a_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    if a_area == 0 or b_area == 0:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    return inter / float(a_area + b_area - inter)


def suppress_within_category(
    rows: Sequence[MatchRow],
    iou_thresh: float,
) -> list[MatchRow]:
    """Greedy NMS, applied within each label independently.

    Same-label boxes that overlap above `iou_thresh` collapse to the higher
    confidence one. Different-label overlaps are preserved (resolved
    elsewhere by the cross-category clusterer + judge)."""

    by_label: dict[str, list[MatchRow]] = {}
    for r in rows:
        by_label.setdefault(r.label, []).append(r)

    kept: list[MatchRow] = []
    for _label, group in by_label.items():
        group.sort(key=lambda r: r.conf, reverse=True)
        survivors: list[MatchRow] = []
        for cand in group:
            if all(iou(cand.bbox, s.bbox) < iou_thresh for s in survivors):
                survivors.append(cand)
        kept.extend(survivors)
    return kept


@dataclass(frozen=True)
class Cluster:
    rows: list[MatchRow]

    def distinct_labels(self) -> list[str]:
        seen: list[str] = []
        for r in self.rows:
            if r.label not in seen:
                seen.append(r.label)
        return seen

    def is_conflict(self) -> bool:
        return len(self.rows) >= 2 and len(self.distinct_labels()) >= 2


@dataclass(frozen=True)
class JudgePayload:
    cluster: Cluster
    crop: np.ndarray
    crop_origin: tuple[int, int]                # (x_min, y_min) in scene coords
    competing: list[tuple[str, str]]            # (label, ref_data_url), deduped


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self) -> list[list[int]]:
        gmap: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            gmap.setdefault(self.find(i), []).append(i)
        return list(gmap.values())


def cluster_for_judge(
    rows: Sequence[MatchRow],
    iou_thresh: float,
) -> list[Cluster]:
    """Greedy connected-components over the IoU graph.

    Two rows share an edge iff their IoU >= `iou_thresh`. Connected
    components become clusters. Singletons and same-label-only clusters are
    not conflicts; multi-label clusters of size >= 2 are."""

    rows = list(rows)
    n = len(rows)
    if n == 0:
        return []

    uf = _UnionFind(n)
    for i, j in combinations(range(n), 2):
        if iou(rows[i].bbox, rows[j].bbox) >= iou_thresh:
            uf.union(i, j)

    return [Cluster(rows=[rows[k] for k in members]) for members in uf.groups()]


def build_judge_payload(
    cluster: Cluster,
    items: dict[str, str],            # label -> ref_data_url
    scene_bgr: np.ndarray,
    margin_px: int,
) -> JudgePayload:
    """Compute the union bbox of cluster members, expand by `margin_px`,
    clamp to scene bounds, and produce the cropped image + the competing
    label/ref pairs (deduped by label).

    Defensive: an empty cluster yields an empty crop + empty competing list
    rather than raising. cluster_for_judge never emits empty clusters in
    practice, but consumers may construct payloads directly in tests."""

    if not cluster.rows:
        return JudgePayload(
            cluster=cluster,
            crop=np.zeros((0, 0, 3), dtype=scene_bgr.dtype),
            crop_origin=(0, 0),
            competing=[],
        )

    h, w = scene_bgr.shape[:2]
    x1 = min(r.bbox[0] for r in cluster.rows)
    y1 = min(r.bbox[1] for r in cluster.rows)
    x2 = max(r.bbox[2] for r in cluster.rows)
    y2 = max(r.bbox[3] for r in cluster.rows)

    x1c = max(0, x1 - margin_px)
    y1c = max(0, y1 - margin_px)
    x2c = min(w, x2 + margin_px)
    y2c = min(h, y2 + margin_px)
    # .copy() so the judge consumer can draw on the crop without aliasing
    # the live scene buffer (a camera callback may overwrite it later).
    crop = scene_bgr[y1c:y2c, x1c:x2c].copy()

    seen: set[str] = set()
    competing: list[tuple[str, str]] = []
    for r in cluster.rows:
        if r.label in seen:
            continue
        if r.label in items:
            competing.append((r.label, items[r.label]))
            seen.add(r.label)
    return JudgePayload(
        cluster=cluster,
        crop=crop,
        crop_origin=(x1c, y1c),
        competing=competing,
    )
