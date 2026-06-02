"""Scoreboard metrics over an aligned [(GtFrame, PredFrame|None)] list.

Pure functions; never raise on empty or degenerate input. Time spans are derived
from ``t_ns`` deltas (in seconds), so nothing assumes a fixed fps.

Lock definitions (a "pred prefix" means: pred is not None AND not
``target_lost`` AND ``point_xyz`` is not None AND gt.present AND
gt.centroid_3d is not None):

    correct lock : prefix AND dist3d(pred, gt) <= correct_radius_m
    wrong   lock : prefix AND dist3d(pred, gt)  > wrong_radius_m
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .align import PredFrame
from .geometry import dist3d, lateral_range
from .schema import GtFrame

Pair = Tuple[GtFrame, Optional[PredFrame]]


@dataclass
class MetricConfig:
    correct_radius_m: float = 0.50
    wrong_radius_m: float = 0.75
    align_tol_ms: float = 50.0
    sustained_s: float = 0.5


def _has_pred_prefix(pair: Pair) -> bool:
    g, p = pair
    return (
        p is not None
        and not p.target_lost
        and p.point_xyz is not None
        and g.present
        and g.centroid_3d is not None
    )


def _is_correct_lock(pair: Pair, cfg: MetricConfig) -> bool:
    if not _has_pred_prefix(pair):
        return False
    g, p = pair
    return dist3d(p.point_xyz, g.centroid_3d) <= cfg.correct_radius_m


def _is_wrong_lock(pair: Pair, cfg: MetricConfig) -> bool:
    if not _has_pred_prefix(pair):
        return False
    g, p = pair
    return dist3d(p.point_xyz, g.centroid_3d) > cfg.wrong_radius_m


def _median_max(samples: List[float]) -> dict:
    if not samples:
        return {"median": 0.0, "max": 0.0, "samples": []}
    return {
        "median": float(np.median(samples)),
        "max": float(np.max(samples)),
        "samples": [float(s) for s in samples],
    }


def _median_p95(samples: List[float]) -> dict:
    if not samples:
        return {"median": 0.0, "p95": 0.0}
    return {
        "median": float(np.median(samples)),
        "p95": float(np.percentile(samples, 95)),
    }


def compute_metrics(
    aligned: List[Pair],
    throughput_hz: Optional[float] = None,
    cfg: MetricConfig = MetricConfig(),
) -> dict:
    n = len(aligned)
    correct_flags = [_is_correct_lock(pair, cfg) for pair in aligned]
    wrong_flags = [_is_wrong_lock(pair, cfg) for pair in aligned]

    n_present = sum(1 for g, _ in aligned if g.present)
    n_absent = sum(1 for g, _ in aligned if not g.present)

    # --- correct_lock_rate ------------------------------------------------
    n_correct = sum(correct_flags)
    correct_lock_rate = (n_correct / n_present) if n_present else 0.0

    # --- wrong_lock_episodes ---------------------------------------------
    wrong_lock_episodes = 0
    i = 0
    while i < n:
        if wrong_flags[i]:
            j = i
            while j + 1 < n and wrong_flags[j + 1]:
                j += 1
            span_s = (aligned[j][0].t_ns - aligned[i][0].t_ns) / 1e9
            if span_s > cfg.sustained_s:
                wrong_lock_episodes += 1
            i = j + 1
        else:
            i += 1

    # --- reacquire_latency_s ---------------------------------------------
    # Exactly one sample per loss episode: the time from loss onset to the next
    # correct lock. Loss onset is the first present frame after the lock breaks
    # (operator present but wrong/lost), OR — when the operator went absent — the
    # frame they REappear on (we do not penalise the tracker for the time the
    # operator is genuinely out of frame). The initial acquisition is one
    # episode anchored at the first present frame. An episode that never recovers
    # before the data ends is recorded as a censored sample (onset → last frame).
    # Emitting one sample per loss prevents spurious zero-latency samples from
    # deflating the median against the gate.
    reacquire_samples: List[float] = []
    acquired = False
    seek_start_t: Optional[int] = None
    for idx in range(n):
        g = aligned[idx][0]
        correct = correct_flags[idx]
        if not acquired:
            if seek_start_t is None:
                if g.present:
                    seek_start_t = g.t_ns  # anchor the seek at first present frame
                else:
                    continue  # operator not yet (re)present; nothing to time
            if correct:
                reacquire_samples.append((g.t_ns - seek_start_t) / 1e9)
                acquired = True
                seek_start_t = None
        else:
            if not correct:
                # lock broke; start a new seek. If present (wrong/lost), anchor
                # here; if absent, anchor later at the reappearance frame.
                acquired = False
                seek_start_t = g.t_ns if g.present else None
    if (not acquired) and seek_start_t is not None and n:
        reacquire_samples.append((aligned[n - 1][0].t_ns - seek_start_t) / 1e9)

    reacquire_latency = _median_max(reacquire_samples)

    # --- pos_error_lateral_m / pos_error_range_m -------------------------
    lat_errs: List[float] = []
    range_errs: List[float] = []
    for k, pair in enumerate(aligned):
        if correct_flags[k]:
            g, p = pair
            lat, rng = lateral_range(p.point_xyz, g.centroid_3d)
            lat_errs.append(lat)
            range_errs.append(rng)
    pos_error_lateral = _median_p95(lat_errs)
    pos_error_range = _median_p95(range_errs)

    # --- false_target_rate -----------------------------------------------
    n_false = sum(
        1
        for g, p in aligned
        if (not g.present) and p is not None and not p.target_lost
    )
    false_target_rate = (n_false / n_absent) if n_absent else 0.0

    return {
        "correct_lock_rate": correct_lock_rate,
        "wrong_lock_episodes": wrong_lock_episodes,
        "reacquire_latency_s": reacquire_latency,
        "pos_error_lateral_m": pos_error_lateral,
        "pos_error_range_m": pos_error_range,
        "false_target_rate": false_target_rate,
        "throughput_hz": throughput_hz,
    }
