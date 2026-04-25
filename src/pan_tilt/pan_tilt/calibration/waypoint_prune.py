"""Waypoint pruning by end-point pose similarity.

Greedy farthest-point sampling (FPS) over a two-axis SE(3) metric, with an
optional rotation-diversity rescue post-pass that protects the Park-Martin
hand-eye solver's conditioning floor (`optimize._park_martin_solve` rejects
sample-pairs at `||log(A_rot)|| < 0.08 rad` and needs >= 3 surviving pairs).

Two waypoints `i`, `j` are *redundant* iff:

    trans(T_i, T_j) < trans_tol_m   AND   rot(T_i, T_j) < rot_tol_rad

Two-axis with AND matches the existing yaml idiom (`duplicate_ee_match_*`)
and exposes two physical knobs the operator can tune independently.

The pruner is pure-Python with no ROS dependency, so the same logic can be
called from the calib_web HTTP endpoints, from a CLI shim, and from tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .utils import pose_error_scalars


# ---- public API types -------------------------------------------------------

PredictPoseFn = Callable[[int, dict], "Predicted"]
"""Per-item pose predictor.

Called once with `(index, payload)`; the payload is the original item dict
(see ``prune_waypoints`` for the expected shape). Returns a ``Predicted``
record carrying either the 4x4 pose plus a source label, or `None` plus a
human-readable failure reason.
"""


@dataclass
class Predicted:
    pose: Optional[np.ndarray]   # 4x4 SE(3) or None on failure
    source: str                  # predictor name on success, failure reason on miss

    @property
    def ok(self) -> bool:
        return self.pose is not None


@dataclass
class PruneItem:
    index: int
    label: str
    pose: Optional[np.ndarray]            # 4x4 or None if predictor failed
    predictor_source: str
    kept: bool = False
    forced_keep: bool = False
    forced_drop: bool = False
    nearest_kept_index: Optional[int] = None
    nearest_kept_label: Optional[str] = None
    nearest_trans_m: Optional[float] = None
    nearest_rot_rad: Optional[float] = None
    drop_reason: Optional[str] = None     # only meaningful when kept=False

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "label": self.label,
            "kept": self.kept,
            "forced_keep": self.forced_keep,
            "forced_drop": self.forced_drop,
            "predictor_source": self.predictor_source,
            "predicted_pose_xyzquat": (
                _matrix_to_xyzquat(self.pose) if self.pose is not None else None
            ),
            "nearest_kept_index": self.nearest_kept_index,
            "nearest_kept_label": self.nearest_kept_label,
            "nearest_trans_m": self.nearest_trans_m,
            "nearest_rot_deg": (
                math.degrees(self.nearest_rot_rad)
                if self.nearest_rot_rad is not None else None
            ),
            "drop_reason": self.drop_reason,
        }


@dataclass
class PruneResult:
    kept_indices: list[int]
    dropped_indices: list[int]
    items: list[PruneItem]
    factors: dict
    diagnostics: dict = field(default_factory=dict)

    @property
    def headline(self) -> str:
        n_total = len(self.items)
        n_kept = len(self.kept_indices)
        return f"Will keep {n_kept} of {n_total} waypoints (drop {n_total - n_kept})"

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "kept_indices": list(self.kept_indices),
            "dropped_indices": list(self.dropped_indices),
            "items": [it.to_dict() for it in self.items],
            "factors": dict(self.factors),
            "diagnostics": dict(self.diagnostics),
        }


# ---- core ------------------------------------------------------------------

def prune_waypoints(
    items: list[dict],
    predict_pose_fn: PredictPoseFn,
    *,
    trans_tol_m: float,
    rot_tol_deg: float,
    min_count: int,
    min_rot_diversity_pairs: int = 0,
    min_rot_diversity_rad: float = 0.5,
    seed_index: int = 0,
    overrides: Optional[dict[int, str]] = None,
) -> PruneResult:
    """Greedy FPS prune.

    Parameters
    ----------
    items
        Original payload list. Each entry should carry at least a ``label``
        key (any other keys are forwarded to ``predict_pose_fn``).
    predict_pose_fn
        Callable ``(index, payload) -> Predicted``. See ``Predicted``.
    trans_tol_m, rot_tol_deg
        AND-thresholded redundancy gate.
    min_count
        Hard floor on the kept-set size (FPS will not exit below this).
    min_rot_diversity_pairs
        If > 0, after FPS the routine adds dropped items back until the kept
        set has at least this many pairs whose relative rotation magnitude is
        >= ``min_rot_diversity_rad``. Use 0 to disable.
    seed_index
        Anchor index for FPS; defaults to the first item.
    overrides
        Optional ``{index: "keep" | "drop"}`` per-row overrides.
    """
    if not items:
        return PruneResult(
            kept_indices=[], dropped_indices=[], items=[],
            factors=_factors_dict(trans_tol_m, rot_tol_deg, min_count,
                                  min_rot_diversity_pairs, min_rot_diversity_rad,
                                  seed_index),
        )

    overrides = overrides or {}
    rot_tol_rad = math.radians(rot_tol_deg)

    # 1. Run predictor for every item.
    prune_items: list[PruneItem] = []
    for i, payload in enumerate(items):
        try:
            pred = predict_pose_fn(i, payload)
        except Exception as exc:  # pragma: no cover - defensive
            pred = Predicted(pose=None, source=f"predictor raised: {exc!r}")
        if not isinstance(pred, Predicted):
            pred = Predicted(pose=None, source="predictor returned non-Predicted")
        label = str(payload.get("label", f"item/{i}"))
        forced_keep = overrides.get(i) == "keep"
        forced_drop = overrides.get(i) == "drop"
        prune_items.append(PruneItem(
            index=i,
            label=label,
            pose=pred.pose,
            predictor_source=pred.source,
            forced_keep=forced_keep,
            forced_drop=forced_drop,
        ))

    # 2. Build the candidate set: items with a valid pose AND not force-dropped.
    candidate_indices = [
        it.index for it in prune_items
        if it.pose is not None and not it.forced_drop
    ]

    # Force-keep entries with no pose are honoured (kept regardless of metric).
    kept_set: list[int] = []
    for it in prune_items:
        if it.forced_keep:
            kept_set.append(it.index)
    # Seed: first force-keep, else seed_index if it has a pose, else first
    # candidate. This keeps FPS deterministic across calls with the same
    # inputs.
    if not kept_set:
        if seed_index in candidate_indices:
            kept_set.append(seed_index)
        elif candidate_indices:
            kept_set.append(candidate_indices[0])

    remaining = [i for i in candidate_indices if i not in kept_set]

    # 3. FPS loop.
    early_exit_at: Optional[int] = None
    while remaining:
        best_c = None
        best_min = (-1.0, -1.0)        # (trans, rot) — both want to be large
        best_neighbour = None
        for c in remaining:
            min_pair = _min_distance_to_kept(
                prune_items[c].pose,
                [(k, prune_items[k].pose) for k in kept_set if prune_items[k].pose is not None],
            )
            # min_pair = (trans, rot, neighbour_index)
            if min_pair is None:
                continue
            d_t, d_r, neighbour = min_pair
            if (d_t, d_r) > best_min:
                best_min = (d_t, d_r)
                best_c = c
                best_neighbour = neighbour
        if best_c is None:
            break

        d_t, d_r = best_min
        below = (d_t < trans_tol_m) and (d_r < rot_tol_rad)
        if below and len(kept_set) >= min_count:
            early_exit_at = best_c
            break

        kept_set.append(best_c)
        remaining.remove(best_c)

    # 4. Rotation-diversity rescue (Phase-1 Park-Martin guard).
    rescued: list[int] = []
    if min_rot_diversity_pairs > 0:
        dropped_pool = [i for i in candidate_indices if i not in kept_set]
        while True:
            n_diverse = _count_rot_diverse_pairs(
                [prune_items[k].pose for k in kept_set if prune_items[k].pose is not None],
                min_rot_diversity_rad,
            )
            if n_diverse >= min_rot_diversity_pairs:
                break
            # Pick the dropped item whose addition adds the most rotation-
            # diverse pairs against the current kept set.
            best_gain = -1
            best_add = None
            for c in dropped_pool:
                cand_pose = prune_items[c].pose
                if cand_pose is None:
                    continue
                gain = sum(
                    1 for k in kept_set
                    if prune_items[k].pose is not None
                    and _rot_dist(prune_items[k].pose, cand_pose) >= min_rot_diversity_rad
                )
                if gain > best_gain:
                    best_gain = gain
                    best_add = c
            if best_add is None or best_gain <= 0:
                break
            kept_set.append(best_add)
            dropped_pool.remove(best_add)
            rescued.append(best_add)

    kept_set_sorted = sorted(set(kept_set))
    kept_lookup = {i: prune_items[i].label for i in kept_set_sorted}

    # 5. Annotate every item with kept/dropped status + nearest kept neighbour.
    for it in prune_items:
        if it.index in kept_set_sorted:
            it.kept = True
            # Even kept rows expose their nearest peer (informational).
            min_pair = _min_distance_to_kept(
                it.pose,
                [(k, prune_items[k].pose) for k in kept_set_sorted
                 if k != it.index and prune_items[k].pose is not None],
            )
            if min_pair is not None:
                it.nearest_trans_m, it.nearest_rot_rad, it.nearest_kept_index = min_pair
                it.nearest_kept_label = kept_lookup.get(it.nearest_kept_index)
            continue
        # Dropped.
        it.kept = False
        if it.forced_drop:
            it.drop_reason = "forced_drop"
        elif it.pose is None:
            it.drop_reason = "no_pose_prediction"
        elif it.index == early_exit_at:
            it.drop_reason = "below_tol_after_floor_met"
        else:
            it.drop_reason = "below_tol"
        min_pair = _min_distance_to_kept(
            it.pose,
            [(k, prune_items[k].pose) for k in kept_set_sorted
             if prune_items[k].pose is not None],
        )
        if min_pair is not None:
            it.nearest_trans_m, it.nearest_rot_rad, it.nearest_kept_index = min_pair
            it.nearest_kept_label = kept_lookup.get(it.nearest_kept_index)

    dropped = [it.index for it in prune_items if not it.kept]
    diagnostics = {
        "predictor_sources": _count_sources(prune_items),
        "n_predict_failed": sum(1 for it in prune_items if it.pose is None),
        "n_forced_keep": sum(1 for it in prune_items if it.forced_keep),
        "n_forced_drop": sum(1 for it in prune_items if it.forced_drop),
        "n_rescued_for_rot_diversity": len(rescued),
        "rot_diverse_pairs_in_kept": _count_rot_diverse_pairs(
            [prune_items[k].pose for k in kept_set_sorted if prune_items[k].pose is not None],
            min_rot_diversity_rad,
        ),
    }

    return PruneResult(
        kept_indices=kept_set_sorted,
        dropped_indices=dropped,
        items=prune_items,
        factors=_factors_dict(trans_tol_m, rot_tol_deg, min_count,
                              min_rot_diversity_pairs, min_rot_diversity_rad,
                              seed_index),
        diagnostics=diagnostics,
    )


# ---- helpers ---------------------------------------------------------------

def _min_distance_to_kept(
    pose: Optional[np.ndarray],
    kept_with_idx: list[tuple[int, np.ndarray]],
) -> Optional[tuple[float, float, int]]:
    """Return (trans_m, rot_rad, neighbour_index) for the nearest kept pose
    by FPS metric, picking the *smallest* (trans, rot) tuple lexicographically.
    None if pose is None or kept set is empty.
    """
    if pose is None or not kept_with_idx:
        return None
    best = None
    for idx, k_pose in kept_with_idx:
        d_t, d_r = pose_error_scalars(pose, k_pose)
        if best is None or (d_t, d_r) < (best[0], best[1]):
            best = (d_t, d_r, idx)
    return best


def _rot_dist(T1: np.ndarray, T2: np.ndarray) -> float:
    """Geodesic rotation distance in radians."""
    R_err = T1[:3, :3].T @ T2[:3, :3]
    cos = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.arccos(cos))


def _count_rot_diverse_pairs(poses: list[np.ndarray], threshold_rad: float) -> int:
    n = 0
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            if _rot_dist(poses[i], poses[j]) >= threshold_rad:
                n += 1
    return n


def _count_sources(prune_items: list[PruneItem]) -> dict:
    out: dict[str, int] = {}
    for it in prune_items:
        key = it.predictor_source if it.pose is not None else f"FAILED:{it.predictor_source}"
        out[key] = out.get(key, 0) + 1
    return out


def _matrix_to_xyzquat(T: np.ndarray) -> dict:
    quat = Rotation.from_matrix(T[:3, :3]).as_quat().tolist()
    return {
        "translation": T[:3, 3].tolist(),
        "rotation": quat,
    }


def _factors_dict(trans_tol_m, rot_tol_deg, min_count,
                  min_rot_diversity_pairs, min_rot_diversity_rad,
                  seed_index) -> dict:
    return {
        "trans_tol_m": float(trans_tol_m),
        "rot_tol_deg": float(rot_tol_deg),
        "min_count": int(min_count),
        "min_rot_diversity_pairs": int(min_rot_diversity_pairs),
        "min_rot_diversity_rad": float(min_rot_diversity_rad),
        "min_rot_diversity_deg": float(math.degrees(min_rot_diversity_rad)),
        "seed_index": int(seed_index),
    }
