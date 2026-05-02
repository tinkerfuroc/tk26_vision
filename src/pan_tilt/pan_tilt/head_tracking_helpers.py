"""Small stateful helpers for follow_head: person identity + target smoothing.

Kept local to pan_tilt so gaze tracking does not depend on the heavier
`vision_track` ReID stack — the sub-second, small-displacement regime these
helpers target is orthogonal to re-identification across long gaps.
"""

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


Xyz = Tuple[float, float, float]


class PersonTracker:
    """Sticky single-target lock on a 3D candidate stream.

    Lifecycle per `update`:

    * **Fresh lock** (no `locked_xyz` yet): pick the candidate with the
      smallest cost. The cost is supplied by the caller via
      `fresh_lock_costs`; if not supplied, the tracker defaults to
      ``hypot(x, y)`` — i.e. the candidate closest to the robot origin in
      pan-tilt-root XY ("closest person", the original upstream behavior).
      A `math.inf` cost excludes a candidate; if every candidate is
      excluded the tracker returns `None` instead of locking onto a
      bystander.

    * **Sticky** (lock active): pick the candidate within
      ``reassoc_dist_m`` of `locked_xyz`. If no candidate is close enough,
      hold the previous lock without refreshing `last_seen_mono` so the
      TTL still expires.

    * **TTL expiry**: if no candidate updates the lock for `ttl_sec`, the
      tracker resets and the next call starts a fresh lock again.

    The tracker is frame-agnostic: callers may pass coordinates in any
    Cartesian frame as long as it's consistent with `reassoc_dist_m`.
    follow_head uses pan-tilt-root.
    """

    def __init__(self, reassoc_dist_m: float = 0.4, ttl_sec: float = 0.8):
        self.reassoc_dist_m = float(reassoc_dist_m)
        self.ttl_sec = float(ttl_sec)
        self.locked_xyz: Optional[np.ndarray] = None
        self.last_seen_mono: Optional[float] = None

    def reset(self) -> None:
        self.locked_xyz = None
        self.last_seen_mono = None

    def update(
        self,
        candidates_xyz: Iterable[Xyz],
        now_mono: float,
        fresh_lock_costs: Optional[Sequence[float]] = None,
    ) -> Optional[Xyz]:
        cand = [np.asarray(c, dtype=np.float64) for c in candidates_xyz]
        if not cand:
            if (
                self.last_seen_mono is not None
                and (now_mono - self.last_seen_mono) > self.ttl_sec
            ):
                self.reset()
            return None

        if fresh_lock_costs is not None and len(fresh_lock_costs) != len(cand):
            raise ValueError(
                "PersonTracker.update: fresh_lock_costs length "
                f"({len(fresh_lock_costs)}) does not match candidates "
                f"length ({len(cand)})."
            )

        # Expire lock if we haven't seen it in too long.
        if (
            self.locked_xyz is not None
            and self.last_seen_mono is not None
            and (now_mono - self.last_seen_mono) > self.ttl_sec
        ):
            self.reset()

        if self.locked_xyz is None:
            if fresh_lock_costs is None:
                # Default fresh-lock: closest to robot origin in XY.
                costs = [float(np.hypot(c[0], c[1])) for c in cand]
            else:
                costs = [float(c) for c in fresh_lock_costs]
            best_idx = int(np.argmin(costs))
            if not math.isfinite(costs[best_idx]):
                # All candidates are excluded by the caller's cost (e.g.
                # seed set but nothing inside seed_radius). Refuse to lock.
                return None
            chosen = cand[best_idx]
        else:
            dists = [float(np.linalg.norm(c - self.locked_xyz)) for c in cand]
            best_idx = int(np.argmin(dists))
            if dists[best_idx] > self.reassoc_dist_m:
                # No candidate close to the lock — hold the previous lock
                # (don't re-lock onto a different person) but don't refresh
                # last_seen so TTL can expire.
                return tuple(float(v) for v in self.locked_xyz)
            chosen = cand[best_idx]

        self.locked_xyz = chosen
        self.last_seen_mono = now_mono
        return tuple(float(v) for v in self.locked_xyz)


class WorldTargetEMA:
    """Exponential moving average on a 3D point, with a time-to-live.

    If no update arrives within `ttl_sec`, `get()` returns None so callers do
    not act on stale targets.
    """

    def __init__(self, alpha: float = 0.4, ttl_sec: float = 0.8):
        self.alpha = float(alpha)
        self.ttl_sec = float(ttl_sec)
        self._state: Optional[np.ndarray] = None
        self._last_update_mono: Optional[float] = None

    def reset(self) -> None:
        self._state = None
        self._last_update_mono = None

    def update(self, new_xyz: Xyz, now_mono: float) -> Xyz:
        new_arr = np.asarray(new_xyz, dtype=np.float64)
        if self._state is None:
            self._state = new_arr
        else:
            self._state = self.alpha * new_arr + (1.0 - self.alpha) * self._state
        self._last_update_mono = now_mono
        return tuple(float(v) for v in self._state)

    def get(self, now_mono: float) -> Optional[Xyz]:
        if self._state is None or self._last_update_mono is None:
            return None
        if (now_mono - self._last_update_mono) > self.ttl_sec:
            return None
        return tuple(float(v) for v in self._state)
