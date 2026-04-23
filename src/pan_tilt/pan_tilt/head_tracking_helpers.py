"""Small stateful helpers for follow_head: person identity + target smoothing.

Kept local to pan_tilt so gaze tracking does not depend on the heavier
`vision_track` ReID stack — the sub-second, small-displacement regime these
helpers target is orthogonal to re-identification across long gaps.
"""

from typing import Iterable, Optional, Tuple

import numpy as np


Xyz = Tuple[float, float, float]


class PersonTracker:
    """Sticky nearest-neighbor lock on a single target in world coordinates.

    Maintains `locked_xyz` between frames. New candidates are accepted only
    if one lies within `reassoc_dist_m` of the previous lock; otherwise the
    lock is held (marked stale) until `ttl_sec` elapses, at which point the
    tracker resets and picks the closest candidate to the robot origin.
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
    ) -> Optional[Xyz]:
        cand = [np.asarray(c, dtype=np.float64) for c in candidates_xyz]
        if not cand:
            if (
                self.last_seen_mono is not None
                and (now_mono - self.last_seen_mono) > self.ttl_sec
            ):
                self.reset()
            return None

        # Expire lock if we haven't seen it in too long.
        if (
            self.locked_xyz is not None
            and self.last_seen_mono is not None
            and (now_mono - self.last_seen_mono) > self.ttl_sec
        ):
            self.reset()

        if self.locked_xyz is None:
            # Fresh lock: choose the candidate nearest to the robot origin
            # in the XY plane. (Height is irrelevant for person-selection.)
            dists = [float(np.hypot(c[0], c[1])) for c in cand]
            idx = int(np.argmin(dists))
        else:
            dists = [float(np.linalg.norm(c - self.locked_xyz)) for c in cand]
            idx = int(np.argmin(dists))
            if dists[idx] > self.reassoc_dist_m:
                # No candidate close to the lock — hold the previous lock
                # (don't re-lock onto a different person) but don't refresh
                # last_seen so TTL can expire.
                return tuple(float(v) for v in self.locked_xyz)

        self.locked_xyz = cand[idx]
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
