"""Curated multi-view ReID gallery for precision-safe reacquisition.

Holds a bounded bank of diverse, high-quality operator feature views and scores
a candidate as the max cosine over them (or a stricter fallback mode). The
caller is responsible for passing only quality-gated features (the appearance
update path already applies ``crop_quality_ok`` before admission); the gallery
adds a novelty gate so the bank spans genuinely different views. Pure numpy — no
torch / ROS — so it lives in ``core`` and is unit-testable in isolation.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _l2norm(v: np.ndarray) -> np.ndarray:
    """Return the L2-normalized vector (unchanged if near-zero norm)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of two already-L2-normalized vectors."""
    return float(np.dot(a, b))


class ReIDGallery:
    """Bounded bank of diverse operator feature views for reacquisition."""

    def __init__(self, enabled: bool = True, size: int = 6,
                 novelty_max: float = 0.85, score_mode: str = "max") -> None:
        """Store policy; the bank starts empty (index 0 is the pinned anchor)."""
        self.enabled = bool(enabled)
        self.size = max(1, int(size))
        self.novelty_max = float(novelty_max)
        self.score_mode = score_mode if score_mode in ("max", "top2_mean") else "max"
        self._views: List[np.ndarray] = []

    def configure(self, *, enabled: bool, size: int, novelty_max: float,
                  score_mode: str) -> None:
        """Apply runtime config (from ROS params) without dropping views."""
        self.enabled = bool(enabled)
        self.size = max(1, int(size))
        self.novelty_max = float(novelty_max)
        self.score_mode = score_mode if score_mode in ("max", "top2_mean") else "max"

    def __len__(self) -> int:
        """Number of stored views."""
        return len(self._views)

    def clear(self) -> None:
        """Drop all views (e.g. on tracker reset)."""
        self._views = []

    def _matching(self, dim: int) -> List[np.ndarray]:
        """Views whose dimension matches ``dim`` (guards backbone swaps)."""
        return [v for v in self._views if v.shape[0] == dim]

    def maybe_add(self, feature: Optional[np.ndarray]) -> bool:
        """Admit an (already quality-gated) feature if novel. Return admitted."""
        if feature is None or feature.ndim != 1 or not np.all(np.isfinite(feature)):
            return False
        f = _l2norm(feature.astype(np.float32))
        if not self._views:
            self._views.append(f)  # anchor, pinned at index 0
            return True
        same = self._matching(f.shape[0])
        if same and max(_cos(f, v) for v in same) >= self.novelty_max:
            return False
        self._views.append(f)
        if len(self._views) > self.size:
            self._evict_most_redundant()
        return True

    def _evict_most_redundant(self) -> None:
        """Drop the most-redundant non-anchor view (keep diversity)."""
        if len(self._views) <= 1:
            return
        non_anchor = list(range(1, len(self._views)))

        def redundancy(idx: int) -> float:
            vi = self._views[idx]
            others = [v for j, v in enumerate(self._views)
                      if j != idx and v.shape[0] == vi.shape[0]]
            return float(np.mean([_cos(vi, o) for o in others])) if others else -1.0

        drop = max(non_anchor, key=redundancy)
        self._views.pop(drop)

    def score(self, feature: Optional[np.ndarray]) -> Optional[float]:
        """Max cosine over matching views (or top2_mean). None if unusable."""
        if not self.enabled or feature is None or feature.ndim != 1:
            return None
        f = _l2norm(feature.astype(np.float32))
        sims = sorted((_cos(f, v) for v in self._matching(f.shape[0])), reverse=True)
        if not sims:
            return None
        if self.score_mode == "top2_mean":
            return float(np.mean(sims[:2]))
        return float(sims[0])
