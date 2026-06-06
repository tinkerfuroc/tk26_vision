import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .reid_gallery import ReIDGallery

logger = logging.getLogger(__name__)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (0.0 on near-zero norm)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class TrackerState(Enum):
    """Enumeration for tracker states."""

    UNINITIALIZED = "uninitialized"
    INITIALIZED = "initialized"
    TRACKING = "tracking"
    LOST = "lost"
    REIDENTIFYING = "reidentifying"


@dataclass
class TrackingResult:
    """Data class to hold tracking results."""

    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray]  # Segmentation mask
    confidence: float
    class_id: int
    class_name: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert tracking result to dictionary."""
        return {
            "track_id": self.track_id,
            "bbox": self.bbox,
            "mask": self.mask,
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
        }


@dataclass
class LockDecision:
    """Output of LockStateMachine.step — what the node should do this frame.

    Attributes:
        publish: emit the 3D point this frame (provisional or committed).
        target_lost: feedback flag; True during any coast (asymmetric hysteresis).
        committed_id: stable original track id, or None once hard-lost.
        state: one of 'tracking' | 'reidentifying' | 'lost'.
    """

    publish: bool
    target_lost: bool
    committed_id: Optional[int]
    state: str


@dataclass
class TargetAppearance:
    """
    Stores appearance features for re-identification.

    Maintains a history of feature embeddings and visual characteristics
    to enable robust re-identification after occlusion or off-screen events.
    """

    feature_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    gallery: ReIDGallery = field(default_factory=ReIDGallery)
    color_hist_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    upper_color_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    lower_color_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    body_color_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    size_history: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=30))
    anchor_feature: Optional[np.ndarray] = None
    anchor_body_color: Optional[np.ndarray] = None
    anchor_color_hist: Optional[np.ndarray] = None
    anchor_upper_color: Optional[np.ndarray] = None
    anchor_lower_color: Optional[np.ndarray] = None
    best_similarity: float = 0.0
    last_refresh_time: float = 0.0
    position_history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    velocity: Tuple[float, float] = (0.0, 0.0)
    last_seen_time: float = 0.0
    class_id: int = -1
    class_name: str = ""

    def get_average_feature(self) -> Optional[np.ndarray]:
        """Get averaged feature embedding from history."""
        if not self.feature_history:
            return None

        try:
            dims = [f.shape[0] for f in self.feature_history]
            if not dims:
                return None

            target_dim = dims[-1]
            matching_features = [f for f in self.feature_history if f.shape[0] == target_dim]
            if not matching_features:
                return None

            return np.mean(np.array(matching_features), axis=0)
        except Exception:
            return self.feature_history[-1] if self.feature_history else None

    def configure_gallery(self, *, enabled: bool, size: int,
                          novelty_max: float, score_mode: str) -> None:
        """Apply gallery config (from ROS params) to this target's gallery."""
        self.gallery.configure(enabled=enabled, size=size,
                               novelty_max=novelty_max, score_mode=score_mode)

    def deep_score(self, candidate_reid: Optional[np.ndarray]) -> Optional[float]:
        """Deep-ReID similarity of a candidate to this target's appearance.

        Uses the multi-view gallery (max over diverse views) when enabled and
        populated, never doing worse than the pinned anchor; otherwise falls
        back to the legacy max(average, anchor) cosine. Returns a raw cosine in
        [-1, 1], or None when no usable target feature exists.
        """
        if candidate_reid is None or not np.all(np.isfinite(candidate_reid)):
            return None
        dim = candidate_reid.shape[0]
        if self.gallery.enabled and len(self.gallery) > 0:
            g = self.gallery.score(candidate_reid)
            if g is not None:
                if (self.anchor_feature is not None
                        and self.anchor_feature.shape[0] == dim):
                    return max(g, _cosine(self.anchor_feature, candidate_reid))
                return g
        best = None
        avg = self.get_average_feature()
        if avg is not None and avg.shape[0] == dim:
            best = _cosine(avg, candidate_reid)
        if self.anchor_feature is not None and self.anchor_feature.shape[0] == dim:
            a = _cosine(self.anchor_feature, candidate_reid)
            best = a if best is None else max(best, a)
        return best

    def get_average_color_hist(self) -> Optional[np.ndarray]:
        """Get averaged color histogram from history."""
        if not self.color_hist_history:
            return None
        try:
            return np.mean(np.array(list(self.color_hist_history)), axis=0)
        except Exception:
            return self.color_hist_history[-1] if self.color_hist_history else None

    def get_body_color(self) -> Optional[np.ndarray]:
        """Get averaged body part color histogram from history."""
        if not self.body_color_history:
            return None
        try:
            return np.mean(np.array(list(self.body_color_history)), axis=0)
        except Exception:
            return self.body_color_history[-1] if self.body_color_history else None

    def get_average_size(self) -> Optional[Tuple[float, float]]:
        """Get average size from history."""
        if not self.size_history:
            return None
        try:
            sizes = np.array(list(self.size_history))
            return (float(np.mean(sizes[:, 0])), float(np.mean(sizes[:, 1])))
        except Exception:
            return self.size_history[-1] if self.size_history else None

    def predict_position(self, dt: float = 1.0) -> Optional[Tuple[float, float]]:
        """Predict next position based on velocity."""
        if not self.position_history:
            return None
        last_pos = self.position_history[-1]
        return (last_pos[0] + self.velocity[0] * dt, last_pos[1] + self.velocity[1] * dt)
