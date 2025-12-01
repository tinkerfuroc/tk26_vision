import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
class TargetAppearance:
    """
    Stores appearance features for re-identification.

    Maintains a history of feature embeddings and visual characteristics
    to enable robust re-identification after occlusion or off-screen events.
    """

    feature_history: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
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
