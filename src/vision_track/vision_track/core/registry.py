import logging
from typing import Dict, Optional, Tuple, Callable, Any

import numpy as np

from .tracking_types import TargetAppearance

logger = logging.getLogger(__name__)


class PersonRegistry:
    """
    Registry of known persons with their distinctive features.

    This prevents wrong ID assignment by:
    1. Storing features of all tracked persons (not just the target)
    2. Checking that a candidate doesn't better match another known person
    3. Requiring candidates to be distinctively closer to the target than to others
    """

    # An "other person" whose similarity to the candidate reaches this level is
    # not a distinct person — it is a self-ghost of the candidate's own track
    # (e.g. the returning target was registered from its own crop a frame
    # earlier). Two genuinely different people never fuse this high (cross-person
    # OSNet cosine ~0.5), so excluding it costs nothing and prevents the
    # self-match from masquerading as a competing identity.
    SELF_MATCH_CUTOFF = 0.98

    def __init__(self):
        self.known_persons: Dict[int, TargetAppearance] = {}
        self.distinctiveness_threshold = 0.10

    def register_person(self, display_id: int, appearance: TargetAppearance):
        """Register a person with their appearance features."""
        self.known_persons[display_id] = appearance
        logger.debug(f"Registered person ID {display_id} in registry (total: {len(self.known_persons)})")

    def update_person(self, display_id: int, appearance: TargetAppearance):
        """Update a person's appearance features."""
        self.known_persons[display_id] = appearance

    def get_person(self, display_id: int) -> Optional[TargetAppearance]:
        """Get a person's appearance by display ID."""
        return self.known_persons.get(display_id)

    def remove_person(self, display_id: int):
        """Remove a person from registry."""
        if display_id in self.known_persons:
            del self.known_persons[display_id]

    def clear(self):
        """Clear all entries."""
        self.known_persons.clear()

    def clear_temporary_ids(self):
        """Clear temporary IDs (negative IDs used for non-targets)."""
        self.known_persons = {k: v for k, v in self.known_persons.items() if k >= 0}

    def check_distinctiveness(
        self,
        target_id: int,
        best_features: Dict[str, np.ndarray],
        best_similarity: float,
        similarity_func: Callable[[TargetAppearance, Dict[str, np.ndarray]], float],
    ) -> bool:
        """
        Check if candidate is sufficiently distinct from other known persons.

        Args:
            target_id: The ID of the target (original, stable ID)
            best_features: Feature set of the candidate
            best_similarity: Similarity score to target
            similarity_func: Function to compute similarity to other persons
        """
        target_appearance = self.known_persons.get(target_id)
        if not target_appearance:
            return True

        max_other_similarity = 0.0
        for pid, appearance in self.known_persons.items():
            if pid == target_id:
                continue

            try:
                similarity = similarity_func(appearance, best_features)
                if similarity >= self.SELF_MATCH_CUTOFF:
                    # Self-ghost of the candidate's own track, not a real rival.
                    logger.debug(
                        f"Ignoring near-self-match other ID {pid} "
                        f"(sim {similarity:.3f} >= {self.SELF_MATCH_CUTOFF})"
                    )
                    continue
                max_other_similarity = max(max_other_similarity, similarity)
            except Exception as exc:
                logger.debug(f"Distinctiveness check failed for ID {pid}: {exc}")
                continue

        margin = best_similarity - max_other_similarity
        if margin <= self.distinctiveness_threshold:
            logger.warning(
                f"Candidate not distinctive enough: best {best_similarity:.3f}, "
                f"max other {max_other_similarity:.3f}, margin {margin:.3f} "
                f"(threshold {self.distinctiveness_threshold})"
            )
            return False

        return True

    def find_best_match(
        self, appearance: TargetAppearance, features: Dict[str, np.ndarray], similarity_func: Callable
    ) -> Optional[Tuple[int, float]]:
        """
        Find the best matching person in the registry for a given appearance.

        Args:
            appearance: Target appearance features
            features: Features to compare
            similarity_func: Similarity computation function

        Returns:
            Tuple of (person_id, similarity) or None
        """
        best_id = None
        best_similarity = 0.0

        for pid, known_app in self.known_persons.items():
            similarity = similarity_func(known_app, features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_id = pid

        if best_id is None:
            return None

        return best_id, best_similarity
