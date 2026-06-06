import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .reid import ReIDMatcher
from .identity_gate import (
    deep_ratio_ambiguous,
    spatial_switch_allowed,
    DEFAULT_RATIO_MAX,
    DEFAULT_DEEP_SWITCH_MARGIN,
)
from ..core.tracking_types import TargetAppearance, TrackingResult

logger = logging.getLogger(__name__)


def find_best_match_reid(
    tracker,
    frame: np.ndarray,
    results: List[TrackingResult],
) -> Optional[Tuple[TrackingResult, float]]:
    """
    Find the detection that best matches the stored appearance using re-identification.

    Returns:
        Best matching (TrackingResult, similarity) or None
    """
    if tracker.target_appearance is None or tracker.appearance_extractor is None:
        return None

    current_time = time.time()
    candidates = _filter_candidates(tracker, results)
    if not candidates:
        logger.info(
            "ReID: No valid person candidates (track_id >= 0) for re-identification. "
            f"Total results: {len(results)}, persons with invalid IDs: "
            f"{sum(1 for r in results if r.class_id == 0 and r.track_id < 0)}"
        )
        return None

    is_person = tracker.target_appearance.class_id == 0
    candidate_scores = _score_candidates(tracker, frame, candidates, current_time, is_person)
    if not candidate_scores:
        logger.debug("No candidates with valid features")
        return None

    candidate_scores.sort(key=lambda x: x[1], reverse=True)
    # Phase 2: surface the best-vs-second similarity margin for the lock FSM's
    # distinctiveness gate (inf when a single candidate). The pipeline reads
    # tracker.last_reid_margin after this call.
    if len(candidate_scores) > 1:
        tracker.last_reid_margin = float(candidate_scores[0][1] - candidate_scores[1][1])
    else:
        tracker.last_reid_margin = float("inf")
    logger.info(
        f"ReID candidates (threshold={tracker.reid_threshold}): "
        f"{[(r.track_id, f'{s:.3f}') for r, s, _, _ in candidate_scores]}"
    )

    best_match, best_similarity, best_features, best_deep = candidate_scores[0]
    if best_similarity <= tracker.reid_threshold:
        logger.info(
            f"ReID FAILED: Best similarity {best_similarity:.3f} <= threshold {tracker.reid_threshold} "
            f"(best candidate: ID {best_match.track_id})"
        )
        return None

    _update_candidate_consistency(tracker, candidate_scores[:2])
    best_match, best_similarity, best_features = _resolve_ambiguity(
        tracker,
        candidate_scores,
        current_time,
        is_person,
        best_match,
        best_similarity,
        best_features,
        results,
    )
    if best_match is None:
        return None

    if not _passes_distinctiveness(tracker, is_person, candidate_scores, best_match, best_features, best_similarity):
        return None

    if not _single_candidate_guard(is_person, candidate_scores, best_similarity):
        return None

    if tracker.target_appearance.class_id == 0 and best_match.class_id != 0:
        logger.warning(
            f"Rejecting match: expected person but got class {best_match.class_name} (id={best_match.class_id})"
        )
        return None

    logger.info(
        f"Re-identified target (class='{best_match.class_name}') as ID {best_match.track_id} "
        f"with similarity {best_similarity:.3f}"
    )
    return best_match, best_similarity


def _filter_candidates(tracker, results: List[TrackingResult]) -> List[TrackingResult]:
    """Filter detections to same class with valid track ids."""
    target_class_id = tracker.target_appearance.class_id
    target_class_name = tracker.target_appearance.class_name
    if target_class_id == 0:
        candidates = [
            r for r in results if r.class_id == 0 and r.class_name.lower() == "person" and r.track_id >= 0
        ]
    else:
        candidates = [r for r in results if r.class_id == target_class_id and r.track_id >= 0]

    logger.debug(
        f"ReID: {len(candidates)} candidates of class '{target_class_name}' from {len(results)} total detections"
    )
    return candidates


def _score_candidates(
    tracker,
    frame: np.ndarray,
    candidates: List[TrackingResult],
    current_time: float,
    is_person: bool,
) -> List[Tuple[TrackingResult, float, Dict[str, np.ndarray], float]]:
    """Extract features and compute similarity scores for candidates.

    All candidate crops are embedded in a SINGLE batched deep forward pass
    (extract_features_batch) instead of one forward per candidate — this removes
    the multi-person ReID throughput cliff. The returned per-candidate tuples
    (result, similarity, features, raw_cosine) are unchanged, computed exactly as
    the old per-crop loop.
    """
    candidate_scores: List[Tuple[TrackingResult, float, Dict[str, np.ndarray], float]] = []

    # Phase 3: prime the per-frame embedding cache so the downstream verify /
    # confirm / periodic-validation call sites reuse this score-pass feature dict
    # instead of re-embedding the same crop. begin_frame is defensive — put()
    # also auto-begins.
    tracker.embedding_cache.begin_frame(tracker.frame_count)

    bboxes = [r.bbox for r in candidates]
    masks = [r.mask for r in candidates]
    class_ids = [r.class_id for r in candidates]
    feature_dicts = tracker.appearance_extractor.extract_features_batch(
        frame, bboxes, masks, class_ids
    )

    for result, features in zip(candidates, feature_dicts):
        if not features:
            logger.debug(f"ID {result.track_id}: No features extracted")
            continue

        # Cache only stable ByteTrack ids (>= 0); negative temp ids collide.
        if result.track_id >= 0:
            tracker.embedding_cache.put(result.track_id, tracker.frame_count, features)

        raw_cosine = 0.0
        if is_person and "reid" in features:
            ds = tracker.target_appearance.deep_score(features["reid"])
            if ds is not None:
                raw_cosine = ds
                logger.debug(f"ID {result.track_id}: gallery deep score={raw_cosine:.3f}")

        if is_person and "body_color" in features:
            target_body = tracker.target_appearance.get_body_color()
            if target_body is not None:
                body_sim = ReIDMatcher._histogram_similarity(target_body, features["body_color"])
                logger.debug(f"ID {result.track_id}: body color similarity={body_sim:.3f}")

        similarity = ReIDMatcher.compute_similarity(
            tracker.target_appearance,
            features,
            result.bbox,
            current_time,
            is_person=is_person,
        )
        candidate_scores.append((result, similarity, features, raw_cosine))

    return candidate_scores


def _resolve_ambiguity(
    tracker,
    candidate_scores,
    current_time: float,
    is_person: bool,
    best_match: TrackingResult,
    best_similarity: float,
    best_features: Dict[str, np.ndarray],
    results: List[TrackingResult],
):
    """Handle margin checks and tie-breaking between top candidates."""
    if len(candidate_scores) == 1:
        return best_match, best_similarity, best_features

    second_best_match, second_best_similarity, _, second_deep = candidate_scores[1]
    best_deep = candidate_scores[0][3]
    margin = best_similarity - second_best_similarity

    # Lowe-style ratio test on the DEEP term: if the runner-up is deep-
    # indistinguishable from the best, the identities are not separable.
    if deep_ratio_ambiguous(best_deep, second_deep, ratio_max=DEFAULT_RATIO_MAX):
        logger.info(
            f"ReID FAILED (deep ratio): best_deep={best_deep:.3f} second_deep={second_deep:.3f} "
            f"ratio>{DEFAULT_RATIO_MAX} — identities not separable"
        )
        return None, 0.0, {}

    if margin >= ReIDMatcher.REID_MARGIN:
        return best_match, best_similarity, best_features

    if tracker.camera_motion_detected:
        resolved = _resolve_with_camera_motion(
            tracker,
            best_match,
            best_similarity,
            best_features,
            second_best_match,
            second_best_similarity,
            results,
            best_deep,
            second_deep,
        )
        if resolved is None:
            return None, 0.0, {}
        return resolved

    if tracker.last_known_center is not None:
        resolved = _resolve_with_spatial_gate(
            tracker,
            best_match,
            best_similarity,
            best_features,
            second_best_match,
            second_best_similarity,
            best_deep,
            second_deep,
        )
        if resolved is None:
            return None, 0.0, {}
        return resolved

    logger.info(
        f"ReID FAILED: margin {margin:.3f} < {ReIDMatcher.REID_MARGIN}, "
        f"ambiguous between ID {best_match.track_id} and ID {second_best_match.track_id}"
    )
    return None, 0.0, {}


def _resolve_with_camera_motion(
    tracker,
    best_match,
    best_similarity,
    best_features,
    second_best_match,
    second_best_similarity,
    results,
    best_deep,
    second_deep,
):
    """Use velocity and relative position cues to resolve ambiguous matches during camera motion."""
    logger.info("Camera motion detected - using advanced disambiguation")
    predicted_pos = tracker._predict_target_position()
    if predicted_pos is not None:
        pred_x, pred_y = predicted_pos

        def get_distance_to_predicted(result: TrackingResult) -> float:
            x1, y1, x2, y2 = result.bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return ((cx - pred_x) ** 2 + (cy - pred_y) ** 2) ** 0.5

        dist_best = get_distance_to_predicted(best_match)
        dist_second = get_distance_to_predicted(second_best_match)
        logger.info(
            f"Velocity prediction: predicted ({pred_x:.0f}, {pred_y:.0f}), "
            f"Best ID {best_match.track_id} dist={dist_best:.0f}px, "
            f"Second ID {second_best_match.track_id} dist={dist_second:.0f}px"
        )
        prediction_threshold = 80.0
        if (
            dist_second < dist_best - prediction_threshold
            and second_best_similarity > tracker.reid_threshold
            and spatial_switch_allowed(best_deep, second_deep, margin=DEFAULT_DEEP_SWITCH_MARGIN)
        ):
            best_match, best_similarity, best_features = second_best_match, second_best_similarity, None
        elif dist_best > dist_second + prediction_threshold:
            logger.info(f"Velocity prediction confirms Best ID {best_match.track_id}")

    best_rel_consistent, best_rel_score = tracker._check_relative_position_consistency(best_match, results)
    second_rel_consistent, second_rel_score = tracker._check_relative_position_consistency(second_best_match, results)
    logger.info(
        f"Relative position consistency: Best ID {best_match.track_id}={best_rel_score:.2f}, "
        f"Second ID {second_best_match.track_id}={second_rel_score:.2f}"
    )
    if (
        second_rel_score > best_rel_score + 0.3
        and second_best_similarity > tracker.reid_threshold
        and spatial_switch_allowed(best_deep, second_deep, margin=DEFAULT_DEEP_SWITCH_MARGIN)
    ):
        best_match, best_similarity, best_features = second_best_match, second_best_similarity, None

    best_consistency = tracker._get_candidate_consistency_score(best_match.track_id)
    second_consistency = tracker._get_candidate_consistency_score(second_best_match.track_id)
    if (
        best_consistency < 0.3
        and second_consistency > 0.6
        and second_best_similarity > tracker.reid_threshold
        and spatial_switch_allowed(best_deep, second_deep, margin=DEFAULT_DEEP_SWITCH_MARGIN)
    ):
        logger.info(
            f"Consistency check: Best ID {best_match.track_id} is erratic ({best_consistency:.2f}), "
            f"Second ID {second_best_match.track_id} is stable ({second_consistency:.2f})"
        )
        best_match, best_similarity, best_features = second_best_match, second_best_similarity, None

    if best_similarity <= tracker.reid_threshold:
        logger.info("ReID FAILED during camera motion: insufficient confidence after all checks")
        return None

    return best_match, best_similarity, best_features


def _resolve_with_spatial_gate(
    tracker,
    best_match,
    best_similarity,
    best_features,
    second_best_match,
    second_best_similarity,
    best_deep,
    second_deep,
):
    """Use spatial continuity when the camera is stable."""
    def get_distance_to_last(result: TrackingResult) -> float:
        x1, y1, x2, y2 = result.bbox
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        lx, ly = tracker.last_known_center
        return ((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5

    dist_best = get_distance_to_last(best_match)
    dist_second = get_distance_to_last(second_best_match)
    logger.info(
        f"Using spatial continuity: Best ID {best_match.track_id} dist={dist_best:.1f}px, "
        f"Second ID {second_best_match.track_id} dist={dist_second:.1f}px"
    )
    spatial_threshold = 100.0
    if (
        dist_second < dist_best - spatial_threshold
        and second_best_similarity > tracker.reid_threshold
        and spatial_switch_allowed(best_deep, second_deep, margin=DEFAULT_DEEP_SWITCH_MARGIN)
    ):
        logger.info(
            f"Spatial tiebreaker: preferring closer ID {second_best_match.track_id} "
            f"(dist={dist_second:.1f}px vs {dist_best:.1f}px, deep-gated)"
        )
        return second_best_match, second_best_similarity, None
    if dist_best < dist_second - spatial_threshold:
        logger.info(
            f"Spatial confirmation: ID {best_match.track_id} is closer "
            f"(dist={dist_best:.1f}px vs {dist_second:.1f}px), accepting despite small margin"
        )
        return best_match, best_similarity, best_features

    logger.info(
        f"ReID FAILED: margin {best_similarity - second_best_similarity:.3f} < {ReIDMatcher.REID_MARGIN}, "
        f"and spatial positions similar (ambiguous between ID {best_match.track_id} and ID {second_best_match.track_id})"
    )
    return None


def _passes_distinctiveness(
    tracker,
    is_person: bool,
    candidate_scores,
    best_match: TrackingResult,
    best_features: Optional[Dict[str, np.ndarray]],
    best_similarity: float,
) -> bool:
    """Check distinctiveness against other known persons in the registry."""
    if not (is_person and tracker.original_track_id is not None and len(candidate_scores) > 1):
        return True

    if not best_features:
        return True

    def similarity_func(appearance: TargetAppearance, features: Dict[str, np.ndarray]) -> float:
        return ReIDMatcher.compute_similarity(appearance, features, best_match.bbox, time.time(), is_person=True)

    if not tracker.person_registry.check_distinctiveness(
        tracker.original_track_id, best_features, best_similarity, similarity_func
    ):
        logger.info(
            f"ReID FAILED: candidate ID {best_match.track_id} not distinctive enough from other known persons"
        )
        return False
    return True


def _single_candidate_guard(is_person: bool, candidate_scores, best_similarity: float) -> bool:
    """Handle single-person scenes with stricter threshold."""
    if not (is_person and len(candidate_scores) == 1):
        return True

    single_person_threshold = 0.72
    if best_similarity < single_person_threshold:
        logger.info(
            f"ReID FAILED: only one person visible but similarity {best_similarity:.3f} < {single_person_threshold} "
            "(requires high confidence when no comparison is possible)"
        )
        return False

    logger.info(
        f"Single person mode: similarity {best_similarity:.3f} >= {single_person_threshold}, accepting"
    )
    return True


def _update_candidate_consistency(tracker, candidates):
    """Update consistency tracking for the top candidates."""
    for match, similarity, _, _ in candidates:
        tracker._update_candidate_consistency(match.track_id, similarity)
