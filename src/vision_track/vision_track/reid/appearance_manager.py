import logging
import time
import cv2
import numpy as np
from typing import Optional

from ..core.tracking_types import TargetAppearance, TrackingResult
from .quality import crop_quality_ok, DEFAULT_GATE

logger = logging.getLogger(__name__)


def update_appearance(
    tracker,
    frame: np.ndarray,
    result: TrackingResult,
    similarity: Optional[float] = None,
) -> None:
    """
    Update the tracked target's appearance model.

    Args:
        tracker: YOLOTracker instance
        frame: Current frame
        result: Detection result to use for appearance update
        similarity: Optional similarity score of the observation
    """
    if tracker.appearance_extractor is None:
        return

    features = tracker.appearance_extractor.extract_features(
        frame, result.bbox, result.mask, class_id=result.class_id
    )
    if not features:
        return

    current_time = time.time()
    refresh_allowed = (
        similarity is not None
        and similarity > 0.82
        and (
            tracker.target_appearance is None
            or current_time - tracker.target_appearance.last_refresh_time > tracker.feature_refresh_interval
        )
    )

    # --- gallery hygiene: skip poisoning inserts ---------------------------------
    x1, y1, x2, y2 = result.bbox
    crop_h, crop_w = max(0, y2 - y1), max(0, x2 - x1)
    aspect_ratio = crop_w / max(crop_h, 1e-6)
    mask_coverage = None
    if "mask_coverage" in features and features["mask_coverage"].size:
        mask_coverage = float(features["mask_coverage"][0])
    blur_var = 0.0
    h, w = frame.shape[:2]
    cx1, cy1, cx2, cy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if cx2 > cx1 and cy2 > cy1:
        gray = cv2.cvtColor(frame[cy1:cy2, cx1:cx2], cv2.COLOR_RGB2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if tracker.target_appearance is not None and not crop_quality_ok(
        crop_h=crop_h, crop_w=crop_w, mask_coverage=mask_coverage,
        blur_var=blur_var, aspect_ratio=aspect_ratio, **DEFAULT_GATE,
    ):
        logger.debug(
            f"Gallery insert skipped (low quality): h={crop_h} cov={mask_coverage} "
            f"blur={blur_var:.0f} ar={aspect_ratio:.2f}"
        )
        # still refresh motion so velocity/last_seen stay current
        _update_motion(
            appearance=tracker.target_appearance,
            result=result,
            current_time=current_time,
        )
        if tracker.original_track_id is not None and result.class_id == 0:
            tracker.person_registry.update_person(
                tracker.original_track_id, tracker.target_appearance
            )
        return

    if tracker.target_appearance is None:
        tracker.target_appearance = TargetAppearance(class_id=result.class_id, class_name=result.class_name)

    _update_feature_history(tracker, features, similarity, current_time, refresh_allowed)
    _update_color_histories(tracker, features, similarity, refresh_allowed)
    _update_motion(appearance=tracker.target_appearance, result=result, current_time=current_time)

    if tracker.original_track_id is not None and result.class_id == 0:
        tracker.person_registry.update_person(tracker.original_track_id, tracker.target_appearance)


def _update_feature_history(tracker, features, similarity, current_time, refresh_allowed):
    """Update feature embeddings and anchors."""
    feature_key = "reid" if "reid" in features else "cnn" if "cnn" in features else None
    if not feature_key:
        return

    new_feature = features[feature_key]
    if tracker.target_appearance.feature_history:
        last_dim = tracker.target_appearance.feature_history[-1].shape[0]
        new_dim = new_feature.shape[0]
        if last_dim != new_dim:
            logger.debug(f"Feature dimension changed from {last_dim} to {new_dim}, clearing history")
            tracker.target_appearance.feature_history.clear()

    tracker.target_appearance.feature_history.append(new_feature)
    tracker.target_appearance.gallery.maybe_add(new_feature)
    if tracker.target_appearance.anchor_feature is None:
        tracker.target_appearance.anchor_feature = new_feature
        tracker.target_appearance.best_similarity = similarity if similarity is not None else 0.0
        tracker.target_appearance.last_refresh_time = current_time
    elif refresh_allowed and similarity is not None and similarity >= tracker.target_appearance.best_similarity:
        tracker.target_appearance.anchor_feature = new_feature
        tracker.target_appearance.best_similarity = similarity
        tracker.target_appearance.last_refresh_time = current_time


def _update_color_histories(tracker, features, similarity, refresh_allowed):
    """Update color-based appearance cues."""
    appearance = tracker.target_appearance

    if "color_hist" in features:
        appearance.color_hist_history.append(features["color_hist"])
        if appearance.anchor_color_hist is None:
            appearance.anchor_color_hist = features["color_hist"]
        elif refresh_allowed and similarity is not None and similarity >= appearance.best_similarity:
            appearance.anchor_color_hist = features["color_hist"]

    if "body_color" in features:
        appearance.body_color_history.append(features["body_color"])
        if appearance.anchor_body_color is None:
            appearance.anchor_body_color = features["body_color"]
        elif refresh_allowed and similarity is not None and similarity >= appearance.best_similarity:
            appearance.anchor_body_color = features["body_color"]

    if "upper_color" in features:
        appearance.upper_color_history.append(features["upper_color"])
        if appearance.anchor_upper_color is None:
            appearance.anchor_upper_color = features["upper_color"]
        elif refresh_allowed and similarity is not None and similarity >= appearance.best_similarity:
            appearance.anchor_upper_color = features["upper_color"]

    if "lower_color" in features:
        appearance.lower_color_history.append(features["lower_color"])
        if appearance.anchor_lower_color is None:
            appearance.anchor_lower_color = features["lower_color"]
        elif refresh_allowed and similarity is not None and similarity >= appearance.best_similarity:
            appearance.anchor_lower_color = features["lower_color"]

    if "size" in features:
        appearance.size_history.append(tuple(features["size"]))


def _update_motion(appearance: TargetAppearance, result: TrackingResult, current_time: float) -> None:
    """Update position and velocity history for motion cues."""
    center_x = (result.bbox[0] + result.bbox[2]) / 2
    center_y = (result.bbox[1] + result.bbox[3]) / 2

    if appearance.position_history:
        last_pos = appearance.position_history[-1]
        dt = current_time - appearance.last_seen_time
        if dt > 0:
            vx = (center_x - last_pos[0]) / dt
            vy = (center_y - last_pos[1]) / dt
            alpha = 0.3
            old_vx, old_vy = appearance.velocity
            appearance.velocity = (alpha * vx + (1 - alpha) * old_vx, alpha * vy + (1 - alpha) * old_vy)

    appearance.position_history.append((center_x, center_y))
    appearance.last_seen_time = current_time
