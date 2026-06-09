import logging
import time
import cv2
import numpy as np
from typing import Optional

from ..core.tracking_types import TargetAppearance, TrackingResult
from .quality import crop_quality_ok, DEFAULT_GATE

logger = logging.getLogger(__name__)


def _make_thumb(frame, bbox, mask=None, max_h: int = 192):
    """Person-only, aspect-preserving gallery thumbnail with a transparent
    background.

    Returns a 4-channel **BGRA** array (alpha = mask) so the publisher can
    ``cv2.imencode('.png', thumb)`` directly and the dashboard renders the
    segmented person with everything else fully transparent. The RGB channels
    are stored as **BGR** (PNG/imencode convention) — the publish path encodes
    them with no colour conversion, and the on-disk vision_log writer uses
    ``cv2.imwrite`` (also BGR), so both consumers stay consistent.

    With a ``mask`` (full-frame, indexed like ``mask[y1:y2, x1:x2]``): tight-crop
    to the mask's bbox within ``bbox`` (mirrors the deep-crop segmentation intent
    — person-centered, bystander/background dropped), set the alpha channel to
    255 where ``mask>0`` else 0, then resize preserving aspect (the whole
    4-channel image is resized with INTER_NEAREST so the alpha edge stays crisp).

    With ``mask is None``: returns an opaque BGRA thumb (alpha all 255) of the
    clamped ``bbox`` crop, so behaviour degrades gracefully when no seg model is
    present.

    ``frame`` is RGB in this pipeline; the returned RGB channels are flipped to
    BGR. Returns None when the bbox / mask region is degenerate.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2]

    alpha = None
    if mask is not None:
        mask_crop = mask[y1:y2, x1:x2]
        if mask_crop.shape[:2] == crop.shape[:2] and mask_crop.size:
            m = (mask_crop > 0)
            if m.any():
                ys, xs = np.where(m)
                ty1, ty2 = int(ys.min()), int(ys.max()) + 1
                tx1, tx2 = int(xs.min()), int(xs.max()) + 1
                if ty2 - ty1 >= 2 and tx2 - tx1 >= 2:
                    crop = crop[ty1:ty2, tx1:tx2]
                    alpha = np.where(m[ty1:ty2, tx1:tx2], 255, 0).astype(np.uint8)

    # RGB -> BGR so cv2.imencode('.png') / cv2.imwrite write the right colours.
    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
    if alpha is None:
        alpha = np.full(bgr.shape[:2], 255, dtype=np.uint8)
    bgra = np.dstack([bgr, alpha])

    if bgra.shape[0] > max_h:
        new_w = max(1, round(bgra.shape[1] * max_h / bgra.shape[0]))
        # INTER_NEAREST on the 4-ch image keeps the alpha edge a hard cut.
        bgra = cv2.resize(bgra, (new_w, max_h), interpolation=cv2.INTER_NEAREST)
    return bgra.copy()


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

    # Issue 3: gate gallery admission on MASK-FILL (mask_coverage), not bbox
    # aspect ratio. The operator stands throughout, so pose uprightness is
    # guaranteed by behaviour, not box shape — occlusion/clipping makes a clean
    # upright operator's bbox square-ish (w/h ~= 1.0), which the old 0.9 aspect
    # reject wrongly dropped. Gate on mask-fill (default 0.35, configurable at
    # launch) and relax the aspect to a wide degenerate backstop. Untouched keys
    # fall back to DEFAULT_GATE.
    gate = dict(DEFAULT_GATE)
    gate["min_mask_coverage"] = float(
        getattr(tracker, "gallery_min_mask_fill", DEFAULT_GATE["min_mask_coverage"]))
    gate["max_aspect_ratio"] = float(
        getattr(tracker, "gallery_max_aspect_ratio", DEFAULT_GATE["max_aspect_ratio"]))

    if tracker.target_appearance is not None and not crop_quality_ok(
        crop_h=crop_h, crop_w=crop_w, mask_coverage=mask_coverage,
        blur_var=blur_var, aspect_ratio=aspect_ratio, **gate,
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
        tracker._configure_gallery(tracker.target_appearance)

    thumb = None
    if getattr(tracker, "keep_gallery_thumbs", False):
        thumb = _make_thumb(frame, result.bbox, result.mask)
    _update_feature_history(tracker, features, similarity, current_time, refresh_allowed, thumb)
    _update_color_histories(tracker, features, similarity, refresh_allowed)
    _update_motion(appearance=tracker.target_appearance, result=result, current_time=current_time)

    if tracker.original_track_id is not None and result.class_id == 0:
        tracker.person_registry.update_person(tracker.original_track_id, tracker.target_appearance)


def _update_feature_history(
    tracker, features, similarity, current_time, refresh_allowed, thumb=None
):
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
    tracker.target_appearance.gallery.maybe_add(new_feature, thumb=thumb)
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
