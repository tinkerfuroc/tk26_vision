import copy
import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from ..reid.appearance_manager import update_appearance
from ..reid.reid import ReIDMatcher
from ..reid.reid_search import find_best_match_reid
from .tracking_types import TargetAppearance, TrackerState, TrackingResult

logger = logging.getLogger(__name__)

# A detection scoring at/above this similarity to the target could BE the target,
# so it must never be cemented as an "other person" distractor. Reuses the
# single-person reacquisition acceptance bar (reid_search._single_candidate_guard)
# — below it the candidate is confidently a different person worth registering.
OTHER_PERSON_MAX_TARGET_SIM = 0.72


def _get_or_extract_features(tracker, frame, track_id, bbox, mask, class_id):
    """Cache-aware single-detection feature extraction.

    Reuses the score-pass embedding for (track_id, frame_count) within one frame,
    eliminating the up-to-4x/frame re-embed. track_id < 0 (unstable) is never
    cached (it collides across detections); falls through to a direct extract.
    """
    cache = getattr(tracker, "embedding_cache", None)
    seq = getattr(tracker, "frame_count", None)
    if cache is not None and seq is not None and track_id is not None and track_id >= 0:
        hit = cache.get(track_id, seq)
        if hit is not None:
            return hit
        features = tracker.appearance_extractor.extract_features(frame, bbox, mask, class_id=class_id)
        if features:
            cache.put(track_id, seq, features)
        return features
    return tracker.appearance_extractor.extract_features(frame, bbox, mask, class_id=class_id)


def update_tracker(tracker, frame: np.ndarray, target_id: Optional[int] = None) -> Optional[TrackingResult]:
    """Top-level tracking orchestration."""
    if tracker.state == TrackerState.UNINITIALIZED:
        logger.warning("Tracker not initialized. Call initialize_tracking first.")
        return None

    _switch_target(tracker, target_id)
    tracker.frame_count += 1
    # Phase 3: frame_count is the single source of truth for the embedding
    # cache's frame_seq. Begin the new frame here so the cache is clean for
    # every processed frame even if no candidate is scored this update.
    if getattr(tracker, "embedding_cache", None) is not None:
        tracker.embedding_cache.begin_frame(tracker.frame_count)
    # Cleared each frame; reidentify_target sets it True when it authoritatively
    # steps the FSM, telling the node not to re-step present=True for a recovery
    # frame (which would defeat the asymmetric hysteresis on a partial confirm).
    tracker.last_frame_recovery = False

    _t_yolo = time.perf_counter()
    results = tracker.track(frame, persist=True)
    tracker._t_yolo_ms = (time.perf_counter() - _t_yolo) * 1000.0
    tracker.last_results = results or []

    if results:
        update_scene_motion(tracker, results, frame)
        _log_detections(tracker, results)

    _t_pipe = time.perf_counter()
    match = track_by_id(tracker, frame, results)
    if match is not None:
        tracker._t_pipeline_ms = (time.perf_counter() - _t_pipe) * 1000.0
        return match

    out = reidentify_target(tracker, frame, results)
    tracker._t_pipeline_ms = (time.perf_counter() - _t_pipe) * 1000.0
    return out


def _switch_target(tracker, target_id: Optional[int]) -> None:
    if target_id is None or target_id == tracker.target_track_id:
        return
    logger.info(f"Switching target from {tracker.target_track_id} to {target_id}")
    tracker.target_track_id = target_id
    tracker.original_track_id = target_id
    tracker.frames_lost = 0
    tracker.target_appearance = None


def _log_detections(tracker, results: List[TrackingResult]) -> None:
    persons = [r.track_id for r in results if r.class_id == 0]
    logger.debug(
        f"Frame detections: {len(persons)} persons with IDs {persons}. "
        f"Looking for target YOLO ID: {tracker.target_track_id}"
    )


def track_by_id(tracker, frame: np.ndarray, results: List[TrackingResult]) -> Optional[TrackingResult]:
    """Stage 1: rely on persisted ByteTrack IDs with safety checks."""
    if not results:
        return None

    current_time = time.time()
    target_ids = [tracker.target_track_id]
    if tracker.pending_reid_match is not None:
        pending_id, _ = tracker.pending_reid_match
        target_ids.append(pending_id)

    for result in results:
        if result.track_id not in target_ids:
            continue
        if tracker.target_class_id is not None and result.class_id != tracker.target_class_id:
            continue
        if not _passes_spatial_gate_check(tracker, result, current_time):
            continue

        occlusion_choice = _handle_occlusion_state(tracker, frame, result, results, current_time)
        if occlusion_choice is not None:
            return occlusion_choice

        if tracker.enable_reid and tracker.target_appearance is not None and result.class_id == 0:
            verified = _verify_person_candidate(tracker, frame, result, results)
            if verified is not None:
                return verified
            continue

        return _finalize_track_match(tracker, frame, results, result, is_person=False)

    _log_stage1_failure(tracker, results)
    return None


def _passes_spatial_gate_check(tracker, result: TrackingResult, current_time: float) -> bool:
    if tracker.frames_lost >= 20:
        return True
    use_motion_gate = (
        tracker.camera_motion_detected
        or (current_time - tracker.last_camera_motion_time) < tracker.camera_motion_recent_window
    )
    passes_gate, move_dist, gate_threshold = tracker._passes_spatial_gate(result.bbox, use_camera_gate=use_motion_gate)
    if not passes_gate:
        logger.info(
            f"Spatial gate reject: Track ID {result.track_id} moved {move_dist:.1f}px "
            f"(gate {gate_threshold:.1f}px){' during camera motion' if use_motion_gate else ''}"
        )
    return passes_gate


def _handle_occlusion_state(
    tracker,
    frame: np.ndarray,
    result: TrackingResult,
    results: List[TrackingResult],
    current_time: float,
) -> Optional[TrackingResult]:
    is_occluded, occluder = detect_occlusion(tracker, result, results)
    if is_occluded and tracker.enable_reid and result.class_id == 0:
        if not tracker.is_occluded:
            save_pre_occlusion_state(tracker)
            tracker.is_occluded = True
            tracker.occlusion_start_time = current_time
            logger.warning(f"Occlusion started! Target ID {result.track_id} being occluded by ID {occluder.track_id}")

        if tracker.appearance_extractor is not None:
            features = _get_or_extract_features(tracker, frame, result.track_id, result.bbox, result.mask, 0)
            if features:
                similarity = ReIDMatcher.compute_similarity(
                    tracker.target_appearance, features, result.bbox, current_time, is_person=True
                )
                if similarity < 0.70:
                    logger.warning(
                        f"Track ID {result.track_id} appearance degraded during occlusion "
                        f"(similarity={similarity:.3f} < 0.70). Treating as lost."
                    )
                    return None
        # Phase 2: a crosser causing the occlusion must not be adopted as the
        # target. Reject if its depth jumped toward the camera vs the operator.
        from .depth_gate import should_reject_candidate
        cand_depth = getattr(tracker, "candidate_depths_m", {}).get(result.track_id)
        if should_reject_candidate(
            cand_depth, getattr(tracker, "operator_last_depth_m", None),
            getattr(tracker, "crosser_depth_jump_m", 0.6),
        ):
            logger.warning(
                f"Depth gate reject during occlusion: ID {result.track_id} is a crosser."
            )
            return None
        tracker.state = TrackerState.TRACKING
        tracker.frames_lost = 0
        return tracker._with_original_id(result)

    if tracker.is_occluded and not is_occluded:
        tracker.frames_since_occlusion_ended += 1
        if tracker.frames_since_occlusion_ended <= tracker.occlusion_recovery_frames:
            if not verify_post_occlusion(tracker, frame, result, current_time):
                logger.warning(
                    f"Post-occlusion verification failed for Track ID {result.track_id}; will re-identify."
                )
                return None
            logger.info(
                f"Post-occlusion frame {tracker.frames_since_occlusion_ended}/{tracker.occlusion_recovery_frames}"
            )
        else:
            tracker.is_occluded = False
            tracker.occlusion_start_time = None
            tracker.pre_occlusion_appearance = None
            tracker.frames_since_occlusion_ended = 0

    return None


def _verify_person_candidate(
    tracker,
    frame: np.ndarray,
    result: TrackingResult,
    results: List[TrackingResult],
) -> Optional[TrackingResult]:
    features = _get_or_extract_features(tracker, frame, result.track_id, result.bbox, result.mask, 0)
    if not features:
        return None

    # Phase 2: depth-gated crosser rejection. A candidate whose median depth
    # jumps toward the camera beyond crosser_depth_jump_m (vs the operator's last
    # depth) is geometrically a crosser between robot and operator — a cue
    # appearance cannot spoof. Reject before trusting the appearance similarity.
    from .depth_gate import should_reject_candidate
    cand_depth = getattr(tracker, "candidate_depths_m", {}).get(result.track_id)
    op_depth = getattr(tracker, "operator_last_depth_m", None)
    jump = getattr(tracker, "crosser_depth_jump_m", 0.6)
    if should_reject_candidate(cand_depth, op_depth, jump):
        logger.warning(
            f"Depth gate reject: Track ID {result.track_id} candidate depth "
            f"{cand_depth} jumped toward camera vs operator {op_depth} "
            f"(jump>{jump} m); treating as crosser."
        )
        return None

    current_time = time.time()
    similarity = ReIDMatcher.compute_similarity(
        tracker.target_appearance, features, result.bbox, current_time, is_person=True
    )
    if similarity < 0.50:
        logger.warning(
            f"ID switch detected! Track ID {result.track_id} similarity={similarity:.3f} < 0.50; seeking ReID."
        )
        return None

    keep_current, better_match = periodic_reid_validation(tracker, frame, results, result, current_similarity=similarity)
    if not keep_current and better_match is not None:
        tracker.pending_reid_match = (better_match.track_id, current_time)
        tracker.consecutive_reid_frames = 1
        return None

    if _confirm_pending_reid(tracker, result):
        return tracker._with_original_id(result)

    tracker.state = TrackerState.TRACKING
    tracker.frames_lost = 0
    tracker.pending_reid_match = None
    tracker.consecutive_reid_frames = 0

    if not tracker.fast_tracking_mode and tracker.frame_count > 30:
        tracker.fast_tracking_mode = True

    tracker.target_class_id = result.class_id
    tracker.target_class_name = result.class_name

    if similarity >= 0.80:
        tracker._update_appearance(frame, result, similarity=similarity)

    center = ((result.bbox[0] + result.bbox[2]) / 2.0, (result.bbox[1] + result.bbox[3]) / 2.0)
    tracker._update_target_velocity(center)
    tracker._update_relative_positions(result, results)

    if tracker.is_occluded:
        tracker.is_occluded = False
        tracker.occlusion_start_time = None
        tracker.pre_occlusion_appearance = None
        tracker.frames_since_occlusion_ended = 0

    return tracker._with_original_id(result)


def _confirm_pending_reid(tracker, result: TrackingResult) -> bool:
    if tracker.pending_reid_match is None:
        return False
    pending_id, _ = tracker.pending_reid_match
    if result.track_id != pending_id or result.track_id == tracker.target_track_id:
        return False

    tracker.consecutive_reid_frames += 1
    if tracker.consecutive_reid_frames >= tracker.reid_confirmation_frames:
        old_id = tracker.target_track_id
        tracker.target_track_id = result.track_id
        tracker.pending_reid_match = None
        tracker.consecutive_reid_frames = 0
        tracker.person_registry.clear()
        if tracker.original_track_id is not None:
            tracker.person_registry.register_person(tracker.original_track_id, tracker.target_appearance)
        logger.info(f"ReID confirmed via Stage 1: YOLO ID {old_id} -> {tracker.target_track_id}")
    tracker.state = TrackerState.TRACKING
    tracker.frames_lost = 0
    return True


def _finalize_track_match(
    tracker,
    frame: np.ndarray,
    results: List[TrackingResult],
    result: TrackingResult,
    is_person: bool,
) -> TrackingResult:
    tracker.state = TrackerState.TRACKING
    tracker.frames_lost = 0
    tracker.pending_reid_match = None
    tracker.consecutive_reid_frames = 0

    tracker.target_class_id = result.class_id
    tracker.target_class_name = result.class_name

    if tracker.enable_reid and not is_person:
        tracker._update_appearance(frame, result)

    center = ((result.bbox[0] + result.bbox[2]) / 2.0, (result.bbox[1] + result.bbox[3]) / 2.0)
    tracker._update_target_velocity(center)
    tracker._update_relative_positions(result, results)
    return tracker._with_original_id(result)


def _log_stage1_failure(tracker, results: List[TrackingResult]) -> None:
    if not results:
        logger.info("Stage 1: No detections at all")
        return
    person_results = [r for r in results if r.class_id == 0]
    person_ids = [r.track_id for r in person_results]
    valid_person_ids = [r.track_id for r in person_results if r.track_id >= 0]
    if tracker.target_track_id in person_ids:
        logger.info("Stage 1: Target found but rejected (appearance mismatch or class change)")
    else:
        logger.info(
            f"Stage 1: Target ID {tracker.target_track_id} NOT in detections. "
            f"Found {len(person_results)} persons with IDs: {person_ids} (valid: {valid_person_ids})"
        )


def reidentify_target(tracker, frame: np.ndarray, results: List[TrackingResult]) -> Optional[TrackingResult]:
    """Stage 2: appearance-based re-identification."""
    if not tracker.enable_reid or tracker.frames_lost > tracker.max_frames_lost:
        tracker.frames_lost += 1
        if tracker.frames_lost > tracker.max_frames_lost:
            tracker.state = TrackerState.LOST
        return None

    tracker.state = TrackerState.REIDENTIFYING

    # Phase 2: the lock FSM is the publish/target_lost authority for the recovery
    # path. It is stepped EXACTLY ONCE per frame here with the real per-frame
    # inputs — never twice — because the asymmetric hysteresis relies on a
    # monotone provisional streak; a no-op "seed" step at sim 0.0 would reset
    # that streak every frame and the high-bar commit would never accumulate. The
    # early-return branches below therefore each step the FSM themselves (coast),
    # and the match path steps it once with the resolved similarity. The pipeline
    # remains the identity-swap authority — the FSM never touches target_track_id.
    fsm = getattr(tracker, "lock_state_machine", None)
    # This frame is decided by the recovery path; the node must defer to the
    # last_lock_decision produced here rather than re-step present=True.
    tracker.last_frame_recovery = True

    def _step_coast() -> None:
        """Step the FSM as an absent/unconfirmed coast frame (no provisional)."""
        if fsm is None:
            return
        present = any(r.track_id == tracker.target_track_id for r in results)
        cands = len([r for r in results if r.class_id == 0 and r.track_id >= 0])
        tracker.last_lock_decision = fsm.step(
            sim_score=0.0, present=present, frames_since_loss=tracker.frames_lost,
            num_candidates=cands, distinct_margin=0.0, depth_consistent=True,
        )

    if len(results) > 1:
        register_other_persons(tracker, frame, results)

    reid_match = find_best_match_reid(tracker, frame, results)
    if reid_match is None:
        tracker.frames_lost += 1
        tracker.reid_fit_streak = 0
        tracker.reid_fit_id = None
        if tracker.fast_tracking_mode:
            tracker.fast_tracking_mode = False
        if tracker.frames_lost > 3:
            tracker.pending_reid_match = None
            tracker.consecutive_reid_frames = 0
        if tracker.frames_lost > tracker.max_frames_lost:
            tracker.state = TrackerState.LOST
        _step_coast()
        return None

    match_result, best_similarity = reid_match
    if match_result.track_id < 0:
        logger.warning(f"ReID returned invalid track ID {match_result.track_id}, ignoring")
        _step_coast()
        return None

    # _confirm_reid_candidate is the SOLE writer of target_track_id (the id-swap
    # at its ~:397). It also returns non-None on PARTIAL confirms (pending /
    # sim>=reid_threshold pre-commit), so a non-None return does NOT imply a
    # commit. The committed-vs-provisional signal is target_track_id changing
    # across the call: only the real id-swap mutates it. Capture before/after.
    prev_target_id = tracker.target_track_id
    confirmed = _confirm_reid_candidate(tracker, frame, match_result, best_similarity)
    committed_swap = confirmed is not None and tracker.target_track_id != prev_target_id

    num_cands = len([r for r in results if r.class_id == 0 and r.track_id >= 0])
    margin = float(getattr(tracker, "last_reid_margin", 0.0) or 0.0)

    # Phase 2: real depth-consistency for the chosen recovery candidate (replaces
    # the Task-1 hardcoded True). A toward-camera crosser is depth-inconsistent;
    # the FSM treats it as a failed confirm so it is never surfaced as a valid
    # provisional. None operator/candidate depth ⇒ permissive (True).
    from .depth_gate import should_reject_candidate
    cand_depth = getattr(tracker, "candidate_depths_m", {}).get(match_result.track_id)
    depth_consistent = not should_reject_candidate(
        cand_depth, getattr(tracker, "operator_last_depth_m", None),
        getattr(tracker, "crosser_depth_jump_m", 0.6),
    )

    if committed_swap:
        # A genuine id-swap committed THIS frame. Mirror it in the FSM as a
        # present frame so it reports target_lost=False — it does NOT re-decide
        # the id (the pipeline owns that). Always publishes the committed point.
        if fsm is not None:
            tracker.last_lock_decision = fsm.step(
                sim_score=float(best_similarity), present=True, frames_since_loss=0,
                num_candidates=num_cands, distinct_margin=margin,
                depth_consistent=depth_consistent,
            )
        return confirmed

    tracker.state = TrackerState.REIDENTIFYING

    # Partial confirm (sim>=reid_threshold but pre-commit) OR no confirm: this is
    # a PROVISIONAL recovery frame. Step the FSM present=False with the real
    # similarity + distinctiveness margin + depth-consistency (Task 2 — a
    # toward-camera crosser is now flagged depth-inconsistent so the FSM will not
    # promote it). The asymmetric high-bar + commit_frames hysteresis governs
    # target_lost; decision.publish gates whether we surface the provisional
    # point. A sim in [reid_threshold, high_bar) does NOT publish and does NOT
    # clear target_lost — the partial-confirm leak is closed here.
    if fsm is not None:
        decision = fsm.step(
            sim_score=float(best_similarity), present=False,
            frames_since_loss=tracker.frames_lost, num_candidates=num_cands,
            distinct_margin=margin, depth_consistent=depth_consistent,
        )
        tracker.last_lock_decision = decision
        if not decision.publish:
            return None

    # Provisional publish allowed (cleared the high bar). Reuse the partial
    # confirm's result if present (already original-id-stamped); otherwise stamp
    # the raw match.
    if confirmed is not None:
        return confirmed
    return tracker._with_original_id(match_result)


def _confirm_reid_candidate(
    tracker,
    frame: np.ndarray,
    reid_match: TrackingResult,
    best_similarity: float,
) -> Optional[TrackingResult]:
    current_time = time.time()
    new_yolo_id = reid_match.track_id
    match_similarity = best_similarity

    if tracker.appearance_extractor is not None:
        features = _get_or_extract_features(tracker, frame, reid_match.track_id, reid_match.bbox, reid_match.mask, reid_match.class_id)
        if features:
            match_similarity = ReIDMatcher.compute_similarity(
                tracker.target_appearance, features, reid_match.bbox, current_time, is_person=True
            )
            if tracker.is_occluded and tracker.pre_occlusion_appearance is not None:
                pre_sim = ReIDMatcher.compute_similarity(
                    tracker.pre_occlusion_appearance, features, reid_match.bbox, current_time, is_person=True
                )
                if pre_sim < 0.65:
                    logger.info(
                        f"ReID candidate ID {reid_match.track_id} rejected: pre-occlusion similarity {pre_sim:.3f} < 0.65"
                    )
                    return None
        else:
            match_similarity = 0.5

    post_shake_extra = 5 if (current_time - tracker.last_camera_motion_time) < 2.0 else 0
    required_confirmation = tracker.reid_confirmation_frames + post_shake_extra

    if tracker.pending_reid_match is not None and tracker.pending_reid_match[0] == new_yolo_id:
        if match_similarity >= tracker.reid_threshold:
            tracker.consecutive_reid_frames += 1
            tracker.state = TrackerState.REIDENTIFYING
            tracker.frames_lost = 0
            if tracker.consecutive_reid_frames >= required_confirmation:
                if (current_time - tracker.last_reid_switch_time) >= tracker.reid_switch_cooldown:
                    old_yolo_id = tracker.target_track_id
                    tracker.target_track_id = new_yolo_id
                    tracker.state = TrackerState.TRACKING
                    tracker.frames_lost = 0
                    tracker.last_reid_switch_time = current_time
                    tracker.pending_reid_match = None
                    tracker.consecutive_reid_frames = 0
                    tracker.reid_fit_streak = 0
                    tracker.reid_fit_id = None
                    tracker.person_registry.clear()
                    if tracker.original_track_id is not None:
                        tracker.person_registry.register_person(tracker.original_track_id, tracker.target_appearance)
                    logger.info(f"Confirmed ReID: YOLO ID {old_yolo_id} -> {tracker.target_track_id}")
                    return tracker._with_original_id(reid_match)
        return tracker._with_original_id(reid_match)

    if match_similarity >= tracker.reid_threshold:
        if tracker.reid_fit_id == new_yolo_id:
            tracker.reid_fit_streak += 1
        else:
            tracker.reid_fit_id = new_yolo_id
            tracker.reid_fit_streak = 1
        if tracker.reid_fit_streak >= tracker.reid_preconfirm_frames:
            tracker.pending_reid_match = (new_yolo_id, current_time)
            tracker.consecutive_reid_frames = 1
        tracker.frames_lost = 0
        tracker.state = TrackerState.REIDENTIFYING
        return tracker._with_original_id(reid_match)

    tracker.reid_fit_streak = 0
    tracker.reid_fit_id = None
    return None


def register_other_persons(tracker, frame: np.ndarray, results: List[TrackingResult]) -> None:
    """Register other visible persons to improve distinctiveness checks."""
    if tracker.appearance_extractor is None:
        return
    if tracker.fast_tracking_mode and tracker.frame_count % tracker.reid_extraction_interval != 0:
        return

    registered = 0
    for result in results:
        if registered >= 2:
            break
        if result.class_id != 0 or result.track_id < 0 or result.track_id == tracker.target_track_id:
            continue
        if tracker.pending_reid_match is not None and result.track_id == tracker.pending_reid_match[0]:
            continue
        temp_display_id = -result.track_id if result.track_id > 0 else result.track_id - 1000
        if tracker.person_registry.get_person(temp_display_id) is not None:
            continue

        features = tracker.appearance_extractor.extract_features(frame, result.bbox, result.mask, class_id=result.class_id)
        if not features:
            continue

        # Never cement a plausible TARGET as an "other person". After a transient
        # detection gap ByteTrack re-emits the lone operator under a fresh id, so
        # `target_track_id` (the stale pre-loss id) no longer excludes it above.
        # If the candidate matches the target above the acceptance bar it could BE
        # the operator — registering it would make it fail its own distinctiveness
        # check against its self-ghost forever (a self-poisoning deadlock, since
        # the confirmed-swap that would clear() the registry is gated behind that
        # very check). Skip it; genuine distractors score well below the bar.
        target_app = getattr(tracker, "target_appearance", None)
        if target_app is not None:
            sim_to_target = ReIDMatcher.compute_similarity(
                target_app, features, result.bbox, time.time(), is_person=True)
            if sim_to_target >= OTHER_PERSON_MAX_TARGET_SIM:
                logger.debug(
                    f"Not registering ID {result.track_id} as other-person: "
                    f"sim_to_target {sim_to_target:.3f} >= {OTHER_PERSON_MAX_TARGET_SIM} "
                    "(likely the returning target)"
                )
                continue

        other = TargetAppearance(class_id=result.class_id, class_name=result.class_name)
        if "reid" in features:
            other.feature_history.append(features["reid"])
        elif "cnn" in features:
            other.feature_history.append(features["cnn"])
        if "color_hist" in features:
            other.color_hist_history.append(features["color_hist"])
        if "body_color" in features:
            other.body_color_history.append(features["body_color"])
        if "size" in features:
            other.size_history.append(tuple(features["size"]))
        other.last_seen_time = time.time()
        tracker.person_registry.register_person(temp_display_id, other)
        registered += 1


def update_scene_motion(tracker, results: List[TrackingResult], frame: Optional[np.ndarray] = None) -> None:
    """Detect camera motion using lightweight scene centroid tracking."""
    current_time = time.time()
    if tracker.frame_count % 2 != 0:
        return

    persons = [r for r in results if r.class_id == 0 and r.track_id >= 0]
    motion_detected = False
    motion_magnitude = 0.0

    if len(persons) >= 2:
        sum_cx = sum((r.bbox[0] + r.bbox[2]) * 0.5 for r in persons)
        sum_cy = sum((r.bbox[1] + r.bbox[3]) * 0.5 for r in persons)
        scene_cx = sum_cx / len(persons)
        scene_cy = sum_cy / len(persons)
        if tracker.scene_center_history:
            prev_cx, prev_cy = tracker.scene_center_history[-1]
            centroid_motion = abs(scene_cx - prev_cx) + abs(scene_cy - prev_cy)
            if centroid_motion > tracker.CAMERA_MOTION_THRESHOLD * 1.2:
                motion_detected = True
                motion_magnitude = centroid_motion
                tracker.camera_motion_vector = (scene_cx - prev_cx, scene_cy - prev_cy)
        tracker.scene_center_history.append((scene_cx, scene_cy))
        if len(tracker.scene_center_history) > 3:
            tracker.scene_center_history.pop(0)

    if motion_detected:
        was_stable = not tracker.camera_motion_detected
        tracker.camera_motion_detected = True
        tracker.last_camera_motion_time = current_time
        logger.info(f"Camera motion detected! Magnitude: {motion_magnitude:.1f}px")
        if was_stable and tracker.pending_reid_match is not None:
            tracker.pending_reid_match = None
            tracker.consecutive_reid_frames = 0
    elif tracker.camera_motion_detected:
        if current_time - tracker.last_camera_motion_time > tracker.CAMERA_MOTION_COOLDOWN:
            tracker.camera_motion_detected = False
            tracker.camera_motion_vector = (0.0, 0.0)
            logger.info("Camera stabilized, re-enabling spatial continuity")


def periodic_reid_validation(
    tracker,
    frame: np.ndarray,
    results: List[TrackingResult],
    current_result: TrackingResult,
    current_similarity: Optional[float] = None,
) -> Tuple[bool, Optional[TrackingResult]]:
    """Run a scheduled ReID sanity check while already tracking."""
    if (
        tracker.reid_verification_interval <= 0
        or tracker.frame_count % tracker.reid_verification_interval != 0
        or not tracker.enable_reid
        or tracker.target_appearance is None
        or tracker.appearance_extractor is None
        or current_result.class_id != 0
    ):
        return True, None

    if current_similarity is None:
        features_cur = _get_or_extract_features(
            tracker, frame, current_result.track_id, current_result.bbox, current_result.mask, current_result.class_id
        )
        current_similarity = (
            ReIDMatcher.compute_similarity(
                tracker.target_appearance, features_cur, current_result.bbox, time.time(), is_person=True
            )
            if features_cur
            else 0.0
        )

    # Dashboard telemetry: the periodic re-verification of the currently-tracked
    # target's similarity. _find_best_match_reid below overwrites this with the
    # full per-candidate map when there are valid candidates, but stashing here
    # guarantees the tracked target's own score surfaces even when the search
    # returns no candidates.
    tracker.last_debug_scores = {int(current_result.track_id): float(current_similarity)}

    match = tracker._find_best_match_reid(frame, results)
    if match is None:
        return True, None

    best_match, best_similarity = match
    if best_match.track_id == current_result.track_id:
        return True, None

    margin = best_similarity - (current_similarity or 0.0)
    margin_required = max(ReIDMatcher.REID_MARGIN, 0.15)
    if best_similarity > tracker.reid_threshold and margin > margin_required:
        logger.info(
            f"Periodic ReID prefers ID {best_match.track_id} (sim={best_similarity:.3f}) "
            f"over current {current_result.track_id} (sim={current_similarity:.3f}, margin={margin:.3f})"
        )
        return False, best_match

    return True, None


def detect_occlusion(
    tracker,
    target_result: TrackingResult,
    all_results: List[TrackingResult],
) -> Tuple[bool, Optional[TrackingResult]]:
    """Detect if the target is being occluded by another person."""
    if target_result is None:
        return False, None

    target_bbox = target_result.bbox
    target_area = (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1])

    for result in all_results:
        if result.track_id == target_result.track_id or result.class_id != 0:
            continue
        iou = tracker._calculate_iou(target_bbox, result.bbox)
        if iou > tracker.occlusion_iou_threshold:
            occluder_area = (result.bbox[2] - result.bbox[0]) * (result.bbox[3] - result.bbox[1])
            x1 = max(target_bbox[0], result.bbox[0])
            y1 = max(target_bbox[1], result.bbox[1])
            x2 = min(target_bbox[2], result.bbox[2])
            y2 = min(target_bbox[3], result.bbox[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            overlap_ratio = intersection / target_area if target_area > 0 else 0
            if overlap_ratio > 0.4 or (occluder_area > target_area * 1.2 and iou > 0.35):
                logger.info(
                    f"Occlusion detected: Person ID {result.track_id} overlapping target "
                    f"(IoU={iou:.2f}, overlap_ratio={overlap_ratio:.2f})"
                )
                return True, result

    return False, None


def verify_post_occlusion(
    tracker,
    frame: np.ndarray,
    result: TrackingResult,
    current_time: float,
) -> bool:
    """Verify that the tracked person after occlusion is still the same target."""
    if tracker.pre_occlusion_appearance is None or tracker.appearance_extractor is None:
        return True

    features = tracker.appearance_extractor.extract_features(frame, result.bbox, result.mask, class_id=0)
    if not features:
        return True

    similarity = ReIDMatcher.compute_similarity(
        tracker.pre_occlusion_appearance, features, result.bbox, current_time, is_person=True
    )
    if similarity < 0.65:
        logger.warning(
            f"Post-occlusion verification FAILED: similarity={similarity:.3f} < 0.65. "
            f"Track ID {result.track_id} may have switched to occluder!"
        )
        return False

    logger.info(f"Post-occlusion verification PASSED: similarity={similarity:.3f}")
    return True


def save_pre_occlusion_state(tracker) -> None:
    """Save the target appearance before occlusion for later verification."""
    if tracker.target_appearance is not None:
        tracker.pre_occlusion_appearance = copy.deepcopy(tracker.target_appearance)
        logger.info("Saved pre-occlusion appearance for later verification")
