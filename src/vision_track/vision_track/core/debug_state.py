"""Pure snapshot of tracker state for the track_web dashboard.

JSON-serializable dict, built defensively (getattr everywhere) so a partially
initialized or bare tracker never raises. No ROS, no cv2 — unit-testable.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _gallery(tracker: Any) -> Tuple[int, int]:
    """(len, version) of the appearance gallery; (0, 0) when absent."""
    app = getattr(tracker, "target_appearance", None)
    gal = getattr(app, "gallery", None) if app is not None else None
    if gal is None:
        return 0, 0
    version = int(getattr(gal, "version", 0) or 0)
    try:
        return len(gal), version
    except Exception:
        return 0, version


def build_debug_state(
    tracker: Any,
    *,
    ts: float,
    target_lost: bool,
    reacquisition_state: int,
    time_since_seen: float,
    awaiting_help: bool,
    active_help_after_sec: float,
    active_help_timeout_sec: float,
) -> Dict[str, Any]:
    """Snapshot tracker + node loss-state into a JSON-serializable dict."""
    decision = getattr(tracker, "last_lock_decision", None)
    scores = getattr(tracker, "last_debug_scores", None) or {}

    candidates = []
    for r in getattr(tracker, "last_results", None) or []:
        tid = getattr(r, "track_id", None)
        if getattr(r, "class_id", None) != 0 or tid is None or tid < 0:
            continue
        bbox = getattr(r, "bbox", None)
        sc: Optional[float] = scores.get(tid)
        candidates.append({
            "id": int(tid),
            "bbox": [int(v) for v in bbox] if bbox is not None else None,
            "score": float(sc) if sc is not None else None,
        })

    ranked = sorted((s for s in scores.values() if s is not None), reverse=True)
    gallery_len, gallery_version = _gallery(tracker)
    return {
        "ts": float(ts),
        "fsm_state": getattr(decision, "state", None),
        "target_lost": bool(target_lost),
        "reacquisition_state": int(reacquisition_state),
        "frames_lost": int(getattr(tracker, "frames_lost", 0) or 0),
        "time_since_seen": float(time_since_seen),
        "awaiting_help": bool(awaiting_help),
        "active_help_after_sec": float(active_help_after_sec),
        "active_help_timeout_sec": float(active_help_timeout_sec),
        "target_track_id": getattr(tracker, "target_track_id", None),
        "original_track_id": getattr(tracker, "original_track_id", None),
        "candidates": candidates,
        "best_sim": float(ranked[0]) if ranked else None,
        "second_sim": float(ranked[1]) if len(ranked) > 1 else None,
        "gallery_len": gallery_len,
        "gallery_version": gallery_version,
    }
