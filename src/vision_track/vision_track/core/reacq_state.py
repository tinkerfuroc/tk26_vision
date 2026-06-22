"""Pure reacquisition-state hysteresis for the active-reID escalation signal.

The tracker is the publish authority; this maps (tracked?, wall-clock seconds
since the last confirmed lock) to an advisory state a consumer (behaviour tree)
can act on. It NEVER calls out itself — it only debounces "lost long enough that
active help is warranted" so the BT doesn't escalate (and incur a points
penalty) prematurely. Escalation is wall-clock, not frame-count: tournament GPU
contention makes the frame rate unreliable, so a frame threshold would give an
unpredictable real-time window.
"""
from __future__ import annotations

REACQ_TRACKING = 0
REACQ_PASSIVE = 1
REACQ_NEEDS_HELP = 2


def reacq_state(tracked: bool, time_since_lost: float, help_after_sec: float) -> int:
    """Map tracking status to a reacquisition state.

    Args:
        tracked: True if the target was matched/published this frame.
        time_since_lost: wall-clock seconds since the last CONFIRMED lock.
        help_after_sec: escalate to NEEDS_HELP once time_since_lost reaches this
            many seconds. ``<= 0`` escalates immediately when lost.

    Returns:
        REACQ_TRACKING while held; REACQ_PASSIVE while lost but within the
        passive-recovery window; REACQ_NEEDS_HELP once lost for
        >= help_after_sec seconds.
    """
    if tracked:
        return REACQ_TRACKING
    if time_since_lost >= help_after_sec:
        return REACQ_NEEDS_HELP
    return REACQ_PASSIVE
