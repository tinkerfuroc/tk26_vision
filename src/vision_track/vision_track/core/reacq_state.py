"""Pure reacquisition-state hysteresis for the active-reID escalation signal.

The tracker is the publish authority; this maps (tracked?, consecutive frames
lost) to an advisory state a consumer (behaviour tree) can act on. It NEVER
calls out itself — it only debounces "lost long enough that active help is
warranted" so the BT doesn't escalate (and incur a points penalty) prematurely.
"""
from __future__ import annotations

REACQ_TRACKING = 0
REACQ_PASSIVE = 1
REACQ_NEEDS_HELP = 2


def reacq_state(tracked: bool, frames_lost: int, help_after: int) -> int:
    """Map tracking status to a reacquisition state.

    Args:
        tracked: True if the target was matched/published this frame.
        frames_lost: consecutive frames since the target was last held.
        help_after: escalate to NEEDS_HELP once frames_lost reaches this.

    Returns:
        REACQ_TRACKING while held; REACQ_PASSIVE while lost but within the
        passive-recovery window; REACQ_NEEDS_HELP once lost >= help_after.
    """
    if tracked:
        return REACQ_TRACKING
    if frames_lost >= help_after:
        return REACQ_NEEDS_HELP
    return REACQ_PASSIVE
