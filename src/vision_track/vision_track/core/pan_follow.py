# vision_track/core/pan_follow.py
"""Pure horizontal pan-to-center controller for the head pan-tilt.

ROS-free + numpy-free so the control law is unit-testable with synthetic inputs.
Keeps the tracked person centered horizontally by commanding the pan servo in
ABSOLUTE mode: target = current_pan + pan_sign * atan2(u - cx, fx). Because the
target is recomputed from the live PanTiltState pan AND the live pixel error on
every command, error never accumulates (the explicit reason ABSOLUTE is used
instead of RELATIVE). Tilt is held fixed by the caller; this class owns pan only.

Modes:
  - center(...)  CENTER: a bbox is visible -> pan toward atan2(u - cx, fx).
  - HOLD: no bbox -> the caller does not call center(), so no command issues and
    the servo keeps pointing where the person was last seen. This holds through
    BOTH the PASSIVE coast and the NEEDS_HELP hold: the head must NOT swing away
    during NEEDS_HELP, because keeping it on the operator's last direction is what
    lets the tracker re-detect them when they walk back into that view.

Every command passes a common gate: EMA-smooth the target, clamp to limits, then
suppress it if (a) the servo already points within deadband_rad of it, (b) it is
within min_change_rad of the last command (anti-chatter), or (c) fewer than
min_interval_s have passed since the last command (rate-limit the 30 Hz loop).
"""
from __future__ import annotations

import math
from typing import Optional


def pan_follow_suppressed(enable_follow: bool, has_publisher: bool,
                          help_latched: bool) -> bool:
    """True when the tracker must NOT issue a pan command this tick.

    Suppress when pan-follow is disabled, when there is no command publisher, or
    when NEEDS_HELP is latched — in NEEDS_HELP the behavior tree owns the head
    (the two-pass recovery scan), so the tracker holds off until re-lock clears
    the latch and hands head control back.
    """
    return (not enable_follow) or (not has_publisher) or bool(help_latched)


class PanFollower:
    def __init__(
        self,
        *,
        pan_sign: float = 1.0,   # +1 matches follow_head (cur_pan + atan2(x_cam,z_cam))
        deadband_rad: float = math.radians(3.0),
        min_change_rad: float = math.radians(1.0),
        min_interval_s: float = 0.15,
        ema_alpha: float = 0.5,
        pan_min_rad: float = math.radians(-90.0),
        pan_max_rad: float = math.radians(90.0),
    ) -> None:
        self.pan_sign = float(pan_sign)
        self.deadband_rad = float(deadband_rad)
        self.min_change_rad = float(min_change_rad)
        self.min_interval_s = float(min_interval_s)
        self.ema_alpha = float(ema_alpha)
        self.pan_min_rad = float(pan_min_rad)
        self.pan_max_rad = float(pan_max_rad)
        self._ema_target: Optional[float] = None
        self._last_cmd_pan: Optional[float] = None
        self._last_cmd_t: float = -1e9

    def reset(self) -> None:
        """Drop EMA + throttle state (call on goal start/end)."""
        self._ema_target = None
        self._last_cmd_pan = None
        self._last_cmd_t = -1e9

    def center(self, u, cx, fx, current_pan, now) -> Optional[float]:
        """CENTER: pan toward the bbox center-x. None if it must hold."""
        if current_pan is None or fx is None or float(fx) == 0.0:
            return None
        theta = math.atan2(float(u) - float(cx), float(fx))
        raw_target = float(current_pan) + self.pan_sign * theta
        return self._gate(raw_target, current_pan, now)

    def _clamp(self, p: float) -> float:
        return max(self.pan_min_rad, min(self.pan_max_rad, p))

    def _gate(self, raw_target, current_pan, now) -> Optional[float]:
        if self._ema_target is None:
            self._ema_target = raw_target
        else:
            a = self.ema_alpha
            self._ema_target = a * raw_target + (1.0 - a) * self._ema_target
        target = self._clamp(self._ema_target)

        if current_pan is not None and abs(target - float(current_pan)) < self.deadband_rad:
            return None
        if (self._last_cmd_pan is not None
                and abs(target - self._last_cmd_pan) < self.min_change_rad):
            return None
        if (now - self._last_cmd_t) < self.min_interval_s:
            return None

        self._last_cmd_pan = target
        self._last_cmd_t = now
        return target
