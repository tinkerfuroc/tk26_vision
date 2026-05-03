"""Unit tests for the slim PersonTracker contract.

Covers:

- default fresh lock = closest to robot origin (XY)
- caller-supplied `fresh_lock_costs` selects argmin
- all-inf costs return None (refuse to lock)
- pixel-distance simulation -> centermost candidate wins despite a
  spatially-closer rival (the `track_centermost` use case)
- once locked, sticky reassoc within ``reassoc_dist_m`` re-acquires
- a far-only candidate set holds the previous lock
- empty candidate set past ``ttl_sec`` resets the lock
"""

import math

import pytest

from pan_tilt.head_tracking_helpers import PersonTracker


def test_fresh_lock_default_picks_closest_to_origin():
    pt = PersonTracker()
    cands = [
        (3.0, 1.5, 1.2),   # far + lateral
        (1.5, 0.05, 1.3),  # closest in XY
        (2.0, -1.0, 1.3),  # mid
    ]
    out = pt.update(cands, now_mono=0.0)
    assert out == pytest.approx((1.5, 0.05, 1.3))


def test_fresh_lock_costs_picks_argmin():
    pt = PersonTracker()
    cands = [
        (1.0, 0.0, 1.3),   # closest to origin XY
        (2.0, -1.0, 1.3),  # caller wants this one
        (3.0, 0.5, 1.3),
    ]
    # Caller's preference overrides default closest-to-origin.
    costs = [10.0, 1.0, 5.0]
    out = pt.update(cands, now_mono=0.0, fresh_lock_costs=costs)
    assert out == pytest.approx((2.0, -1.0, 1.3))


def test_fresh_lock_all_inf_returns_none():
    pt = PersonTracker()
    cands = [(1.0, 0.0, 1.3), (2.0, 0.0, 1.3)]
    out = pt.update(
        cands, now_mono=0.0, fresh_lock_costs=[math.inf, math.inf],
    )
    assert out is None
    assert pt.locked_xyz is None


def test_fresh_lock_centermost_simulation():
    # Two candidates: the spatially closer one is OFF-CENTER in pixels;
    # the further one is image-centered. With pixel-distance costs the
    # centered candidate wins.
    pt = PersonTracker()
    cands = [
        (1.0, 0.5, 1.3),   # closer to robot
        (2.5, 0.05, 1.3),  # further but image-centered
    ]
    pixel_costs = [200.0, 5.0]   # px distance from image center
    out = pt.update(cands, now_mono=0.0, fresh_lock_costs=pixel_costs)
    assert out == pytest.approx((2.5, 0.05, 1.3))


def test_fresh_lock_costs_length_mismatch_raises():
    pt = PersonTracker()
    with pytest.raises(ValueError):
        pt.update(
            [(1.0, 0.0, 1.3)], now_mono=0.0, fresh_lock_costs=[1.0, 2.0],
        )


def test_sticky_within_reassoc_holds():
    pt = PersonTracker(reassoc_dist_m=0.4)
    pt.update([(2.0, 0.0, 1.3)], now_mono=0.0)
    # New candidate within reassoc distance -> picked even though a closer-to-
    # origin one also appeared.
    out = pt.update(
        [(2.05, 0.05, 1.3), (0.5, 0.0, 1.3)],
        now_mono=0.05,
    )
    assert out == pytest.approx((2.05, 0.05, 1.3))


def test_sticky_far_candidates_hold_previous_lock():
    pt = PersonTracker(reassoc_dist_m=0.4, ttl_sec=1.0)
    pt.update([(2.0, 0.0, 1.3)], now_mono=0.0)
    # Only candidate is far outside reassoc; hold the previous lock.
    out = pt.update([(5.0, 0.0, 1.3)], now_mono=0.05)
    assert out == pytest.approx((2.0, 0.0, 1.3))
    # last_seen_mono not refreshed -> TTL still ticks.
    assert pt.last_seen_mono == 0.0


def test_no_candidates_then_ttl_resets():
    pt = PersonTracker(ttl_sec=0.5)
    pt.update([(2.0, 0.0, 1.3)], now_mono=0.0)
    pt.update([], now_mono=1.0)
    assert pt.locked_xyz is None
    # Fresh re-lock works.
    out = pt.update([(1.0, 0.0, 1.3)], now_mono=1.05)
    assert out == pytest.approx((1.0, 0.0, 1.3))


def test_seeded_fresh_lock_simulation():
    # The seed-mode the action server emits: cost = XY-dist-to-seed if
    # within radius, else inf.
    pt = PersonTracker()
    seed_xy = (1.5, -0.6)
    radius = 0.5
    cands = [
        (1.0, 0.0, 1.3),    # far from seed (~0.78 m)
        (1.5, -0.6, 1.3),   # exactly on the seed
        (1.5, -0.65, 1.3),  # near the seed
    ]
    costs = []
    for x, y, _z in cands:
        d = math.hypot(x - seed_xy[0], y - seed_xy[1])
        costs.append(d if d <= radius else math.inf)
    out = pt.update(cands, now_mono=0.0, fresh_lock_costs=costs)
    # Either of the two within-radius candidates is fine; seed-on wins by
    # being exactly zero cost.
    assert out == pytest.approx((1.5, -0.6, 1.3))


def test_reset_clears_state():
    pt = PersonTracker()
    pt.update([(1.0, 0.0, 1.3)], now_mono=0.0)
    pt.reset()
    assert pt.locked_xyz is None
    assert pt.last_seen_mono is None
