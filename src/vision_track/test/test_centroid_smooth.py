"""Tests for pure torso-band reducer + EMA point smoother."""
import numpy as np
import pytest

from vision_track.core.centroid_smooth import torso_band_mask, PointEMA


class TestTorsoBandMask:
    def test_band_selects_chest_rows(self):
        """A bbox of height 100 with band (0.15, 0.55) keeps rows 15..55."""
        bbox = (10, 0, 60, 100)  # x1,y1,x2,y2 — height 100
        m = torso_band_mask(bbox, lo=0.15, hi=0.55)
        y1_band, y2_band = m
        assert y1_band == 15
        assert y2_band == 55

    def test_band_clamped_to_bbox(self):
        bbox = (0, 40, 50, 60)  # height 20
        y1_band, y2_band = torso_band_mask(bbox, lo=0.15, hi=0.55)
        assert y1_band == 40 + 3   # 0.15 * 20 = 3
        assert y2_band == 40 + 11  # 0.55 * 20 = 11

    def test_degenerate_band_returns_full(self):
        """Tiny bbox where lo*h == hi*h → fall back to full bbox rows."""
        bbox = (0, 0, 50, 2)  # height 2 → 0.15*2=0, 0.55*2=1
        y1_band, y2_band = torso_band_mask(bbox, lo=0.15, hi=0.55, min_rows=4)
        # band too thin → returns full bbox y-range
        assert y1_band == 0
        assert y2_band == 2


class TestPointEMA:
    def test_first_sample_passes_through(self):
        ema = PointEMA(alpha=0.5)
        out = ema.update((1.0, 2.0, 3.0))
        assert out == (1.0, 2.0, 3.0)

    def test_ema_blends(self):
        ema = PointEMA(alpha=0.5)
        ema.update((0.0, 0.0, 0.0))
        out = ema.update((2.0, 4.0, 6.0))
        assert out == pytest.approx((1.0, 2.0, 3.0))

    def test_alpha_one_is_passthrough(self):
        ema = PointEMA(alpha=1.0)
        ema.update((0.0, 0.0, 0.0))
        out = ema.update((5.0, 5.0, 5.0))
        assert out == pytest.approx((5.0, 5.0, 5.0))

    def test_reset_clears_state(self):
        ema = PointEMA(alpha=0.5)
        ema.update((10.0, 10.0, 10.0))
        ema.reset()
        out = ema.update((1.0, 2.0, 3.0))
        assert out == (1.0, 2.0, 3.0)   # first-after-reset passes through

    def test_none_sample_does_not_corrupt_state(self):
        ema = PointEMA(alpha=0.5)
        ema.update((2.0, 2.0, 2.0))
        assert ema.update(None) is None
        out = ema.update((4.0, 4.0, 4.0))
        # state preserved across the None gap
        assert out == pytest.approx((3.0, 3.0, 3.0))
