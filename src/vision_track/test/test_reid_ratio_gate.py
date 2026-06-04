from vision_track.reid.identity_gate import (
    deep_ratio_ambiguous,
    spatial_switch_allowed,
    DEFAULT_RATIO_MAX,
    DEFAULT_DEEP_SWITCH_MARGIN,
)


def test_clear_winner_not_ambiguous():
    # best 0.85, second 0.55 -> ratio 0.65 well under the cap
    assert deep_ratio_ambiguous(0.85, 0.55, ratio_max=DEFAULT_RATIO_MAX) is False


def test_deep_indistinguishable_is_ambiguous():
    # best 0.80, second 0.79 -> ratio ~0.99 -> ambiguous
    assert deep_ratio_ambiguous(0.80, 0.79, ratio_max=DEFAULT_RATIO_MAX) is True


def test_zero_or_negative_best_is_ambiguous():
    assert deep_ratio_ambiguous(0.0, 0.0, ratio_max=DEFAULT_RATIO_MAX) is True
    assert deep_ratio_ambiguous(-0.1, -0.2, ratio_max=DEFAULT_RATIO_MAX) is True


def test_spatial_switch_blocked_when_runner_up_loses_deep():
    # spatially closer runner-up but its deep cosine is much worse -> block switch
    assert spatial_switch_allowed(
        deep_best=0.82, deep_candidate=0.55, margin=DEFAULT_DEEP_SWITCH_MARGIN
    ) is False


def test_spatial_switch_allowed_when_runner_up_ties_or_wins_deep():
    # runner-up at least ties deep within margin -> proximity may break the tie
    assert spatial_switch_allowed(
        deep_best=0.80, deep_candidate=0.78, margin=DEFAULT_DEEP_SWITCH_MARGIN
    ) is True
    assert spatial_switch_allowed(
        deep_best=0.75, deep_candidate=0.85, margin=DEFAULT_DEEP_SWITCH_MARGIN
    ) is True
