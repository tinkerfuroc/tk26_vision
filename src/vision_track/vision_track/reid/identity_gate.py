"""Pure-logic identity gates for ReID association.

Operate on raw deep cosines / scalars so they unit-test without a model.
Used by reid_search to stop spatial proximity from overriding identity.
"""

# Lowe ratio cap on the DEEP term: if second/best > this, the two top
# candidates are deep-indistinguishable and the match is ambiguous.
DEFAULT_RATIO_MAX = 0.92

# A spatially-closer runner-up may only steal the lock if its deep cosine is
# within this margin of (or better than) the best candidate's deep cosine.
DEFAULT_DEEP_SWITCH_MARGIN = 0.05


def deep_ratio_ambiguous(deep_best: float, deep_second: float, *, ratio_max: float) -> bool:
    """True if the runner-up is too close to the best on the deep term."""
    if deep_best <= 1e-6:
        return True  # no usable deep signal -> treat as ambiguous
    ratio = deep_second / deep_best
    return ratio > ratio_max


def spatial_switch_allowed(deep_best: float, deep_candidate: float, *, margin: float) -> bool:
    """True if a spatially-closer candidate also wins/ties the deep term.

    Proximity may break the tie only when the candidate's identity evidence is
    at least as strong (within `margin`) as the current best's.
    """
    return deep_candidate >= deep_best - margin
