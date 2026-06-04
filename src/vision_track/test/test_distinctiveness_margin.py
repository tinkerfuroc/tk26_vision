import numpy as np
from vision_track.core.registry import PersonRegistry
from vision_track.core.tracking_types import TargetAppearance


def _registry_with_two():
    reg = PersonRegistry()
    reg.register_person(0, TargetAppearance(class_id=0, class_name="person"))   # target
    reg.register_person(1, TargetAppearance(class_id=0, class_name="person"))   # other
    return reg


def test_threshold_raised_to_0_10():
    assert abs(PersonRegistry().distinctiveness_threshold - 0.10) < 1e-9


def test_lookalike_rejected_at_tight_margin():
    reg = _registry_with_two()
    # other person scores 0.78 vs target candidate score 0.83 -> margin 0.05 < 0.10
    sim_func = lambda appearance, feats: 0.78
    assert reg.check_distinctiveness(0, {"reid": np.zeros(4)}, 0.83, sim_func) is False


def test_distinct_candidate_accepted():
    reg = _registry_with_two()
    sim_func = lambda appearance, feats: 0.55   # other much worse
    assert reg.check_distinctiveness(0, {"reid": np.zeros(4)}, 0.83, sim_func) is True


def test_no_other_persons_always_distinct():
    reg = PersonRegistry()
    reg.register_person(0, TargetAppearance(class_id=0, class_name="person"))
    assert reg.check_distinctiveness(0, {"reid": np.zeros(4)}, 0.5, lambda a, f: 0.99) is True
