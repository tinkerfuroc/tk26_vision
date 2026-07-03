"""Unit tests for _waving_bench_eval.py — rclpy-free scenario suite core."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from tk_vision_specialized._waving_bench_eval import (
    CallRecord,
    CaseSpec,
    Expect,
    distance_of,
    evaluate_call,
    evaluate_case,
    expected_status,
    load_suite,
    suite_passed,
)

_SHIPPED_CONFIG = Path(__file__).resolve().parent.parent / "config" / "waving_bench.yaml"


def _suite(scenarios, defaults=None):
    config = {"scenarios": scenarios}
    if defaults is not None:
        config["defaults"] = defaults
    return load_suite(config)


def _case(expect, *, best_effort=False, pass_ratio=0.8, request=None):
    req = {"threshold_meters": 8.0, "target_frame": "", "min_waving_persons": 0}
    if request:
        req.update(request)
    return CaseSpec(
        scenario="s", index=0, prompt="p", request=req,
        calls_per_case=5, interval_sec=1.0, pass_ratio=pass_ratio,
        expect=expect, best_effort=best_effort,
    )


# --- load_suite: defaults & overrides ---------------------------------------

def test_defaults_applied_when_case_has_no_overrides():
    suite = _suite({"s": [{"prompt": "p"}]})
    case = suite["s"][0]
    assert case.request == {
        "threshold_meters": 8.0, "target_frame": "", "min_waving_persons": 0,
    }
    assert case.calls_per_case == 5
    assert case.pass_ratio == 0.8


def test_overrides_win_over_defaults():
    suite = _suite({
        "s": [{
            "prompt": "p",
            "overrides": {
                "threshold_meters": 3.0,
                "target_frame": "base_link",
                "pass_ratio": 0.5,
            },
        }],
    })
    case = suite["s"][0]
    assert case.request["threshold_meters"] == 3.0
    assert case.request["target_frame"] == "base_link"
    assert case.pass_ratio == 0.5


# --- distance_of -------------------------------------------------------------

@pytest.mark.parametrize("frame_id", ["", "camera_color_optical_frame"])
def test_distance_of_camera_frame_uses_z(frame_id):
    assert distance_of((1.0, 2.0, 3.0), frame_id) == 3.0


@pytest.mark.parametrize("frame_id", ["base_link", "map"])
def test_distance_of_robot_frame_uses_hypot_xy(frame_id):
    assert distance_of((3.0, 4.0, 9.0), frame_id) == pytest.approx(5.0)


# --- expected_status -----------------------------------------------------------

def test_expected_status_derived_one_from_count_zero():
    assert expected_status(Expect(count=0)) == 1


def test_expected_status_derived_zero_from_positive_count():
    assert expected_status(Expect(count=2)) == 0


def test_expected_status_explicit_status_wins_over_derived():
    assert expected_status(Expect(status=1, count=2)) == 1


def test_expected_status_none_when_no_count_or_status():
    assert expected_status(Expect()) is None


# --- evaluate_call -------------------------------------------------------------

def test_count_mismatch_reason_names_expected_and_got():
    case = _case(Expect(count=2))
    call = CallRecord(status=0, points=[(0.0, 0.0, 1.0)], frame_ids=[""])
    verdict = evaluate_call(case, call)
    assert verdict.passed is False
    assert any("expected 2" in r and "got 1" in r for r in verdict.reasons)


def test_z_range_boundary_lo_and_hi_pass_outside_fails():
    case = _case(Expect(z_range_m=(1.0, 2.0)))
    call_lo = CallRecord(status=0, points=[(0.0, 0.0, 1.0)], frame_ids=[""])
    call_hi = CallRecord(status=0, points=[(0.0, 0.0, 2.0)], frame_ids=[""])
    call_out = CallRecord(status=0, points=[(0.0, 0.0, 2.01)], frame_ids=[""])
    assert evaluate_call(case, call_lo).passed is True
    assert evaluate_call(case, call_hi).passed is True
    assert evaluate_call(case, call_out).passed is False


def test_ordering_closest_first():
    case = _case(Expect(ordering="closest_first"))
    ascending = CallRecord(
        status=0, points=[(0.0, 0.0, 2.0), (0.0, 0.0, 4.0)], frame_ids=["", ""])
    descending = CallRecord(
        status=0, points=[(0.0, 0.0, 4.0), (0.0, 0.0, 2.0)], frame_ids=["", ""])
    within_tolerance = CallRecord(
        status=0, points=[(0.0, 0.0, 2.00), (0.0, 0.0, 1.96)], frame_ids=["", ""])
    assert evaluate_call(case, ascending).passed is True
    assert evaluate_call(case, descending).passed is False
    assert evaluate_call(case, within_tolerance).passed is True


def test_frame_id_mismatch_fails():
    case = _case(Expect(frame_id="base_link"))
    call = CallRecord(status=0, points=[(1.0, 1.0, 0.0)], frame_ids=["map"])
    assert evaluate_call(case, call).passed is False


# --- evaluate_case / suite_passed -----------------------------------------------

def test_pass_ratio_boundary_at_defaults():
    case = _case(Expect(count=1))
    passing_call = CallRecord(status=0, points=[(0.0, 0.0, 1.0)], frame_ids=[""])
    failing_call = CallRecord(status=0, points=[], frame_ids=[])

    four_of_five = [passing_call] * 4 + [failing_call]
    result_pass = evaluate_case(case, four_of_five)
    assert result_pass.n_passed == 4
    assert result_pass.passed is True

    three_of_five = [passing_call] * 3 + [failing_call] * 2
    result_fail = evaluate_case(case, three_of_five)
    assert result_fail.n_passed == 3
    assert result_fail.passed is False

    result_zero = evaluate_case(case, [])
    assert result_zero.passed is False


def test_best_effort_failing_case_does_not_fail_suite():
    case = _case(Expect(count=1), best_effort=True)
    failing_calls = [CallRecord(status=0, points=[], frame_ids=[])] * 5
    result = evaluate_case(case, failing_calls)
    assert result.passed is False
    assert suite_passed([result]) is True


def test_non_best_effort_failing_case_fails_suite():
    case = _case(Expect(count=1), best_effort=False)
    failing_calls = [CallRecord(status=0, points=[], frame_ids=[])] * 5
    result = evaluate_case(case, failing_calls)
    assert result.passed is False
    assert suite_passed([result]) is False


# --- load_suite validation -------------------------------------------------------

def test_unknown_expect_key_raises():
    with pytest.raises(ValueError):
        _suite({"s": [{"prompt": "p", "expect": {"bogus": 1}}]})


def test_unknown_overrides_key_raises():
    with pytest.raises(ValueError):
        _suite({"s": [{"prompt": "p", "overrides": {"bogus": 1}}]})


def test_invalid_ordering_raises():
    with pytest.raises(ValueError):
        _suite({"s": [{"prompt": "p", "expect": {"ordering": "farthest_first"}}]})


def test_descending_z_range_raises():
    with pytest.raises(ValueError):
        _suite({"s": [{"prompt": "p", "expect": {"z_range_m": [2.0, 1.0]}}]})


def test_unknown_defaults_key_raises():
    with pytest.raises(ValueError):
        _suite({"s": [{"prompt": "p"}]}, defaults={"bogus": 1})


# --- shipped config -------------------------------------------------------------

def test_shipped_config_parses_and_has_expected_shape():
    with _SHIPPED_CONFIG.open() as f:
        config = yaml.safe_load(f)
    suite = load_suite(config)

    assert set(suite) == {
        "smoke", "range_ladder", "gesture_matrix",
        "two_person_arbitration", "threshold_gate", "frames", "vlm_fallback",
    }

    range_ladder_8m = suite["range_ladder"][-1]
    assert range_ladder_8m.best_effort is True

    vlm_case = suite["vlm_fallback"][0]
    assert vlm_case.best_effort is True

    frames_case = suite["frames"][0]
    assert frames_case.expect.frame_id == "base_link"
