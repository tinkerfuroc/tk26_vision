"""Tests for ptbench.common.scoreboard — PASS/WARN/FAIL gates + table + JSON."""
import json

import pytest

from ptbench.common.scoreboard import GateConfig, Scoreboard, score


def base_metrics(**overrides):
    """A metrics dict that yields all-PASS by default; override individual keys."""
    m = {
        "correct_lock_rate": 0.95,                         # >= 0.92 PASS
        "wrong_lock_episodes": 0,                          # == 0 PASS
        "reacquire_latency_s": {"median": 0.5, "max": 1.0, "samples": [0.5]},  # <= 1.0 PASS
        "pos_error_lateral_m": {"median": 0.10, "p95": 0.2},   # <= 0.25 PASS
        "pos_error_range_m": {"median": 0.10, "p95": 0.2},
        "false_target_rate": 0.01,                         # <= 0.05 PASS
        "throughput_hz": 30.0,                             # >= 12 PASS
    }
    m.update(overrides)
    return m


def verdict_for(sb: Scoreboard, name: str) -> str:
    for n, _v, verdict in sb.rows:
        if n == name:
            return verdict
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Overall happy path
# ---------------------------------------------------------------------------

class TestOverall:
    def test_all_pass(self):
        sb = score(base_metrics())
        assert sb.overall == "PASS"
        for _n, _v, verdict in sb.rows:
            assert verdict == "PASS"

    def test_worst_wins_warn(self):
        sb = score(base_metrics(correct_lock_rate=0.88))  # WARN
        assert verdict_for(sb, "correct_lock_rate") == "WARN"
        assert sb.overall == "WARN"

    def test_worst_wins_fail(self):
        sb = score(base_metrics(correct_lock_rate=0.88, throughput_hz=5.0))  # WARN + FAIL
        assert sb.overall == "FAIL"


# ---------------------------------------------------------------------------
# correct_lock_rate (higher-better)
# ---------------------------------------------------------------------------

class TestCorrectLockRate:
    def test_pass_boundary(self):
        assert verdict_for(score(base_metrics(correct_lock_rate=0.92)), "correct_lock_rate") == "PASS"

    def test_warn_boundary(self):
        assert verdict_for(score(base_metrics(correct_lock_rate=0.85)), "correct_lock_rate") == "WARN"

    def test_just_below_warn_fails(self):
        assert verdict_for(score(base_metrics(correct_lock_rate=0.8499)), "correct_lock_rate") == "FAIL"


# ---------------------------------------------------------------------------
# throughput (higher-better)
# ---------------------------------------------------------------------------

class TestThroughput:
    def test_pass_boundary(self):
        assert verdict_for(score(base_metrics(throughput_hz=12.0)), "throughput_hz") == "PASS"

    def test_warn_boundary(self):
        assert verdict_for(score(base_metrics(throughput_hz=8.0)), "throughput_hz") == "WARN"

    def test_below_warn_fails(self):
        assert verdict_for(score(base_metrics(throughput_hz=7.99)), "throughput_hz") == "FAIL"


# ---------------------------------------------------------------------------
# reacquire_latency (lower-better, uses median)
# ---------------------------------------------------------------------------

class TestReacquireLatency:
    def test_pass_boundary(self):
        m = base_metrics(reacquire_latency_s={"median": 1.0, "max": 1.0, "samples": [1.0]})
        assert verdict_for(score(m), "reacquire_latency_s") == "PASS"

    def test_warn_boundary(self):
        m = base_metrics(reacquire_latency_s={"median": 2.0, "max": 2.0, "samples": [2.0]})
        assert verdict_for(score(m), "reacquire_latency_s") == "WARN"

    def test_above_warn_fails(self):
        m = base_metrics(reacquire_latency_s={"median": 2.01, "max": 3.0, "samples": [2.01]})
        assert verdict_for(score(m), "reacquire_latency_s") == "FAIL"


# ---------------------------------------------------------------------------
# pos_error_lateral (lower-better, uses median)
# ---------------------------------------------------------------------------

class TestPosErrorLateral:
    def test_pass_boundary(self):
        m = base_metrics(pos_error_lateral_m={"median": 0.25, "p95": 0.3})
        assert verdict_for(score(m), "pos_error_lateral_m") == "PASS"

    def test_warn_boundary(self):
        m = base_metrics(pos_error_lateral_m={"median": 0.40, "p95": 0.5})
        assert verdict_for(score(m), "pos_error_lateral_m") == "WARN"

    def test_above_warn_fails(self):
        m = base_metrics(pos_error_lateral_m={"median": 0.41, "p95": 0.5})
        assert verdict_for(score(m), "pos_error_lateral_m") == "FAIL"


# ---------------------------------------------------------------------------
# false_target_rate (lower-better)
# ---------------------------------------------------------------------------

class TestFalseTargetRate:
    def test_pass_boundary(self):
        assert verdict_for(score(base_metrics(false_target_rate=0.05)), "false_target_rate") == "PASS"

    def test_warn_boundary(self):
        assert verdict_for(score(base_metrics(false_target_rate=0.10)), "false_target_rate") == "WARN"

    def test_above_warn_fails(self):
        assert verdict_for(score(base_metrics(false_target_rate=0.11)), "false_target_rate") == "FAIL"


# ---------------------------------------------------------------------------
# wrong_lock_episodes (zero-only)
# ---------------------------------------------------------------------------

class TestWrongLockEpisodes:
    def test_zero_passes(self):
        assert verdict_for(score(base_metrics(wrong_lock_episodes=0)), "wrong_lock_episodes") == "PASS"

    def test_nonzero_fails(self):
        assert verdict_for(score(base_metrics(wrong_lock_episodes=1)), "wrong_lock_episodes") == "FAIL"

    def test_nonzero_forces_overall_fail(self):
        # everything else PASS, but one episode -> overall FAIL
        sb = score(base_metrics(wrong_lock_episodes=2))
        assert sb.overall == "FAIL"


# ---------------------------------------------------------------------------
# None -> N/A, excluded from overall
# ---------------------------------------------------------------------------

class TestNA:
    def test_none_value_is_na(self):
        sb = score(base_metrics(throughput_hz=None))
        assert verdict_for(sb, "throughput_hz") == "N/A"

    def test_na_excluded_from_overall(self):
        # throughput N/A but everything else PASS -> overall PASS
        sb = score(base_metrics(throughput_hz=None))
        assert sb.overall == "PASS"

    def test_all_na_overall_na(self):
        m = {
            "correct_lock_rate": None,
            "wrong_lock_episodes": None,
            "reacquire_latency_s": {"median": None},
            "pos_error_lateral_m": {"median": None},
            "pos_error_range_m": {"median": None},
            "false_target_rate": None,
            "throughput_hz": None,
        }
        sb = score(m)
        assert sb.overall == "N/A"


# ---------------------------------------------------------------------------
# table + dict serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_to_table_contains_overall(self):
        sb = score(base_metrics())
        table = sb.to_table()
        assert "OVERALL" in table
        assert "PASS" in table

    def test_to_table_lists_all_metrics(self):
        sb = score(base_metrics())
        table = sb.to_table()
        for name in (
            "correct_lock_rate",
            "wrong_lock_episodes",
            "reacquire_latency_s",
            "pos_error_lateral_m",
            "false_target_rate",
            "throughput_hz",
        ):
            assert name in table

    def test_to_dict_json_serializable(self):
        sb = score(base_metrics())
        d = sb.to_dict()
        s = json.dumps(d)  # must not raise
        reloaded = json.loads(s)
        assert reloaded["overall"] == "PASS"
        assert isinstance(reloaded["rows"], list)

    def test_to_dict_includes_na(self):
        sb = score(base_metrics(throughput_hz=None))
        d = sb.to_dict()
        json.dumps(d)  # still serializable with N/A present
        verdicts = {r["metric"]: r["verdict"] for r in d["rows"]}
        assert verdicts["throughput_hz"] == "N/A"


class TestRangeGate:
    def test_range_gate_pass(self):
        from ptbench.common.scoreboard import score
        board = score({"pos_error_range_m": {"median": 0.20}})
        row = [r for r in board.rows if r[0] == "pos_error_range_m"][0]
        assert row[2] == "PASS"

    def test_range_gate_warn(self):
        from ptbench.common.scoreboard import score
        board = score({"pos_error_range_m": {"median": 0.45}})
        row = [r for r in board.rows if r[0] == "pos_error_range_m"][0]
        assert row[2] == "WARN"

    def test_range_gate_fail(self):
        from ptbench.common.scoreboard import score
        board = score({"pos_error_range_m": {"median": 0.80}})
        row = [r for r in board.rows if r[0] == "pos_error_range_m"][0]
        assert row[2] == "FAIL"

    def test_range_gate_default_thresholds(self):
        from ptbench.common.scoreboard import GateConfig
        g = GateConfig()
        assert g.pos_error_range_pass_m == 0.30
        assert g.pos_error_range_warn_m == 0.50
