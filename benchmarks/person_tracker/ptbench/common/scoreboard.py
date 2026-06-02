"""PASS/WARN/FAIL gates + table + JSON dump over a metrics dict.

Verdict precedence (worst wins for the overall): PASS < WARN < FAIL. Metrics
whose value is None map to "N/A" and are excluded from the overall verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
NA = "N/A"

_RANK = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class GateConfig:
    correct_lock_rate_pass: float = 0.92
    correct_lock_rate_warn: float = 0.85
    reacquire_latency_pass_s: float = 1.0
    reacquire_latency_warn_s: float = 2.0
    pos_error_lateral_pass_m: float = 0.25
    pos_error_lateral_warn_m: float = 0.40
    false_target_rate_pass: float = 0.05
    false_target_rate_warn: float = 0.10
    throughput_pass_hz: float = 12.0
    throughput_warn_hz: float = 8.0


@dataclass
class Scoreboard:
    rows: List[Tuple[str, str, str]]
    overall: str

    def to_table(self) -> str:
        header = ("METRIC", "VALUE", "VERDICT")
        all_rows = [header] + list(self.rows) + [("OVERALL", "", self.overall)]
        w0 = max(len(r[0]) for r in all_rows)
        w1 = max(len(r[1]) for r in all_rows)
        w2 = max(len(r[2]) for r in all_rows)
        sep = "-" * (w0 + w1 + w2 + 6)

        def fmt(r):
            return f"{r[0]:<{w0}}  {r[1]:<{w1}}  {r[2]:<{w2}}"

        lines = [fmt(header), sep]
        lines += [fmt(r) for r in self.rows]
        lines.append(sep)
        lines.append(fmt(("OVERALL", "", self.overall)))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "rows": [
                {"metric": name, "value": value, "verdict": verdict}
                for name, value, verdict in self.rows
            ],
        }


def _verdict_higher_better(value, pass_thr, warn_thr) -> str:
    if value is None:
        return NA
    if value >= pass_thr:
        return PASS
    if value >= warn_thr:
        return WARN
    return FAIL


def _verdict_lower_better(value, pass_thr, warn_thr) -> str:
    if value is None:
        return NA
    if value <= pass_thr:
        return PASS
    if value <= warn_thr:
        return WARN
    return FAIL


def _verdict_zero_only(value) -> str:
    if value is None:
        return NA
    return PASS if value == 0 else FAIL


def _fmt_value(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def score(metrics: dict, gates: Optional[GateConfig] = None) -> Scoreboard:
    gates = gates or GateConfig()  # avoid a shared mutable default instance
    correct = metrics.get("correct_lock_rate")
    wrong_eps = metrics.get("wrong_lock_episodes")
    reacq = metrics.get("reacquire_latency_s")
    reacq_med = reacq.get("median") if isinstance(reacq, dict) else None
    pos_lat = metrics.get("pos_error_lateral_m")
    pos_lat_med = pos_lat.get("median") if isinstance(pos_lat, dict) else None
    false_rate = metrics.get("false_target_rate")
    throughput = metrics.get("throughput_hz")

    rows: List[Tuple[str, str, str]] = [
        (
            "correct_lock_rate",
            _fmt_value(correct),
            _verdict_higher_better(
                correct, gates.correct_lock_rate_pass, gates.correct_lock_rate_warn
            ),
        ),
        (
            "wrong_lock_episodes",
            _fmt_value(wrong_eps),
            _verdict_zero_only(wrong_eps),
        ),
        (
            "reacquire_latency_s",
            _fmt_value(reacq_med),
            _verdict_lower_better(
                reacq_med,
                gates.reacquire_latency_pass_s,
                gates.reacquire_latency_warn_s,
            ),
        ),
        (
            "pos_error_lateral_m",
            _fmt_value(pos_lat_med),
            _verdict_lower_better(
                pos_lat_med,
                gates.pos_error_lateral_pass_m,
                gates.pos_error_lateral_warn_m,
            ),
        ),
        (
            "false_target_rate",
            _fmt_value(false_rate),
            _verdict_lower_better(
                false_rate, gates.false_target_rate_pass, gates.false_target_rate_warn
            ),
        ),
        (
            "throughput_hz",
            _fmt_value(throughput),
            _verdict_higher_better(
                throughput, gates.throughput_pass_hz, gates.throughput_warn_hz
            ),
        ),
    ]

    non_na = [v for _, _, v in rows if v != NA]
    overall = max(non_na, key=lambda v: _RANK[v]) if non_na else NA

    return Scoreboard(rows=rows, overall=overall)
