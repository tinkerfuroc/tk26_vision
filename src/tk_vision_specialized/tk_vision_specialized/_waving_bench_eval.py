"""rclpy-free evaluation core for the waving_bench scenario suite.

Parses a YAML-derived dict into CaseSpecs and judges DetectWaving responses
(already reduced to plain CallRecords) against per-case expectations.
Kept free of ROS imports so it runs under plain pytest.
"""
import math
from dataclasses import dataclass, field
from typing import Iterable

DEFAULTS = {
    "threshold_meters": 8.0,
    "target_frame": "",
    "min_waving_persons": 0,
    "calls_per_case": 5,
    "interval_sec": 1.0,
    "pass_ratio": 0.8,
}

REQUEST_KEYS = ("threshold_meters", "target_frame", "min_waving_persons")
_CASE_OVERRIDE_KEYS = ("calls_per_case", "interval_sec", "pass_ratio")
_ALLOWED_OVERRIDE_KEYS = frozenset(REQUEST_KEYS) | frozenset(_CASE_OVERRIDE_KEYS)
_ALLOWED_ORDERINGS = frozenset(("closest_first",))
ORDERING_TOLERANCE_M = 0.05


@dataclass(frozen=True)
class Expect:
    """A case's pass/fail criteria, mirroring the YAML `expect:` block."""

    status: int | None = None      # explicit override; when None, derived from count
    count: int | None = None
    z_range_m: tuple[float, float] | None = None
    ordering: str | None = None    # only "closest_first" is valid
    frame_id: str | None = None


_ALLOWED_EXPECT_KEYS = frozenset(f.name for f in Expect.__dataclass_fields__.values())


@dataclass(frozen=True)
class CaseSpec:
    """One fully-resolved sub-case: defaults + overrides merged, ready to run."""

    scenario: str
    index: int                     # 0-based within scenario
    prompt: str
    request: dict                  # exactly the REQUEST_KEYS, defaults+overrides merged
    calls_per_case: int
    interval_sec: float
    pass_ratio: float
    expect: Expect
    best_effort: bool = False


@dataclass(frozen=True)
class CallRecord:
    """A single DetectWaving response, already reduced to plain Python values."""

    status: int
    points: list                   # list of (x, y, z) tuples
    frame_ids: list                # parallel list of frame_id strings
    error_msg: str = ""


@dataclass(frozen=True)
class CallVerdict:
    """Pass/fail outcome for a single call against a CaseSpec."""

    passed: bool
    reasons: list = field(default_factory=list)   # empty when passed


@dataclass(frozen=True)
class CaseResult:
    """Aggregate pass/fail outcome for a case across its repeated calls."""

    case: CaseSpec
    n_passed: int
    n_calls: int
    passed: bool
    reasons: list = field(default_factory=list)   # distinct failure reasons, max 5


def load_suite(config: dict) -> dict[str, list[CaseSpec]]:
    """Parse a YAML-derived dict into scenario name -> list[CaseSpec]."""
    raw_defaults = config.get("defaults", {})
    unknown_defaults = set(raw_defaults) - set(DEFAULTS)
    if unknown_defaults:
        raise ValueError(f"Unknown default key(s): {sorted(unknown_defaults)}")
    defaults = {**DEFAULTS, **raw_defaults}

    suite: dict[str, list[CaseSpec]] = {}
    for scenario_name, case_dicts in config.get("scenarios", {}).items():
        suite[scenario_name] = [
            _parse_case(scenario_name, index, case_dict, defaults)
            for index, case_dict in enumerate(case_dicts)
        ]
    return suite


def _parse_case(scenario: str, index: int, case_dict: dict, defaults: dict) -> CaseSpec:
    prompt = case_dict.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"{scenario}[{index}]: 'prompt' is required and must be a str")

    overrides = case_dict.get("overrides", {})
    unknown_overrides = set(overrides) - _ALLOWED_OVERRIDE_KEYS
    if unknown_overrides:
        raise ValueError(
            f"{scenario}[{index}]: unknown overrides key(s) {sorted(unknown_overrides)}")

    request = {key: overrides.get(key, defaults[key]) for key in REQUEST_KEYS}
    calls_per_case = overrides.get("calls_per_case", defaults["calls_per_case"])
    interval_sec = overrides.get("interval_sec", defaults["interval_sec"])
    pass_ratio = overrides.get("pass_ratio", defaults["pass_ratio"])

    expect = _parse_expect(scenario, index, case_dict.get("expect", {}))

    return CaseSpec(
        scenario=scenario,
        index=index,
        prompt=prompt,
        request=request,
        calls_per_case=calls_per_case,
        interval_sec=interval_sec,
        pass_ratio=pass_ratio,
        expect=expect,
        best_effort=case_dict.get("best_effort", False),
    )


def _parse_expect(scenario: str, index: int, expect_dict: dict) -> Expect:
    unknown_expect = set(expect_dict) - _ALLOWED_EXPECT_KEYS
    if unknown_expect:
        raise ValueError(f"{scenario}[{index}]: unknown expect key(s) {sorted(unknown_expect)}")

    ordering = expect_dict.get("ordering")
    if ordering is not None and ordering not in _ALLOWED_ORDERINGS:
        raise ValueError(f"{scenario}[{index}]: invalid ordering {ordering!r}")

    z_range_m = expect_dict.get("z_range_m")
    if z_range_m is not None:
        if (not isinstance(z_range_m, (list, tuple)) or len(z_range_m) != 2
                or not (z_range_m[0] <= z_range_m[1])):
            raise ValueError(
                f"{scenario}[{index}]: z_range_m must be a 2-item ascending list")
        z_range_m = (float(z_range_m[0]), float(z_range_m[1]))

    return Expect(
        status=expect_dict.get("status"),
        count=expect_dict.get("count"),
        z_range_m=z_range_m,
        ordering=ordering,
        frame_id=expect_dict.get("frame_id"),
    )


def distance_of(point, frame_id: str) -> float:
    """Distance of `point` from the sensor, per Global Constraint 5.

    Frame `""` or any frame name ending `_optical_frame` -> `point.z`
    (camera-forward depth). Any other frame (e.g. `base_link`, `map`) ->
    `math.hypot(x, y)` (floor-plane distance).
    """
    x, y, z = point
    if frame_id == "" or frame_id.endswith("_optical_frame"):
        return z
    return math.hypot(x, y)


def expected_status(expect: Expect) -> int | None:
    """Derive the expected DetectWaving response status from `expect`."""
    if expect.status is not None:
        return expect.status
    if expect.count == 0:
        return 1
    if expect.count is not None and expect.count > 0:
        return 0
    return None


def evaluate_call(case: CaseSpec, call: CallRecord) -> CallVerdict:
    """Judge a single CallRecord against `case.expect`."""
    expect = case.expect
    frame_id = case.request["target_frame"]
    reasons = []

    # status=-1 is reserved for server/transport errors (e.g. the CLI's
    # timeout sentinel) and must fail the call regardless of `expect` —
    # unless the case explicitly expects a transport error itself.
    if call.status == -1 and expect.status != -1:
        reasons.append(
            f"transport/server error (status=-1): {call.error_msg or 'no response'}")
        return CallVerdict(passed=False, reasons=reasons)

    want_status = expected_status(expect)
    if want_status is not None and call.status != want_status:
        reasons.append(f"status: expected {want_status}, got {call.status}")

    if expect.count is not None and len(call.points) != expect.count:
        reasons.append(f"count: expected {expect.count}, got {len(call.points)}")

    if expect.z_range_m is not None:
        lo, hi = expect.z_range_m
        for point in call.points:
            d = distance_of(point, frame_id)
            if not (lo <= d <= hi):
                reasons.append(f"z_range_m: point distance {d:.3f} outside [{lo}, {hi}]")
                break

    if expect.ordering == "closest_first":
        distances = [distance_of(point, frame_id) for point in call.points]
        for prev, cur in zip(distances, distances[1:]):
            if cur < prev - ORDERING_TOLERANCE_M:
                reasons.append(f"ordering: expected closest_first, got {distances}")
                break

    if expect.frame_id is not None and any(
            fid != expect.frame_id for fid in call.frame_ids):
        reasons.append(
            f"frame_id: expected all {expect.frame_id!r}, got {call.frame_ids}")

    return CallVerdict(passed=not reasons, reasons=reasons)


def evaluate_case(case: CaseSpec, calls: list) -> CaseResult:
    """Aggregate per-call verdicts into a pass/fail CaseResult."""
    n_calls = len(calls)
    n_passed = 0
    distinct_reasons = []
    for call in calls:
        verdict = evaluate_call(case, call)
        if verdict.passed:
            n_passed += 1
            continue
        for reason in verdict.reasons:
            if reason not in distinct_reasons and len(distinct_reasons) < 5:
                distinct_reasons.append(reason)

    passed = n_calls > 0 and n_passed >= math.ceil(case.pass_ratio * n_calls)

    return CaseResult(
        case=case,
        n_passed=n_passed,
        n_calls=n_calls,
        passed=passed,
        reasons=distinct_reasons,
    )


def suite_passed(results: Iterable[CaseResult]) -> bool:
    """True iff every non-best_effort case passed (vacuously True if none)."""
    return all(result.passed for result in results if not result.case.best_effort)
