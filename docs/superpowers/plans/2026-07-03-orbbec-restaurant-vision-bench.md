# Implementation Plan: Orbbec-Only Restaurant Vision Bench (Tiers 0–2)

- **Date:** 2026-07-03
- **Design:** `docs/superpowers/specs/2026-07-03-orbbec-only-restaurant-vision-bench-design.md` (approved)
- **Repos touched:** `src/tk26_vision` (Tasks 1–2), `src/tk25_decision` (Task 3) — two separate git repos, both on branch `tinker2-net`.
- **Execution:** subagent-driven, sequential tasks.

## Global Constraints

1. **Commits:** author every commit as `git commit --author="Ccindy0171 <cindy.w0135@gmail.com>"`. End every commit message with these two trailer lines:
   ```
   Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
   Claude-Session: https://claude.ai/code/session_012gBi9MRKBNiDyrcyQACeyg
   ```
2. **Do not touch or stage pre-existing dirty files.** Stage only the files your task creates or edits, by explicit path (`git add <path> <path>`; never `git add -A`/`.`). Off-limits WIP: tk25_decision `src/behavior_tree/setup.py`, `src/behavior_tree/behavior_tree/PickAndPlace/pick_and_place_rulebook.py`, `src/behavior_tree/behavior_tree/Restaurant/test_order_confirm.py`; tk26_vision `.claude/worktrees/static-disk-ref`.
3. **`_waving_bench_eval.py` and `test_waving_bench_eval.py` import no ROS modules** — no `rclpy`, no `*_msgs`, no `ament_index_python`. Pure Python 3.10 + PyYAML only.
4. **Exact suite defaults:** `threshold_meters: 8.0`, `target_frame: ""`, `min_waving_persons: 0`, `calls_per_case: 5`, `interval_sec: 1.0`, `pass_ratio: 0.8`.
5. **Distance semantics:** frame `""` or any frame name ending `_optical_frame` → distance is `point.z`; any other frame → `math.hypot(x, y)`. `closest_first` ordering check uses tolerance 0.05 m (non-decreasing within tolerance).
6. **`vision_live_bench.json` differs from `full_mock.json` by exactly two things:** the top-level `description` string, and `mock_mode.subsystems.vision.nodes.BtNode_ScanForWavingPerson` set to `"NO_MOCK"`. Nothing else.
7. **`test_scan.py` change is exactly one kwarg added** to the `BtNode_ScanForWavingPerson(...)` call in `scan_once`; no other edits to that file (the pan-angle list stays as-is on purpose).
8. Python 3.10. Match the surrounding code style of each package. No new pip dependencies (PyYAML is already available; if `import yaml` fails in the venv, stop and report instead of installing anything).

---

## Task 1: rclpy-free scenario suite — eval helper, YAML, pytest (TDD)

**Repo:** `/home/tinker/tk25_ws/src/tk26_vision` (all paths below relative to workspace root `/home/tinker/tk25_ws`).

**Files to create:**
1. `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/_waving_bench_eval.py`
2. `src/tk26_vision/src/tk_vision_specialized/config/waving_bench.yaml` (new `config/` dir in the package)
3. `src/tk26_vision/src/tk_vision_specialized/test/test_waving_bench_eval.py`

**TDD is required**: write failing tests first (RED), then implement (GREEN). Record both in your report.

### 1a. `_waving_bench_eval.py` — module API (implement exactly this)

```python
"""rclpy-free evaluation core for the waving_bench scenario suite.

Parses a YAML-derived dict into CaseSpecs and judges DetectWaving responses
(already reduced to plain CallRecords) against per-case expectations.
Kept free of ROS imports so it runs under plain pytest.
"""
from dataclasses import dataclass, field

DEFAULTS = {
    "threshold_meters": 8.0,
    "target_frame": "",
    "min_waving_persons": 0,
    "calls_per_case": 5,
    "interval_sec": 1.0,
    "pass_ratio": 0.8,
}

REQUEST_KEYS = ("threshold_meters", "target_frame", "min_waving_persons")
ORDERING_TOLERANCE_M = 0.05

@dataclass(frozen=True)
class Expect:
    status: int | None = None      # explicit override; when None, derived from count
    count: int | None = None
    z_range_m: tuple[float, float] | None = None
    ordering: str | None = None    # only "closest_first" is valid
    frame_id: str | None = None

@dataclass(frozen=True)
class CaseSpec:
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
    status: int
    points: list                   # list of (x, y, z) tuples
    frame_ids: list                # parallel list of frame_id strings
    error_msg: str = ""

@dataclass(frozen=True)
class CallVerdict:
    passed: bool
    reasons: list = field(default_factory=list)   # empty when passed

@dataclass(frozen=True)
class CaseResult:
    case: CaseSpec
    n_passed: int
    n_calls: int
    passed: bool
    reasons: list = field(default_factory=list)   # distinct failure reasons, max 5
```

Functions:

- `load_suite(config: dict) -> dict[str, list[CaseSpec]]`
  - Merge `config.get("defaults", {})` over `DEFAULTS` (unknown default keys → `ValueError`).
  - `config["scenarios"]` maps scenario name → list of case dicts. Each case dict has `prompt` (required str), optional `overrides` (only REQUEST_KEYS plus `calls_per_case`/`interval_sec`/`pass_ratio` allowed → else `ValueError`), optional `expect` dict (keys limited to Expect fields → else `ValueError`), optional `best_effort` bool.
  - `expect.ordering` other than `"closest_first"` → `ValueError`. `z_range_m` must be a 2-item ascending list → else `ValueError`.
- `distance_of(point, frame_id: str) -> float` — per Global Constraint 5.
- `expected_status(expect: Expect) -> int | None` — explicit `expect.status` wins; else `1` if `expect.count == 0`; else `0` if `expect.count` and `expect.count > 0`; else `None` (no status check).
- `evaluate_call(case: CaseSpec, call: CallRecord) -> CallVerdict` — checks, each contributing a human-readable reason string on failure:
  1. status (via `expected_status`),
  2. count: `len(call.points) == expect.count` (when count is not None),
  3. z_range_m: `distance_of` of EVERY point within `[lo, hi]`,
  4. ordering `closest_first`: `distance_of` sequence non-decreasing within `ORDERING_TOLERANCE_M`,
  5. frame_id: every entry of `call.frame_ids` equals `expect.frame_id` (when set).
- `evaluate_case(case: CaseSpec, calls: list[CallRecord]) -> CaseResult` — `passed` iff `n_passed >= math.ceil(case.pass_ratio * len(calls))` and `len(calls) > 0`; `reasons` = first 5 distinct failure reasons across calls.
- `suite_passed(results: Iterable[CaseResult]) -> bool` — `all(r.passed for r in results if not r.case.best_effort)`; vacuously True.

### 1b. `config/waving_bench.yaml` — exact content

```yaml
# Scenario suite for `ros2 run tk_vision_specialized waving_bench`.
# Bench rig + operator protocol: docs/superpowers/specs/
# 2026-07-03-orbbec-only-restaurant-vision-bench-design.md §4.
# Distances are floor-tape marks measured from the camera plane.

defaults:
  threshold_meters: 8.0        # match production DETECT_WAVING_THRESHOLD_M
  target_frame: ""             # camera optical frame unless overridden
  min_waving_persons: 0        # VLM fallback off unless a scenario opts in
  calls_per_case: 5
  interval_sec: 1.0
  pass_ratio: 0.8              # >=4/5 calls must satisfy the case expectation

scenarios:
  smoke:
    - prompt: "one person, wave at 2 m"
      expect: {count: 1, z_range_m: [1.7, 2.3]}
  range_ladder:
    - {prompt: "wave at 1 m", expect: {count: 1, z_range_m: [0.7, 1.3]}}
    - {prompt: "wave at 2 m", expect: {count: 1, z_range_m: [1.7, 2.3]}}
    - {prompt: "wave at 4 m", expect: {count: 1, z_range_m: [3.7, 4.3]}}
    - {prompt: "wave at 6 m", expect: {count: 1, z_range_m: [5.6, 6.4]}}
    - {prompt: "wave at 8 m", expect: {count: 1, z_range_m: [7.5, 8.5]}, best_effort: true}
  gesture_matrix:
    - {prompt: "static raised hand (above head)", expect: {count: 1}}
    - {prompt: "hand raised above elbow, elbow at shoulder height", expect: {count: 1}}
    - {prompt: "arms down, walk around", expect: {status: 1, count: 0}}
    - {prompt: "point sideways at shoulder height", expect: {status: 1, count: 0}}
  two_person_arbitration:
    - prompt: "A waves at 2 m, B waves at 4 m"
      expect: {count: 2, ordering: closest_first}
    - prompt: "A idle at 2 m, B waves at 4 m"
      expect: {count: 1, z_range_m: [3.7, 4.3]}
  threshold_gate:
    - prompt: "wave at 4 m"
      overrides: {threshold_meters: 3.0}
      expect: {status: 1, count: 0}
  frames:                       # requires the static-TF shim (design §4.2)
    - prompt: "wave at 2 m"
      overrides: {target_frame: base_link}
      expect: {count: 1, frame_id: base_link}
  vlm_fallback:                 # requires DASHSCOPE/OPENROUTER key + network
    - prompt: "one person waves clearly, second sits waving small/far"
      overrides: {min_waving_persons: 2}
      expect: {count: 2}
      best_effort: true
```

### 1c. `test_waving_bench_eval.py` — required coverage

Plain pytest, no ROS. Cover at minimum:
1. defaults applied when a case has no overrides (threshold 8.0, calls 5, ratio 0.8);
2. overrides win (threshold_meters, target_frame, pass_ratio);
3. `distance_of`: `""` and `camera_color_optical_frame` → z; `base_link`/`map` → hypot(x,y);
4. `expected_status`: derived 1 from count 0, derived 0 from count 2, explicit status wins, None when no count/status;
5. count mismatch → failed verdict with a reason naming expected vs got;
6. z_range boundary: point at exactly lo and hi passes, outside fails;
7. ordering: [2.0, 4.0] passes; [4.0, 2.0] fails; [2.00, 1.96] passes (within 0.05 tolerance);
8. frame_id mismatch fails;
9. pass_ratio boundary at defaults: 4/5 passing calls → case passes, 3/5 → fails; zero calls → fails;
10. best_effort failing case does not fail `suite_passed`; non-best_effort failing case does;
11. `ValueError` on: unknown expect key, unknown overrides key, `ordering: farthest_first`, descending `z_range_m`, unknown defaults key;
12. the shipped `config/waving_bench.yaml` (load with `yaml.safe_load`, path relative to the test file: `../config/waving_bench.yaml`) parses through `load_suite` and yields exactly the scenario names {smoke, range_ladder, gesture_matrix, two_person_arbitration, threshold_gate, frames, vlm_fallback}; the 8 m range_ladder case and the vlm_fallback case are `best_effort`; the frames case expects `frame_id == "base_link"`.

**Test command** (no ROS sourcing needed), run from `/home/tinker/tk25_ws`:
```bash
src/tk26_vision/.venv-vision-main/bin/python -m pytest \
    src/tk26_vision/src/tk_vision_specialized/test/test_waving_bench_eval.py -q
```

**Commit** (in `/home/tinker/tk25_ws/src/tk26_vision`): the three new files only. Suggested message: `Add rclpy-free waving_bench eval core + scenario suite (design 2026-07-03 bench §4.3)`.

---

## Task 2: `waving_bench` ROS CLI, packaging, session-log template, build

**Repo:** `/home/tinker/tk25_ws/src/tk26_vision`. Depends on Task 1 (its module API is final; do not modify Task 1 files except — if genuinely needed — adding a helper is NOT allowed; report instead).

**Files:**
1. Create `src/tk26_vision/src/tk_vision_specialized/tk_vision_specialized/waving_bench.py`
2. Edit `src/tk26_vision/src/tk_vision_specialized/setup.py`: add data_files entry `('share/' + package_name + '/config', glob('config/*.yaml'))` and console script `'waving_bench = tk_vision_specialized.waving_bench:main'`
3. Create `src/tk26_vision/scripts/tests/logs/waving_bench_session_TEMPLATE.md`

### 2a. `waving_bench.py` behavior

Interactive scenario client for `/detect_waving_persons` (`tinker_vision_msgs_26/srv/DetectWaving`). Model the node/client structure on the existing `waving_client.py` (same package); all judging goes through `_waving_bench_eval` — no evaluation logic in this file.

argparse (parse BEFORE `rclpy.init` so `--help` works with no ROS graph):
- `--scenario NAME` — repeatable (`action="append"`); run the named scenario(s)
- `--all` — run every scenario in the suite
- `--config PATH` — suite YAML; default: `ament_index_python.get_package_share_directory('tk_vision_specialized') + '/config/waving_bench.yaml'`, falling back (on `PackageNotFoundError` / missing file) to the source-tree copy `Path(__file__).resolve().parent.parent / 'config' / 'waving_bench.yaml'`
- `--out-dir PATH` — default `/tmp/waving_bench_<YYYYmmdd_HHMMSS>`; created; results written to `<out-dir>/results.jsonl`
- `--service NAME` — default `/detect_waving_persons`
- `--calls N` — override `calls_per_case` for every case (quick smoke)
- `--yes` — non-interactive: skip operator prompts (still runs all selected cases)

No `--scenario` and no `--all` → print available scenario names and exit 2.

Flow: load+`load_suite` the YAML → `rclpy.init` → node `waving_bench` → `wait_for_service(timeout_sec=5.0)`; unavailable → single clear error line, exit 1 (no traceback). Per case: print `[scenario/index] prompt`; unless `--yes`, `input("Position the scene, then press Enter (s=skip, q=quit): ")` — `s` records the case as skipped (excluded from the suite verdict), `q` stops the run (summary still written). Fire `calls_per_case` requests at `interval_sec` spacing, building `DetectWaving.Request` from `case.request`. Convert each response with a module-level function:

```python
def record_from_response(resp) -> CallRecord:
    return CallRecord(
        status=resp.status,
        points=[(p.point.x, p.point.y, p.point.z) for p in resp.waving_persons],
        frame_ids=[p.header.frame_id for p in resp.waving_persons],
        error_msg=resp.error_msg,
    )
```

Evaluate per call (`evaluate_call`) and per case (`evaluate_case`); print one line per call (`✓`/`✗`, status, n points, distances to 2 dp, reasons on failure) and a case summary line.

`results.jsonl`: one JSON object per call `{"ts": <iso8601>, "scenario", "case_index", "prompt", "request", "status", "error_msg", "points": [{"x","y","z","frame_id"}], "passed", "reasons"}`; after each case `{"type": "case_summary", "scenario", "case_index", "n_passed", "n_calls", "passed", "best_effort", "skipped"}`; at the end `{"type": "suite_summary", "passed": <bool>, "cases_run": N, "cases_skipped": N}`.

Exit code: 0 iff `suite_passed` over the non-skipped results, else 1.

### 2b. `waving_bench_session_TEMPLATE.md` — content

```markdown
# Waving bench session — YYYY-MM-DD

Copy this file to `waving_bench_session_<date>.md` and fill in. One file per rig session.
Protocol: docs/superpowers/specs/2026-07-03-orbbec-only-restaurant-vision-bench-design.md

## Rig
- Tripod height (m, floor→camera_link):
- Down-tilt (rad, positive = down):
- Resolution (720p default / 1080p): 
- `ros2 topic hz /camera/color/image_raw` (expect ~30):
- TF shim running (y/n + the two static_transform_publisher command lines used):
- VLM fallback: off (default) / keyed run

## Results
| Scenario | Cases passed | Notes |
|---|---|---|
| smoke |  |  |
| range_ladder |  |  |
| gesture_matrix |  |  |
| two_person_arbitration |  |  |
| threshold_gate |  |  |
| frames |  |  |
| vlm_fallback (best-effort) |  |  |

- `results.jsonl` archived at:
- Frames copied into `detect_waving_test/{waving,not_waving}/` (corpus growth rule, design §3):
```

### 2c. Build + acceptance

From `/home/tinker/tk25_ws`:
```bash
./src/tk26_vision/scripts/build.sh --packages-select tk_vision_specialized
source install/setup.bash
ros2 run tk_vision_specialized waving_bench --help            # exit 0
ros2 run tk_vision_specialized waving_bench                    # exit 2, lists scenarios
ros2 run tk_vision_specialized waving_bench --all --yes        # exit 1, one clean "service not available" line, NO traceback
```
Also confirm the installed share config resolves: the exit-2 run must list the 7 scenario names (proves the YAML was found via ament share). Re-run the Task 1 pytest once (unchanged code, cheap regression).

**Commit** (in `/home/tinker/tk25_ws/src/tk26_vision`): the two new files + setup.py. Suggested message: `Add waving_bench scenario CLI + packaging + bench session template`.

---

## Task 3: tk25_decision — vision_live_bench.json + test_scan crop patch

**Repo:** `/home/tinker/tk25_ws/src/tk25_decision`. Reminder: `src/behavior_tree/setup.py` has unrelated uncommitted WIP — do not touch or stage it (no new entry points are needed; `test_restaurant_scan` already exists).

**Files:**
1. Create `src/tk25_decision/src/behavior_tree/config/vision_live_bench.json`: byte-for-byte copy of the sibling `full_mock.json` with exactly two changes (Global Constraint 6):
   - top-level `description`: `"Tier-2 Orbbec bench config, derived from full_mock.json. Delta: vision.BtNode_ScanForWavingPerson=NO_MOCK (live detect_waving_persons service); everything else stays mocked and keyboard stays disabled so KEYPRESS auto-advances. Use: BT_MOCK_CONFIG=<this file> ros2 run behavior_tree test_restaurant_scan. See tk26_vision docs/superpowers/specs/2026-07-03-orbbec-only-restaurant-vision-bench-design.md §5."`
   - `mock_mode.subsystems.vision.nodes.BtNode_ScanForWavingPerson`: `"KEYPRESS"` → `"NO_MOCK"`
2. Edit `src/tk25_decision/src/behavior_tree/behavior_tree/Restaurant/test_scan.py`: in `scan_once`, add the kwarg `bb_key_pictures="test_scan_waving_pictures",` to the `BtNode_ScanForWavingPerson(...)` call (after `min_waving_persons=2`). Exactly this one change (Global Constraint 7).

### Verification (all from `/home/tinker/tk25_ws`)

1. **Delta check** — a throwaway Python snippet loading both JSONs and asserting the parsed objects are identical after (a) deleting both `description` fields and (b) setting `BtNode_ScanForWavingPerson` to the same value; and asserting the new file has `"NO_MOCK"` there. Paste the snippet + output in your report.
2. **Mock resolution check** (headless, no ROS graph needed): `source install/setup.bash`, then run python asserting, with `BT_MOCK_CONFIG=/home/tinker/tk25_ws/src/tk25_decision/src/behavior_tree/config/vision_live_bench.json` and `BT_MOCK_MODE` unset:
   - node mocked: `BtNode_TurnPanTilt` → True, `BtNode_Announce` → True
   - `BtNode_ScanForWavingPerson` → False
   Check `behavior_tree/config.py` for the exact import (there is a module-level `is_node_mocked` used by the template nodes; if the name differs, use what the template nodes import). The check runs against the *installed* behavior_tree (config code is unchanged by this task), so it works before the rebuild.
3. **Rebuild** so the installed `test_scan.py` picks up the patch: `./tkbuild tk25_decision --packages-select behavior_tree` from the workspace root (this wrapper refreshes the ROOT install with correct shebangs — do not use the mini-workspace build script). If `tkbuild` is missing, fall back to `colcon build --packages-select behavior_tree` from `/home/tinker/tk25_ws`. Then `grep -n 'bb_key_pictures' install/behavior_tree/lib/python3.10/site-packages/behavior_tree/Restaurant/test_scan.py` (adjust path if the grep shows the install layout differs) must hit.

**Commit** (in `/home/tinker/tk25_ws/src/tk25_decision`): the new JSON + test_scan.py only. Suggested message: `Add vision_live_bench mock config + exercise crop path in test_scan (Orbbec bench §5)`.
