# Person-Tracker Benchmark Harness (`ptbench`)

Validate the `vision_track` person tracker (`/track_person`, `PersonTrackNode` →
`YOLOTracker` ResNet50 + color ReID) against its real deployment domain — the
RoboCup@Home arena, single-target follow — via two tiers: **Tier A**, the
production gate, scores self-recorded **Orbbec rosbags** of Carry-My-Luggage
style follow scenarios against the node's real output contract (the one we
improve against); **Tier B** is the **TPT-Bench** external smoke-test, a relative
regression signal on a public dataset (ZED2-recorded, so it measures
appearance-tracking generalisation, not on-robot numbers). Full design,
contract, and rationale live in [`PLAN.md`](./PLAN.md).

## Scenario taxonomy (Tier A recordings)

Stage each clip Carry-My-Luggage style: one operator (the tracked target) plus
the listed distractors, recorded with the robot's Orbbec RGB-D.

- **`cml_crossing`** — operator walks while one or more bystanders cross the path
  between the operator and the camera; tests lock persistence through occluders.
- **`occlusion_reentry`** — operator leaves frame (or is fully occluded by a
  pillar/door) and re-enters; tests reacquire latency on the same identity.
- **`lookalike_distractors`** — one or more bystanders dressed similarly to the
  operator share the frame; tests that ReID does not jump to a look-alike
  (wrong-lock guard).
- **`back_to_camera`** — operator turns away / walks with their back to the
  robot for sustained stretches; tests tracking without a frontal appearance cue.
- **`range_lighting`** — operator moves across the usable depth range (near →
  far) and through lighting changes (bright → shadow); tests range accuracy and
  detection robustness under exposure shifts.

## Workflow

All `ptbench` commands run from `benchmarks/person_tracker/` so `import ptbench`
resolves via the `pyproject.toml` `pythonpath=["."]` setting (no install needed).
`<venv>` below is `/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main`.

### 1. Run the unit tests (no data / model / ROS needed)

The pure-logic modules are fully unit-tested with synthetic fixtures:

```bash
cd benchmarks/person_tracker
<venv>/bin/python -m pytest -q          # 171 tests
```

### 2. Label a recorded bag → `gt.json`

cv2 UI: step the color frames, draw/adjust the operator box, toggle
present/absent, save. On save it samples depth → 3D centroid and writes a
schema-valid `GtClip` to `<dir>/gt.json` (override with `--out`).

```bash
cd benchmarks/person_tracker
<venv>/bin/python -m ptbench.labeler.label_cli \
    --bag /path/to/cml_crossing_01 \
    --scenario cml_crossing
# → writes /path/to/cml_crossing_01/gt.json
```

Keys: `n`/`→` next, `p`/`←` prev, `space` toggle present, `b`/`r` (re)draw box,
`c` clear, `s` save, `q`/`Esc` quit. The box is copied forward as the default for
each untouched frame. Topic overrides: `--color-topic`, `--depth-topic`,
`--camera-info-topic`, `--clip-id`, `--notes`, `--fps-hint`.

### 3. Score one bag against its `gt.json`

Aligns the tracker's predictions to GT frames by timestamp, computes the
scoreboard metrics, scores them against the gates, and prints a PASS/WARN/FAIL
table (add `--json out.json` to dump the scoreboard dict):

```bash
cd benchmarks/person_tracker
<venv>/bin/python -m ptbench.replay.score_cli \
    --bag /path/to/cml_crossing_01 \
    --gt  /path/to/cml_crossing_01/gt.json \
    --backend offline
```

Backends: `offline` (default — drives `YOLOTracker` in-process, deterministic,
no ROS daemon) or `action` (replays onto a live `/track_person` server). Offline
tuning: `--imgsz` (default 1280), `--conf` (default 0.5).

### 4. Batch via the T4 tier

`scripts/tests/t4_hardware.sh follow_regression` scores every `<clip>/gt.json`
under the bags dir and folds the verdicts into the shared T4 `summary`. Point it
at your clips with `PTBENCH_BAGS_DIR` (default
`$WS_ROOT/benchmarks/person_tracker/bags`):

```bash
PTBENCH_BAGS_DIR=/path/to/labeled/bags \
    scripts/tests/t4_hardware.sh follow_regression
```

A clip passes when its scoreboard `OVERALL` row reads `PASS`; `WARN`/`FAIL` or a
scorer error counts as a failure. With no labeled bags present the case SKIPs
cleanly (exit 0) — the expected state until recordings exist. Per-clip scorer
output is captured to `scripts/tests/logs/T4.5_<clip>.out`.

### 5. TPT-Bench (Tier B external smoke-test)

The dataset is multi-GB and is **not** auto-downloaded. See
[`ptbench/tpt_bench/DOWNLOAD.md`](./ptbench/tpt_bench/DOWNLOAD.md) for how to
fetch it and the annotation-format details, then:

```bash
cd benchmarks/person_tracker
<venv>/bin/python -m ptbench.tpt_bench.score_cli \
    --seq /path/to/tpt-bench/<seq_name>
```

Reports precision / recall / f_score / ao / amr. Tuning: `--iou` (default 0.5),
`--imgsz` (default 1280), `--conf` (default 0.5), `--json out.json`.

## Gate table (Tier A production thresholds)

The overall verdict is the worst per-metric verdict.

| metric | PASS | WARN | else FAIL |
|---|---|---|---|
| `wrong_lock_episodes` | `== 0` | — | `≥ 1` |
| `correct_lock_rate` | `≥ 0.92` | `≥ 0.85` | `< 0.85` |
| `reacquire_latency_s` (median) | `≤ 1.0` | `≤ 2.0` | `> 2.0` |
| `pos_error_lateral_m` (median) | `≤ 0.25` | `≤ 0.40` | `> 0.40` |
| `false_target_rate` | `≤ 0.05` | `≤ 0.10` | `> 0.10` |
| `throughput_hz` | `≥ 12` | `≥ 8` | `< 8` |

## Note on the offline backend

The `offline` backend (and the `action` backend, and the TPT-Bench scorer)
import the real `vision_track` tracker and load the YOLO model, so the **ROS
workspace must be built and sourced** first:

```bash
source /home/tinker/tk25_ws/install/setup.bash
```

Always run the scorers from `benchmarks/person_tracker/` so `import ptbench`
resolves. The unit tests (`pytest`) are the only thing that needs **neither** the
workspace nor a model.
