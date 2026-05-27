# Benchmark scripts for `object_match_all`

Two sibling scripts:

## `produce_match_ground_truth.py`

Generates ground-truth detections for a directory of scenes by running the
single-category VLM (today's `/object_match` production path) for every
`(scene, category)` pair. The output JSON is what
`benchmark_match_batch_size.py` scores against.

This is **VLM ground truth**, not human ground truth — it measures
agreement with the single-category service we already trust. See the
design doc for the rationale (spec §8.3.1).

### Usage

```bash
source /opt/ros/humble/setup.bash
source /home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/activate
source /home/tinker/tk25_ws/src/tk26_vision/install/setup.bash

python3 /home/tinker/tk25_ws/src/tk26_vision/src/tk_vision_specialized/scripts/produce_match_ground_truth.py \
    --scenes-dir /path/to/scenes \
    --items-dir  /home/tinker/tk25_ws/src/tk26_vision/src/items \
    --out        /tmp/gt_$(date +%Y%m%d_%H%M%S).json
```

Requires `DASHSCOPE_API_KEY` (or the typo'd `DASHCOPE_API_KEY`) in env or
`.env`.

Cost: `N_scenes * N_categories` single-category calls (~$1–3 per
regeneration on the default 10-scene × 10-item dataset).

Manual edits: the JSON is a plain dict; correct known-bad single-cat
predictions by editing the file before running the benchmark.

## `benchmark_match_batch_size.py`

Sweeps `batch_size` against the GT JSON and reports
precision/recall/F1/latency/token-cost per (provider, batch_size).

### Usage

```bash
python3 /home/tinker/tk25_ws/src/tk26_vision/src/tk_vision_specialized/scripts/benchmark_match_batch_size.py \
    --scenes-dir /path/to/scenes \
    --items-dir  /home/tinker/tk25_ws/src/tk26_vision/src/items \
    --ground-truth /tmp/gt_TIMESTAMP.json \
    --batch-sizes 1 2 3 5 8 \
    --provider qwen \
    --repeats 3 \
    --out-prefix /tmp/bench_$(date +%Y%m%d_%H%M%S)
```

Output: a CSV (one row per `(scene, provider, batch_size, repeat)`) plus
a Markdown summary with the recommended `batch_size` default.

The recommendation is **advisory**. Update the `batch_size` ROS
parameter in your launch params after reviewing the summary. The choice
doesn't auto-update.

### When to re-run

- Items added to or removed from `items_map.yaml`.
- VLM provider switch.
- Accuracy regression observed in T3/T4 integration tests.
