# TPT-Bench — dataset & offline scorer

**TPT-Bench** is a large-scale, long-term, robot-egocentric dataset for
benchmarking *target person tracking* (single target, follow-the-person), with
exhaustively annotated 2D bounding boxes of the target across 48 indoor/outdoor
sequences featuring frequent occlusions and crowds.

- Project page: <https://medlartea.github.io/tpt-bench/>
- Paper: arXiv **2505.07446** — *"TPT-Bench: A Large-Scale, Long-Term and
  Robot-Egocentric Dataset for Benchmarking Target Person Tracking"*
- Code/repo: <https://github.com/MedlarTea/TPT-Bench>

This sub-package (`ptbench.tpt_bench`) is an **offline scorer**: it runs
Tinker's `vision_track` YOLO tracker against a downloaded TPT-Bench sequence and
reports tracking precision / recall / F-score / AO / AMR. It is an **external
regression smoke-test**, not a Tinker integration test.

> **Sensor caveat.** TPT-Bench RGB-D frames were collected with a **ZED2 stereo
> camera** (plus a RICOH Theta Z1 panoramic camera for the primary annotated
> stream). Tinker's robot uses an **Orbbec** depth camera, so absolute numbers
> here measure the tracker's *appearance-tracking generalisation*, not its
> on-robot performance. Treat scores as relative regression signal across code
> changes, not as a hardware claim.

## Annotation format (confirmed from the paper, Sec. 4)

TPT-Bench uses the **LaSOT** single-target annotation standard:

- If the target appears in a frame, the box is the tightest up-right rectangle
  around any visible part of the target.
- Bounding boxes are stored as **`[u, v, w, h]`** — upper-left corner `(u, v)`
  plus `(width, height)`, one box per frame.
- When the target is **absent**, the box is written as `0,0,0,0` (LaSOT-style
  absent label).

The paper's evaluation metrics (Sec. 4.1), reproduced in `metrics.py`:

- **Tracking Precision** = correct predictions ÷ frames with a prediction
  (`N_p`). A box on an absent-GT frame is a false positive.
- **Tracking Recall** = correct predictions ÷ frames where the target exists
  (`N_g`).
- **F-score** = harmonic mean of precision and recall.
- **AO** (Average Overlap) = mean IoU over target-present frames at overlap
  threshold `tau_Omega = 0` (pred absent ⇒ IoU 0).
- **AMR** (Average Max Recall) = max recall achievable while precision stays at
  100%, averaged over IoU thresholds. Our tracker emits only a coarse per-frame
  confidence, so the scorer approximates AMR at a single IoU threshold by
  sweeping the confidence threshold and taking the max recall among thresholds
  that keep precision == 1.0 (see `metrics.py` docstring).

## Directory layout the loader expects

The loader (`dataset.load_sequence`) is tolerant of the common LaSOT-derived
release variants. Per **sequence** directory:

```
<seq_dir>/
  img/                 # frames, sorted lexically (zero-padded names)
    00000001.jpg       #   *.jpg / *.jpeg / *.png / *.bmp accepted
    00000002.jpg
    ...
  groundtruth.txt      # one "x,y,w,h" per line (comma OR whitespace delimited)
  absent.txt           # OPTIONAL: one 0/1 flag per line (1 = target absent)
```

Variants handled automatically:

- Frames may live directly under `<seq_dir>` if there is no `img/` subdir.
- Ground-truth file may be `groundtruth.txt` or `groundtruth_rect.txt`.
- The absence-flag file may be `absent.txt`, `out_of_view.txt`, or
  `full_occlusion.txt` (optional; when present its line count must match the
  ground-truth line count).
- Delimiters may be commas or whitespace.
- An absent frame is recognised either from a set absence flag **or** a
  `0,0,0,0` / zero-area / empty ground-truth line.

The loader raises `TptDatasetError` on missing files, unparseable lines, or
line/frame count mismatches.

### Format assumptions not verified online

The paper and project page confirm the LaSOT-style `[u,v,w,h]` convention and
the `0,0,0,0` absent label, but do **not** publish the exact on-disk *file
names* for every release split. The file-name variants above
(`groundtruth_rect.txt`, `out_of_view.txt`, `full_occlusion.txt`, optional
`img/` subdir) are implemented defensively from LaSOT precedent. If a real
download uses different names, extend `_GT_NAMES` / `_ABSENT_NAMES` /
`_IMAGE_EXTS` in `dataset.py`.

## How to obtain the data

**Do not auto-download** — TPT-Bench is multi-GB. Follow the download
instructions on the project page and clone/extract sequences manually:

- <https://medlartea.github.io/tpt-bench/>
- <https://github.com/MedlarTea/TPT-Bench>

Extract one or more sequence directories somewhere local, e.g.
`~/datasets/tpt-bench/<seq_name>/`.

## Running the scorer

The runner imports the heavy `vision_track` tracker, so source the workspace
first (the tracker lives at
`/home/tinker/tk25_ws/src/tk26_vision/src/vision_track`):

```bash
source /home/tinker/tk25_ws/install/setup.bash
cd /home/tinker/tk25_ws/src/tk26_vision/benchmarks/person_tracker

python -m ptbench.tpt_bench.score_cli \
    --seq ~/datasets/tpt-bench/<seq_name> \
    --iou 0.5 --imgsz 1280 \
    --json /tmp/tpt_<seq_name>.json
```

This prints an ASCII metrics table (precision / recall / f_score / ao / amr)
and, with `--json`, dumps the metrics plus run config.

The pure-logic modules (`dataset.py`, `metrics.py`) are unit-tested with
synthetic fixtures and need **no** dataset or model:

```bash
/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python \
    -m pytest tests/test_tpt_dataset.py tests/test_tpt_metrics.py -q
```
