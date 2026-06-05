# Offline person-tracking scorer — TPT-Bench (intended) / LaSOT `person` (realized)

This sub-package (`ptbench.tpt_bench`) is an **offline scorer**: it runs Tinker's
`vision_track` YOLO tracker against a downloaded LaSOT-style single-target
sequence and reports tracking precision / recall / F-score / AO / AMR. It is an
**external regression smoke-test**, not a Tinker integration test.

## Reality check (2026-06-05): TPT-Bench is download-blocked → use LaSOT `person`

The originally-intended Tier-B set was **TPT-Bench** (arXiv **2505.07446**,
robot-egocentric target-person tracking). Its data is published **only** on
OneDrive (pw `rcvtptbench`) and Baidu (pw `pf25`) — both browser/password-gated;
the `1drv.ms` link returns HTTP 403 under `curl`, so it is **not scriptable**
from the workstation. The real release layout is also `panoramic_images/<seq>/` +
`GTs/<seq>.json`, **not** the LaSOT `img/`+`groundtruth.txt` this loader expects.

So the **realized** external benchmark is **LaSOT's `person` category** — 20
single-target person sequences, directly downloadable from HuggingFace (no auth),
and a genuine drop-in for this scorer. The two are interchangeable for our
purpose (single-target, first-frame-seeded, LaSOT annotation standard); LaSOT is
3rd-person web video rather than robot-egocentric, so treat the numbers as an
**appearance-tracking / ReID + occlusion-recovery regression signal**, not an
on-robot (Orbbec) performance claim.

> **What the offline scorer does and does NOT exercise.** The runner constructs
> `YOLOTracker` directly, so it measures the **tracker core** (YOLO + ByteTrack +
> OSNet-AIN/MSMT17 ReID + identity gates). The node-level `LockStateMachine`
> recovery FSM and the depth-gated crosser rejection are attached/driven by
> `person_track_node` and are **depth-dependent** (the node only publishes a
> target when depth yields a 3D position), so they are NOT in this RGB-only
> path. `run_lasot_person_benchmark.py --fsm` attaches the FSM for an ablation
> (depth permissive); fully testing the depth half needs Tier-A Orbbec bags.

## How to obtain the data (LaSOT `person`)

Directly downloadable from the HuggingFace mirror `l-lt/LaSOT` (no login, no
EULA). The whole `person` category is one zip (4.29 GB, all 20 sequences):

```bash
mkdir -p ~/datasets/lasot && cd ~/datasets/lasot
curl -L -o person.zip \
  "https://huggingface.co/datasets/l-lt/LaSOT/resolve/main/person.zip?download=true"
unzip -q person.zip          # -> person-1/ ... person-20/
```

The official Protocol-II **test** split for this category is
`person-1`, `person-5`, `person-10`, `person-12`.

## Annotation format (VERIFIED against the real LaSOT release)

Per **sequence** directory:

```
<seq_dir>/
  img/                 # frames 00000001.jpg ...  (sorted lexically, zero-padded)
  groundtruth.txt      # one "x,y,w,h" per line (comma-delimited)
  full_occlusion.txt   # absence flags — SINGLE comma-separated line of 0/1
  out_of_view.txt      # absence flags — SINGLE comma-separated line of 0/1
  nlp.txt              # natural-language description (ignored)
```

Real-format details the loader handles (confirmed by inspecting `person-*`):

- **Absence flag files are a single comma-separated line** of 0/1 (one per
  frame), NOT one flag per line. LaSOT ships **both** `out_of_view.txt` and
  `full_occlusion.txt`; a frame is "absent" if **either** is set — the loader
  unions every flag file that exists. (`_read_flags` + the union loop in
  `dataset.py`.)
- In this release, fully-occluded frames also carry a `0,0,0,0` ground-truth
  box, which the loader independently maps to absent (zero-area ⇒ `None`).
- The loader is also tolerant of variants: frames directly under `<seq_dir>`,
  `groundtruth_rect.txt`, comma OR whitespace delimiters, one-flag-per-line.

Metrics (paper Sec. 4.1, implemented in `metrics.py`): **Precision** = correct ÷
predicted frames; **Recall** = correct ÷ target-present frames; **F-score** =
their harmonic mean; **AO** = mean IoU over present frames (pred absent ⇒ 0);
**AMR** = max recall while precision stays 1.0 (swept over the per-frame score).

The loader raises `TptDatasetError` on missing files, unparseable lines, or
line/frame count mismatches.

## Running the scorer

The runner imports the heavy `vision_track` tracker. The reproducible driver
derives all paths from its own location (no workspace sourcing needed) and runs
with **production-faithful** settings (imgsz=736, conf=0.5; OSNet-AIN/MSMT17 +
fp16 + yolo_track_conf=0.15 are `YOLOTracker` defaults):

```bash
VENV=/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python

# all 20 person sequences (tracker core):
$VENV benchmarks/person_tracker/demo/run_lasot_person_benchmark.py --json /tmp/lasot.json
# just the Protocol-II test split:
$VENV benchmarks/person_tracker/demo/run_lasot_person_benchmark.py \
    --seqs person-1 person-5 person-10 person-12
# FSM ablation (attach node LockStateMachine; depth permissive on RGB):
$VENV benchmarks/person_tracker/demo/run_lasot_person_benchmark.py --fsm \
    --seqs person-1 person-5 person-10 person-12
```

Or score a single sequence with the lower-level CLI (note `--imgsz 736`):

```bash
cd benchmarks/person_tracker
$VENV -m ptbench.tpt_bench.score_cli --seq ~/datasets/lasot/person-1 \
    --iou 0.5 --imgsz 736 --json /tmp/person-1.json
```

## Latest run (2026-06-05, all 20 person sequences, tracker core)

Mean **P=0.928 · R=0.490 · F=0.559 · AO=0.422 · AMR=0.279 @ ~29 Hz**. Bimodal:
8/20 excellent (F≥0.9), 9/20 poor (F<0.4). Of the 9 poor, **7 are
"lost-conservative"** (P≥0.85, very low recall — loses the target and refuses to
re-lock, ~0 false targets) and **only 2 are wrong-lock** (P≈0.5). Headline:
**precision 0.93 with only 2/20 wrong-lock** validates the overhaul's primary
goal (kill wrong-locks); the open frontier is **reacquisition recall (0.49)**.
FSM ablation: cuts wrong-locks (person-12 precision 0.50→0.61) but does not fix
the conservative reacquire (driven by the distinctiveness gate upstream of the
FSM). See `../../demo/lasot_benchmark_contact_sheet.png` for GT-vs-pred frames.

## Pure-logic unit tests (no dataset / model needed)

```bash
cd benchmarks/person_tracker
/home/tinker/tk25_ws/src/tk26_vision/.venv-vision-main/bin/python \
    -m pytest tests/test_tpt_dataset.py tests/test_tpt_metrics.py -q
```
