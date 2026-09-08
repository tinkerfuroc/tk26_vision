# object_scan tuning WebUI

Standalone (ROS-free) harness to validate the **batched labels-only VLM scan**
on real photos before wiring it into `kimi_api/object_scan` + `BtNode_ObjectScan`
(design: `docs/superpowers/specs/2026-07-03-object-scan-design.md`).

`scan_core.py` is the **reference implementation** of the batching + Gemini→Qwen
fallback logic that later lifts into `kimi_api/_scan_vlm.py`.

## What it does

- **Shoot photos** — three ways:
  - browser webcam (with a camera picker),
  - **robot ROS camera** — a **live MJPEG preview** of a color topic (so you can
    aim) + one-click **Capture frame** (needs the camera launched + ROS sourced;
    this is the camera the real service uses),
  - drag/drop / upload photos you took elsewhere (phone).
  Saved under `./photos/`.
- **Run tests** — split the 32-class PickAndPlace vocabulary into batches, run
  one Gemini call per batch (Qwen fallback), **all batches in parallel**, union
  the results. Shows found labels on the vocab grid (green = found), a per-batch
  table (items / found / provider / latency), and total time.
- **Sweep (find the best batch_size)** — run several `batch_size` values, each
  **`repeats` times** (VLM answers vary run-to-run, so repeats reveal the
  *reliable* best), and compare avg-found + run-to-run range + latency.
  **★-click the vocab chips** to mark what's *actually* in the photo (ground
  truth) and the sweep ranks by **recall/precision/F1** instead of raw count, so
  false positives don't inflate the winner.

## Vocabulary

Read live from
`src/tk25_decision/src/behavior_tree/behavior_tree/PickAndPlace/constants.json`
(`table_scan_prompt`, 32 classes). No copy — edit it there and restart.

## Run

```bash
cd src/tk26_vision/scripts/object_scan_webui
./run.sh                       # -> http://127.0.0.1:8000
./run.sh --host 0.0.0.0 --port 8080   # reach it from another device on the LAN
```

Needs `OPENROUTER_API_KEY` (Gemini) and `DASHSCOPE_API_KEY` / `DASHCOPE_API_KEY`
(Qwen) in the workspace-root `.env` — the same keys the other kimi_api nodes use.
The server prints which keys it found at startup.

> Webcam capture needs a secure context: `localhost` is fine. If you open the UI
> over the LAN by IP, browsers block `getUserMedia` on plain HTTP — use the
> upload path instead (or an SSH tunnel to localhost).

## CLI (no browser)

```bash
../../.venv-vision-main/bin/python scan_core.py photo.jpg --batch-size 8
../../.venv-vision-main/bin/python scan_core.py photo.jpg --sweep 4,8,16
```

## Notes

- Runs under `.venv-vision-main` (needs only `openai` + `python-dotenv`, already
  installed). No new pip installs, no ROS build.
- Models: `google/gemini-2.5-flash` (OpenRouter) → `qwen3-vl-plus` (DashScope),
  matching `object_detection_generalist`.
- Labels only — no bboxes/masks/depth, matching the approved design.
