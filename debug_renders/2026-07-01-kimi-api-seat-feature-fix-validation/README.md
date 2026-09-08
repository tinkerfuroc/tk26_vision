# kimi_api seat_recommend_bbox + feature_extraction fix validation — 2026-07-01

Contact sheet validating the same-day fixes to `seat_recommend_bbox`
(`_seat_bbox_vlm.py`) and `feature_recognition.select_best_person_idx`
against **new** vision_log scenes captured after the original two bug
reports (`vision_log/20260701_203130/`, `vision_log/20260701_203616/`) —
not a re-run of the original bug-report images.

Built with `tkbuild tk26_vision --packages-select kimi_api` (build only;
this script calls the library functions directly, not the running node).

## What changed, validated here

**Seat recommendation** — prompt strengthened for partial-occupancy care
(cross-legged/reclined sitters) and seat suitability (prefer sofa/chair
over stool, widest cushion among ties), backed by a deterministic
`_best_unoccupied_seat` / `_seat_rank` re-rank, plus a self-consistency
check: recover when the model's `choice` names a seat its own `seats`
entry marks `occupied: true`.

**Feature extraction** — `select_best_person_idx` gates out candidates
below a minimum apparent size, then uses depth only as a near-tie
breaker (not an additive score term) with non-positive depth treated as
invalid rather than "0 m away".

## Files
- `contact_sheet.jpg` — 2×3 grid, top row seat recommendation, bottom row
  person selection
- `seat_panel_{1,2,3}.jpg`, `feature_panel_{1,2,3}.jpg` — individual panels
- `make_contact_sheet.py` — the script that produced everything in this dir

## Panel notes

- **SEAT 1/3, 2/3** — live `qwen3-vl-plus` calls via
  `request_seat_bbox_chain`, both correct on the first pass (no override
  needed): a genuinely empty sofa cushion in each scene.
- **SEAT 3/3** — replays a real captured `qwen3-vl-plus` response where
  `choice="left spot on sofa"` while that exact seat's own entry says
  `"occupied": true"` — a live self-contradiction, not a synthetic test
  case. Red box = what the model said (rejected); green box = what
  `select_box()` recovers to (`right spot on sofa`, the genuinely empty,
  non-stool cushion). Re-running the same live call several times shows
  this triggers intermittently (~1 in 4 calls) — the backstop is there for
  when it does.
- **PERSON 1/3, 3/3** — one obviously dominant subject; correct under both
  the old and new selection logic (sanity check, not a regression case).
- **PERSON 2/3** — the case that mattered: replaying real
  `object_detection_generalist` detections (5 persons) through an early
  version of the fix (additive `offset + weight·depth` score) picked a
  small, off-center, invalid-depth (`centroid.z == 0.0` sentinel) detection
  over this correct, dominant, real-depth (0.75 m) subject — the sentinel
  read as "closer than anything real". Fixed by making depth a tie-breaker
  only, and treating non-positive depth as invalid. This panel is the
  fixed, correct result.

## Reproduce

```bash
tkbuild tk26_vision --packages-select kimi_api
source /opt/ros/humble/setup.bash
source src/tk26_vision/.venv-vision-main/bin/activate
cd src/tk26_vision/debug_renders/2026-07-01-kimi-api-seat-feature-fix-validation
python3 make_contact_sheet.py
```

Seat panels 1/2 make live OpenRouter/DashScope calls (`OPENROUTER_API_KEY` /
`DASHCOPE_API_KEY` via the workspace-root `.env`); panel 3 and all feature
panels replay captured `vision_log/` JSON offline, no network needed.
