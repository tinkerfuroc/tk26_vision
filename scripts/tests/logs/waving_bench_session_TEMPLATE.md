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
