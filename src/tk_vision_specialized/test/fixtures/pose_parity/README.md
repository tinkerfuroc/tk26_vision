# pose_parity fixture

Person crops + the mediapipe **0.10.9** `mp.solutions.pose` output
(`static_image_mode=True, min_detection_confidence=0.5, model_complexity=1`)
recorded by `scripts/tests/record_pose_fixture.py` on 2026-08-22.

`test_pose_parity.py` replays these through `_pose_backend.PoseBackend`
(mediapipe ≥ 1.0, Tasks API, `pose_landmarker_full.task`) and requires
identical `is_waving` verdicts and near-identical landmarks.

Sources: ultralytics `bus.jpg`, `zidane.jpg` (crops 00–05); live Orbbec frames
none — camera was not running when recorded.

Each crop entry in `expected_0.10.9.json` also carries a `source` key naming
the originating image filename; it is informational only and ignored by
`test_pose_parity.py`.

Do not regenerate under a newer mediapipe; the value of this fixture is that
it encodes the legacy behaviour.
