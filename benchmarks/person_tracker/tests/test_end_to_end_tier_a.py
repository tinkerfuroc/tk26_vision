"""End-to-end Tier-A cohesion test — model-free, no bag, no live tracker.

Proves the production Tier-A loop composes *today*, before any arena recordings
exist: ``labeler.build_gt_clip`` (GT authoring) -> ``save_gt``/``load_gt``
(schema validation boundary) -> ``align`` -> ``metrics`` -> ``scoreboard``.
The tracker half (YOLO model / rosbag) is stubbed with fabricated PredFrames, so
this runs on pure synthetic fixtures and locks the cross-module contract that the
per-module tests only cover in isolation.
"""
import numpy as np

from ptbench.common.align import PredFrame
from ptbench.common.schema import load_gt, save_gt
from ptbench.labeler.label_io import FrameAnnotation, build_gt_clip
from ptbench.replay.score_cli import score_preds

S = 1_000_000_000  # 1 second in ns


def _K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


def _depth(depth_m=2.0, bbox=None, h=480, w=640):
    arr = np.zeros((h, w), dtype=np.uint16)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        arr[y1:y2, x1:x2] = int(depth_m * 1000)
    return arr


def _clip(tmp_path, anns, depth_list, scenario):
    K = _K()
    clip = build_gt_clip(
        anns,
        depth_list,
        K,
        clip_id="e2e",
        bag_path="bags/e2e",
        scenario=scenario,
        color_topic="/camera/color/image_raw",
        depth_topic="/camera/depth/image_raw",
        camera_info_topic="/camera/color/camera_info",
    )
    # Round-trip through the schema validation boundary the scorer relies on.
    out = tmp_path / "gt.json"
    save_gt(clip, out)
    return load_gt(out)


def test_tier_a_full_loop_perfect_predictions_pass(tmp_path):
    bbox0 = (100, 80, 200, 160)
    bbox1 = (140, 80, 240, 160)
    depth_list = [
        (0 * S, _depth(bbox=bbox0)),
        (1 * S, _depth(bbox=bbox1)),
        (2 * S, _depth()),  # operator absent this frame
    ]
    anns = [
        FrameAnnotation(t_ns=0 * S, present=True, bbox=bbox0),
        FrameAnnotation(t_ns=1 * S, present=True, bbox=bbox1),
        FrameAnnotation(t_ns=2 * S, present=False, bbox=None),
    ]
    clip = _clip(tmp_path, anns, depth_list, "cml_crossing")

    assert len(clip.frames) == 3
    assert clip.frames[0].centroid_3d is not None
    assert clip.frames[2].present is False

    # Perfect tracker: locks exactly onto each GT centroid; reports lost while
    # the operator is absent (so no false target).
    preds = []
    for f in clip.frames:
        if f.present and f.centroid_3d is not None:
            preds.append(
                PredFrame(t_ns=f.t_ns, target_lost=False, target_track_id=7,
                          point_xyz=tuple(f.centroid_3d))
            )
        else:
            preds.append(
                PredFrame(t_ns=f.t_ns, target_lost=True, target_track_id=-1,
                          point_xyz=None)
            )

    board = score_preds(preds, clip, throughput_hz=15.0)
    verdicts = {r["metric"]: r["verdict"] for r in board.to_dict()["rows"]}
    assert verdicts["correct_lock_rate"] == "PASS"
    assert verdicts["wrong_lock_episodes"] == "PASS"
    assert verdicts["false_target_rate"] == "PASS"
    assert board.overall == "PASS"


def test_tier_a_full_loop_discriminates_wrong_lock(tmp_path):
    # A tracker drifted 5 m off the operator for >0.5 s must surface as FAIL —
    # proves the loop actually discriminates rather than rubber-stamping.
    bbox = (100, 80, 200, 160)
    depth_list = [(t * S, _depth(bbox=bbox)) for t in range(4)]
    anns = [FrameAnnotation(t_ns=t * S, present=True, bbox=bbox) for t in range(4)]
    clip = _clip(tmp_path, anns, depth_list, "lookalike_distractors")

    preds = [
        PredFrame(
            t_ns=f.t_ns, target_lost=False, target_track_id=9,
            point_xyz=(f.centroid_3d[0] + 5.0, f.centroid_3d[1], f.centroid_3d[2]),
        )
        for f in clip.frames
    ]
    board = score_preds(preds, clip, throughput_hz=15.0)
    verdicts = {r["metric"]: r["verdict"] for r in board.to_dict()["rows"]}
    assert verdicts["wrong_lock_episodes"] == "FAIL"
    assert board.overall == "FAIL"
