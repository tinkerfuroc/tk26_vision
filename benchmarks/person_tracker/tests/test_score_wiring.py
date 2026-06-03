"""Tests for ptbench.replay.score_cli.score_preds — full common pipeline glue.

Feeds hand-built predictions + a small GtClip through ``score_preds`` (align →
metrics → score) without importing the tracker, asserting the scoreboard verdict
and per-row verdicts. This exercises the committed ``common`` pipeline end to end
through the replay layer with no model and no ROS.
"""
from __future__ import annotations

from ptbench.common.align import PredFrame
from ptbench.common.schema import GtClip, GtFrame
from ptbench.replay.score_cli import score_preds

S = 1_000_000_000  # 1 second in ns


def _clip(frames) -> GtClip:
    return GtClip(
        schema_version="1.0",
        clip_id="unit",
        bag_path="n/a",
        scenario="unit-test",
        frames=frames,
    )


def _gt(t_s, present=True, centroid=(0.0, 0.0, 2.0)) -> GtFrame:
    return GtFrame(
        t_ns=int(t_s * S),
        present=present,
        bbox=(0, 0, 10, 10) if present else None,
        centroid_field=centroid if present else None,
        centroid_track=centroid if present else None,
    )


def _pred(t_s, lost=False, xyz=(0.0, 0.0, 2.0), tid=1) -> PredFrame:
    return PredFrame(
        t_ns=int(t_s * S), target_lost=lost, target_track_id=tid, point_xyz=xyz
    )


def test_perfect_track_passes():
    """Pred dead-on the GT centroid every frame, fast throughput -> PASS."""
    frames = [_gt(t) for t in range(10)]
    preds = [_pred(t) for t in range(10)]
    board = score_preds(preds, _clip(frames), throughput_hz=30.0)
    verdicts = {name: verdict for name, _val, verdict in board.rows}
    assert verdicts["correct_lock_rate"] == "PASS"
    assert verdicts["wrong_lock_episodes"] == "PASS"
    assert verdicts["throughput_hz"] == "PASS"
    assert board.overall == "PASS"


def test_throughput_none_scores_na_not_fail():
    """Throughput None -> N/A row; overall driven by the rest (PASS here)."""
    frames = [_gt(t) for t in range(10)]
    preds = [_pred(t) for t in range(10)]
    board = score_preds(preds, _clip(frames), throughput_hz=None)
    verdicts = {name: verdict for name, _val, verdict in board.rows}
    assert verdicts["throughput_hz"] == "N/A"
    assert board.overall == "PASS"


def test_sustained_wrong_lock_fails():
    """Pred locked far from GT for >0.5 s -> wrong_lock_episodes>0 -> FAIL."""
    # 11 frames at 0.0,0.1,...,1.0 s; pred 5 m off the GT centroid throughout.
    times = [i * 0.1 for i in range(11)]
    frames = [_gt(t) for t in times]
    preds = [_pred(t, xyz=(5.0, 0.0, 2.0)) for t in times]
    board = score_preds(preds, _clip(frames), throughput_hz=30.0)
    verdicts = {name: verdict for name, _val, verdict in board.rows}
    assert verdicts["wrong_lock_episodes"] == "FAIL"
    assert verdicts["correct_lock_rate"] == "FAIL"  # 0 correct locks
    assert board.overall == "FAIL"


def test_false_target_on_absent_frames():
    """Operator absent but tracker claims a lock -> false_target_rate fires."""
    frames = [_gt(t, present=False) for t in range(10)]
    preds = [_pred(t, lost=False, xyz=(0.0, 0.0, 2.0)) for t in range(10)]
    board = score_preds(preds, _clip(frames), throughput_hz=30.0)
    verdicts = {name: verdict for name, val, verdict in board.rows}
    # All absent + all "found" -> false_target_rate = 1.0 -> FAIL.
    assert verdicts["false_target_rate"] == "FAIL"
    assert board.overall == "FAIL"
