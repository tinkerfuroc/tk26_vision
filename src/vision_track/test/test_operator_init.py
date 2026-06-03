"""Unit tests for the ROS-free operator-init heuristic."""
from dataclasses import dataclass
from typing import Tuple

from vision_track.core.operator_init import select_operator_detection


@dataclass
class Det:
    track_id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_name: str = "person"


IMG_W, IMG_H = 640, 480


def test_picks_central_when_depth_equal():
    # Two people, equal (None) depth → the more central one wins.
    dets = [
        Det(1, (0, 0, 80, 400), 0.9),       # far left
        Det(2, (280, 40, 360, 440), 0.9),   # centered
    ]
    chosen = select_operator_detection(
        dets, image_wh=(IMG_W, IMG_H), depth_lookup=lambda b: None
    )
    assert chosen.track_id == 2


def test_nearer_wins_over_more_central():
    # A slightly off-center but much nearer person beats a centered far one.
    dets = [
        Det(1, (300, 40, 380, 440), 0.9),   # centered, far (5 m)
        Det(2, (120, 40, 240, 440), 0.9),   # off-center, near (1 m)
    ]
    # Map each bbox (immutable tuple key) to its depth in meters.
    depth_by_bbox = {dets[0].bbox: 5.0, dets[1].bbox: 1.0}
    chosen = select_operator_detection(
        dets, image_wh=(IMG_W, IMG_H),
        depth_lookup=lambda b: depth_by_bbox[b],
    )
    assert chosen.track_id == 2


def test_confidence_breaks_ties():
    # Two identical-geometry detections → higher confidence wins.
    dets = [
        Det(1, (280, 40, 360, 440), 0.6),
        Det(2, (280, 40, 360, 440), 0.95),
    ]
    chosen = select_operator_detection(
        dets, image_wh=(IMG_W, IMG_H), depth_lookup=lambda b: 2.0
    )
    assert chosen.track_id == 2


def test_only_persons_considered():
    dets = [
        Det(1, (300, 40, 380, 440), 0.99, class_name="chair"),
        Det(2, (120, 40, 240, 440), 0.5, class_name="person"),
    ]
    chosen = select_operator_detection(
        dets, image_wh=(IMG_W, IMG_H), depth_lookup=lambda b: None,
        target_class="person",
    )
    assert chosen.track_id == 2


def test_empty_returns_none():
    assert select_operator_detection(
        [], image_wh=(IMG_W, IMG_H), depth_lookup=lambda b: None
    ) is None
