"""Unit tests for nms.py — pure-function NMS and clustering helpers."""

from __future__ import annotations

import pytest

from tk_vision_specialized.nms import (
    iou,
    suppress_within_category,
    MatchRow,
)


def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # box A = 10x10, box B shifted right by 5 -> intersection 5x10=50, union 150
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_iou_zero_area_box_returns_zero():
    assert iou((0, 0, 0, 0), (0, 0, 10, 10)) == 0.0


def test_within_category_keeps_one_per_overlapping_pair():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(1, 1, 11, 11), conf=0.7),  # IoU > 0.5 with first
    ]
    kept = suppress_within_category(rows, iou_thresh=0.5)
    assert len(kept) == 1
    assert kept[0].conf == 0.9


def test_within_category_keeps_disjoint_same_label():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(50, 50, 60, 60), conf=0.5),  # disjoint
    ]
    kept = suppress_within_category(rows, iou_thresh=0.5)
    assert len(kept) == 2


def test_within_category_does_not_suppress_across_labels():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='cola', bbox=(0, 0, 10, 10), conf=0.8),  # same box, different label
    ]
    kept = suppress_within_category(rows, iou_thresh=0.5)
    assert len(kept) == 2  # cross-label overlap is not this function's job


def test_within_category_idempotent():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(1, 1, 11, 11), conf=0.7),
        MatchRow(label='milk', bbox=(50, 50, 60, 60), conf=0.5),
    ]
    once = suppress_within_category(rows, iou_thresh=0.5)
    twice = suppress_within_category(once, iou_thresh=0.5)
    assert once == twice


def test_within_category_empty_input():
    assert suppress_within_category([], iou_thresh=0.5) == []


def test_within_category_suppresses_at_threshold_equality():
    # Pin the strict `<` semantics: IoU == iou_thresh -> suppress.
    # A=(0,0,10,10) area=100, B=(0,0,10,5) area=50, intersection=50
    # -> IoU = 50 / (100 + 50 - 50) = 0.5.
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(0, 0, 10, 5), conf=0.5),
    ]
    kept = suppress_within_category(rows, iou_thresh=0.5)
    assert len(kept) == 1
    assert kept[0].conf == 0.9


from tk_vision_specialized.nms import (
    Cluster,
    JudgePayload,
    cluster_for_judge,
    build_judge_payload,
)
import numpy as np


def test_cluster_singletons_when_disjoint():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='cola', bbox=(50, 50, 60, 60), conf=0.8),
    ]
    clusters = cluster_for_judge(rows, iou_thresh=0.5)
    assert len(clusters) == 2
    assert all(c.is_conflict() is False for c in clusters)


def test_cluster_groups_overlapping_cross_label():
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='cola', bbox=(1, 1, 11, 11), conf=0.85),  # IoU > 0.5
    ]
    clusters = cluster_for_judge(rows, iou_thresh=0.5)
    assert len(clusters) == 1
    assert clusters[0].is_conflict() is True
    assert {r.label for r in clusters[0].rows} == {'milk', 'cola'}


def test_cluster_same_label_overlap_not_conflict():
    # After within-cat NMS this shouldn't happen, but defensively:
    rows = [
        MatchRow(label='milk', bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='milk', bbox=(1, 1, 11, 11), conf=0.7),
    ]
    clusters = cluster_for_judge(rows, iou_thresh=0.5)
    assert len(clusters) == 1
    assert clusters[0].is_conflict() is False  # only one distinct label


def test_cluster_transitive_overlap_collapses_into_one():
    # A overlaps B, B overlaps C, A may not overlap C — still one cluster.
    rows = [
        MatchRow(label='milk',   bbox=(0, 0, 10, 10), conf=0.9),
        MatchRow(label='cola',   bbox=(5, 0, 15, 10), conf=0.8),
        MatchRow(label='sprite', bbox=(10, 0, 20, 10), conf=0.7),
    ]
    clusters = cluster_for_judge(rows, iou_thresh=0.3)
    assert len(clusters) == 1
    assert clusters[0].is_conflict() is True


def test_build_judge_payload_crops_with_margin_clamped_to_bounds():
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    rows = [
        MatchRow(label='milk', bbox=(10, 10, 30, 30), conf=0.9),
        MatchRow(label='cola', bbox=(20, 20, 40, 40), conf=0.85),
    ]
    cluster = Cluster(rows=rows)
    items = {
        'milk': 'data:image/jpeg;base64,FAKE_MILK',
        'cola': 'data:image/jpeg;base64,FAKE_COLA',
    }
    payload = build_judge_payload(cluster, items, scene, margin_px=20)
    # Union bbox is (10,10,40,40); +20 margin -> (-10,-10,60,60) clamped to (0,0,60,60)
    assert payload.crop.shape == (60, 60, 3)
    competing_labels = {label for label, _url in payload.competing}
    assert competing_labels == {'milk', 'cola'}


def test_build_judge_payload_collapses_duplicate_labels():
    scene = np.zeros((100, 100, 3), dtype=np.uint8)
    # Cluster has two 'milk' rows (somehow survived within-cat NMS at this
    # IoU threshold) plus one 'cola' — competing list collapses duplicates.
    rows = [
        MatchRow(label='milk', bbox=(10, 10, 30, 30), conf=0.9),
        MatchRow(label='milk', bbox=(12, 12, 32, 32), conf=0.85),
        MatchRow(label='cola', bbox=(20, 20, 40, 40), conf=0.8),
    ]
    cluster = Cluster(rows=rows)
    items = {'milk': 'A', 'cola': 'B'}
    payload = build_judge_payload(cluster, items, scene, margin_px=0)
    assert len(payload.competing) == 2
