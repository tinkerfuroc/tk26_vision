"""Unit tests for nms.py — pure-function NMS and clustering helpers."""

from __future__ import annotations

from dataclasses import dataclass

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
