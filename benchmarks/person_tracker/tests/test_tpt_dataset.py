"""Unit tests for ptbench.tpt_bench.dataset (pure, synthetic fixtures)."""

from __future__ import annotations

import os

import pytest

from ptbench.tpt_bench.dataset import (
    TptDatasetError,
    TptFrame,
    load_sequence,
)


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _make_seq(
    root,
    *,
    gt_text: str,
    n_frames: int,
    absent_text: str | None = None,
    gt_name: str = "groundtruth.txt",
    absent_name: str = "absent.txt",
    use_img_subdir: bool = True,
    img_ext: str = ".jpg",
) -> str:
    """Build a synthetic sequence dir; returns its path."""
    seq_dir = os.path.join(str(root), "seq01")
    os.makedirs(seq_dir, exist_ok=True)
    img_dir = os.path.join(seq_dir, "img") if use_img_subdir else seq_dir
    os.makedirs(img_dir, exist_ok=True)
    for i in range(1, n_frames + 1):
        # dummy image bytes; loader only enumerates, does not decode
        _write(os.path.join(img_dir, f"{i:08d}{img_ext}"), "x")
    _write(os.path.join(seq_dir, gt_name), gt_text)
    if absent_text is not None:
        _write(os.path.join(seq_dir, absent_name), absent_text)
    return seq_dir


def test_basic_parse_and_xywh_to_xyxy(tmp_path):
    # 3 frames, comma-delimited x,y,w,h
    gt = "10,20,30,40\n0,0,0,0\n5,5,15,25\n"
    seq = _make_seq(tmp_path, gt_text=gt, n_frames=3)
    frames = load_sequence(seq)

    assert len(frames) == 3
    assert all(isinstance(f, TptFrame) for f in frames)
    assert [f.index for f in frames] == [0, 1, 2]

    # xywh (10,20,30,40) -> xyxy (10,20,40,60)
    assert frames[0].gt_bbox == (10.0, 20.0, 40.0, 60.0)
    # 0,0,0,0 => absent => None
    assert frames[1].gt_bbox is None
    # (5,5,15,25) -> (5,5,20,30)
    assert frames[2].gt_bbox == (5.0, 5.0, 20.0, 30.0)

    # paths are sorted and point at real files
    assert frames[0].image_path.endswith("00000001.jpg")
    assert frames[2].image_path.endswith("00000003.jpg")
    assert all(os.path.isfile(f.image_path) for f in frames)


def test_whitespace_delimited_and_no_img_subdir(tmp_path):
    gt = "1 2 3 4\n10 10 5 5\n"
    seq = _make_seq(tmp_path, gt_text=gt, n_frames=2, use_img_subdir=False)
    frames = load_sequence(seq)
    assert frames[0].gt_bbox == (1.0, 2.0, 4.0, 6.0)
    assert frames[1].gt_bbox == (10.0, 10.0, 15.0, 15.0)


def test_absence_flag_overrides_present_box(tmp_path):
    # box present on every line, but absent flag set on frame 2
    gt = "10,20,30,40\n11,21,31,41\n12,22,32,42\n"
    absent = "0\n1\n0\n"
    seq = _make_seq(tmp_path, gt_text=gt, n_frames=3, absent_text=absent)
    frames = load_sequence(seq)
    assert frames[0].gt_bbox is not None
    assert frames[1].gt_bbox is None  # flagged absent
    assert frames[2].gt_bbox is not None


def test_empty_gt_line_is_absent(tmp_path):
    gt = "10,20,30,40\n\n5,5,5,5\n"
    seq = _make_seq(tmp_path, gt_text=gt, n_frames=3)
    frames = load_sequence(seq)
    assert frames[1].gt_bbox is None


def test_png_frames_and_alt_gt_name(tmp_path):
    gt = "1,1,2,2\n3,3,4,4\n"
    seq = _make_seq(
        tmp_path,
        gt_text=gt,
        n_frames=2,
        gt_name="groundtruth_rect.txt",
        img_ext=".png",
    )
    frames = load_sequence(seq)
    assert len(frames) == 2
    assert frames[0].image_path.endswith(".png")


def test_alt_absent_filename_out_of_view(tmp_path):
    gt = "1,1,2,2\n3,3,4,4\n"
    seq = _make_seq(
        tmp_path,
        gt_text=gt,
        n_frames=2,
        absent_text="1\n0\n",
        absent_name="out_of_view.txt",
    )
    frames = load_sequence(seq)
    assert frames[0].gt_bbox is None
    assert frames[1].gt_bbox is not None


def test_comma_single_line_absence_flags(tmp_path):
    # Real LaSOT out_of_view.txt / full_occlusion.txt are ONE comma-separated
    # line of 0/1 flags, not one-per-line. The loader must flatten them.
    gt = "1,1,2,2\n3,3,4,4\n5,5,6,6\n"
    seq = _make_seq(
        tmp_path,
        gt_text=gt,
        n_frames=3,
        absent_text="0,1,0",  # single comma line, 3 flags, trailing newline-free
        absent_name="out_of_view.txt",
    )
    frames = load_sequence(seq)
    assert frames[0].gt_bbox is not None
    assert frames[1].gt_bbox is None  # flagged absent via single-line flags
    assert frames[2].gt_bbox is not None


def test_absence_is_union_of_all_flag_files(tmp_path):
    # LaSOT ships BOTH out_of_view.txt and full_occlusion.txt; a frame is
    # absent if EITHER flag is set (union), not just the first file found.
    gt = "1,1,2,2\n3,3,4,4\n5,5,6,6\n7,7,8,8\n"
    seq = _make_seq(tmp_path, gt_text=gt, n_frames=4)  # no absence file yet
    _write(os.path.join(seq, "out_of_view.txt"), "0,1,0,0")
    _write(os.path.join(seq, "full_occlusion.txt"), "0,0,1,0")
    frames = load_sequence(seq)
    assert frames[0].gt_bbox is not None
    assert frames[1].gt_bbox is None  # out_of_view
    assert frames[2].gt_bbox is None  # full_occlusion
    assert frames[3].gt_bbox is not None


def test_frame_count_mismatch_raises(tmp_path):
    gt = "1,1,2,2\n3,3,4,4\n5,5,6,6\n"  # 3 gt lines
    seq = _make_seq(tmp_path, gt_text=gt, n_frames=2)  # only 2 images
    with pytest.raises(TptDatasetError, match="frame count"):
        load_sequence(seq)


def test_absence_count_mismatch_raises(tmp_path):
    gt = "1,1,2,2\n3,3,4,4\n"
    seq = _make_seq(
        tmp_path, gt_text=gt, n_frames=2, absent_text="0\n0\n0\n"
    )  # 3 flags vs 2 gt
    with pytest.raises(TptDatasetError, match="absence flag count"):
        load_sequence(seq)


def test_missing_groundtruth_raises(tmp_path):
    seq_dir = os.path.join(str(tmp_path), "noseq")
    os.makedirs(os.path.join(seq_dir, "img"))
    _write(os.path.join(seq_dir, "img", "00000001.jpg"), "x")
    with pytest.raises(TptDatasetError, match="no ground-truth"):
        load_sequence(seq_dir)


def test_missing_dir_raises(tmp_path):
    with pytest.raises(TptDatasetError, match="sequence dir not found"):
        load_sequence(os.path.join(str(tmp_path), "does_not_exist"))


def test_bad_gt_field_count_raises(tmp_path):
    gt = "1,2,3\n4,5,6,7\n"  # first line has 3 fields
    seq = _make_seq(tmp_path, gt_text=gt, n_frames=2)
    with pytest.raises(TptDatasetError, match="expected 4 values"):
        load_sequence(seq)


def test_non_numeric_gt_raises(tmp_path):
    gt = "a,b,c,d\n1,2,3,4\n"
    seq = _make_seq(tmp_path, gt_text=gt, n_frames=2)
    with pytest.raises(TptDatasetError, match="non-numeric"):
        load_sequence(seq)


def test_no_images_raises(tmp_path):
    seq_dir = os.path.join(str(tmp_path), "seqnoimg")
    os.makedirs(seq_dir)
    _write(os.path.join(seq_dir, "groundtruth.txt"), "1,1,2,2\n")
    with pytest.raises(TptDatasetError, match="no frame images"):
        load_sequence(seq_dir)
