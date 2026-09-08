import hashlib
from pathlib import Path

import pytest

from foundation_stereo.scripts import fetch_fast_fs_weights as f


def test_checkpoint_path_layout(tmp_path):
    p = f.checkpoint_path(str(tmp_path))
    assert p == tmp_path / "Fast-FoundationStereo" / "weights" / "23-36-37" / "model_best_bp2_serialize.pth"


def test_sums_written_then_verified(tmp_path):
    ckpt_dir = tmp_path / "23-36-37"; ckpt_dir.mkdir()
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"abc")
    digest = f.verify_or_write_sums(ckpt_dir, expected=None)
    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert (ckpt_dir / "SHA256SUMS").read_text().split()[0] == digest
    assert f.verify_or_write_sums(ckpt_dir, expected=None) == digest  # second run verifies


def test_sums_mismatch_raises(tmp_path):
    ckpt_dir = tmp_path / "23-36-37"; ckpt_dir.mkdir()
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"abc")
    f.verify_or_write_sums(ckpt_dir, expected=None)
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"xyz")
    with pytest.raises(RuntimeError):
        f.verify_or_write_sums(ckpt_dir, expected=None)


def test_expected_digest_mismatch_raises(tmp_path):
    ckpt_dir = tmp_path / "23-36-37"; ckpt_dir.mkdir()
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"abc")
    with pytest.raises(RuntimeError):
        f.verify_or_write_sums(ckpt_dir, expected="0" * 64)
    # The mismatch must be caught before SHA256SUMS gets (re)written.
    assert not (ckpt_dir / "SHA256SUMS").exists()


def test_expected_digest_match_ok(tmp_path):
    ckpt_dir = tmp_path / "23-36-37"; ckpt_dir.mkdir()
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    assert f.verify_or_write_sums(ckpt_dir, expected=digest) == digest
    assert (ckpt_dir / "SHA256SUMS").read_text().split()[0] == digest


def test_default_expected_sha256_matches_validated_checkpoint():
    # Locks in the constant against silent edits — the actual value is
    # cross-checked against the cached SHA256SUMS by the final-fix brief,
    # not re-derived here (the cache may not be present in every test env).
    assert f.EXPECTED_SHA256 == (
        "af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990"
    )
