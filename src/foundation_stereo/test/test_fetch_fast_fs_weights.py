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
    digest = f.verify_or_write_sums(ckpt_dir)
    assert digest == hashlib.sha256(b"abc").hexdigest()
    assert (ckpt_dir / "SHA256SUMS").read_text().split()[0] == digest
    assert f.verify_or_write_sums(ckpt_dir) == digest          # second run verifies


def test_sums_mismatch_raises(tmp_path):
    ckpt_dir = tmp_path / "23-36-37"; ckpt_dir.mkdir()
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"abc")
    f.verify_or_write_sums(ckpt_dir)
    (ckpt_dir / "model_best_bp2_serialize.pth").write_bytes(b"xyz")
    with pytest.raises(RuntimeError):
        f.verify_or_write_sums(ckpt_dir)
