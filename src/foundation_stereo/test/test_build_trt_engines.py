from pathlib import Path

import pytest

from foundation_stereo.scripts import build_trt_engines as b


def test_variant_dir_layout(tmp_path):
    assert b.variant_dir(str(tmp_path)) == tmp_path / "Fast-FoundationStereo" / "output_two_stage"
    assert b.variant_dir(str(tmp_path), "x") == tmp_path / "Fast-FoundationStereo" / "x"


def test_make_onnx_command_shape(tmp_path):
    cmd = b.make_onnx_command(Path("/v/Fast-FoundationStereo"), Path("/ck.pth"), tmp_path, 576, 960, 4, 192)
    assert cmd[1].endswith("scripts/make_onnx.py")
    for flag, val in (("--model_dir", "/ck.pth"), ("--save_path", str(tmp_path)), ("--height", "576"),
                      ("--width", "960"), ("--valid_iters", "4"), ("--max_disp", "192")):
        assert val == cmd[cmd.index(flag) + 1]


def test_refuses_overwrite_without_force(tmp_path):
    out = tmp_path / "Fast-FoundationStereo" / "output_two_stage"; out.mkdir(parents=True)
    for n in ("feature_runner.engine", "post_runner.engine", "onnx.yaml"):
        (out / n).write_bytes(b"x")
    with pytest.raises(SystemExit):
        b.ensure_writable(out, force=False)
    b.ensure_writable(out, force=True)   # no raise


def test_build_engine_requires_tensorrt_and_onnx(tmp_path):
    trt = pytest.importorskip("tensorrt")
    with pytest.raises(FileNotFoundError):
        b.build_engine(tmp_path / "missing.onnx", tmp_path / "o.engine")
