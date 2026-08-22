"""Import-shape tests for stereo_runner.

These exercise the namespace-swap logic (FoundationStereo vs
Fast-FoundationStereo both ship a top-level `core/` package with
overlapping module names). The tests require torch and the vendored
thirdparty trees — they're skipped if either is missing, so the rest
of the foundation_stereo suite still runs in a vanilla venv.
"""

import importlib.util
import os

import pytest

torch = pytest.importorskip("torch")

_VENDOR_ROOT = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "thirdparty",
                 "foundation_stereo")
)
_FS = os.path.join(_VENDOR_ROOT, "FoundationStereo")
_FAST = os.path.join(_VENDOR_ROOT, "Fast-FoundationStereo")

if not (os.path.isdir(_FS) and os.path.isdir(_FAST)):
    pytest.skip("vendored foundation_stereo trees not present",
                allow_module_level=True)


def test_runner_module_imports():
    """The runner module itself must import without instantiating any model."""
    from foundation_stereo import stereo_runner
    assert hasattr(stereo_runner, "StereoRunner")
    assert hasattr(stereo_runner, "InferResult")
    assert hasattr(stereo_runner, "TRT_VARIANTS")


def test_namespace_swap_to_upstream_then_fast():
    """After swapping into upstream then Fast, the right `core.foundation_stereo`
    is on sys.path each time and the cached version doesn't leak across."""
    from foundation_stereo import stereo_runner
    stereo_runner._swap_namespace(_FS)
    import core.foundation_stereo as upstream_core   # noqa: F401
    assert hasattr(upstream_core, "FoundationStereo")

    stereo_runner._swap_namespace(_FAST)
    import core.foundation_stereo as fast_core
    assert hasattr(fast_core, "TrtRunner")


def test_default_iters_table_complete():
    """Every PyTorch backend kind must have a default-iters entry."""
    from foundation_stereo import stereo_runner
    for kind in ("vitl", "vits", "fast_fp32", "fast_fp16"):
        assert kind in stereo_runner._DEFAULT_ITERS


def test_weights_root_is_user_expanded(tmp_path, monkeypatch):
    """`~` in weights_root must be expanded so the yaml default works."""
    from foundation_stereo import stereo_runner
    monkeypatch.setenv("HOME", str(tmp_path))
    runner = stereo_runner.StereoRunner(weights_root="~/wr")
    assert runner._weights_root == os.path.join(str(tmp_path), "wr")
    assert runner._fast_pickle.startswith(str(tmp_path))
