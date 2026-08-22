"""Unit tests for ``vision_util.weights_cache.resolve_weights``.

Network-hitting paths (auto-download) are covered via monkey-patched
Ultralytics classes so the test suite runs offline.
"""
from __future__ import annotations

import os
import threading
import types
from pathlib import Path

import pytest

from vision_util import weights_cache
from vision_util.weights_cache import find_cached, resolve_weights


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the resolver at a clean tmp cache and clear env override."""
    monkeypatch.setattr(weights_cache, "_DEFAULT_CACHE", tmp_path)
    monkeypatch.delenv(weights_cache._ENV_VAR, raising=False)
    return tmp_path


def _touch(dir_: Path, name: str) -> Path:
    dir_.mkdir(parents=True, exist_ok=True)
    path = dir_ / name
    path.write_bytes(b"\x00")  # non-empty so filesystems that care don't fold it
    return path


def test_absolute_path_existing(tmp_path):
    stub = _touch(tmp_path, "custom.pt")
    assert resolve_weights(str(stub)) == stub


def test_absolute_path_missing_raises():
    with pytest.raises(FileNotFoundError):
        resolve_weights("/nonexistent/weights.pt")


def test_relative_with_separator_rejected():
    with pytest.raises(ValueError, match="relative paths with separators"):
        resolve_weights("subdir/yolo11n-seg.pt")


def test_empty_name_rejected():
    with pytest.raises(ValueError):
        resolve_weights("")


def test_default_cache_hit(isolated_cache):
    target = _touch(isolated_cache, "yolo11n-seg.pt")
    assert resolve_weights("yolo11n-seg.pt") == target


def test_env_override_wins(tmp_path, monkeypatch):
    default = tmp_path / "default"
    override = tmp_path / "override"
    monkeypatch.setattr(weights_cache, "_DEFAULT_CACHE", default)
    _touch(default, "shared.pt")
    override_weight = _touch(override, "shared.pt")
    monkeypatch.setenv(weights_cache._ENV_VAR, str(override))

    assert resolve_weights("shared.pt") == override_weight


def test_auto_download_lands_in_cache(isolated_cache, monkeypatch):
    """Simulate Ultralytics by writing the file into CWD, as it does IRL."""

    class FakeYOLO:
        def __init__(self, name: str):
            Path.cwd().joinpath(name).write_bytes(b"fake-weights")

    fake_module = types.SimpleNamespace(
        YOLO=FakeYOLO, FastSAM=FakeYOLO, YOLOWorld=FakeYOLO, SAM=FakeYOLO,
    )
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_module)

    result = resolve_weights("imaginary-yolo.pt")
    assert result == isolated_cache / "imaginary-yolo.pt"
    assert result.read_bytes() == b"fake-weights"


def test_concurrent_resolve_downloads_once(isolated_cache, monkeypatch):
    call_count = {"n": 0}
    lock = threading.Lock()

    class CountingYOLO:
        def __init__(self, name: str):
            with lock:
                call_count["n"] += 1
            Path.cwd().joinpath(name).write_bytes(b"x")

    fake_module = types.SimpleNamespace(
        YOLO=CountingYOLO, FastSAM=CountingYOLO, YOLOWorld=CountingYOLO,
        SAM=CountingYOLO,
    )
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_module)

    results: list[Path] = []

    def worker():
        results.append(resolve_weights("race.pt"))

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 3
    assert all(r == isolated_cache / "race.pt" for r in results)
    assert call_count["n"] == 1


def test_pick_class_dispatch(monkeypatch):
    """Filename prefix must route to the right Ultralytics entrypoint."""
    fake_module = types.SimpleNamespace(
        YOLO=type("YOLO", (), {}),
        FastSAM=type("FastSAM", (), {}),
        YOLOWorld=type("YOLOWorld", (), {}),
        SAM=type("SAM", (), {}),
    )
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_module)
    pick = weights_cache._pick_ultralytics_cls
    assert pick("yolo11n-seg.pt").__name__ == "YOLO"
    assert pick("yolov8s-worldv2.pt").__name__ == "YOLOWorld"
    assert pick("FastSAM-s.pt").__name__ == "FastSAM"
    assert pick("mobile_sam.pt").__name__ == "SAM"
    assert pick("sam_b.pt").__name__ == "SAM"
    assert pick("sam2_t.pt").__name__ == "SAM"


def test_mobile_sam_auto_download_routes_to_sam_branch(isolated_cache, monkeypatch):
    """resolve_weights('mobile_sam.pt') must instantiate the SAM class."""
    called = {"name": None}

    class FakeSAM:
        def __init__(self, name: str):
            called["name"] = name
            Path.cwd().joinpath(name).write_bytes(b"sam-weights")

    class FakeFastSAM:
        def __init__(self, name: str):
            raise AssertionError(
                f"FastSAM should not be invoked for {name!r}"
            )

    fake_module = types.SimpleNamespace(
        YOLO=FakeFastSAM, FastSAM=FakeFastSAM, YOLOWorld=FakeFastSAM,
        SAM=FakeSAM,
    )
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_module)

    result = resolve_weights("mobile_sam.pt")
    assert result == isolated_cache / "mobile_sam.pt"
    assert called["name"] == "mobile_sam.pt"
    assert result.read_bytes() == b"sam-weights"


def test_cwd_is_restored_after_download(isolated_cache, monkeypatch, tmp_path):
    """Resolver must not leak CWD changes back to the caller."""

    class FakeYOLO:
        def __init__(self, name: str):
            Path.cwd().joinpath(name).write_bytes(b"x")

    fake_module = types.SimpleNamespace(
        YOLO=FakeYOLO, FastSAM=FakeYOLO, YOLOWorld=FakeYOLO, SAM=FakeYOLO,
    )
    monkeypatch.setitem(__import__("sys").modules, "ultralytics", fake_module)

    starting_cwd = tmp_path / "caller_cwd"
    starting_cwd.mkdir()
    os.chdir(starting_cwd)
    try:
        resolve_weights("restore-check.pt")
        assert Path.cwd() == starting_cwd
    finally:
        os.chdir("/")


def test_find_cached_returns_cache_hit(isolated_cache):
    path = _touch(isolated_cache, "pose_landmarker_full.task")
    assert find_cached("pose_landmarker_full.task") == path


def test_find_cached_returns_none_on_miss_without_downloading(isolated_cache, monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("find_cached must never download")
    monkeypatch.setattr(weights_cache, "_download", _boom)
    assert find_cached("pose_landmarker_full.task") is None


def test_find_cached_honours_env_override(isolated_cache, monkeypatch, tmp_path):
    override = tmp_path / "override"
    path = _touch(override, "pose_landmarker_full.task")
    monkeypatch.setenv(weights_cache._ENV_VAR, str(override))
    assert find_cached("pose_landmarker_full.task") == path


def test_find_cached_absolute_path(tmp_path):
    path = _touch(tmp_path, "x.task")
    assert find_cached(str(path)) == path
    assert find_cached(str(tmp_path / "missing.task")) is None


def test_find_cached_rejects_relative_with_separator(isolated_cache):
    with pytest.raises(ValueError):
        find_cached("models/pose_landmarker_full.task")
