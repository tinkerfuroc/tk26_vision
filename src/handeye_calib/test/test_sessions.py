"""Unit tests for handeye_sessions: on-disk capture-session persistence + history.

Pure filesystem; HANDEYE_DUMP_DIR points each test at a tmp dir so nothing
touches the real calibration_data tree.
"""
import json
import os
import time
import pytest

from handeye_calib import handeye_sessions as hsx


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("HANDEYE_DUMP_DIR", str(tmp_path))
    return tmp_path


def _payload(n_samples=3, with_result=False):
    p = {
        "schema": "wrist_handeye_session/1",
        "timestamp": "20260628_140000",
        "robot": "tinker2",
        "calib_frame": "color",
        "board": {"squares_x": 5, "squares_y": 5, "square_len_m": 0.04},
        "samples": [{"idx": i, "capture_reproj_px": 0.2} for i in range(n_samples)],
    }
    if with_result:
        p["result"] = {"status": "PASS", "rejected_sample_indices": [1]}
    return p


def test_write_and_read_roundtrip(root):
    name = "wrist_handeye_20260628_140000"
    d = hsx.write_session(name, _payload(4), base=str(root))
    assert os.path.isdir(d) and os.path.isdir(os.path.join(d, "thumbs"))
    data = hsx.read_session(name, base=str(root))
    assert len(data["samples"]) == 4 and data["robot"] == "tinker2"


def test_list_sessions_newest_first_with_summary(root):
    hsx.write_session("wrist_handeye_A", _payload(2), base=str(root))
    time.sleep(0.01)
    hsx.write_session("wrist_handeye_B", _payload(5, with_result=True), base=str(root))
    sessions = hsx.list_sessions(base=str(root))
    assert [s["name"] for s in sessions] == ["wrist_handeye_B", "wrist_handeye_A"]
    b = sessions[0]
    assert b["n_samples"] == 5 and b["has_solve"] is True
    assert b["status"] == "PASS" and b["n_rejected"] == 1
    a = sessions[1]
    assert a["n_samples"] == 2 and a["has_solve"] is False and a["status"] is None


def test_list_sessions_empty_when_root_absent(root):
    assert hsx.list_sessions(base=str(root / "nope")) == []


def test_list_skips_corrupt_dir(root):
    hsx.write_session("wrist_handeye_ok", _payload(1), base=str(root))
    bad = hsx.session_dir("wrist_handeye_bad", base=str(root))
    os.makedirs(bad, exist_ok=True)
    with open(os.path.join(bad, "session.json"), "w") as f:
        f.write("{ this is not json")
    names = [s["name"] for s in hsx.list_sessions(base=str(root))]
    assert names == ["wrist_handeye_ok"]  # corrupt one skipped, not raised


def test_thumbs_write_and_rewrite_recompacts(root):
    name = "wrist_handeye_thumbs"
    hsx.write_session(name, _payload(3), base=str(root))
    for i in range(3):
        hsx.write_thumb(name, i, f"jpg{i}".encode(), base=str(root))
    assert os.path.isfile(hsx.thumb_path(name, 1, base=str(root)))
    # Simulate deleting sample 1: in-memory recompacts 2->1; rewrite the dir.
    hsx.rewrite_thumbs(name, {0: b"jpg0", 1: b"jpg2"}, base=str(root))
    files = sorted(os.listdir(os.path.join(hsx.session_dir(name, base=str(root)), "thumbs")))
    assert files == ["0.jpg", "1.jpg"]  # 2.jpg gone after recompaction
    with open(hsx.thumb_path(name, 1, base=str(root)), "rb") as f:
        assert f.read() == b"jpg2"


def test_delete_session(root):
    hsx.write_session("wrist_handeye_del", _payload(1), base=str(root))
    assert hsx.delete_session("wrist_handeye_del", base=str(root)) is True
    assert hsx.list_sessions(base=str(root)) == []
    assert hsx.delete_session("wrist_handeye_del", base=str(root)) is False  # idempotent


@pytest.mark.parametrize("bad", ["", "..", ".", "../escape", "a/b", ".hidden", "x\\y"])
def test_safe_name_rejects_traversal(root, bad):
    with pytest.raises(ValueError):
        hsx.session_dir(bad, base=str(root))


def test_new_session_name_is_deterministic_with_injected_time():
    nm = hsx.new_session_name(time.strptime("20260628_140102", "%Y%m%d_%H%M%S"))
    assert nm == "wrist_handeye_20260628_140102"


def test_new_session_name_carries_robot():
    st = time.strptime('2026-07-03 10:00:00', '%Y-%m-%d %H:%M:%S')
    assert hsx.new_session_name(st, robot='tinker1') == 'wrist_handeye_tinker1_20260703_100000'
    assert hsx.new_session_name(st) == 'wrist_handeye_20260703_100000'


def test_new_session_name_ignores_unsafe_robot():
    st = time.strptime('2026-07-03 10:00:00', '%Y-%m-%d %H:%M:%S')
    # unsafe robot tag falls back to the untagged legacy name — persistence
    # must never die on a bad tag (that's the bug class this task fixes)
    assert hsx.new_session_name(st, robot='../evil') == 'wrist_handeye_20260703_100000'


def test_build_session_dict_has_no_dead_attr():
    import inspect
    from handeye_calib import handeye_web
    assert 'self._robot_name' not in inspect.getsource(handeye_web)


# ---- flatten_samples / flat_thumb_path (v2 multi-placement history browsing) ----
# The detail/thumb HTTP endpoints (and the webui gallery) were written against
# the v1 flat-'samples' schema and never updated for v2 placements — masked by
# _persist_session dying silently on the dead self._robot_name attribute, so no
# real v2 session ever reached them. These are the pure-filesystem helpers the
# endpoints now use to bridge both schemas.

def _v2_payload():
    return {
        "schema": "wrist_handeye_session/2",
        "timestamp": "20260703_100000",
        "robot": "tinker1",
        "placements": [
            {"id": "front", "label": "front",
             "samples": [{"idx": 0}, {"idx": 1}], "result": None},
            {"id": "side", "label": "side",
             "samples": [{"idx": 0}], "result": None},
        ],
        "combined_result": None,
    }


def test_flatten_samples_concatenates_v2_placements():
    assert hsx.flatten_samples(_v2_payload()) == [
        {"idx": 0}, {"idx": 1}, {"idx": 0},
    ]


def test_flatten_samples_v1_passthrough():
    assert hsx.flatten_samples(_payload(3)) == [
        {"idx": 0, "capture_reproj_px": 0.2},
        {"idx": 1, "capture_reproj_px": 0.2},
        {"idx": 2, "capture_reproj_px": 0.2},
    ]


def test_flat_thumb_path_maps_flat_index_into_owning_placement(root):
    name = "wrist_handeye_v2thumbs"
    data = _v2_payload()
    hsx.write_session(name, data, base=str(root))
    hsx.write_placement_thumb(name, "front", 0, b"front0", base=str(root))
    hsx.write_placement_thumb(name, "front", 1, b"front1", base=str(root))
    hsx.write_placement_thumb(name, "side", 0, b"side0", base=str(root))

    # flat idx 0,1 -> front[0], front[1]; flat idx 2 -> side[0]
    assert hsx.flat_thumb_path(name, 0, data=data, base=str(root)) == \
        hsx.placement_thumb_path(name, "front", 0, base=str(root))
    assert hsx.flat_thumb_path(name, 2, data=data, base=str(root)) == \
        hsx.placement_thumb_path(name, "side", 0, base=str(root))
    with open(hsx.flat_thumb_path(name, 2, data=data, base=str(root)), "rb") as f:
        assert f.read() == b"side0"


def test_flat_thumb_path_out_of_range_returns_none(root):
    assert hsx.flat_thumb_path("wrist_handeye_v2thumbs", 99,
                                data=_v2_payload(), base=str(root)) is None


def test_flat_thumb_path_v1_passthrough(root):
    name = "wrist_handeye_v1thumbs"
    hsx.write_session(name, _payload(2), base=str(root))
    hsx.write_thumb(name, 1, b"legacy1", base=str(root))
    assert hsx.flat_thumb_path(name, 1, data=_payload(2), base=str(root)) == \
        hsx.thumb_path(name, 1, base=str(root))


# ---- multi-placement additions ----

def test_write_read_multi_placement(tmp_path):
    """write a v2 session dict with 2 placements, read back, assert structure."""
    name = "wrist_handeye_v2"
    payload = {
        "schema": "wrist_handeye_session/2",
        "timestamp": "20260628_150000",
        "robot": "tinker2",
        "calib_frame": "color",
        "placements": [
            {"id": "placement_0", "label": "front",
             "samples": [{"idx": 0}, {"idx": 1}], "result": None},
            {"id": "placement_1", "label": "side",
             "samples": [{"idx": 0}], "result": {"status": "PASS"}},
        ],
        "combined_result": None,
    }
    hsx.write_session(name, payload, base=str(tmp_path))
    data = hsx.read_session(name, base=str(tmp_path))
    assert len(data["placements"]) == 2
    assert data["placements"][0]["label"] == "front"
    assert len(data["placements"][0]["samples"]) == 2
    assert len(data["placements"][1]["samples"]) == 1

    sessions = hsx.list_sessions(base=str(tmp_path))
    assert len(sessions) == 1
    s = sessions[0]
    assert s["n_placements"] == 2
    assert s["n_samples_total"] == 3
    assert s["has_combined_solve"] is False
    assert s["placements"][1]["has_solve"] is True
    assert s["placements"][1]["status"] == "PASS"


def test_v1_load_summary_compat(tmp_path):
    """a v1 session (flat 'samples', no 'placements') gets n_placements=1 in summary."""
    name = "wrist_handeye_v1_compat"
    payload = {
        "schema": "wrist_handeye_session/1",
        "timestamp": "20260628_160000",
        "robot": "tinker2",
        "calib_frame": "color",
        "samples": [{"idx": 0}, {"idx": 1}, {"idx": 2}],
    }
    hsx.write_session(name, payload, base=str(tmp_path))
    sessions = hsx.list_sessions(base=str(tmp_path))
    s = sessions[0]
    assert s["n_placements"] == 1
    assert s["n_samples_total"] == 3
    assert s["has_combined_solve"] is False
    assert s["combined_status"] is None
    assert len(s["placements"]) == 1
    assert s["placements"][0]["id"] == "default"
    assert s["placements"][0]["n_samples"] == 3


def test_placement_thumb_path(tmp_path):
    """placement_thumb_path returns <session_dir>/thumbs/<pid>/<idx>.jpg."""
    name = "wrist_handeye_pt"
    hsx.write_session(name, {"samples": []}, base=str(tmp_path))
    p = hsx.placement_thumb_path(name, "placement_0", 3, base=str(tmp_path))
    sdir = hsx.session_dir(name, base=str(tmp_path))
    assert p == os.path.join(sdir, "thumbs", "placement_0", "3.jpg")


def test_rewrite_placement_thumbs_idempotent(tmp_path):
    """calling rewrite_placement_thumbs twice replaces, doesn't accumulate."""
    name = "wrist_handeye_rpt"
    hsx.write_session(name, {"samples": []}, base=str(tmp_path))
    pid = "placement_0"
    hsx.rewrite_placement_thumbs(name, pid, {0: b"a", 1: b"b", 2: b"c"}, base=str(tmp_path))
    hsx.rewrite_placement_thumbs(name, pid, {0: b"x", 1: b"y"}, base=str(tmp_path))
    d = os.path.join(hsx.session_dir(name, base=str(tmp_path)), "thumbs", pid)
    files = sorted(os.listdir(d))
    assert files == ["0.jpg", "1.jpg"]
    with open(os.path.join(d, "0.jpg"), "rb") as f:
        assert f.read() == b"x"
