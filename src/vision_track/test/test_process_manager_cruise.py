# test/test_process_manager_cruise.py
import yaml
from vision_track.process_manager import (
    follow_server_argv, DEFAULT_CRUISE, ProcessManager)


def test_argv_appends_cruise_overrides():
    base = ["ros2", "run", "following", "follow_server",
            "--ros-args", "-p", "working_frame:=map"]
    argv = follow_server_argv(base, {"enable_cruise_goalgate": True,
                                     "enable_cruise_carrot": False,
                                     "cruise_min_gap": 0.6})
    assert argv[:len(base)] == base                 # base preserved, prefix
    assert "-p" in argv and "enable_cruise_goalgate:=true" in argv
    assert "enable_cruise_carrot:=false" in argv
    assert "cruise_min_gap:=0.6" in argv


def test_argv_clamps_out_of_range_cruise_min_gap():
    base = ["ros2", "run", "following", "follow_server",
            "--ros-args", "-p", "working_frame:=map"]
    # An out-of-range stored gap (e.g. a hand-corrupted sidecar) is clamped to
    # the [0.2, 1.5] bounds at the launch boundary.
    hi = follow_server_argv(base, {"enable_cruise_goalgate": True,
                                   "enable_cruise_carrot": False,
                                   "cruise_min_gap": 5.0})
    assert "cruise_min_gap:=1.5" in hi
    lo = follow_server_argv(base, {"enable_cruise_goalgate": True,
                                   "enable_cruise_carrot": False,
                                   "cruise_min_gap": 0.05})
    assert "cruise_min_gap:=0.2" in lo


def test_get_set_round_trip_and_persist(tmp_path):
    side = tmp_path / "follow_cruise.yaml"
    pm = ProcessManager(cruise_sidecar=str(side))
    assert pm.get_follow_cruise() == DEFAULT_CRUISE
    pm.set_follow_cruise({"enable_cruise_goalgate": False,
                          "enable_cruise_carrot": True,
                          "cruise_min_gap": 0.55})
    assert pm.get_follow_cruise()["enable_cruise_carrot"] is True
    # persisted to sidecar...
    assert yaml.safe_load(side.read_text())["cruise_min_gap"] == 0.55
    # ...and reloaded by a fresh manager
    pm2 = ProcessManager(cruise_sidecar=str(side))
    assert pm2.get_follow_cruise()["enable_cruise_goalgate"] is False


def test_set_ignores_unknown_keys_and_coerces(tmp_path):
    pm = ProcessManager(cruise_sidecar=str(tmp_path / "c.yaml"))
    pm.set_follow_cruise({"cruise_min_gap": "0.7", "bogus": 1})
    s = pm.get_follow_cruise()
    assert s["cruise_min_gap"] == 0.7 and "bogus" not in s
