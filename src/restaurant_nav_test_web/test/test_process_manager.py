"""Unit tests for the allowlisted ProcessManager (ROS-free)."""
import time

from restaurant_nav_test_web.process_manager import ProcessManager


def _pm():
    return ProcessManager(
        registry={"a": ["sleep", "30"], "b": ["sleep", "30"]},
        groups={"all": ["a", "b"]},
        stagger_sec=0.0,
    )


def test_unknown_name_is_rejected_not_run():
    pm = _pm()
    out = pm.start("evil")
    assert "error" in out and "unknown" in out["error"]
    assert pm.status("evil").get("error")


def test_start_status_stop_cycle():
    pm = _pm()
    st = pm.start("a")
    assert st["running"] is True and st["pid"]
    assert pm.start("a")["running"] is True
    stopped = pm.stop("a")
    assert stopped["running"] is False
    pm.shutdown_all()


def test_group_starts_all_members():
    pm = _pm()
    out = pm.start_group("all")
    assert isinstance(out, list) and len(out) == 2
    assert all(m["running"] for m in out)
    pm.shutdown_all()
    time.sleep(0.1)
    assert pm.status("a")["running"] is False
