"""ROS-free tests for the FastAPI factory using a fake bridge."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from restaurant_nav_test_web.restaurant_nav_test_web_app import create_app


class FakeBridge:
    def __init__(self):
        self.started = False
        self.stopped = False
    def snapshot(self):
        return {"state": {"phase": "idle"}, "readiness": {"goto": False},
                "proc": {"camera_femto": {"running": False}}}
    def latest_state(self):
        return 1, {"phase": "scanning", "waver_count": 2}
    def latest_jpeg(self):
        return 0, None
    def start_test(self, mock=False):
        self.started = True
        return {"name": "test_bt", "running": True}
    def stop_test(self):
        self.stopped = True
        return {"name": "test_bt", "running": False}
    def proc_status(self):
        return {"camera_femto": {"running": False}}
    def proc_start(self, name):
        return {"name": name, "running": True}
    def proc_stop(self, name):
        return {"name": name, "running": False}
    def proc_group_start(self, group):
        return [{"name": "camera_femto", "running": True}]
    def proc_group_stop(self, group):
        return [{"name": "camera_femto", "running": False}]


@pytest.fixture
def client(tmp_path):
    webui = tmp_path / "webui"
    webui.mkdir()
    (webui / "index.html").write_text("<html>nav test</html>")
    (webui / "app.js").write_text("// js")
    (webui / "style.css").write_text("/* css */")
    return TestClient(create_app(FakeBridge(), webui_dir=webui))


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "nav test" in r.text

def test_status_endpoint(client):
    r = client.get("/api/status")
    assert r.status_code == 200 and r.json()["readiness"] == {"goto": False}

def test_start_and_stop_test(client):
    assert client.post("/api/test/start").json()["running"] is True
    assert client.post("/api/test/stop").json()["running"] is False

def test_proc_group_route_before_name(client):
    r = client.post("/api/proc/group/prereqs/start")
    assert r.status_code == 200 and isinstance(r.json(), list)

def test_proc_named_start(client):
    r = client.post("/api/proc/camera_femto/start")
    assert r.status_code == 200 and r.json()["name"] == "camera_femto"
