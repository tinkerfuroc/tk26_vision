"""FastAPI route tests for the custom hand-eye dataset endpoints.

Builds the real `make_app` against a bare CalibWebNode (object.__new__ with the
handful of attributes the dataset/waypoint routes touch) so the actual route
wiring + the real store methods are exercised, without standing up rclpy.
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pan_tilt.calib_web import CalibWebNode, make_app


WP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


@pytest.fixture
def client():
    node = object.__new__(CalibWebNode)
    node.lock = threading.Lock()
    node._loaded_cfg = {}
    node._waypoints = {
        "phase1_waypoints": [],
        "phase2_waypoints": [],
        "sanity_xarm_angles_rad": [],
    }
    node._custom_datasets = []
    webui_dir = Path(__file__).resolve().parents[1] / "webui"
    return TestClient(make_app(node, webui_dir))


def test_custom_dataset_crud_and_waypoints(client):
    # starts empty
    assert client.get("/api/calib/custom_datasets").json() == {"datasets": []}

    # create (name sanitized)
    r = client.post("/api/calib/custom_datasets", json={"name": "High Shelf"})
    assert r.status_code == 200, r.text
    assert r.json()["dataset"]["name"] == "high_shelf"

    # duplicate rejected
    assert client.post("/api/calib/custom_datasets",
                       json={"name": "high_shelf"}).status_code == 400
    # bad name rejected
    assert client.post("/api/calib/custom_datasets",
                       json={"name": "2bad"}).status_code == 400

    # waypoints addressable via the dynamic phase key
    key = "phase1_waypoints_custom:high_shelf"
    assert client.get(f"/api/waypoints/{key}").json()["waypoints"] == []
    r = client.post(f"/api/waypoints/{key}", json={"waypoints": [WP, WP]})
    assert r.status_code == 200
    assert len(r.json()["waypoints"]) == 2

    # park set + envelope enforcement
    assert client.post("/api/calib/custom_datasets/high_shelf/park",
                       json={"pan_deg": 15.0, "tilt_deg": 30.0}).status_code == 200
    assert client.post("/api/calib/custom_datasets/high_shelf/park",
                       json={"pan_deg": 99.0, "tilt_deg": 30.0}).status_code == 400

    # the dataset reflects the edits
    ds = client.get("/api/calib/custom_datasets").json()["datasets"]
    assert ds[0]["park_pan_deg"] == 15.0 and ds[0]["park_tilt_deg"] == 30.0
    assert len(ds[0]["waypoints"]) == 2

    # the generic /api/waypoints view stays static-only
    assert set(client.get("/api/waypoints").json().keys()) == {
        "phase1_waypoints", "phase2_waypoints", "sanity_xarm_angles_rad"}

    # delete; the dynamic key then 404s
    assert client.delete("/api/calib/custom_datasets/high_shelf").status_code == 200
    assert client.get(f"/api/waypoints/{key}").status_code == 404
    assert client.delete("/api/calib/custom_datasets/high_shelf").status_code == 404


def test_unknown_custom_phase_rejected(client):
    assert client.get(
        "/api/waypoints/phase1_waypoints_custom:never_made").status_code == 404
