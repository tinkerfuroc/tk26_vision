"""Tests for the calib_web prune endpoints.

These exercise the FastAPI surface in isolation from rclpy by passing a
stand-in node object to ``make_app``. The stand-in implements only the
attributes the prune endpoints touch — the remaining (websocket, /api/state,
camera, xArm-move) routes are exercised elsewhere.

The tests assert the operator-preview-first invariants:
  * Preview returns the headline + per-row breakdown and writes nothing.
  * Apply without ``confirm=true`` is rejected (HTTP 400).
  * Apply with ``confirm=true`` writes a sidecar yaml + report under the
    promote-target dir, leaves the original yaml untouched, and produces
    distinct filenames on rapid re-application.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from pan_tilt.calib_web import make_app


# ---- minimal stand-in node --------------------------------------------------

class _StubCalibRunner:
    def __init__(self, sessions_dir: Path):
        self.sessions_dir = sessions_dir


class _StubCalibNode:
    """Minimal duck-typed CalibWebNode for the prune tests.

    Implements only the attributes/methods the prune endpoints touch."""

    def __init__(self, *, config_path: Path, promote_yaml_out: Path,
                 sessions_dir: Path, waypoints: dict, loaded_cfg: dict):
        self.config_path = str(config_path)
        self.promote_yaml_out = promote_yaml_out
        self.calib_runner = _StubCalibRunner(sessions_dir)
        self.lock = threading.Lock()
        self._loaded_cfg = loaded_cfg
        self._waypoints = waypoints

    def list_waypoints(self, phase: str) -> list:
        return list(self._waypoints.get(phase, []))

    # Mirrors the real `_atomic_write` so the test exercises the same path.
    def _atomic_write(self, target: Path, text: str) -> None:
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(text)
        tmp.replace(target)


@pytest.fixture
def fake_robot(tmp_path: Path):
    """Spin up a stand-in node + TestClient with a small but realistic yaml."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    cfg_path = config_dir / "calibration.yaml"

    sample_cfg = {
        "collector": {
            "phase1_waypoints": [
                [0.0, 0.0, 0.0, 1.2, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.2, 0.0, -1.0, 0.5],
                [0.0, -0.3, 0.3, 1.2, 0.0, -1.0, 0.0],
                [0.0, -0.3, 0.3, 1.2, 0.0, -0.7, 0.0],
                [-0.5, 0.0, 0.5, 1.4, 0.0, -0.5, 0.0],
                [-0.9, 0.0, 0.5, 1.6, 0.0, -0.5, 0.0],
                [0.5, 0.5, -0.5, 1.0, 0.0, -1.2, 0.5],
                [0.7, -0.5, 0.5, 1.5, 0.0, -0.6, -0.3],
            ],
            "phase1_waypoints_custom": [],
            "phase2_waypoints": [[0.0, 0.0, 0.0, 1.2, 0.0, -1.0, 0.0]],
            "pan_grid_deg": [-30.0, 0.0, 30.0],
            "tilt_grid_deg": [15.0, 30.0, 45.0],
        },
    }
    cfg_path.write_text(yaml.safe_dump(sample_cfg, sort_keys=False))

    sessions_dir = tmp_path / "calibration_data"
    sessions_dir.mkdir()
    # Drop a fake prior-run file so the prior_runs picker has something to
    # show + so the replay predictor exercises the keyed-by-label path.
    fake_run = sessions_dir / "20260101_dummy"
    fake_run.mkdir()
    fake_handeye = fake_run / "phase1_handeye.json"
    handeye_payload = {
        "samples": [
            {
                "label": f"phase1/{i}",
                "theta_pan_rad": 0.0, "theta_tilt_rad": 0.0,
                "t_base_ee": {
                    "translation": [0.1 * i, 0.0, 1.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
                "t_cam_marker_body": {
                    "translation": [0.0, 0.0, 0.5],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
                "image_stamp_ns": 0, "state_stamp_ns": 0,
                "detection_quality": 24, "reprojection_rms_px": 0.1,
            }
            for i in range(8)
        ],
    }
    fake_handeye.write_text(json.dumps(handeye_payload))

    node = _StubCalibNode(
        config_path=cfg_path,
        promote_yaml_out=cfg_path,         # promote target = source for the test
        sessions_dir=sessions_dir,
        waypoints={
            "phase1_waypoints": sample_cfg["collector"]["phase1_waypoints"],
            "phase1_waypoints_custom": [],
            "phase2_waypoints": sample_cfg["collector"]["phase2_waypoints"],
            "sanity_xarm_angles_rad": [],
        },
        loaded_cfg=sample_cfg["collector"],
    )

    webui_dir = Path(__file__).resolve().parents[1] / "webui"
    app = make_app(node, webui_dir)
    client = TestClient(app)
    yield {
        "client": client,
        "node": node,
        "config_path": cfg_path,
        "config_dir": config_dir,
        "fake_handeye": fake_handeye,
    }


# ---- tests ------------------------------------------------------------------

def _config_dir_files(config_dir: Path) -> set[str]:
    return {p.name for p in config_dir.iterdir() if p.is_file()}


def _file_md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def test_prune_inputs_lists_phase_and_prior_runs(fake_robot):
    r = fake_robot["client"].get("/api/calib/prune_inputs?phase=phase1_waypoints")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["phase"] == "phase1_waypoints"
    assert data["n_items"] == 8
    assert "trans_tol_m" in data["default_factors"]
    assert any(run["path"].endswith("phase1_handeye.json")
               for run in data["prior_runs"])


def test_prune_inputs_rejects_unknown_phase(fake_robot):
    r = fake_robot["client"].get("/api/calib/prune_inputs?phase=garbage")
    assert r.status_code == 404


def test_preview_returns_headline_no_write(fake_robot):
    before_md5 = _file_md5(fake_robot["config_path"])
    before_files = _config_dir_files(fake_robot["config_dir"])

    body = {
        "phase": "phase1_waypoints",
        "factors": {"trans_tol_m": 0.05, "rot_tol_deg": 8.0,
                    "min_count": 4, "min_rot_diversity_pairs": 0},
        "predictor_choice": "auto",
        "prior_run_path": str(fake_robot["fake_handeye"]),
    }
    r = fake_robot["client"].post("/api/calib/prune_preview", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "headline" in data and data["headline"].startswith("Will keep")
    assert isinstance(data["kept_indices"], list)
    assert isinstance(data["dropped_indices"], list)
    assert len(data["items"]) == 8
    assert data["wrote"] is None

    # No files written by Preview.
    assert _config_dir_files(fake_robot["config_dir"]) == before_files
    assert _file_md5(fake_robot["config_path"]) == before_md5


def test_apply_without_confirm_rejects(fake_robot):
    body = {
        "phase": "phase1_waypoints",
        "factors": {"min_count": 4, "min_rot_diversity_pairs": 0},
        "predictor_choice": "auto",
        "prior_run_path": str(fake_robot["fake_handeye"]),
    }
    r = fake_robot["client"].post("/api/calib/prune_apply", json=body)
    assert r.status_code == 400
    assert "confirm" in r.text


def test_apply_with_confirm_writes_sidecar(fake_robot):
    config_dir = fake_robot["config_dir"]
    before_md5 = _file_md5(fake_robot["config_path"])

    body = {
        "phase": "phase1_waypoints",
        "factors": {"trans_tol_m": 0.05, "rot_tol_deg": 8.0,
                    "min_count": 4, "min_rot_diversity_pairs": 0},
        "predictor_choice": "auto",
        "prior_run_path": str(fake_robot["fake_handeye"]),
        "confirm": True,
    }
    r = fake_robot["client"].post("/api/calib/prune_apply", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    sidecar = Path(data["wrote"]["sidecar_yaml"])
    report = Path(data["wrote"]["report_json"])
    assert sidecar.exists()
    assert report.exists()
    assert sidecar.parent == config_dir

    # Original is unchanged.
    assert _file_md5(fake_robot["config_path"]) == before_md5

    # Sidecar parses, has the same shape, and `phase1_waypoints` is the
    # filtered subset.
    sidecar_data = yaml.safe_load(sidecar.read_text())
    src_data = yaml.safe_load(fake_robot["config_path"].read_text())
    src_phase1 = src_data["collector"]["phase1_waypoints"]
    side_phase1 = sidecar_data["collector"]["phase1_waypoints"]
    assert len(side_phase1) == len(data["kept_indices"])
    # Every kept entry is verbatim from the source.
    for kept_idx, side_row in zip(sorted(data["kept_indices"]), side_phase1):
        assert side_row == src_phase1[kept_idx]

    # Report json round-trips and carries the same kept_indices.
    report_data = json.loads(report.read_text())
    assert report_data["kept_indices"] == data["kept_indices"]
    assert report_data["headline"] == data["headline"]
    assert report_data["phase"] == "phase1_waypoints"


def test_apply_idempotent_filenames(fake_robot):
    body = {
        "phase": "phase1_waypoints",
        "factors": {"min_count": 4, "min_rot_diversity_pairs": 0},
        "predictor_choice": "auto",
        "prior_run_path": str(fake_robot["fake_handeye"]),
        "confirm": True,
    }
    r1 = fake_robot["client"].post("/api/calib/prune_apply", json=body)
    r2 = fake_robot["client"].post("/api/calib/prune_apply", json=body)
    assert r1.status_code == 200 and r2.status_code == 200
    p1 = r1.json()["wrote"]["sidecar_yaml"]
    p2 = r2.json()["wrote"]["sidecar_yaml"]
    assert p1 != p2, "two applies must produce distinct filenames even within 1 second"


def test_phase2_grid_apply_emits_grid_pairs(fake_robot):
    body = {
        "phase": "phase2_grid",
        "factors": {"trans_tol_m": 1.0, "rot_tol_deg": 60.0,
                    "min_count": 2, "min_rot_diversity_pairs": 0},
        "predictor_choice": "fk_only",
        "confirm": True,
    }
    r = fake_robot["client"].post("/api/calib/prune_apply", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    sidecar = Path(data["wrote"]["sidecar_yaml"])
    sd = yaml.safe_load(sidecar.read_text())
    assert "phase2_grid_pairs" in sd["collector"]
    assert isinstance(sd["collector"]["phase2_grid_pairs"], list)
    assert len(sd["collector"]["phase2_grid_pairs"]) == len(data["kept_indices"])
    # Each kept pair lies on the original grid.
    for pan, tilt in sd["collector"]["phase2_grid_pairs"]:
        assert pan in (-30.0, 0.0, 30.0)
        assert tilt in (15.0, 30.0, 45.0)


def test_unknown_predictor_choice_rejected(fake_robot):
    body = {
        "phase": "phase1_waypoints",
        "predictor_choice": "telepathy",
    }
    r = fake_robot["client"].post("/api/calib/prune_preview", json=body)
    assert r.status_code == 400
    assert "predictor_choice" in r.text


def test_overwrite_without_confirm_rejects(fake_robot):
    body = {
        "phase": "phase1_waypoints",
        "factors": {"min_count": 4, "min_rot_diversity_pairs": 0},
        "predictor_choice": "auto",
        "prior_run_path": str(fake_robot["fake_handeye"]),
    }
    r = fake_robot["client"].post("/api/calib/prune_overwrite", json=body)
    assert r.status_code == 400
    assert "confirm" in r.text


def test_overwrite_writes_source_and_backs_up(fake_robot):
    config_path = fake_robot["config_path"]
    config_dir = fake_robot["config_dir"]
    pre_md5 = _file_md5(config_path)
    pre_text = config_path.read_text()

    body = {
        "phase": "phase1_waypoints",
        "factors": {"trans_tol_m": 0.05, "rot_tol_deg": 8.0,
                    "min_count": 4, "min_rot_diversity_pairs": 0},
        "predictor_choice": "auto",
        "prior_run_path": str(fake_robot["fake_handeye"]),
        "confirm": True,
    }
    r = fake_robot["client"].post("/api/calib/prune_overwrite", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    written = data["wrote"]
    assert Path(written["wrote_yaml"]) == config_path
    backup = Path(written["backup_yaml"])
    assert backup.exists()
    assert backup.parent == config_dir
    assert backup.name.startswith("calibration.yaml.old-")

    # Backup is byte-identical to the pre-call source.
    assert _file_md5(backup) == pre_md5
    assert backup.read_text() == pre_text

    # Source now carries the pruned set + a generated header.
    new_text = config_path.read_text()
    assert "Overwritten by calib_web prune-overwrite" in new_text
    new_data = yaml.safe_load(new_text)
    src_phase1 = yaml.safe_load(pre_text)["collector"]["phase1_waypoints"]
    side_phase1 = new_data["collector"]["phase1_waypoints"]
    assert len(side_phase1) == len(data["kept_indices"])
    for kept_idx, side_row in zip(sorted(data["kept_indices"]), side_phase1):
        assert side_row == src_phase1[kept_idx]

    # Audit report exists + carries the same kept_indices.
    report = Path(written["report_json"])
    assert report.exists()
    report_data = json.loads(report.read_text())
    assert report_data["kept_indices"] == data["kept_indices"]
    assert report_data["overwrote_yaml"] == str(config_path)
    assert report_data["backup_yaml"] == str(backup)


def test_overwrite_refuses_without_promote_target(fake_robot):
    fake_robot["node"].promote_yaml_out = None
    body = {
        "phase": "phase1_waypoints",
        "factors": {"min_count": 4, "min_rot_diversity_pairs": 0},
        "predictor_choice": "auto",
        "prior_run_path": str(fake_robot["fake_handeye"]),
        "confirm": True,
    }
    r = fake_robot["client"].post("/api/calib/prune_overwrite", json=body)
    assert r.status_code == 400
    assert "promote" in r.text or "source-tree" in r.text


def test_overwrite_phase2_grid_emits_grid_pairs(fake_robot):
    body = {
        "phase": "phase2_grid",
        "factors": {"trans_tol_m": 1.0, "rot_tol_deg": 60.0,
                    "min_count": 2, "min_rot_diversity_pairs": 0},
        "predictor_choice": "fk_only",
        "confirm": True,
    }
    r = fake_robot["client"].post("/api/calib/prune_overwrite", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    target = Path(data["wrote"]["wrote_yaml"])
    sd = yaml.safe_load(target.read_text())
    assert "phase2_grid_pairs" in sd["collector"]
    assert len(sd["collector"]["phase2_grid_pairs"]) == len(data["kept_indices"])


def test_preview_surfaces_predict_failures(fake_robot):
    # Use the prior-run path that only has phase1/0..7 — the synthetic yaml
    # also has 8 entries so all labels match. Force the missing case by
    # asking replay_only with a path that has fewer labels: create a sparse
    # fake handeye with just 3 labels.
    sparse = fake_robot["fake_handeye"].parent / "phase1_handeye_sparse.json"
    sparse.write_text(json.dumps({
        "samples": [
            {
                "label": f"phase1/{i}",
                "theta_pan_rad": 0.0, "theta_tilt_rad": 0.0,
                "t_base_ee": {
                    "translation": [0.1 * i, 0.0, 1.0],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
                "t_cam_marker_body": {
                    "translation": [0.0, 0.0, 0.5],
                    "rotation": [0.0, 0.0, 0.0, 1.0],
                },
                "image_stamp_ns": 0, "state_stamp_ns": 0,
                "detection_quality": 24, "reprojection_rms_px": 0.1,
            }
            for i in range(3)
        ],
    }))

    body = {
        "phase": "phase1_waypoints",
        "factors": {"min_count": 2, "min_rot_diversity_pairs": 0},
        "predictor_choice": "replay_only",
        "prior_run_path": str(sparse),
    }
    r = fake_robot["client"].post("/api/calib/prune_preview", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["diagnostics"]["n_predict_failed"] >= 5
    # 5 of 8 missing => 62.5% > 20% threshold => stale-prior warning fires.
    assert "warning" in data["diagnostics"]
    assert "stale" in data["diagnostics"]["warning"]
