# test/test_track_web_cruise.py
from fastapi.testclient import TestClient
from vision_track.track_web_app import create_app


class FakeBridge:
    def __init__(self):
        self._c = {"enable_cruise_goalgate": True,
                   "enable_cruise_carrot": False, "cruise_min_gap": 0.6}
    # minimal bridge surface create_app may touch on these routes:
    def get_follow_cruise(self):
        return dict(self._c)
    def set_follow_cruise(self, s):
        self._c.update({k: s[k] for k in self._c if k in s})
        return dict(self._c)


def test_get_cruise():
    c = TestClient(create_app(FakeBridge()))
    r = c.get("/api/follow/cruise")
    assert r.status_code == 200
    assert r.json()["enable_cruise_goalgate"] is True


def test_post_cruise_updates():
    c = TestClient(create_app(FakeBridge()))
    r = c.post("/api/follow/cruise",
               json={"enable_cruise_goalgate": False,
                     "enable_cruise_carrot": True, "cruise_min_gap": 0.55})
    assert r.status_code == 200
    body = r.json()
    assert body["enable_cruise_goalgate"] is False
    assert body["enable_cruise_carrot"] is True
    assert body["cruise_min_gap"] == 0.55
