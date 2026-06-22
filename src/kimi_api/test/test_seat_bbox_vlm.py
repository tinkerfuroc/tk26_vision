"""Unit tests for the production bbox+select seat client (no network)."""
import pytest
from kimi_api import _seat_bbox_vlm as m
from kimi_api._seat_bbox_vlm import SeatBboxResult, VlmSeatBboxError


# --- decode_box_xyxy ---
def test_decode_box_scales_and_orders():
    assert m.decode_box_xyxy([500, 500, 250, 250], 1000, 1000) == (250, 250, 500, 500)


def test_decode_box_degenerate_is_none():
    assert m.decode_box_xyxy([100, 100, 100, 100], 1000, 1000) is None


def test_decode_box_malformed_is_none():
    assert m.decode_box_xyxy("nope", 640, 480) is None
    assert m.decode_box_xyxy([1, 2, 3], 640, 480) is None


# --- select_box ---
_SEATS = [
    {"label": "left chair", "box_2d": [100, 100, 200, 300], "occupied": False},
    {"label": "right chair", "box_2d": [600, 100, 700, 300], "occupied": True},
]


def test_select_box_valid_choice_returns_box():
    res = m.select_box({"seats": _SEATS, "choice": "left chair"}, 1000, 1000, None)
    assert res.error is None
    assert res.box_xyxy == [100, 100, 200, 300]
    assert res.label == "left chair"


def test_select_box_none_is_clean_no_error():
    res = m.select_box({"seats": _SEATS, "choice": "none"}, 1000, 1000, None)
    assert res.error is None
    assert res.box_xyxy is None
    assert res.label == "none"


def test_select_box_choice_not_in_seats_is_error():
    res = m.select_box({"seats": _SEATS, "choice": "sofa"}, 1000, 1000, None)
    assert res.error is not None
    assert res.box_xyxy is None


def test_select_box_out_of_catalog_is_error():
    res = m.select_box({"seats": _SEATS, "choice": "left chair"}, 1000, 1000,
                       ["only this seat"])
    assert res.error is not None


def test_select_box_undecodable_box_is_error():
    seats = [{"label": "x", "box_2d": [5, 5, 5, 5], "occupied": False}]
    res = m.select_box({"seats": seats, "choice": "x"}, 1000, 1000, None)
    assert res.error is not None


# --- request_seat_bbox_chain (monkeypatch the per-provider call) ---
def _fake(monkeypatch, by_provider):
    """by_provider: dict provider -> (result_or_exc)."""
    def fake_request(rgb, names, features, *, provider, model, **kw):
        v = by_provider[provider]
        if isinstance(v, Exception):
            raise v
        v.provider = provider
        return v
    monkeypatch.setattr(m, "request_seat_bbox", fake_request)


def test_chain_first_success_short_circuits(monkeypatch):
    good = SeatBboxResult(label="left", box_xyxy=[1, 2, 3, 4])
    _fake(monkeypatch, {"qwen": good, "gemini": RuntimeError("should not call")})
    res = m.request_seat_bbox_chain(None, [], [],
                                    provider_models=[("qwen", "q"), ("gemini", "g")])
    assert res.box_xyxy == [1, 2, 3, 4] and res.provider == "qwen"


def test_chain_hard_error_falls_back(monkeypatch):
    good = SeatBboxResult(label="r", box_xyxy=[5, 6, 7, 8])
    _fake(monkeypatch, {"qwen": VlmSeatBboxError("boom"), "gemini": good})
    res = m.request_seat_bbox_chain(None, [], [],
                                    provider_models=[("qwen", "q"), ("gemini", "g")])
    assert res.provider == "gemini" and res.box_xyxy == [5, 6, 7, 8]


def test_chain_soft_error_falls_back(monkeypatch):
    soft = SeatBboxResult(label="bad", error="out-of-catalog")
    good = SeatBboxResult(label="r", box_xyxy=[5, 6, 7, 8])
    _fake(monkeypatch, {"qwen": soft, "gemini": good})
    res = m.request_seat_bbox_chain(None, [], [],
                                    provider_models=[("qwen", "q"), ("gemini", "g")])
    assert res.provider == "gemini"


def test_chain_legit_none_does_not_fall_back(monkeypatch):
    none_res = SeatBboxResult(label="none", box_xyxy=None, error=None)
    _fake(monkeypatch, {"qwen": none_res, "gemini": RuntimeError("should not call")})
    res = m.request_seat_bbox_chain(None, [], [],
                                    provider_models=[("qwen", "q"), ("gemini", "g")])
    assert res.label == "none" and res.box_xyxy is None and res.provider == "qwen"


def test_chain_all_fail_raises(monkeypatch):
    _fake(monkeypatch, {"qwen": VlmSeatBboxError("a"), "gemini": VlmSeatBboxError("b")})
    with pytest.raises(VlmSeatBboxError):
        m.request_seat_bbox_chain(None, [], [],
                                  provider_models=[("qwen", "q"), ("gemini", "g")])
