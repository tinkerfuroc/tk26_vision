"""Unit tests for the production bbox+select seat client (no network)."""
import pytest
from kimi_api import _seat_bbox_vlm as m
from kimi_api._seat_bbox_vlm import SeatBboxResult, VlmSeatBboxError


# --- _SYSTEM prompt regression (2026-07-01: wrong-seat incident) ---
# A real arena scene had the VLM mark a cushion occupied by a seated person
# as occupied=false; guards against silently dropping the mitigating prompt
# guidance for partially reclined / cross-legged / occluded sitters.
def test_system_prompt_warns_about_partial_occupancy():
    text = m._SYSTEM.lower()
    assert "cross-legged" in text
    assert "reclined" in text
    assert "mark it occupied=true" in text or "occupied=true rather" in text


# --- _SYSTEM prompt regression (2026-07-01: seat-suitability follow-up) ---
def test_system_prompt_prefers_sofa_and_chair_over_stool():
    text = m._SYSTEM.lower()
    assert "stool" in text and "bench" in text
    assert "wider" in text


# --- _SYSTEM prompt regression (2026-07-02: chaise-mislabeled-as-stool
# follow-up) ---
def test_system_prompt_distinguishes_chaise_from_stool():
    text = m._SYSTEM.lower()
    assert "chaise" in text or "ottoman" in text
    assert "never call it a stool" in text or "never" in text
    assert "not visible" in text


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


# --- suitability re-rank (2026-07-01: "prefer wide sofa spot / standalone
# chair over a stool" follow-up) ---
def test_seat_rank_stool_ranks_behind_sofa():
    sofa_key = m._seat_rank("right spot on sofa", (0, 0, 100, 50))
    stool_key = m._seat_rank("left stool", (0, 0, 100, 50))
    assert sofa_key < stool_key


def test_seat_rank_wider_seat_ranks_first_among_same_type():
    narrow = m._seat_rank("middle spot on sofa", (0, 0, 100, 50))
    wide = m._seat_rank("right spot on sofa", (0, 0, 250, 50))
    assert wide < narrow


# --- stool-label sanity check (2026-07-02: a live capture had the model
# mislabel the sofa's wide chaise/ottoman section as "left stool" — 258 px
# wide, wider than the real sofa cushions in the same frame, vs. a
# genuinely narrow real stool elsewhere at ~34 px in the same normalized
# units. Trusting the "stool" text label made the suitability re-rank
# recommend a cramped sliver next to an occupied cushion over the
# obviously-best, spacious, genuinely-empty chaise.) ---
def test_seat_rank_suspiciously_wide_stool_is_not_penalized():
    # "stool" as wide as the widest real sofa cushion in the same response
    # -> geometry overrides the label; it competes as high-comfort.
    wide_mislabeled_stool = m._seat_rank("left stool", (0, 0, 200, 50), max_high_comfort_width=200)
    narrow_sofa = m._seat_rank("right spot on sofa", (0, 0, 90, 50), max_high_comfort_width=200)
    assert wide_mislabeled_stool < narrow_sofa


def test_seat_rank_narrow_real_stool_stays_penalized():
    real_stool = m._seat_rank("front middle stool", (0, 0, 34, 50), max_high_comfort_width=213)
    sofa_spot = m._seat_rank("right spot on sofa", (0, 0, 30, 50), max_high_comfort_width=213)
    # Even though the sofa spot here happens to be narrower in raw pixels,
    # the stool is only ~16% of the comfort reference width -- nowhere near
    # the suspect ratio -- so it still ranks behind a real sofa spot.
    assert sofa_spot < real_stool


def test_max_high_comfort_width_ignores_stool_labeled_entries():
    seats = [
        {"label": "left stool", "box_2d": [0, 0, 900, 100], "occupied": False},
        {"label": "right spot on sofa", "box_2d": [0, 0, 200, 100], "occupied": False},
    ]
    # Only "right spot on sofa" (non-stool) counts toward the reference width.
    assert m._max_high_comfort_width(seats, 1000, 1000) == 200


# Real 2026-07-02 arena capture: the model called the sofa's wide chaise/
# ottoman (with a pillow, genuinely empty) "left stool" -- 202/1000 wide,
# comparable to the real sofa cushions (196-213/1000) and nowhere near the
# ~34/1000 width of a genuine stool seen elsewhere in this same room. It is
# in fact the single best, most spacious, genuinely-unoccupied seat here.
_CATALOG_SEATS = [
    {"label": "left stool", "box_2d": [38, 567, 240, 888], "occupied": False},
    {"label": "left spot on sofa", "box_2d": [223, 519, 420, 830], "occupied": True},
    {"label": "middle spot on sofa", "box_2d": [375, 588, 588, 822], "occupied": True},
    {"label": "right spot on sofa", "box_2d": [540, 482, 635, 712], "occupied": False},
]
_CATALOG = [
    "left stool", "front middle stool", "right stool",
    "left spot on sofa", "middle spot on sofa", "right spot on sofa",
]


def test_select_box_recognizes_mislabeled_wide_stool_as_best():
    res = m.select_box({"seats": _CATALOG_SEATS, "choice": "right spot on sofa"},
                       1000, 1000, _CATALOG)
    assert res.error is None
    assert res.label == "left stool"
    assert res.overridden_from == "right spot on sofa"


def test_select_box_no_override_when_mislabeled_stool_already_chosen():
    res = m.select_box({"seats": _CATALOG_SEATS, "choice": "left stool"},
                       1000, 1000, _CATALOG)
    assert res.label == "left stool"
    assert res.overridden_from is None


def test_select_box_prefers_sofa_over_a_genuinely_narrow_stool():
    # Same shape as _CATALOG_SEATS but the "stool" here is realistically
    # narrow (34/1000, matching the real stool measured in a different
    # scene) -- geometry no longer contradicts the label, so sofa still wins.
    seats = [
        {"label": "left stool", "box_2d": [38, 567, 72, 888], "occupied": False},
        {"label": "right spot on sofa", "box_2d": [540, 482, 635, 712], "occupied": False},
    ]
    res = m.select_box({"seats": seats, "choice": "left stool"}, 1000, 1000, _CATALOG)
    assert res.label == "right spot on sofa"
    assert res.overridden_from == "left stool"


def test_select_box_overrides_narrower_sofa_for_wider_one():
    seats = [
        {"label": "left spot on sofa", "box_2d": [0, 500, 150, 800], "occupied": False},
        {"label": "right spot on sofa", "box_2d": [400, 500, 900, 800], "occupied": False},
    ]
    res = m.select_box({"seats": seats, "choice": "left spot on sofa"}, 1000, 1000, None)
    assert res.label == "right spot on sofa"
    assert res.overridden_from == "left spot on sofa"


def test_select_box_override_ignores_out_of_catalog_alternative():
    seats = [
        {"label": "left stool", "box_2d": [0, 500, 200, 800], "occupied": False},
        {"label": "unlisted couch", "box_2d": [400, 500, 900, 800], "occupied": False},
    ]
    res = m.select_box({"seats": seats, "choice": "left stool"}, 1000, 1000,
                       ["left stool", "right stool"])
    # "unlisted couch" isn't in the catalog, so the stool choice stands
    # (its 200-wide box is also nowhere near the 900-wide reference, so it
    # isn't even flagged as suspiciously wide here).
    assert res.label == "left stool"
    assert res.overridden_from is None


# --- self-consistency backstop (2026-07-01: real arena replay found the
# model return choice="left spot on sofa" while its own seats entry for
# that exact label had "occupied": true — a plain self-contradiction, not
# a vision judgment call) ---
_SELF_CONTRADICTORY_SCENE = [
    {"label": "left stool", "box_2d": [38, 560, 240, 887], "occupied": False},
    {"label": "left spot on sofa", "box_2d": [223, 510, 420, 830], "occupied": True},
    {"label": "middle spot on sofa", "box_2d": [392, 580, 588, 770], "occupied": True},
    {"label": "right spot on sofa", "box_2d": [550, 480, 636, 710], "occupied": False},
    {"label": "front middle stool", "box_2d": [636, 520, 720, 650], "occupied": False},
    {"label": "right stool", "box_2d": [720, 500, 800, 630], "occupied": False},
]


def test_select_box_recovers_from_self_contradictory_choice():
    res = m.select_box(
        {"seats": _SELF_CONTRADICTORY_SCENE, "choice": "left spot on sofa"},
        1000, 1000, _CATALOG,
    )
    assert res.error is None
    # "left spot on sofa" is occupied per its own entry; of the genuinely
    # unoccupied seats (left stool, right spot on sofa, front middle stool,
    # right stool), "left stool" is actually the sofa's wide chaise/ottoman
    # mislabeled as a stool (202/1000 wide, comparable to the real sofa
    # cushions here) -- geometry overrides the label and it wins outright.
    assert res.label == "left stool"
    assert res.overridden_from == "left spot on sofa"


def test_select_box_self_contradictory_choice_with_no_recovery_is_error():
    seats = [
        {"label": "left spot on sofa", "box_2d": [223, 510, 420, 830], "occupied": True},
        {"label": "middle spot on sofa", "box_2d": [392, 580, 588, 770], "occupied": True},
    ]
    res = m.select_box({"seats": seats, "choice": "left spot on sofa"}, 1000, 1000, None)
    assert res.error is not None
    assert res.box_xyxy is None


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
