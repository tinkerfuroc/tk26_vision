from seat_bench import score

GT = {
    "id": "s0", "image_wh": [1000, 1000],
    "seats": [
        {"label": "left",  "occupied": False, "cushion_bbox": [0, 0, 100, 100]},
        {"label": "right", "occupied": True,  "cushion_bbox": [200, 200, 300, 300]},
    ],
}


def test_point_in_empty_cushion_is_hit():
    r = score.classify({"point_xy": [50, 50], "chosen_label": "left"}, GT)
    assert r["outcome"] == "hit"


def test_point_in_occupied_cushion_is_wrong_seat():
    r = score.classify({"point_xy": [250, 250], "chosen_label": "right"}, GT)
    assert r["outcome"] == "wrong_seat"


def test_point_outside_all_is_miss():
    r = score.classify({"point_xy": [900, 900], "chosen_label": "left"}, GT)
    assert r["outcome"] == "miss"


def test_none_when_seat_available_is_false_none():
    r = score.classify({"point_xy": None, "chosen_label": "none"}, GT)
    assert r["outcome"] == "false_none"


def test_none_when_no_empty_seats_is_correct_reject():
    gt = {"image_wh": [1000, 1000],
          "seats": [{"label": "x", "occupied": True, "cushion_bbox": [0, 0, 50, 50]}]}
    r = score.classify({"point_xy": None, "chosen_label": "none"}, gt)
    assert r["outcome"] == "correct_reject"


def test_aggregate_counts_hit_rate():
    rows = [{"outcome": "hit"}, {"outcome": "miss"}, {"outcome": "hit"},
            {"outcome": "wrong_seat"}]
    agg = score.aggregate(rows)
    assert agg["n"] == 4
    assert agg["hits"] == 2
    assert abs(agg["hit_rate"] - 0.5) < 1e-9
