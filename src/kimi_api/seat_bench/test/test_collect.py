from seat_bench import collect


def test_dedupe_keeps_first_of_identical_bytes(tmp_path):
    a = tmp_path / "a.jpg"; a.write_bytes(b"IMG1")
    b = tmp_path / "b.jpg"; b.write_bytes(b"IMG1")   # duplicate of a
    c = tmp_path / "c.jpg"; c.write_bytes(b"IMG2")
    distinct = collect.dedupe_by_content([a, b, c])
    assert len(distinct) == 2
    assert a in distinct and c in distinct
    assert b not in distinct


def test_req_path_for_orig_swaps_tokens(tmp_path):
    orig = tmp_path / "node_seat_recommend_bbox_orig_20260503_120414_420.jpg"
    expected = tmp_path / "node_seat_recommend_bbox_req_20260503_120414_420.json"
    assert collect.req_path_for_orig(orig) == expected
