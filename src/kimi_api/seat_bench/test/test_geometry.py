from seat_bench import geometry as g


def test_decode_point_yx_scales_to_pixels():
    # [y, x] = [500, 250] over a 1000x2000 (h x w) image -> (x=500, y=500)
    assert g.decode_point_yx([500, 250], w=2000, h=1000) == (500, 500)


def test_decode_point_yx_zero_sentinel_is_none():
    assert g.decode_point_yx([0, 0], w=640, h=480) is None


def test_decode_point_yx_malformed_is_none():
    assert g.decode_point_yx("nope", w=640, h=480) is None
    assert g.decode_point_yx([5], w=640, h=480) is None


def test_decode_box_xyxy_scales_and_orders():
    # swapped corners get normalized; 0-1000 -> pixels
    box = g.decode_box_xyxy([500, 500, 250, 250], w=1000, h=1000)
    assert box == (250, 250, 500, 500)


def test_decode_box_xyxy_degenerate_is_none():
    assert g.decode_box_xyxy([100, 100, 100, 100], w=1000, h=1000) is None


def test_point_in_box():
    assert g.point_in_box((50, 50), (0, 0, 100, 100)) is True
    assert g.point_in_box((150, 50), (0, 0, 100, 100)) is False


def test_box_center():
    assert g.box_center((0, 0, 100, 200)) == (50, 100)


def test_iou_identical_is_one():
    assert g.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero():
    assert g.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
