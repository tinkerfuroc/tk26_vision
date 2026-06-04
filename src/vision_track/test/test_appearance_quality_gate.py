from vision_track.reid.quality import crop_quality_ok, DEFAULT_GATE


def _ok(**over):
    kw = dict(
        crop_h=180, crop_w=70, mask_coverage=0.6, blur_var=200.0,
        aspect_ratio=70 / 180,
    )
    kw.update(over)
    return crop_quality_ok(**kw, **DEFAULT_GATE)


def test_good_crop_passes():
    assert _ok() is True


def test_too_short_crop_rejected():
    assert _ok(crop_h=40) is False


def test_low_mask_coverage_rejected():
    # spec: mask_coverage must exceed 0.4
    assert _ok(mask_coverage=0.3) is False
    assert _ok(mask_coverage=0.41) is True


def test_blurry_crop_rejected():
    assert _ok(blur_var=10.0) is False


def test_back_view_proxy_wide_short_rejected():
    # degenerate wide/short bbox (proxy for non-standing/back-lean) rejected
    assert _ok(aspect_ratio=1.2) is False


def test_missing_mask_coverage_does_not_hard_fail():
    # mask_coverage=None (no mask) must not crash and must not reject on coverage alone
    assert crop_quality_ok(
        crop_h=180, crop_w=70, mask_coverage=None, blur_var=200.0,
        aspect_ratio=70 / 180, **DEFAULT_GATE
    ) is True
