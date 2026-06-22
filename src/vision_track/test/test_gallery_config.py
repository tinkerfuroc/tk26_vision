"""YOLOTracker must apply gallery config to its target_appearance."""
from vision_track.track_yolo import YOLOTracker


def test_tracker_configures_gallery_disabled():
    # Bypass heavy __init__: only check the gallery-config plumbing applies
    # to a freshly created TargetAppearance.
    trk = YOLOTracker.__new__(YOLOTracker)
    trk.reid_gallery_enabled = False
    trk.reid_gallery_size = 4
    trk.reid_gallery_novelty_max = 0.8
    trk.reid_gallery_score_mode = "max"
    from vision_track.core.tracking_types import TargetAppearance
    ta = TargetAppearance(class_id=0, class_name="person")
    YOLOTracker._configure_gallery(trk, ta)
    assert ta.gallery.enabled is False
    assert ta.gallery.size == 4
    assert ta.gallery.novelty_max == 0.8
    assert ta.gallery.score_mode == "max"


def test_tracker_configures_gallery_enabled_top2():
    from vision_track.core.tracking_types import TargetAppearance
    trk = YOLOTracker.__new__(YOLOTracker)
    trk.reid_gallery_enabled = True
    trk.reid_gallery_size = 6
    trk.reid_gallery_novelty_max = 0.9
    trk.reid_gallery_score_mode = "top2_mean"
    ta = TargetAppearance(class_id=0, class_name="person")
    YOLOTracker._configure_gallery(trk, ta)
    assert ta.gallery.enabled is True
    assert ta.gallery.score_mode == "top2_mean"
    assert ta.gallery.novelty_max == 0.9
