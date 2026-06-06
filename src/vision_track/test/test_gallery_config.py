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
