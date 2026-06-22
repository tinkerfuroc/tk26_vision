def test_pan_tilt_calibration_core_importable():
    # The ROS nodes reuse the pan_tilt detection stack; fail loudly if the
    # dependency wiring is wrong once the workspace is built + sourced.
    import importlib
    for mod in ("pan_tilt.calibration.aruco_detect",
                "pan_tilt.calibration.safety"):
        assert importlib.util.find_spec(mod) is not None, mod
