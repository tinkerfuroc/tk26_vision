Calibrates a pan-tilt mounted upon a robot with a camera attatched.
Length of pan joint and tilt joint is known before hand, transformation from `base_link` to `pan_link` and from `tilt_link` to `camera_link` is fixed but unkown.
Data is given in `data/measurements.json`, measurements for angles, length, and orientation are given in degrees, meters, and quaterion, respectively. A tilt angle of `0` degrees means the camera is upright, negative degrees mean the camera is leaning downwards (towards the ground). A positive pan angle means the camera is pointing to the right.
The goal is to calibrate the transformation from `base_link` to `pan_link` and from `tilt_link` to `camera_link` from the measurements given.
