# One observation. Multiple labels may describe the same person.
string[] labels
string description
# Pixel ROI in the returned image; right/bottom bounds are exclusive.
sensor_msgs/RegionOfInterest bbox
# Behaviour-only approximate 3D position; never consume position unless valid.
# This is a person location, not a navigation goal or a body-facing estimate.
bool position_valid
geometry_msgs/PointStamped position
string position_error
