For testing:
```bash
ros2 launch orbbec_camera femto_bolt.launch.py enable_colored_point_cloud:=true depth_registration:=true

ros2 run vision_track person_track_server --ros-args -p reid_mode:=custom 2>&1 | stdbuf -oL -eL tee -a person_track.log || true

ros2 run vision_track person_track_test_client -d
```