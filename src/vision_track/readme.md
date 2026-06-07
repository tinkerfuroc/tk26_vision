For testing:
```bash
ros2 launch orbbec_camera femto_bolt.launch.py enable_colored_point_cloud:=true depth_registration:=true

ros2 run vision_track person_track_server --ros-args -p reid_mode:=custom 2>&1 | stdbuf -oL -eL tee -a person_track.log || true

ros2 run vision_track person_track_test_client -d
```

## Active re-ID (behaviour-tree contract)

The person tracker exposes an active re-acquisition loop: when the target is lost
long enough, `TrackPerson` feedback escalates `reacquisition_state` to
`REACQ_NEEDS_HELP`, the BT asks the operator to raise a hand, and re-seeds the
tracker on the waving box (`ReseedTarget` at `~/reseed_target`, gallery-preserving).
See [`docs/active_reid.md`](docs/active_reid.md) for the full consumer contract.