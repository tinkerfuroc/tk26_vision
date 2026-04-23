> Reference only
> This draft predates the implementation now in this repo.
> The final clean refactor diverged in a few important ways:
> - there is no `pan_tilt_state_tf_node`
> - there is no separate `pan_tilt_msgs` package; the real interfaces live in
>   `tinker_vision_msgs_26`
> - the runtime deliberately does not keep the old `PanTiltCtrl` or
>   `/pan_tilt_ctrl*` control path alive
> - TF now comes from `/joint_states` plus `robot_state_publisher`, not from a
>   dedicated state+TF node
> - `config/specs.json` is no longer loaded at runtime

I’d split it into two ROS 2 nodes:

1. a **low-level serial driver / controller node**
2. a **state + TF node**

That matches ROS 2’s “one node, one main job” style, keeps the serial logic isolated, and makes the transform side easy to debug and replace later. ROS 2 also treats parameters as node-scoped configuration, and tf2 is designed around one or more broadcasters publishing frame relationships, with static transforms handled separately from changing ones. ([ROS Documentation][1])

## 1) Node A: `pan_tilt_controller_node`

This node owns the USB serial port and is the **only** process allowed to talk to the pan-tilt firmware.

### Responsibilities

* open `/dev/...` at `115200`
* wait for boot
* parse newline-terminated JSON feedback
* send motion commands as `{"T":133,...}`
* publish the latest hardware state to ROS
* expose a few maintenance services like torque enable/disable

Because the firmware already auto-streams `T:1001` feedback, this node should be mostly **event-driven** rather than polling.

### ROS interfaces

#### Subscriptions

`~/cmd`
Recommended custom message:

```text
# pan_tilt_msgs/msg/PanTiltCommand.msg
builtin_interfaces/Time stamp
float64 pan_rad
float64 tilt_rad
int32 speed_raw
int32 accel_raw
```

I would keep ROS-side angles in **radians** even though the firmware uses degrees. That makes the rest of the robot stack consistent with standard ROS joint conventions.

#### Publications

`~/state`
Custom message:

```text
# pan_tilt_msgs/msg/PanTiltState.msg
std_msgs/Header header
float64 pan_rad
float64 tilt_rad
bool connected
bool feedback_ok
```

Optional debug publications:

* `~/raw_rx` as `std_msgs/String`
* `~/raw_tx` as `std_msgs/String`

#### Services

Useful runtime services:

```text
# pan_tilt_msgs/srv/SetTorque.srv
bool enable
---
bool success
string message
```

```text
# pan_tilt_msgs/srv/SetZero.srv
uint8 axis   # 0=both, 1=tilt, 2=pan
---
bool success
string message
```

Optional but useful:

* `~/force_gimbal_mode`
* `~/enable_feedback`
* `~/go_home`

I would **not** expose motor ID reassignment as a normal runtime service. That belongs in a separate provisioning tool because it is easy to misuse.

---

## 2) Node behavior of `pan_tilt_controller_node`

### Startup sequence

On startup:

* open serial port
* wait `startup_delay_sec`
* start reading lines
* if no valid `T:1001` arrives within `feedback_startup_timeout_sec`, optionally send:

  * `{"T":4,"cmd":2}` to force gimbal mode
  * `{"T":131,"cmd":1}` to re-enable feedback
* mark the node connected only after the first valid feedback frame

### Command handling

When a `~/cmd` message arrives:

1. convert radians to degrees
2. apply any software inversion flags if needed
3. clamp to firmware limits:

   * pan `[-180, 180]`
   * tilt `[-30, 90]`
4. send:

```json
{"T":133,"X":PAN_DEG,"Y":TILT_DEG,"SPD":speed_raw,"ACC":accel_raw}
```

### Feedback handling

When a line like this arrives:

```json
{"T":1001,"X":0,"Y":0}
```

the node should:

* validate JSON
* ignore unrelated message types
* convert `X`, `Y` to radians
* timestamp with ROS time
* publish `~/state`

### Parameters

Suggested parameters:

```yaml
pan_tilt_controller:
  ros__parameters:
    port: /dev/ttyUSB0
    baudrate: 115200
    startup_delay_sec: 4.0
    feedback_startup_timeout_sec: 2.0
    feedback_stale_timeout_sec: 0.5
    pan_min_deg: -180.0
    pan_max_deg: 180.0
    tilt_min_deg: -30.0
    tilt_max_deg: 90.0
    default_speed_raw: 120
    default_accel_raw: 20
    send_force_gimbal_on_startup: true
    send_enable_feedback_on_startup: true
    invert_pan: false
    invert_tilt: false
```

---

## 3) Node B: `pan_tilt_state_tf_node`

This node does **not** touch the serial port.
Its job is to maintain the latest pan/tilt position and publish ROS-friendly state outputs.

### Responsibilities

* subscribe to `~/state`
* remember the latest pan/tilt angles
* publish `sensor_msgs/msg/JointState`
* publish the dynamic transforms for the pan and tilt chain
* optionally publish diagnostics if feedback becomes stale

### ROS interfaces

#### Subscription

`/pan_tilt_controller/state`
Uses `pan_tilt_msgs/msg/PanTiltState`

#### Publications

`/joint_states` as `sensor_msgs/msg/JointState`

Optional direct TF:

* `/tf` with `geometry_msgs/msg/TransformStamped`

### Internal state

It should maintain:

* `last_pan_rad`
* `last_tilt_rad`
* `last_feedback_stamp`
* `have_state`

and check whether feedback is stale.

---

## 4) Transform design

There are two good ways to do the transforms.

### Recommended ROS-native way

**Best option:** let this node publish only `/joint_states`, and let `robot_state_publisher` produce TF from a URDF.

That is usually cleaner, because the geometry belongs in URDF and the state node only publishes joint values.

In that setup:

* `pan_tilt_state_tf_node` publishes:

  * `pan_joint`
  * `tilt_joint`
* `robot_state_publisher` publishes `/tf`

### Direct-TF way

If you want this second node to publish transforms itself, use a chain like this:

* `pan_tilt_base_link` → `pan_link` : rotation by pan
* `pan_link` → `tilt_link` : fixed translation from pan axis to tilt axis
* `tilt_link` → `camera_mount_link` : rotation by tilt
* `camera_mount_link` → `orbbec_link` : fixed translation/rotation

I would still keep the fixed camera optical transforms elsewhere if the camera driver already publishes them.

---

## 5) Recommended frame model

Assuming:

* pan rotates about local `+Z`
* tilt rotates about local `+Y`

then the chain is:

```text
pan_tilt_base_link
  └── pan_link        (Rz(pan))
        └── tilt_axis_link    (fixed xyz offset to tilt pivot)
              └── tilt_link   (Ry(tilt))
                    └── camera_mount_link (fixed xyz/rpy offset)
```

Important: the README gives angle limits, but it does **not** give reliable mechanical offsets.
So the node should not hardcode dimensions. Put these in parameters.

Suggested parameters:

```yaml
pan_tilt_state_tf:
  ros__parameters:
    base_frame: pan_tilt_base_link
    pan_frame: pan_link
    tilt_axis_frame: tilt_axis_link
    tilt_frame: tilt_link
    camera_frame: camera_mount_link

    tilt_axis_xyz: [0.0, 0.0, 0.045]      # example only
    tilt_axis_rpy: [0.0, 0.0, 0.0]

    camera_xyz: [0.0, 0.0, 0.02]          # example only
    camera_rpy: [0.0, 0.0, 0.0]

    publish_tf: true
    publish_joint_states: true
    stale_timeout_sec: 0.5
```

Since you already know the tilt length is unreliable, this is exactly the kind of quantity that should remain parameterized.

---

## 6) Publishing `JointState`

Every time a fresh state message arrives, publish:

```text
name:     ["pan_joint", "tilt_joint"]
position: [pan_rad, tilt_rad]
velocity: []
effort:   []
```

If no new hardware feedback arrives for too long:

* keep the last value
* set diagnostics to warning
* optionally stop publishing TF if you want stale transforms to be obvious

---

## 7) Suggested package structure

```text
pan_tilt_msgs/
  msg/
    PanTiltCommand.msg
    PanTiltState.msg
  srv/
    SetTorque.srv
    SetZero.srv

pan_tilt_driver/
  src/
    pan_tilt_controller_node.cpp
    pan_tilt_state_tf_node.cpp
  launch/
    pan_tilt.launch.py
  config/
    pan_tilt.yaml
  urdf/
    pan_tilt.urdf.xacro
```

---

## 8) Launch architecture

A clean launch setup is:

* `pan_tilt_controller_node`
* `pan_tilt_state_tf_node`
* `robot_state_publisher` with pan-tilt URDF

ROS 2 launch files are the normal place to wire together node params, namespaces, remaps, and reusable bring-up structure. ([ROS Documentation][2])

---

## 9) Optional lifecycle version

If you want more robust bring-up, make the controller node a **LifecycleNode**:

* `configure`: declare params, open serial
* `activate`: begin reading and publishing
* `deactivate`: stop timers/subscriptions
* `cleanup`: close serial

Managed lifecycles are useful in ROS 2 when you want controlled startup, restart, and replacement of hardware-facing nodes. ([ROS Documentation][3])

---

## 10) Concrete data flow

```text
user / planner / tracker
        │
        ▼
  /pan_tilt_controller/cmd
        │
        ▼
pan_tilt_controller_node
  - serial TX: {"T":133,...}
  - serial RX: {"T":1001,"X":...,"Y":...}
        │
        ▼
 /pan_tilt_controller/state
        │
        ▼
pan_tilt_state_tf_node
  - /joint_states
  - /tf   (or robot_state_publisher does TF)
```

---

## 11) My recommendation

If you want the most maintainable version, I would implement it like this:

* **Node 1**: serial controller, custom command/state messages, torque/zero services
* **Node 2**: subscribes to state, publishes `/joint_states`
* **URDF + robot_state_publisher**: publishes the transform tree

That keeps geometry in URDF, hardware in the driver, and state in standard ROS messages.

If you want, I can turn this into a concrete ROS 2 Humble package skeleton in C++ with message definitions, node class layout, and a launch file.

[1]: https://docs.ros.org/en/humble/Tutorials/Beginner-CLI-Tools/Understanding-ROS2-Nodes/Understanding-ROS2-Nodes.html?utm_source=chatgpt.com "Understanding nodes — ROS 2 Documentation: Humble ..."
[2]: https://docs.ros.org/en/humble/Tutorials/Intermediate/Launch/Using-ROS2-Launch-For-Large-Projects.html?utm_source=chatgpt.com "Managing large projects — ROS 2 Documentation"
[3]: https://docs.ros.org/en/humble/Tutorials/Demos/Managed-Nodes.html?utm_source=chatgpt.com "Managing nodes with managed lifecycles"
