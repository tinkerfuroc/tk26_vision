# tk26_vision / thirdparty

Vendored camera drivers for the 2026 vision stack. Cloned at a pinned release tag and checked in with the upstream `.git` stripped, so the tree is reproducible without submodules.

These supersede the legacy copies under `src/tk23_vision/src/realsense-ros` and `src/tk23_vision/src/OrbbecSDK_ROS2` (both already `COLCON_IGNORE`'d or empty).

## Pinned versions

| Driver             | Upstream                                               | Tag      | Commit SHA                                 |
| ------------------ | ------------------------------------------------------ | -------- | ------------------------------------------ |
| `librealsense`     | https://github.com/realsenseai/librealsense            | `v2.57.7`| `fec2d156e531f417c927262818f3440cfbcde4e9` |
| `realsense-ros`    | https://github.com/realsenseai/realsense-ros           | `4.57.7` | `5c2244ca5cd9867c9ee63769668891430f460dfd` |
| `OrbbecSDK_ROS2`   | https://github.com/orbbec/OrbbecSDK_ROS2               | `v2.7.6` | `c0c14f538a9faf24319d246a27580308c17b1b5e` |

RealSense SDK and ROS wrapper share a major/minor pair — `librealsense v2.X.Y` matches `realsense-ros 4.X.Y`. Upgrade them together.

To refresh a driver:

```bash
cd src/tk26_vision/thirdparty
rm -rf <driver>
git clone --depth=1 --branch <tag> <upstream-url> <driver>
rm -rf <driver>/.git <driver>/.github
# then update the table above with the new tag + `git rev-parse HEAD` SHA
```

## RealSense

### Build `librealsense` from source (one-time)

The source is vendored at `thirdparty/librealsense/` (pinned to `v2.57.7` to match the ROS wrapper). The upstream `realsenseai` apt repo is skipped on purpose — building from source is cleaner and avoids the broken-host / DKMS-rebuild churn that hit the apt path.

Install build-time deps:

```bash
sudo apt install -y \
  git cmake build-essential pkg-config \
  libssl-dev libusb-1.0-0-dev libudev-dev \
  libgtk-3-dev libglfw3-dev \
  libgl1-mesa-dev libglu1-mesa-dev \
  python3-dev
```

Install the udev rules + patch the UVC kernel module (required for non-root USB access and for frame metadata on LTS kernels 5.15 / 5.19 / 6.5 / 6.8 / 6.11 / 6.14):

```bash
cd src/tk26_vision/thirdparty/librealsense
sudo ./scripts/setup_udev_rules.sh
./scripts/patch-realsense-ubuntu-lts-hwe.sh   # prompts for sudo; reboot after if it patches
```

Configure, build, and install system-wide:

```bash
cd src/tk26_vision/thirdparty/librealsense
cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=false \
  -DBUILD_GRAPHICAL_EXAMPLES=false \
  -DBUILD_PYTHON_BINDINGS=false
cmake --build build -j"$(nproc)"
sudo cmake --install build
sudo ldconfig
```

Sanity check: `realsense-viewer` (or `rs-enumerate-devices`) sees the camera.

### rosdep

```bash
rosdep install --from-paths src/tk26_vision/thirdparty/realsense-ros --ignore-src -r -y
```

`librealsense2` is picked up from the system install above via `pkg-config`/`find_package(realsense2)` during the colcon build.

### Why the `librealsense2_ament/` shim

`realsense2_camera/package.xml` declares `<depend>librealsense2</depend>`. That rosdep key maps to the apt package `ros-humble-librealsense2`, which ships ament marker files at `/opt/ros/humble/share/librealsense2/`. Our source build installs to `/usr/local/` and produces CMake config files but **not** ament markers, so colcon refuses to build `realsense2_camera` — it can't find the ament install marker it expects from the `<depend>`.

`thirdparty/librealsense2_ament/` is a tiny `ament_cmake` package named `librealsense2` that `find_package(realsense2 2.57.7 REQUIRED)`s the source-built SDK and `ament_export_dependencies(realsense2)`. Colcon sees a workspace package matching the `<depend>` name, produces the expected ament marker in `install/librealsense2/`, and downstream packages pick up the real library through `/usr/local`'s CMake config. No apt install, no `/opt/ros/humble` pollution.

`thirdparty/librealsense/` carries a `COLCON_IGNORE` so colcon doesn't try to treat the SDK source tree as a workspace package (its `project(librealsense2)` would otherwise collide with the shim).

## Orbbec

OrbbecSDK v2 is **bundled** inside `OrbbecSDK_ROS2/orbbec_camera/SDK/` (headers + prebuilt libs) — no separate SDK install needed.

### Udev rules (one-time, system-level)

Required so non-root processes can open the USB device:

```bash
cd src/tk26_vision/thirdparty/OrbbecSDK_ROS2/orbbec_camera/scripts
sudo bash install_udev_rules.sh
sudo udevadm control --reload && sudo udevadm trigger
```

### rosdep

```bash
rosdep install --from-paths src/tk26_vision/thirdparty/OrbbecSDK_ROS2 --ignore-src -r -y
```

## Build

Both are `ament_cmake` C++ packages — no Python-shebang issue, the normal wrapper build works:

```bash
./src/tk26_vision/scripts/build.sh --packages-select \
  realsense2_camera_msgs realsense2_camera realsense2_description \
  orbbec_camera_msgs orbbec_camera orbbec_description
```

A full `./src/tk26_vision/scripts/build.sh` also picks them up.

## Smoke tests

```bash
# RealSense (e.g. D435 on the xArm wrist)
ros2 launch realsense2_camera rs_launch.py camera_name:=xarm_camera align_depth.enable:=true
ros2 topic hz /xarm_camera/color/image_raw

# Orbbec Femto Bolt
ros2 launch orbbec_camera femto_bolt.launch.py enable_colored_point_cloud:=true depth_registration:=true
ros2 topic hz /camera/color/image_raw
```
