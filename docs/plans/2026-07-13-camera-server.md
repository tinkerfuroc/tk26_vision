# Camera Server Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the per-camera C++ camera servers (synced-frame snapshot, on-demand point cloud, warm-TF lookup services) plus the param-gated legacy compat bridge, per `docs/specs/2026-07-13-camera-server-design.md`.

**Architecture:** One new ament_cmake package `camera_server` (under `src/tk26_vision/src/`) providing a `CameraServerNode` launched twice (wrist RealSense / head Orbbec instances), each owning its camera's only streaming subscriptions and a 180 s TF buffer; three new services in `tinker_vision_msgs_26`; a zero-subscription `camera_compat_bridge` forwarder for the legacy service names, OFF by default. **Servers only — no consumer package is modified.**

**Tech Stack:** ROS 2 Humble, rclcpp + rclcpp_components, message_filters (ApproximateTime), tf2_ros/tf2_eigen, Eigen3, gtest (ament_cmake_gtest), rclpy for T-tier check scripts.

## Global Constraints

- Repo: all work in `/home/tinker/tk25_ws/src/tk26_vision` (its own git repo, branch `tinker2-net`). **Concurrent committer active** — never amend/rebase/reset; commit new commits only; re-check `git status` before every commit and stage only your own files.
- **One task = exactly one git commit** (workspace phase-per-commit policy). No bundling tasks into one commit, no splitting a task across commits.
- Build ONLY via `tkbuild` (alias for `/home/tinker/tk25_ws/tkbuild`): `tkbuild tk26_vision --packages-select <pkg>`. NEVER raw `colcon`. Built artifacts land in `/home/tinker/tk25_ws/build/<pkg>/` and `/home/tinker/tk25_ws/install/<pkg>/`.
- Source environment before ros2/test commands: `source /home/tinker/tk25_ws/install/setup.zsh` (or `setup.bash` in bash).
- `camera_server/README.md` has an append-only `## Changelog` section — every task that touches `camera_server` appends one line **in the same commit**.
- Status convention (spec §4): `int32 status` 0 = OK, non-zero failure classes defined as srv constants; `string error_msg`.
- QoS (spec §5.1): image subs BEST_EFFORT/VOLATILE/KEEP_LAST(5); camera_info subs RELIABLE/VOLATILE/KEEP_LAST(10). Sync: ApproximateTime, queue 10, max interval 0.1 s.
- TF: buffer cache 180 s default, dedicated listener thread, per-lookup timeout default 0.1 s, `GetTransform` timeout capped at 2.0 s.
- Do NOT touch: any consumer node, the three Python utility nodes, `thirdparty/`, or anything in the in-flight kimi_api / tk_vision_specialized work.
- Executors: append your session's standard commit footer (Co-Authored-By etc.) to every commit message shown below.

---

## File Structure (end state)

```
src/tk26_vision/src/tinker_vision_msgs_26/
  srv/GetCameraSnapshot.srv          (new)
  srv/GetCameraPointCloud.srv        (new)
  srv/GetTransform.srv               (new)
  msg/CameraServerStatus.msg         (new)
  CMakeLists.txt                     (modified: 4 entries + builtin_interfaces)
  package.xml                        (modified: builtin_interfaces dep)

src/tk26_vision/src/camera_server/
  package.xml
  CMakeLists.txt
  README.md
  include/camera_server/frame_store.hpp    — thread-safe latest store + captured_after wait
  include/camera_server/deprojector.hpp    — depth→PointCloud2 with cached xy-table
  include/camera_server/camera_server_node.hpp
  src/frame_store.cpp
  src/deprojector.cpp
  src/camera_server_node.cpp               — subs/sync/TF/status + 3 service handlers, component-registered
  src/camera_server_main.cpp                — standalone executable main
  src/compat_bridge_node.cpp                — legacy-name forwarder + main
  launch/camera_server.launch.py
  test/test_frame_store.cpp
  test/test_deprojector.cpp

src/tk26_vision/src/vision_bringup/launch/vision_driver.launch.py  (modified: gated head server)
src/tk26_vision/scripts/tests/manual/camera_server_t1.sh           (new)
src/tk26_vision/scripts/tests/manual/camera_server_t2_parity.py    (new)
src/tk26_vision/DEV_NOTES.md                                       (appended)
```

---

### Task 1: Service/message interfaces in `tinker_vision_msgs_26`

**Files:**
- Create: `src/tk26_vision/src/tinker_vision_msgs_26/srv/GetCameraSnapshot.srv`
- Create: `src/tk26_vision/src/tinker_vision_msgs_26/srv/GetCameraPointCloud.srv`
- Create: `src/tk26_vision/src/tinker_vision_msgs_26/srv/GetTransform.srv`
- Create: `src/tk26_vision/src/tinker_vision_msgs_26/msg/CameraServerStatus.msg`
- Modify: `src/tk26_vision/src/tinker_vision_msgs_26/CMakeLists.txt`
- Modify: `src/tk26_vision/src/tinker_vision_msgs_26/package.xml`
- Modify: `src/tk26_vision/docs/specs/2026-07-13-camera-server-design.md` (one field rename, see Step 1)

**Interfaces:**
- Consumes: nothing.
- Produces (used by Tasks 4–6, 8): `tinker_vision_msgs_26/srv/GetCameraSnapshot`, `.../GetCameraPointCloud`, `.../GetTransform`, `tinker_vision_msgs_26/msg/CameraServerStatus` — exact fields below.

- [ ] **Step 1: Write the four interface files**

Note: the spec's `GetTransform` request field `time` is renamed `lookup_time` (bare `time` collides with IDL keyword handling in some rosidl generators). Update spec §4.3 accordingly in this task's commit.

`srv/GetCameraSnapshot.srv`:

```
# On-demand synced camera frame pair + frame-stamped transforms.
# Spec: docs/specs/2026-07-13-camera-server-design.md §4.1

# Streams to include (payload control; server caches everything regardless).
bool want_color true
bool want_depth true
bool want_camera_info true

# For each entry, response includes transform target_frames[i] <- frame_id
# looked up at the returned pair stamp.
string[] target_frames

# Freshness. All zero => newest cached pair, no waiting.
float32 max_age_sec                      # >0: STALE if cached pair older
builtin_interfaces/Time captured_after   # non-zero: wait until both image stamps >= this
float32 wait_timeout_sec                 # bound on the wait; 0 => server default
---
int32 STATUS_OK=0
int32 STATUS_NO_DATA=1
int32 STATUS_STALE=2
int32 STATUS_WAIT_TIMEOUT=3
int32 STATUS_BAD_REQUEST=5

int32 status
string error_msg
builtin_interfaces/Time stamp            # min(color stamp, depth stamp)
string frame_id                          # common aligned optical frame
sensor_msgs/Image color
sensor_msgs/Image depth
sensor_msgs/CameraInfo color_info
sensor_msgs/CameraInfo depth_info
geometry_msgs/TransformStamped[] transforms   # index-aligned with target_frames
bool[] transforms_ok
```

`srv/GetCameraPointCloud.srv`:

```
# On-demand deprojected point cloud from the latest synced pair.
# Spec: docs/specs/2026-07-13-camera-server-design.md §4.2

uint32 stride              # pixel stride; 0 or 1 = full resolution
bool include_color         # true => XYZRGB, false => XYZ
string target_frame        # empty => native optical frame; else transformed at depth stamp
float32 max_age_sec
builtin_interfaces/Time captured_after
float32 wait_timeout_sec
---
int32 STATUS_OK=0
int32 STATUS_NO_DATA=1
int32 STATUS_STALE=2
int32 STATUS_WAIT_TIMEOUT=3
int32 STATUS_TF_FAIL=4
int32 STATUS_BAD_REQUEST=5

int32 status
string error_msg
builtin_interfaces/Time stamp            # depth image stamp; also points.header
sensor_msgs/PointCloud2 points
```

`srv/GetTransform.srv`:

```
# Time-correct transform lookup against the server's warm long-cache buffer.
# Spec: docs/specs/2026-07-13-camera-server-design.md §4.3

string target_frame
string source_frame
builtin_interfaces/Time lookup_time     # zero => latest available
float32 timeout_sec                     # capped by server param (default cap 2.0)
---
int32 STATUS_OK=0
int32 STATUS_UNAVAILABLE=1
int32 STATUS_BAD_REQUEST=5

int32 status
string error_msg
geometry_msgs/TransformStamped transform
```

`msg/CameraServerStatus.msg`:

```
# 1 Hz health snapshot. Spec §4.4.
builtin_interfaces/Time last_pair_stamp
float32 color_age_sec      # now - last color header stamp (-1 if never seen)
float32 depth_age_sec      # now - last depth header stamp (-1 if never seen)
float32 pair_age_sec       # now - last synced pair stamp (-1 if never)
float32 sync_fps           # synced-pair rate over the last status window
uint64 pair_seq
```

- [ ] **Step 2: Register interfaces in CMakeLists.txt and package.xml**

In `CMakeLists.txt` add after `find_package(sensor_msgs REQUIRED)`:

```cmake
find_package(builtin_interfaces REQUIRED)
```

In the `rosidl_generate_interfaces` block: add `"msg/CameraServerStatus.msg"` after `"msg/PanTiltState.msg"`; add, after `"srv/FoundationStereoDepth.srv"` (alphabetical-ish placement is fine):

```cmake
  "srv/GetCameraSnapshot.srv"
  "srv/GetCameraPointCloud.srv"
  "srv/GetTransform.srv"
```

and change the trailing dependency line to:

```cmake
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs builtin_interfaces
)
```

In `package.xml` add next to the other `<depend>` entries:

```xml
  <depend>builtin_interfaces</depend>
```

- [ ] **Step 3: Build and verify the interfaces generate**

```bash
tkbuild tk26_vision --packages-select tinker_vision_msgs_26
source /home/tinker/tk25_ws/install/setup.zsh
ros2 interface show tinker_vision_msgs_26/srv/GetCameraSnapshot
ros2 interface show tinker_vision_msgs_26/srv/GetCameraPointCloud
ros2 interface show tinker_vision_msgs_26/srv/GetTransform
ros2 interface show tinker_vision_msgs_26/msg/CameraServerStatus
```

Expected: build succeeds; each `ros2 interface show` prints the fields above (constants included), no errors. Freshness check (workspace stale-overlay history): confirm the install tree actually has the new types — `grep -r "GetCameraSnapshot" /home/tinker/tk25_ws/install/tinker_vision_msgs_26/share/ | head -3` returns hits.

- [ ] **Step 4: Update spec §4.3 field name**

In `docs/specs/2026-07-13-camera-server-design.md` §4.3, change `builtin_interfaces/Time time    # zero => latest available` to `builtin_interfaces/Time lookup_time    # zero => latest available`.

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git status --short   # verify only your files are staged-able; do not touch others' WIP
git add src/tinker_vision_msgs_26/srv/GetCameraSnapshot.srv \
        src/tinker_vision_msgs_26/srv/GetCameraPointCloud.srv \
        src/tinker_vision_msgs_26/srv/GetTransform.srv \
        src/tinker_vision_msgs_26/msg/CameraServerStatus.msg \
        src/tinker_vision_msgs_26/CMakeLists.txt \
        src/tinker_vision_msgs_26/package.xml \
        docs/specs/2026-07-13-camera-server-design.md
git commit -m "feat(tinker_vision_msgs_26): camera server interfaces (snapshot/point-cloud/transform + status)"
```

---

### Task 2: `camera_server` package scaffold + FrameStore

**Files:**
- Create: `src/tk26_vision/src/camera_server/package.xml`
- Create: `src/tk26_vision/src/camera_server/CMakeLists.txt`
- Create: `src/tk26_vision/src/camera_server/README.md`
- Create: `src/tk26_vision/src/camera_server/include/camera_server/frame_store.hpp`
- Create: `src/tk26_vision/src/camera_server/src/frame_store.cpp`
- Test: `src/tk26_vision/src/camera_server/test/test_frame_store.cpp`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces (used by Tasks 4–5): `camera_server::FramePair {color, depth : sensor_msgs::msg::Image::ConstSharedPtr; stamp_ns : int64_t; seq : uint64_t}`; `camera_server::FrameStore` with `set_pair(color, depth)`, `set_color_info(info)`, `set_depth_info(info)`, `latest_pair()`, `color_info()`, `depth_info()`, `wait_for_pair_after(after_ns, timeout) -> FramePair`. Stamps are int64 nanoseconds computed directly from `header.stamp`, avoiding rclcpp clock-type comparison throws. A pair is valid only when both color and depth are non-null; incomplete `set_pair` calls are ignored without replacing state, advancing `seq`, or notifying waiters. Freshness uses an inclusive `stamp_ns >= after_ns` boundary.

- [ ] **Step 1: Write package.xml and CMakeLists.txt**

`package.xml`:

```xml
<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>camera_server</name>
  <version>0.1.0</version>
  <description>Per-camera C++ snapshot/point-cloud/TF servers: the only streaming
  camera subscribers; everything else acquires frames on demand via services.
  Design: tk26_vision/docs/specs/2026-07-13-camera-server-design.md</description>
  <maintainer email="cindy.w0135@gmail.com">cindy</maintainer>
  <license>Apache-2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>rclcpp_components</depend>
  <depend>sensor_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>builtin_interfaces</depend>
  <depend>message_filters</depend>
  <depend>tf2</depend>
  <depend>tf2_ros</depend>
  <depend>tf2_eigen</depend>
  <depend>eigen3_cmake_module</depend>
  <depend>eigen</depend>
  <depend>tinker_vision_msgs_26</depend>

  <exec_depend>ros2launch</exec_depend>

  <test_depend>ament_cmake_gtest</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

`CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.8)
project(camera_server)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS calib3d core)
find_package(tinker_vision_msgs_26 REQUIRED)

# Core logic (no rclcpp node): unit-testable pieces.
add_library(camera_server_core
  src/frame_store.cpp
)
set_target_properties(camera_server_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(camera_server_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
ament_target_dependencies(camera_server_core sensor_msgs)

install(TARGETS camera_server_core
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_frame_store test/test_frame_store.cpp)
  target_link_libraries(test_frame_store camera_server_core)
endif()

ament_package()
```

(Node targets, launch install, and the deprojector source are added by Tasks 3–7; each task shows its exact CMake additions.)

- [ ] **Step 2: Write README.md with Changelog**

```markdown
# camera_server

Per-camera C++ servers that own the only streaming subscriptions to the wrist
RealSense / head Orbbec topics and serve frames, point clouds, and
time-correct transforms **on demand**:

- `~/get_snapshot` (`tinker_vision_msgs_26/srv/GetCameraSnapshot`) — latest
  synced color+depth pair + camera infos + transforms at the pair stamp, with
  `max_age` / `captured_after` freshness semantics.
- `~/get_point_cloud` (`GetCameraPointCloud`) — CPU deprojection of the cached
  registered-depth pair (stride / XYZ or XYZRGB / optional target frame at
  the depth image stamp).
- `~/get_transform` (`GetTransform`) — lookup against the server's warm 180 s
  TF buffer, for on-demand consumers with cold local buffers.
- `~/status` (`CameraServerStatus`, 1 Hz) — stream ages, sync fps, pair seq.

Two instances: `wrist_camera_server` (launched by the manipulation bringup
that owns the RealSense) and `head_camera_server` (launched by
`vision_bringup/vision_driver.launch.py`). A separate `camera_compat_bridge`
executable serves the legacy `get_image_service` / `get_point_cloud_service` /
`get_orbbec_pc` names by forwarding to the servers — param-gated, OFF by
default (the Python utility nodes keep those names until cutover).

Design: `../../docs/specs/2026-07-13-camera-server-design.md`.
Consumers are NOT migrated by this package's introduction (Appendix A of the
spec maps the deferred migration).

## Build

    tkbuild tk26_vision --packages-select camera_server

## Changelog

- 2026-07-13: package scaffold + thread-safe FrameStore with captured_after wait.
```

- [ ] **Step 3: Write the failing gtest for FrameStore**

`test/test_frame_store.cpp`:

```cpp
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>

#include "camera_server/frame_store.hpp"

using camera_server::FramePair;
using camera_server::FrameStore;
using sensor_msgs::msg::Image;

namespace {
Image::ConstSharedPtr make_image(int64_t stamp_ns) {
  auto img = std::make_shared<Image>();
  img->header.stamp.sec = static_cast<int32_t>(stamp_ns / 1000000000LL);
  img->header.stamp.nanosec = static_cast<uint32_t>(stamp_ns % 1000000000LL);
  img->header.frame_id = "camera_color_optical_frame";
  return img;
}
}  // namespace

TEST(FrameStore, EmptyStoreReturnsNullPair) {
  FrameStore store;
  FramePair p = store.latest_pair();
  EXPECT_EQ(p.color, nullptr);
  EXPECT_EQ(p.depth, nullptr);
  EXPECT_EQ(p.seq, 0u);
}

TEST(FrameStore, SetPairStoresAndBumpsSeq) {
  FrameStore store;
  store.set_pair(make_image(100), make_image(100));
  FramePair p1 = store.latest_pair();
  ASSERT_NE(p1.color, nullptr);
  EXPECT_EQ(p1.stamp_ns, 100);
  EXPECT_EQ(p1.seq, 1u);
  store.set_pair(make_image(200), make_image(200));
  FramePair p2 = store.latest_pair();
  EXPECT_EQ(p2.stamp_ns, 200);
  EXPECT_EQ(p2.seq, 2u);
}

TEST(FrameStore, InfosStoredIndependently) {
  FrameStore store;
  EXPECT_EQ(store.color_info(), nullptr);
  auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  info->width = 1280;
  store.set_color_info(info);
  ASSERT_NE(store.color_info(), nullptr);
  EXPECT_EQ(store.color_info()->width, 1280u);
  EXPECT_EQ(store.depth_info(), nullptr);
}

// Also cover depth_info storage, shared-pointer identity for frames and infos,
// already-fresh immediate return, the inclusive >= captured_after boundary,
// an empty-store timeout, and rejected null color/depth inputs. Null inputs
// must preserve the prior pair/seq and must not wake a freshness waiter.

TEST(FrameStore, DepthInfoAndPointersArePreserved) {
  FrameStore store;
  const auto color = make_image(100);
  const auto depth = make_image(100);
  auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  store.set_pair(color, depth);
  store.set_depth_info(info);
  EXPECT_EQ(store.latest_pair().color, color);
  EXPECT_EQ(store.latest_pair().depth, depth);
  EXPECT_EQ(store.depth_info(), info);
}

TEST(FrameStore, AlreadyFreshAndExactBoundaryReturnImmediately) {
  FrameStore store;
  store.set_pair(make_image(500), make_image(500));
  EXPECT_EQ(store.wait_for_pair_after(
                400, std::chrono::seconds(1)).stamp_ns, 500);
  EXPECT_EQ(store.wait_for_pair_after(
                500, std::chrono::seconds(1)).stamp_ns, 500);
}

TEST(FrameStore, EmptyStoreWaitTimesOut) {
  FrameStore store;
  auto t0 = std::chrono::steady_clock::now();
  FramePair p = store.wait_for_pair_after(
      0, std::chrono::milliseconds(60));
  auto elapsed = std::chrono::steady_clock::now() - t0;
  EXPECT_GE(elapsed, std::chrono::milliseconds(45));
  EXPECT_EQ(p.color, nullptr);
  EXPECT_EQ(p.depth, nullptr);
}

TEST(FrameStore, NullPairInputsAreIgnored) {
  FrameStore store;
  const auto color = make_image(100);
  const auto depth = make_image(100);
  store.set_pair(color, depth);
  store.set_pair(nullptr, make_image(200));
  store.set_pair(make_image(300), nullptr);
  FramePair p = store.latest_pair();
  EXPECT_EQ(p.color, color);
  EXPECT_EQ(p.depth, depth);
  EXPECT_EQ(p.stamp_ns, 100);
  EXPECT_EQ(p.seq, 1u);
}

TEST(FrameStore, IncompletePairDoesNotSatisfyFreshnessWait) {
  FrameStore store;
  std::thread feeder([&store] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    store.set_pair(make_image(600), nullptr);
  });
  auto t0 = std::chrono::steady_clock::now();
  FramePair p = store.wait_for_pair_after(
      500, std::chrono::milliseconds(80));
  auto elapsed = std::chrono::steady_clock::now() - t0;
  feeder.join();
  EXPECT_GE(elapsed, std::chrono::milliseconds(65));
  EXPECT_EQ(p.seq, 0u);
}

TEST(FrameStore, WaitTimesOutReturningNewest) {
  FrameStore store;
  store.set_pair(make_image(100), make_image(100));
  auto t0 = std::chrono::steady_clock::now();
  FramePair p = store.wait_for_pair_after(500, std::chrono::milliseconds(100));
  auto elapsed = std::chrono::steady_clock::now() - t0;
  EXPECT_GE(elapsed, std::chrono::milliseconds(90));
  EXPECT_EQ(p.stamp_ns, 100);  // newest available, older than requested
}

TEST(FrameStore, WaitUnblocksOnFreshPair) {
  FrameStore store;
  store.set_pair(make_image(100), make_image(100));
  std::thread feeder([&store] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    store.set_pair(make_image(600), make_image(600));
  });
  FramePair p = store.wait_for_pair_after(500, std::chrono::seconds(2));
  feeder.join();
  EXPECT_EQ(p.stamp_ns, 600);
  EXPECT_GE(p.seq, 2u);
}
```

- [ ] **Step 4: Write frame_store.hpp (build must fail first without the impl — compile-fail is the failing state for C++)**

`include/camera_server/frame_store.hpp`:

```cpp
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace camera_server {

struct FramePair {
  sensor_msgs::msg::Image::ConstSharedPtr color;
  sensor_msgs::msg::Image::ConstSharedPtr depth;
  int64_t color_stamp_ns = 0;
  int64_t depth_stamp_ns = 0;
  int64_t stamp_ns = 0;  // min(color_stamp_ns, depth_stamp_ns)
  uint64_t seq = 0;      // monotonic synced-pair counter
};

/// Thread-safe latest-frame store shared between the sync callback (writer)
/// and service handlers (readers). Stores ConstSharedPtrs — no image copies.
class FrameStore {
 public:
  /// Incomplete pairs are ignored: state/seq are unchanged and waiters are
  /// not notified.
  void set_pair(sensor_msgs::msg::Image::ConstSharedPtr color,
                sensor_msgs::msg::Image::ConstSharedPtr depth);
  void set_color_info(sensor_msgs::msg::CameraInfo::ConstSharedPtr info);
  void set_depth_info(sensor_msgs::msg::CameraInfo::ConstSharedPtr info);

  FramePair latest_pair() const;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr color_info() const;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr depth_info() const;

  /// Blocks until both images in a complete pair are stamped >= after_ns, or
  /// timeout elapses. Always returns the newest available pair.
  FramePair wait_for_pair_after(int64_t after_ns,
                                std::chrono::nanoseconds timeout);

 private:
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  FramePair pair_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr color_info_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr depth_info_;
};

}  // namespace camera_server
```

- [ ] **Step 5: Verify the build fails (missing frame_store.cpp implementation)**

Create an empty `src/frame_store.cpp` containing only `#include "camera_server/frame_store.hpp"`, then:

```bash
tkbuild tk26_vision --packages-select camera_server
```

Expected: FAIL at link time — undefined references to `camera_server::FrameStore::set_pair` etc. from `test_frame_store`.

- [ ] **Step 6: Implement frame_store.cpp**

```cpp
#include "camera_server/frame_store.hpp"

#include <utility>

namespace camera_server {

namespace {
int64_t stamp_ns_of(const sensor_msgs::msg::Image& img) {
  return static_cast<int64_t>(img.header.stamp.sec) * 1000000000LL +
         static_cast<int64_t>(img.header.stamp.nanosec);
}
}  // namespace

void FrameStore::set_pair(sensor_msgs::msg::Image::ConstSharedPtr color,
                          sensor_msgs::msg::Image::ConstSharedPtr depth) {
  if (!color || !depth) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    pair_.color = std::move(color);
    pair_.depth = std::move(depth);
    pair_.color_stamp_ns = stamp_ns_of(*pair_.color);
    pair_.depth_stamp_ns = stamp_ns_of(*pair_.depth);
    pair_.stamp_ns =
        std::min(pair_.color_stamp_ns, pair_.depth_stamp_ns);
    pair_.seq += 1;
  }
  cv_.notify_all();
}

void FrameStore::set_color_info(
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info) {
  std::lock_guard<std::mutex> lock(mutex_);
  color_info_ = std::move(info);
}

void FrameStore::set_depth_info(
    sensor_msgs::msg::CameraInfo::ConstSharedPtr info) {
  std::lock_guard<std::mutex> lock(mutex_);
  depth_info_ = std::move(info);
}

FramePair FrameStore::latest_pair() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return pair_;
}

sensor_msgs::msg::CameraInfo::ConstSharedPtr FrameStore::color_info() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return color_info_;
}

sensor_msgs::msg::CameraInfo::ConstSharedPtr FrameStore::depth_info() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return depth_info_;
}

FramePair FrameStore::wait_for_pair_after(int64_t after_ns,
                                          std::chrono::nanoseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait_for(lock, timeout, [this, after_ns] {
    return pair_.color != nullptr && pair_.depth != nullptr &&
           pair_.stamp_ns >= after_ns;
  });
  return pair_;
}

}  // namespace camera_server
```

- [ ] **Step 7: Build and run the gtest**

```bash
tkbuild tk26_vision --packages-select camera_server
/home/tinker/tk25_ws/build/camera_server/test_frame_store
```

Expected: build succeeds; gtest covers empty/latest storage, camera-info and
frame pointer preservation, already-fresh and exact-boundary returns, empty and
stale timeouts, asynchronous wakeup, and rejected null pair inputs.

- [ ] **Step 8: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git status --short
git add src/camera_server/package.xml src/camera_server/CMakeLists.txt \
        src/camera_server/README.md src/camera_server/include \
        src/camera_server/src/frame_store.cpp src/camera_server/test/test_frame_store.cpp
git commit -m "feat(camera_server): package scaffold + thread-safe FrameStore with captured_after wait"
```

---

### Task 3: Deprojector (depth → PointCloud2)

**Files:**
- Create: `src/tk26_vision/src/camera_server/include/camera_server/deprojector.hpp`
- Create: `src/tk26_vision/src/camera_server/src/deprojector.cpp`
- Modify: `src/tk26_vision/src/camera_server/CMakeLists.txt`
- Modify: `src/tk26_vision/src/camera_server/README.md` (changelog line)
- Test: `src/tk26_vision/src/camera_server/test/test_deprojector.cpp`

**Interfaces:**
- Consumes: nothing from other tasks (pure function of msgs).
- Produces (used by Task 5): `camera_server::Deprojector::deproject(depth, depth_info, color_or_null, stride, optional<Eigen::Isometry3f>, out_cloud, error_msg) -> bool`. Output cloud: unorganized, `height=1`, fields `x,y,z` FLOAT32 (+ packed `rgb` FLOAT32 when color given), invalid (z<=0/NaN) pixels dropped, `is_dense=true`. Caller sets `out.header`. Not thread-safe (internal xy-table cache) — callers serialize with a mutex.

**Final Phase 3/4 hardening contract (authoritative over the initial TDD
sketches below):**

- Depth must be registered to color before this API; when color is requested
  its dimensions must exactly match depth. No ratio mapping or cross-camera
  registration is performed. Raw registered images are valid: build rays with
  OpenCV `undistortPoints` for plumb_bob/rational_polynomial and
  `fisheye::undistortPoints` for equidistant.
- The ray cache key includes dimensions, K, distortion model, and every D
  coefficient. Reject unsupported models, non-finite coefficients, incorrect
  coefficient counts, and conflicting nonempty depth/depth-info/color frame
  IDs instead of silently falling back to raw K.
- Accept `16UC1`, `mono16` (millimetres), and `32FC1` (metres), correctly
  decoding both ROS input byte orders. Emit a deterministic little-endian
  cloud with exact `x@0,y@4,z@8[,rgb@12]` FLOAT32 fields and 12/16-byte point
  steps.
- Validate nonzero image and CameraInfo dimensions, exact depth/CameraInfo
  dimensions, finite positive focal lengths, finite principal point, encoding,
  minimum row step, and `step * height` backing length before any access.
  Arithmetic for table, sampled counts, point capacity, and row/data sizes is
  overflow-safe.
- Skip every non-finite or non-positive depth, including NaN, +/-Inf, zero,
  and negative float depth. Reject non-finite transforms/output coordinates.
- Clear `out` and `error_msg` on entry. Every failure leaves `out` empty with
  a deterministic nonempty error; success leaves `error_msg` empty.
- Tests cover exact fields/layout/data sizing, XYZ and RGB/BGR channels,
  rotation plus translation, K/model/D cache invalidation, plumb-bob/rational/
  equidistant rays, valid padded rows, frame-ID compatibility, malformed
  buffers/dimensions, endian cases, invalid/all-invalid depth, and extreme
  strides.

- [ ] **Step 1: Write the failing gtest**

`test/test_deprojector.cpp`:

```cpp
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "camera_server/deprojector.hpp"

using camera_server::Deprojector;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;

namespace {
// 4x4 depth camera: fx=fy=100, cx=cy=2 (principal point at pixel (2,2)).
CameraInfo make_info(uint32_t w = 4, uint32_t h = 4) {
  CameraInfo info;
  info.width = w;
  info.height = h;
  info.k = {100.0, 0.0, 2.0, 0.0, 100.0, 2.0, 0.0, 0.0, 1.0};
  return info;
}

Image make_depth_16u(uint32_t w = 4, uint32_t h = 4, uint16_t mm = 1000) {
  Image img;
  img.width = w;
  img.height = h;
  img.encoding = "16UC1";
  img.step = w * 2;
  img.data.assign(img.step * h, 0);
  for (uint32_t v = 0; v < h; ++v)
    for (uint32_t u = 0; u < w; ++u)
      std::memcpy(&img.data[v * img.step + u * 2], &mm, 2);
  return img;
}

Image make_depth_32f(uint32_t w = 4, uint32_t h = 4, float m = 1.0f) {
  Image img;
  img.width = w;
  img.height = h;
  img.encoding = "32FC1";
  img.step = w * 4;
  img.data.assign(img.step * h, 0);
  for (uint32_t v = 0; v < h; ++v)
    for (uint32_t u = 0; u < w; ++u)
      std::memcpy(&img.data[v * img.step + u * 4], &m, 4);
  return img;
}

Image make_color_rgb8(uint32_t w = 4, uint32_t h = 4) {
  Image img;
  img.width = w;
  img.height = h;
  img.encoding = "rgb8";
  img.step = w * 3;
  img.data.assign(img.step * h, 0);
  // pixel (1,1) = pure red
  img.data[1 * img.step + 1 * 3 + 0] = 255;
  return img;
}
}  // namespace

TEST(Deprojector, Deprojects16UC1FullRes) {
  Deprojector d;
  PointCloud2 out;
  std::string err;
  Image depth = make_depth_16u();
  ASSERT_TRUE(d.deproject(depth, make_info(), nullptr, 0, std::nullopt, out, err)) << err;
  EXPECT_EQ(out.width, 16u);  // 4x4, all valid
  EXPECT_EQ(out.height, 1u);
  sensor_msgs::PointCloud2ConstIterator<float> it_x(out, "x"), it_y(out, "y"), it_z(out, "z");
  // First point = pixel (0,0): x=(0-2)/100*1m=-0.02, y=-0.02, z=1.0
  EXPECT_NEAR(*it_x, -0.02f, 1e-6);
  EXPECT_NEAR(*it_y, -0.02f, 1e-6);
  EXPECT_NEAR(*it_z, 1.0f, 1e-6);
}

TEST(Deprojector, Deprojects32FC1) {
  Deprojector d;
  PointCloud2 out;
  std::string err;
  Image depth = make_depth_32f(4, 4, 2.0f);
  ASSERT_TRUE(d.deproject(depth, make_info(), nullptr, 1, std::nullopt, out, err)) << err;
  sensor_msgs::PointCloud2ConstIterator<float> it_z(out, "z");
  EXPECT_NEAR(*it_z, 2.0f, 1e-6);
}

TEST(Deprojector, StrideReducesCount) {
  Deprojector d;
  PointCloud2 out;
  std::string err;
  Image depth = make_depth_16u();
  ASSERT_TRUE(d.deproject(depth, make_info(), nullptr, 2, std::nullopt, out, err)) << err;
  EXPECT_EQ(out.width, 4u);  // pixels (0,0),(0,2),(2,0),(2,2)
}

TEST(Deprojector, DropsInvalidDepth) {
  Deprojector d;
  PointCloud2 out;
  std::string err;
  Image depth = make_depth_16u();
  uint16_t zero = 0;  // invalidate pixel (0,0)
  std::memcpy(&depth.data[0], &zero, 2);
  ASSERT_TRUE(d.deproject(depth, make_info(), nullptr, 0, std::nullopt, out, err)) << err;
  EXPECT_EQ(out.width, 15u);
}

TEST(Deprojector, PacksRgbFromColor) {
  Deprojector d;
  PointCloud2 out;
  std::string err;
  Image depth = make_depth_16u();
  Image color = make_color_rgb8();
  ASSERT_TRUE(d.deproject(depth, make_info(), &color, 0, std::nullopt, out, err)) << err;
  ASSERT_EQ(out.fields.size(), 4u);
  sensor_msgs::PointCloud2ConstIterator<uint8_t> it_r(out, "r");
  // point index for pixel (u=1, v=1) is 5 in row-major full-res order
  for (int i = 0; i < 5; ++i) ++it_r;
  EXPECT_EQ(it_r[0], 255);  // r channel
}

TEST(Deprojector, AppliesTransform) {
  Deprojector d;
  PointCloud2 out;
  std::string err;
  Image depth = make_depth_16u();
  Eigen::Isometry3f tr = Eigen::Isometry3f::Identity();
  tr.translation() << 10.0f, 0.0f, 0.0f;
  ASSERT_TRUE(d.deproject(depth, make_info(), nullptr, 0, tr, out, err)) << err;
  sensor_msgs::PointCloud2ConstIterator<float> it_x(out, "x");
  EXPECT_NEAR(*it_x, 10.0f - 0.02f, 1e-5);
}

TEST(Deprojector, RejectsUnsupportedEncoding) {
  Deprojector d;
  PointCloud2 out;
  std::string err;
  Image depth = make_depth_16u();
  depth.encoding = "mono8";
  EXPECT_FALSE(d.deproject(depth, make_info(), nullptr, 0, std::nullopt, out, err));
  EXPECT_FALSE(err.empty());
}
```

- [ ] **Step 2: Write deprojector.hpp**

`include/camera_server/deprojector.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace camera_server {

/// CPU depth-image deprojection with a cached per-intrinsics xy-table.
/// Handles 16UC1/mono16 (mm) and 32FC1 (m) depth, rgb8/bgr8 color.
/// Depth must already be registered to same-sized color. Cached rays account
/// for supported CameraInfo distortion models.
/// NOT thread-safe (table cache): callers serialize access externally.
class Deprojector {
 public:
  /// stride: 0 or 1 = full resolution. transform: applied to every point
  /// (target_frame <- optical). Returns false + error_msg on unsupported
  /// encoding or depth/info dimension mismatch. Caller fills out.header.
  bool deproject(const sensor_msgs::msg::Image& depth,
                 const sensor_msgs::msg::CameraInfo& depth_info,
                 const sensor_msgs::msg::Image* color,
                 uint32_t stride,
                 const std::optional<Eigen::Isometry3f>& transform,
                 sensor_msgs::msg::PointCloud2& out,
                 std::string& error_msg);

 private:
  struct TableKey {
    uint32_t w = 0, h = 0;
    double fx = 0, fy = 0, cx = 0, cy = 0;
    std::string distortion_model;
    std::vector<double> distortion;
    bool operator==(const TableKey& o) const {
      return w == o.w && h == o.h && fx == o.fx && fy == o.fy &&
             cx == o.cx && cy == o.cy &&
             distortion_model == o.distortion_model &&
             distortion == o.distortion;
    }
  };
  void rebuild_table(const TableKey& key);

  TableKey key_;
  std::vector<float> xy_table_;  // h*w*2 floats: ((u-cx)/fx, (v-cy)/fy)
};

}  // namespace camera_server
```

- [ ] **Step 3: Add sources/tests to CMakeLists.txt and verify the build fails**

In `CMakeLists.txt`, extend the core library and tests:

```cmake
add_library(camera_server_core
  src/frame_store.cpp
  src/deprojector.cpp
)
target_include_directories(camera_server_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${sensor_msgs_INCLUDE_DIRS})
target_link_libraries(camera_server_core PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
```

and inside `if(BUILD_TESTING)`:

```cmake
  ament_add_gtest(test_deprojector test/test_deprojector.cpp)
  target_link_libraries(test_deprojector camera_server_core)
```

Create `src/deprojector.cpp` containing only the include, run `tkbuild tk26_vision --packages-select camera_server`.
Expected: FAIL — undefined reference to `camera_server::Deprojector::deproject`.

- [ ] **Step 4: Implement deprojector.cpp**

```cpp
#include "camera_server/deprojector.hpp"

#include <cmath>
#include <cstring>

#include <sensor_msgs/point_cloud2_iterator.hpp>

namespace camera_server {

void Deprojector::rebuild_table(const TableKey& key) {
  key_ = key;
  xy_table_.resize(static_cast<size_t>(key.w) * key.h * 2);
  size_t i = 0;
  for (uint32_t v = 0; v < key.h; ++v) {
    for (uint32_t u = 0; u < key.w; ++u) {
      xy_table_[i++] = static_cast<float>((u - key.cx) / key.fx);
      xy_table_[i++] = static_cast<float>((v - key.cy) / key.fy);
    }
  }
}

bool Deprojector::deproject(const sensor_msgs::msg::Image& depth,
                            const sensor_msgs::msg::CameraInfo& depth_info,
                            const sensor_msgs::msg::Image* color,
                            uint32_t stride,
                            const std::optional<Eigen::Isometry3f>& transform,
                            sensor_msgs::msg::PointCloud2& out,
                            std::string& error_msg) {
  const bool is_u16 = depth.encoding == "16UC1";
  const bool is_f32 = depth.encoding == "32FC1";
  if (!is_u16 && !is_f32) {
    error_msg = "unsupported depth encoding: " + depth.encoding;
    return false;
  }
  if (depth_info.k[0] <= 0.0 || depth_info.k[4] <= 0.0) {
    error_msg = "camera_info has non-positive focal length";
    return false;
  }
  if (color && color->encoding != "rgb8" && color->encoding != "bgr8") {
    error_msg = "unsupported color encoding: " + color->encoding;
    return false;
  }

  TableKey key{depth.width, depth.height, depth_info.k[0], depth_info.k[4],
               depth_info.k[2], depth_info.k[5]};
  if (!(key == key_) || xy_table_.empty()) rebuild_table(key);

  const uint32_t s = stride <= 1 ? 1 : stride;
  const size_t max_points =
      (static_cast<size_t>(depth.height + s - 1) / s) *
      (static_cast<size_t>(depth.width + s - 1) / s);

  sensor_msgs::PointCloud2Modifier mod(out);
  if (color) {
    mod.setPointCloud2Fields(
        4, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "rgb", 1, sensor_msgs::msg::PointField::FLOAT32);
  } else {
    mod.setPointCloud2Fields(
        3, "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32);
  }
  mod.resize(max_points);

  sensor_msgs::PointCloud2Iterator<float> it_x(out, "x"), it_y(out, "y"),
      it_z(out, "z");
  std::optional<sensor_msgs::PointCloud2Iterator<uint8_t>> it_rgb;
  if (color) it_rgb.emplace(out, "rgb");

  // The hardened contract requires depth already registered to color and
  // validates identical dimensions before this loop.
  const bool bgr = color && color->encoding == "bgr8";

  size_t n = 0;
  for (uint32_t v = 0; v < depth.height; v += s) {
    const uint8_t* row = depth.data.data() + static_cast<size_t>(v) * depth.step;
    for (uint32_t u = 0; u < depth.width; u += s) {
      float z;
      if (is_u16) {
        uint16_t mm;
        std::memcpy(&mm, row + u * 2, 2);
        z = mm * 0.001f;
      } else {
        std::memcpy(&z, row + u * 4, 4);
      }
      if (!(z > 0.f) || std::isnan(z)) continue;

      const size_t ti = (static_cast<size_t>(v) * depth.width + u) * 2;
      Eigen::Vector3f p(xy_table_[ti] * z, xy_table_[ti + 1] * z, z);
      if (transform) p = (*transform) * p;

      *it_x = p.x(); *it_y = p.y(); *it_z = p.z();
      ++it_x; ++it_y; ++it_z;

      if (color) {
        const uint32_t cu = u;
        const uint32_t cv = v;
        const uint8_t* px =
            color->data.data() + static_cast<size_t>(cv) * color->step + cu * 3;
        // PointCloud2 packed rgb byte order: [b, g, r, _] via the
        // "rgb"-named uint8 iterator (r=idx2? no: iterator exposes float
        // cell; we write bytes little-endian: b,g,r).
        (*it_rgb)[0] = bgr ? px[0] : px[2];  // b
        (*it_rgb)[1] = px[1];                // g
        (*it_rgb)[2] = bgr ? px[2] : px[0];  // r
        (*it_rgb)[3] = 0;
        ++(*it_rgb);
      }
      ++n;
    }
  }
  mod.resize(n);
  out.height = 1;
  out.width = static_cast<uint32_t>(n);
  out.is_dense = true;
  return true;
}

}  // namespace camera_server
```

Note for the test's `PointCloud2ConstIterator<uint8_t>(out, "r")`: sensor_msgs' iterator maps `"r"`/`"g"`/`"b"` names onto the packed `rgb` float field (byte offsets 2/1/0) — this is the standard PCL-compatible layout the existing Python consumers unpack.

- [ ] **Step 5: Build and run both gtests**

```bash
tkbuild tk26_vision --packages-select camera_server
/home/tinker/tk25_ws/build/camera_server/test_deprojector
/home/tinker/tk25_ws/build/camera_server/test_frame_store
```

Expected after the Phase 4 audit: `[  PASSED  ] 17 tests.` for Deprojector
and `[  PASSED  ] 11 tests.` for FrameStore. If the rgb byte-order assertion
fails (iterator maps r↔b), fix the write order in deprojector.cpp — the
authority is: existing consumers unpack with `struct.unpack` little-endian
where byte 0 = b (PCL convention).

- [ ] **Step 6: Append changelog line to README.md**

```markdown
- 2026-07-23: Deprojector — hardened 16UC1/mono16/32FC1 registered depth -> deterministic little-endian XYZ[RGB] cloud, validated buffers/intrinsics, cached xy-table, stride + transform support.
```

- [ ] **Step 7: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git status --short
git add src/camera_server/include/camera_server/deprojector.hpp \
        src/camera_server/src/deprojector.cpp \
        src/camera_server/test/test_deprojector.cpp \
        src/camera_server/CMakeLists.txt src/camera_server/README.md
git commit -m "feat(camera_server): CPU deprojector (depth->XYZ[RGB] cloud, xy-table cache, stride/transform)"
```

---

### Task 4: CameraServerNode — subscriptions, TF, status, `get_snapshot`, `get_transform`

**Landed Phase 4 audit contract (authoritative over the original draft
snippets below):**

- Streaming image/filter and CameraInfo callbacks use an `auto_add=false`
  callback group spun by a node-owned single-thread executor. Destruction
  cancels/joins it. This prevents blocking freshness services from starving
  ingestion in both standalone and component use.
- `num_executor_threads` defaults to 4 and must be >=2. Queue size must be
  >=2; slop is finite/nonnegative; TF cache/lookup/cap, max wait, status
  period, and starvation interval are finite/positive.
- `FramePair` preserves color/depth stamps and uses their minimum for snapshot
  response, TF, freshness, and captured-after. Task 5 uses the depth stamp for
  cloud TF/response.
- Requests validate finite/nonnegative durations and canonical ROS times.
  Snapshot targets are bounded by `max_target_frames` (default 16), blank
  targets and all-empty requests are rejected, and TF arrays remain
  index-aligned on partial failure. Service boundaries catch
  `std::exception` and use response-side status constants.
- Registered color/depth must have compatible nonempty frame IDs; the response
  uses the common aligned frame (depth ID preferred when present). CameraInfo
  remains best-effort with diagnostics. The head default depth intrinsics are
  `/camera/color/camera_info`, matching registered `/camera/depth/image_raw`.
- Status FPS uses actual steady-clock elapsed time; ages use stream header
  stamps; starvation and partner-skew warnings are throttled; `NO_DATA`
  includes available stream ages.
- Automated ROS tests cover no-data/malformed survival, synchronized QoS
  ingestion, payload flags, missing info, stale/timeout behavior, static and
  partial TF, status, and three waits against a two-thread service executor.

**Files:**
- Create: `src/tk26_vision/src/camera_server/include/camera_server/camera_server_node.hpp`
- Create: `src/tk26_vision/src/camera_server/src/camera_server_node.cpp`
- Create: `src/tk26_vision/src/camera_server/src/camera_server_main.cpp`
- Modify: `src/tk26_vision/src/camera_server/CMakeLists.txt`
- Modify: `src/tk26_vision/src/camera_server/README.md` (changelog)

**Interfaces:**
- Consumes: `FrameStore` (Task 2), `Deprojector` (Task 3 — member declared here, used in Task 5), interfaces (Task 1).
- Produces: node class `camera_server::CameraServerNode(const rclcpp::NodeOptions&)`, component-registered; executable `camera_server_node`. Services `~/get_snapshot`, `~/get_transform` live after this task; `~/get_point_cloud` is declared but returns BAD_REQUEST "not implemented" until Task 5 replaces the stub.

Parameters (all declared in the constructor):

| name | default | meaning |
|---|---|---|
| `color_topic` | `/camera/color/image_raw` | head defaults; wrist overrides via launch |
| `depth_topic` | `/camera/depth/image_raw` | |
| `color_info_topic` | `/camera/color/camera_info` | |
| `depth_info_topic` | `/camera/color/camera_info` | registered head depth uses color intrinsics |
| `sync_queue_size` | 10 | |
| `sync_slop_sec` | 0.1 | ApproximateTime max interval |
| `tf_cache_sec` | 180.0 | |
| `tf_lookup_timeout_sec` | 0.1 | snapshot/cloud per-lookup timeout |
| `transform_timeout_cap_sec` | 2.0 | GetTransform request cap |
| `max_wait_sec` | 2.0 | captured_after wait cap / default |
| `status_period_sec` | 1.0 | |
| `starvation_warn_sec` | 2.0 | throttled sync-starvation threshold |
| `max_target_frames` | 16 | bound snapshot TF work |
| `num_executor_threads` | 4 | standalone service executor; minimum 2 |

- [ ] **Step 1: Write camera_server_node.hpp**

```cpp
#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include "camera_server/deprojector.hpp"
#include "camera_server/frame_store.hpp"
#include "tinker_vision_msgs_26/msg/camera_server_status.hpp"
#include "tinker_vision_msgs_26/srv/get_camera_point_cloud.hpp"
#include "tinker_vision_msgs_26/srv/get_camera_snapshot.hpp"
#include "tinker_vision_msgs_26/srv/get_transform.hpp"

namespace camera_server {

class CameraServerNode : public rclcpp::Node {
 public:
  explicit CameraServerNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using GetCameraSnapshot = tinker_vision_msgs_26::srv::GetCameraSnapshot;
  using GetCameraPointCloud = tinker_vision_msgs_26::srv::GetCameraPointCloud;
  using GetTransform = tinker_vision_msgs_26::srv::GetTransform;
  using CameraServerStatus = tinker_vision_msgs_26::msg::CameraServerStatus;
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<Image, Image>;

  void on_synced(Image::ConstSharedPtr color, Image::ConstSharedPtr depth);
  void publish_status();

  /// Shared freshness resolution for snapshot + point cloud (spec §4.1).
  /// Returns a STATUS_* code; `pair` holds the newest available pair either way.
  int32_t acquire_pair(float max_age_sec,
                       const builtin_interfaces::msg::Time& captured_after,
                       float wait_timeout_sec, FramePair& pair,
                       std::string& error_msg);

  void handle_snapshot(GetCameraSnapshot::Request::ConstSharedPtr req,
                       GetCameraSnapshot::Response::SharedPtr res);
  void handle_point_cloud(GetCameraPointCloud::Request::ConstSharedPtr req,
                          GetCameraPointCloud::Response::SharedPtr res);
  void handle_transform(GetTransform::Request::ConstSharedPtr req,
                        GetTransform::Response::SharedPtr res);

  FrameStore store_;
  Deprojector deprojector_;
  std::mutex deproject_mutex_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  message_filters::Subscriber<Image> color_sub_, depth_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Subscription<CameraInfo>::SharedPtr color_info_sub_, depth_info_sub_;

  rclcpp::Service<GetCameraSnapshot>::SharedPtr snapshot_srv_;
  rclcpp::Service<GetCameraPointCloud>::SharedPtr point_cloud_srv_;
  rclcpp::Service<GetTransform>::SharedPtr transform_srv_;
  rclcpp::Publisher<CameraServerStatus>::SharedPtr status_pub_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  rclcpp::CallbackGroup::SharedPtr sub_group_;      // MutuallyExclusive
  rclcpp::CallbackGroup::SharedPtr service_group_;  // Reentrant

  // Per-stream last header stamps (ns) for status ages; -1 = never seen.
  std::atomic<int64_t> last_color_ns_{-1}, last_depth_ns_{-1};
  uint64_t status_last_seq_ = 0;

  double tf_lookup_timeout_sec_ = 0.1;
  double transform_timeout_cap_sec_ = 2.0;
  double max_wait_sec_ = 2.0;
  double status_period_sec_ = 1.0;
};

}  // namespace camera_server
```

- [ ] **Step 2: Write camera_server_node.cpp**

```cpp
#include "camera_server/camera_server_node.hpp"

#include <chrono>
#include <functional>

#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/time.h>
#include <tf2_ros/create_timer_ros.h>

namespace camera_server {

namespace {
int64_t to_ns(const builtin_interfaces::msg::Time& t) {
  return static_cast<int64_t>(t.sec) * 1000000000LL +
         static_cast<int64_t>(t.nanosec);
}
builtin_interfaces::msg::Time from_ns(int64_t ns) {
  builtin_interfaces::msg::Time t;
  t.sec = static_cast<int32_t>(ns / 1000000000LL);
  t.nanosec = static_cast<uint32_t>(ns % 1000000000LL);
  return t;
}
}  // namespace

CameraServerNode::CameraServerNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("camera_server", options) {
  const auto color_topic =
      declare_parameter<std::string>("color_topic", "/camera/color/image_raw");
  const auto depth_topic =
      declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
  const auto color_info_topic = declare_parameter<std::string>(
      "color_info_topic", "/camera/color/camera_info");
  const auto depth_info_topic = declare_parameter<std::string>(
      "depth_info_topic", "/camera/color/camera_info");
  const int sync_queue = declare_parameter<int>("sync_queue_size", 10);
  const double slop = declare_parameter<double>("sync_slop_sec", 0.1);
  const double tf_cache = declare_parameter<double>("tf_cache_sec", 180.0);
  tf_lookup_timeout_sec_ =
      declare_parameter<double>("tf_lookup_timeout_sec", 0.1);
  transform_timeout_cap_sec_ =
      declare_parameter<double>("transform_timeout_cap_sec", 2.0);
  max_wait_sec_ = declare_parameter<double>("max_wait_sec", 2.0);
  status_period_sec_ = declare_parameter<double>("status_period_sec", 1.0);
  declare_parameter<int>("num_executor_threads", 4);  // read by main()

  sub_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  service_group_ =
      create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  // TF: warm long-cache buffer; listener spins its own dedicated thread so TF
  // ingestion never competes with service handlers (spec §5.4).
  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(
      get_clock(), tf2::durationFromSec(tf_cache));
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(
      *tf_buffer_, this, /*spin_thread=*/true);

  // Image subscriptions: BEST_EFFORT KEEP_LAST(5) (spec §5.1).
  rmw_qos_profile_t img_qos = rmw_qos_profile_sensor_data;
  img_qos.depth = 5;
  rclcpp::SubscriptionOptions sub_opts;
  sub_opts.callback_group = sub_group_;
  color_sub_.subscribe(this, color_topic, img_qos, sub_opts);
  depth_sub_.subscribe(this, depth_topic, img_qos, sub_opts);
  // Per-stream arrival tracking for ~/status ages.
  color_sub_.registerCallback([this](Image::ConstSharedPtr msg) {
    last_color_ns_.store(to_ns(msg->header.stamp));
  });
  depth_sub_.registerCallback([this](Image::ConstSharedPtr msg) {
    last_depth_ns_.store(to_ns(msg->header.stamp));
  });

  SyncPolicy policy(static_cast<uint32_t>(sync_queue));
  policy.setMaxIntervalDuration(rclcpp::Duration::from_seconds(slop));
  sync_ = std::make_unique<message_filters::Synchronizer<SyncPolicy>>(
      policy, color_sub_, depth_sub_);
  sync_->registerCallback(std::bind(&CameraServerNode::on_synced, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2));

  const rclcpp::QoS info_qos(10);  // RELIABLE KEEP_LAST(10)
  color_info_sub_ = create_subscription<CameraInfo>(
      color_info_topic, info_qos,
      [this](CameraInfo::ConstSharedPtr msg) { store_.set_color_info(msg); },
      sub_opts);
  depth_info_sub_ = create_subscription<CameraInfo>(
      depth_info_topic, info_qos,
      [this](CameraInfo::ConstSharedPtr msg) { store_.set_depth_info(msg); },
      sub_opts);

  snapshot_srv_ = create_service<GetCameraSnapshot>(
      "~/get_snapshot",
      std::bind(&CameraServerNode::handle_snapshot, this,
                std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, service_group_);
  point_cloud_srv_ = create_service<GetCameraPointCloud>(
      "~/get_point_cloud",
      std::bind(&CameraServerNode::handle_point_cloud, this,
                std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, service_group_);
  transform_srv_ = create_service<GetTransform>(
      "~/get_transform",
      std::bind(&CameraServerNode::handle_transform, this,
                std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, service_group_);

  status_pub_ = create_publisher<CameraServerStatus>("~/status", 10);
  status_timer_ = create_wall_timer(
      std::chrono::duration<double>(status_period_sec_),
      std::bind(&CameraServerNode::publish_status, this), sub_group_);

  RCLCPP_INFO(get_logger(),
              "camera_server up: color=%s depth=%s (sync slop %.3fs), tf cache %.0fs",
              color_topic.c_str(), depth_topic.c_str(), slop, tf_cache);
}

void CameraServerNode::on_synced(Image::ConstSharedPtr color,
                                 Image::ConstSharedPtr depth) {
  store_.set_pair(std::move(color), std::move(depth));
}

void CameraServerNode::publish_status() {
  const int64_t now_ns = get_clock()->now().nanoseconds();
  const FramePair pair = store_.latest_pair();
  CameraServerStatus msg;
  const int64_t c = last_color_ns_.load(), d = last_depth_ns_.load();
  msg.color_age_sec = c < 0 ? -1.f : static_cast<float>((now_ns - c) * 1e-9);
  msg.depth_age_sec = d < 0 ? -1.f : static_cast<float>((now_ns - d) * 1e-9);
  msg.pair_age_sec =
      pair.color ? static_cast<float>((now_ns - pair.stamp_ns) * 1e-9) : -1.f;
  msg.last_pair_stamp = from_ns(pair.color ? pair.stamp_ns : 0);
  msg.pair_seq = pair.seq;
  msg.sync_fps = static_cast<float>((pair.seq - status_last_seq_) /
                                    status_period_sec_);
  status_last_seq_ = pair.seq;
  status_pub_->publish(msg);

  // Starvation warning (spec §8): inputs alive but sync not firing.
  const bool input_alive =
      (msg.color_age_sec >= 0 && msg.color_age_sec < 1.0f) ||
      (msg.depth_age_sec >= 0 && msg.depth_age_sec < 1.0f);
  if (input_alive && (msg.pair_age_sec < 0 || msg.pair_age_sec > 2.0f)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                         "sync starved: color_age=%.2fs depth_age=%.2fs "
                         "pair_age=%.2fs — check partner stream / stamps",
                         msg.color_age_sec, msg.depth_age_sec,
                         msg.pair_age_sec);
  }
}

int32_t CameraServerNode::acquire_pair(
    float max_age_sec, const builtin_interfaces::msg::Time& captured_after,
    float wait_timeout_sec, FramePair& pair, std::string& error_msg) {
  if (max_age_sec < 0.f || wait_timeout_sec < 0.f) {
    error_msg = "negative max_age_sec / wait_timeout_sec";
    return GetCameraSnapshot::Response::STATUS_BAD_REQUEST;
  }
  const int64_t after_ns = to_ns(captured_after);
  if (after_ns > 0) {
    const double wait_s = wait_timeout_sec > 0.f
                              ? std::min<double>(wait_timeout_sec, max_wait_sec_)
                              : max_wait_sec_;
    pair = store_.wait_for_pair_after(
        after_ns, std::chrono::nanoseconds(
                      static_cast<int64_t>(wait_s * 1e9)));
    if (!pair.color) {
      error_msg = "no camera data";
      return GetCameraSnapshot::Response::STATUS_NO_DATA;
    }
    if (pair.stamp_ns < after_ns) {
      error_msg = "wait timed out; newest pair is " +
                  std::to_string((after_ns - pair.stamp_ns) * 1e-9) +
                  "s older than captured_after";
      return GetCameraSnapshot::Response::STATUS_WAIT_TIMEOUT;
    }
  } else {
    pair = store_.latest_pair();
    if (!pair.color) {
      error_msg = "no camera data";
      return GetCameraSnapshot::Response::STATUS_NO_DATA;
    }
  }
  if (max_age_sec > 0.f) {
    const int64_t now_ns = get_clock()->now().nanoseconds();
    const double age = (now_ns - pair.stamp_ns) * 1e-9;
    if (age > max_age_sec) {
      error_msg = "cached pair is " + std::to_string(age) + "s old (max_age " +
                  std::to_string(max_age_sec) + "s)";
      return GetCameraSnapshot::Response::STATUS_STALE;
    }
  }
  return GetCameraSnapshot::Response::STATUS_OK;
}

void CameraServerNode::handle_snapshot(
    GetCameraSnapshot::Request::ConstSharedPtr req,
    GetCameraSnapshot::Response::SharedPtr res) {
  FramePair pair;
  res->status = acquire_pair(req->max_age_sec, req->captured_after,
                             req->wait_timeout_sec, pair, res->error_msg);
  if (res->status == GetCameraSnapshot::Response::STATUS_NO_DATA ||
      res->status == GetCameraSnapshot::Response::STATUS_BAD_REQUEST) {
    return;  // nothing usable to attach
  }
  // OK / STALE / WAIT_TIMEOUT all attach the newest pair (spec §4.1, §7).
  res->stamp = from_ns(pair.stamp_ns);
  res->frame_id = pair.color->header.frame_id;
  if (req->want_color) res->color = *pair.color;
  if (req->want_depth && pair.depth) res->depth = *pair.depth;
  if (req->want_camera_info) {
    auto ci = store_.color_info();
    auto di = store_.depth_info();
    if (ci) res->color_info = *ci;
    if (di) res->depth_info = *di;
    if (!ci || !di) {
      res->error_msg += (res->error_msg.empty() ? "" : "; ");
      res->error_msg += "camera_info not (fully) received yet";
    }
  }
  res->transforms.reserve(req->target_frames.size());
  res->transforms_ok.reserve(req->target_frames.size());
  for (const auto& target : req->target_frames) {
    try {
      res->transforms.push_back(tf_buffer_->lookupTransform(
          target, res->frame_id, rclcpp::Time(res->stamp),
          rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_)));
      res->transforms_ok.push_back(true);
    } catch (const tf2::TransformException& e) {
      res->transforms.emplace_back();
      res->transforms_ok.push_back(false);
      res->error_msg += (res->error_msg.empty() ? "" : "; ");
      res->error_msg += "tf " + target + "<-" + res->frame_id + ": " + e.what();
    }
  }
}

void CameraServerNode::handle_point_cloud(
    GetCameraPointCloud::Request::ConstSharedPtr /*req*/,
    GetCameraPointCloud::Response::SharedPtr res) {
  // Replaced with the real implementation in the following task.
  res->status = GetCameraPointCloud::Response::STATUS_BAD_REQUEST;
  res->error_msg = "get_point_cloud not implemented yet";
}

void CameraServerNode::handle_transform(
    GetTransform::Request::ConstSharedPtr req,
    GetTransform::Response::SharedPtr res) {
  if (req->target_frame.empty() || req->source_frame.empty()) {
    res->status = GetTransform::Response::STATUS_BAD_REQUEST;
    res->error_msg = "target_frame / source_frame must be non-empty";
    return;
  }
  const double timeout =
      req->timeout_sec > 0.f
          ? std::min<double>(req->timeout_sec, transform_timeout_cap_sec_)
          : tf_lookup_timeout_sec_;
  try {
    const int64_t t_ns = to_ns(req->lookup_time);
    if (t_ns == 0) {
      res->transform = tf_buffer_->lookupTransform(
          req->target_frame, req->source_frame, tf2::TimePointZero,
          tf2::durationFromSec(timeout));
    } else {
      res->transform = tf_buffer_->lookupTransform(
          req->target_frame, req->source_frame,
          rclcpp::Time(req->lookup_time),
          rclcpp::Duration::from_seconds(timeout));
    }
    res->status = GetTransform::Response::STATUS_OK;
  } catch (const tf2::TransformException& e) {
    res->status = GetTransform::Response::STATUS_UNAVAILABLE;
    res->error_msg = e.what();
  }
}

}  // namespace camera_server

RCLCPP_COMPONENTS_REGISTER_NODE(camera_server::CameraServerNode)
```

- [ ] **Step 3: Write camera_server_main.cpp**

```cpp
#include <rclcpp/rclcpp.hpp>

#include "camera_server/camera_server_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<camera_server::CameraServerNode>();
  const int threads =
      static_cast<int>(
          node->get_parameter("num_executor_threads").as_int());
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), static_cast<size_t>(threads));
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
```

- [ ] **Step 4: Add node targets to CMakeLists.txt**

Append after the `camera_server_core` install block:

```cmake
add_library(camera_server_component SHARED
  src/camera_server_node.cpp
)
target_link_libraries(camera_server_component camera_server_core)
ament_target_dependencies(camera_server_component
  rclcpp rclcpp_components sensor_msgs geometry_msgs builtin_interfaces
  message_filters tf2 tf2_ros tf2_eigen tinker_vision_msgs_26 Eigen3)
rclcpp_components_register_nodes(camera_server_component
  "camera_server::CameraServerNode")

add_executable(camera_server_node src/camera_server_main.cpp)
target_link_libraries(camera_server_node camera_server_component)
ament_target_dependencies(camera_server_node rclcpp)

install(TARGETS camera_server_component
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS camera_server_node DESTINATION lib/${PROJECT_NAME})
```

- [ ] **Step 5: Build, then smoke-test T1-style (no cameras)**

```bash
tkbuild tk26_vision --packages-select camera_server
source /home/tinker/tk25_ws/install/setup.zsh
ros2 run camera_server camera_server_node &
sleep 3
ros2 service call /camera_server/get_snapshot tinker_vision_msgs_26/srv/GetCameraSnapshot '{}'
ros2 service call /camera_server/get_transform tinker_vision_msgs_26/srv/GetTransform '{target_frame: base_link, source_frame: map, timeout_sec: 0.2}'
ros2 topic echo --once /camera_server/status
kill %1
```

Expected: snapshot responds `status: 1`, `error_msg: 'no camera data'` (no hang, no crash); get_transform responds `status: 1` with a tf2 error string (no TF publisher running); status topic prints one message with `color_age_sec: -1.0`. Node survives all three.

- [ ] **Step 6: Append changelog line to README.md**

```markdown
- 2026-07-13: CameraServerNode — synced subs (BEST_EFFORT/slop 0.1s), 180s TF buffer w/ dedicated listener thread, get_snapshot (freshness + frame-stamped transforms), get_transform, 1 Hz ~/status; component-registered + standalone main.
```

- [ ] **Step 7: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git status --short
git add src/camera_server/include/camera_server/camera_server_node.hpp \
        src/camera_server/src/camera_server_node.cpp \
        src/camera_server/src/camera_server_main.cpp \
        src/camera_server/CMakeLists.txt src/camera_server/README.md
git commit -m "feat(camera_server): CameraServerNode — get_snapshot/get_transform, TF buffer, status heartbeat"
```

---

### Task 5: `get_point_cloud` service implementation

**Files:**
- Modify: `src/tk26_vision/src/camera_server/src/camera_server_node.cpp` (replace `handle_point_cloud` stub)
- Modify: `src/tk26_vision/src/camera_server/README.md` (changelog)

**Interfaces:**
- Consumes: `acquire_pair` + `Deprojector::deproject` + `tf_buffer_` (Tasks 2–4).
- Produces: working `~/get_point_cloud` per spec §4.2 — non-OK statuses return no cloud (unlike snapshot; clouds are expensive), `target_frame` transform at `depth.header.stamp`, response/cloud stamps copied from that same depth stamp, TF failure fails closed with `STATUS_TF_FAIL`.

- [ ] **Step 1: Replace the stub `handle_point_cloud` in camera_server_node.cpp**

Also add these includes at the top of the file:

```cpp
#include <tf2_eigen/tf2_eigen.hpp>
```

New implementation:

```cpp
void CameraServerNode::handle_point_cloud(
    GetCameraPointCloud::Request::ConstSharedPtr req,
    GetCameraPointCloud::Response::SharedPtr res) {
  res->points = sensor_msgs::msg::PointCloud2{};
  FramePair pair;
  const AcquisitionStatus acquisition =
      acquire_pair(req->max_age_sec, req->captured_after,
                   req->wait_timeout_sec, pair, res->error_msg);
  if (acquisition != AcquisitionStatus::kOk) {
    res->status = acquisition == AcquisitionStatus::kNoData
                      ? GetCameraPointCloud::Response::STATUS_NO_DATA
                  : acquisition == AcquisitionStatus::kStale
                      ? GetCameraPointCloud::Response::STATUS_STALE
                  : acquisition == AcquisitionStatus::kWaitTimeout
                      ? GetCameraPointCloud::Response::STATUS_WAIT_TIMEOUT
                      : GetCameraPointCloud::Response::STATUS_BAD_REQUEST;
    return;
  }
  const auto depth_info = store_.depth_info();
  if (!pair.depth || !depth_info ||
      (req->include_color && !pair.color)) {
    res->status = GetCameraPointCloud::Response::STATUS_NO_DATA;
    res->error_msg = !pair.depth ? "no depth frame in synchronized pair"
                    : !depth_info ? "depth camera_info not received yet"
                    : "color requested but no color frame is available";
    return;
  }
  const std::string native_frame = pair.depth->header.frame_id;
  if (native_frame.empty()) {
    res->status = GetCameraPointCloud::Response::STATUS_NO_DATA;
    res->error_msg = "depth frame_id is empty";
    return;
  }
  const auto cloud_stamp = pair.depth->header.stamp;
  std::optional<Eigen::Isometry3f> transform;
  if (!req->target_frame.empty() && req->target_frame != native_frame) {
    if (to_ns(cloud_stamp) == 0) {
      res->status = GetCameraPointCloud::Response::STATUS_TF_FAIL;
      res->error_msg = "cannot perform time-correct target transform for zero depth stamp";
      return;
    }
    try {
      const auto tf_msg = tf_buffer_->lookupTransform(
          req->target_frame, native_frame,
          rclcpp::Time(cloud_stamp, RCL_ROS_TIME),
          rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_));
      transform = tf2::transformToEigen(tf_msg).cast<float>();
      if (!transform->matrix().allFinite()) {
        throw std::runtime_error("TF transform contains non-finite values");
      }
    } catch (const std::exception& e) {
      res->status = GetCameraPointCloud::Response::STATUS_TF_FAIL;
      res->error_msg = "TF " + req->target_frame + "<-" + native_frame +
                       " at depth stamp: " + e.what();
      return;
    }
  }
  const sensor_msgs::msg::Image* color = req->include_color ? pair.color.get() : nullptr;
  std::string error;
  {
    std::lock_guard<std::mutex> lock(deproject_mutex_);
    if (!deprojector_.deproject(*pair.depth, *depth_info, color, req->stride,
                                transform, res->points, error)) {
      res->status = GetCameraPointCloud::Response::STATUS_NO_DATA;
      res->error_msg = "camera data cannot be deprojected: " + error;
      return;
    }
  }
  res->stamp = cloud_stamp;
  res->points.header.stamp = cloud_stamp;
  res->points.header.frame_id = req->target_frame.empty() ? native_frame : req->target_frame;
  res->status = GetCameraPointCloud::Response::STATUS_OK;
}
```

- [ ] **Step 2: Build and smoke-test with a synthetic publisher**

```bash
tkbuild tk26_vision --packages-select camera_server
source /home/tinker/tk25_ws/install/setup.zsh
ros2 run camera_server camera_server_node &
sleep 2
# Without cameras: expect status 1 (NO_DATA), returned promptly, node alive.
ros2 service call /camera_server/get_point_cloud tinker_vision_msgs_26/srv/GetCameraPointCloud '{include_color: false}'
kill %1
```

Expected: `status: 1`, `error_msg: 'no camera data'`, node does not crash. (Live-data verification is Task 8's T2 parity script.)

- [ ] **Step 3: Append changelog line to README.md**

```markdown
- 2026-07-23: get_point_cloud implemented — on-demand deprojection, stride/color, target_frame at depth stamp (TF fail-closed), conservative pair freshness, strict frame/data validation, and empty-cloud failure payloads.
```

- [ ] **Step 4: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git status --short
git add src/camera_server/src/camera_server_node.cpp src/camera_server/README.md
git commit -m "feat(camera_server): implement get_point_cloud (deprojection + frame-stamped transform, fail-closed TF)"
```

---

### Task 6: `camera_compat_bridge` — legacy service names

**Files:**
- Create: `src/tk26_vision/src/camera_server/src/compat_bridge_node.cpp`
- Modify: `src/tk26_vision/src/camera_server/CMakeLists.txt`
- Modify: `src/tk26_vision/src/camera_server/README.md` (changelog)

**Interfaces:**
- Consumes: `GetCameraSnapshot` / `GetCameraPointCloud` services (Tasks 4–5) as a client; legacy types `GetImage`, `GetPointCloud`, `GetOrbbecPC` (pre-existing).
- Produces: executable `camera_compat_bridge` serving `get_image_service`, `get_point_cloud_service`, `get_orbbec_pc` with the exact legacy semantics (`status` 0/1, `error_msg` shapes `Unsupported camera: <x>.` / `No camera data for <x>.`). Parameters: `wrist_server` (default `/wrist_camera_server`), `head_server` (default `/head_camera_server`), `forward_timeout_sec` (default 5.0). **Never launched by default** (spec §6) — Task 7 wires it gated-off.

Implementation correction: the landed bridge uses an auto-add=false client
callback group on a node-owned `SingleThreadedExecutor` thread. This is
required because legacy service callbacks synchronously wait for forwarded
responses; a main executor (including a single-threaded component container)
must never be responsible for servicing those client futures. Forwarding uses
one steady-clock deadline covering discovery and response, removes timed-out
pending requests, catches middleware exceptions, and maps every unavailable or
non-OK upstream result to the exact legacy `No camera data for <camera>.`
shape. The source file is authoritative for these details; the original code
sketch below is retained only as a field-mapping reference.

- [ ] **Step 1: Write compat_bridge_node.cpp**

```cpp
// Legacy-name compatibility bridge: serves get_image_service /
// get_point_cloud_service / get_orbbec_pc by forwarding to the per-camera
// servers. Zero subscriptions. Param-gated OFF by default — the Python
// utility nodes own these names until cutover (spec §6, §11).
#include <chrono>
#include <map>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "tinker_vision_msgs_26/srv/get_camera_point_cloud.hpp"
#include "tinker_vision_msgs_26/srv/get_camera_snapshot.hpp"
#include "tinker_vision_msgs_26/srv/get_image.hpp"
#include "tinker_vision_msgs_26/srv/get_orbbec_pc.hpp"
#include "tinker_vision_msgs_26/srv/get_point_cloud.hpp"

namespace camera_server {

using GetCameraSnapshot = tinker_vision_msgs_26::srv::GetCameraSnapshot;
using GetCameraPointCloud = tinker_vision_msgs_26::srv::GetCameraPointCloud;
using GetImage = tinker_vision_msgs_26::srv::GetImage;
using GetPointCloud = tinker_vision_msgs_26::srv::GetPointCloud;
using GetOrbbecPC = tinker_vision_msgs_26::srv::GetOrbbecPC;

class CompatBridgeNode : public rclcpp::Node {
 public:
  CompatBridgeNode() : rclcpp::Node("camera_compat_bridge") {
    const auto wrist =
        declare_parameter<std::string>("wrist_server", "/wrist_camera_server");
    const auto head =
        declare_parameter<std::string>("head_server", "/head_camera_server");
    forward_timeout_ = std::chrono::duration<double>(
        declare_parameter<double>("forward_timeout_sec", 5.0));

    client_group_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    service_group_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    for (const auto& [cam, ns] :
         std::map<std::string, std::string>{{"realsense", wrist},
                                            {"orbbec", head}}) {
      snapshot_clients_[cam] = create_client<GetCameraSnapshot>(
          ns + "/get_snapshot", rmw_qos_profile_services_default,
          client_group_);
      cloud_clients_[cam] = create_client<GetCameraPointCloud>(
          ns + "/get_point_cloud", rmw_qos_profile_services_default,
          client_group_);
    }

    image_srv_ = create_service<GetImage>(
        "get_image_service",
        std::bind(&CompatBridgeNode::handle_image, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, service_group_);
    cloud_srv_ = create_service<GetPointCloud>(
        "get_point_cloud_service",
        std::bind(&CompatBridgeNode::handle_cloud, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, service_group_);
    orbbec_pc_srv_ = create_service<GetOrbbecPC>(
        "get_orbbec_pc",
        std::bind(&CompatBridgeNode::handle_orbbec_pc, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, service_group_);
    RCLCPP_INFO(get_logger(),
                "compat bridge up (wrist=%s head=%s) — legacy names served",
                wrist.c_str(), head.c_str());
  }

 private:
  /// Forward helper: returns response or nullptr (unavailable/timeout).
  template <typename ClientT, typename ReqT>
  auto forward(ClientT& client, ReqT req)
      -> typename ClientT::element_type::SharedResponse {
    if (!client->wait_for_service(std::chrono::milliseconds(200))) {
      return nullptr;
    }
    auto future = client->async_send_request(req);
    if (future.wait_for(std::chrono::duration_cast<std::chrono::nanoseconds>(
            forward_timeout_)) != std::future_status::ready) {
      return nullptr;
    }
    return future.get();
  }

  void handle_image(GetImage::Request::ConstSharedPtr req,
                    GetImage::Response::SharedPtr res) {
    auto it = snapshot_clients_.find(req->camera);
    if (it == snapshot_clients_.end()) {
      res->status = 1;
      res->error_msg = "Unsupported camera: " + req->camera + ".";
      return;
    }
    auto fwd = std::make_shared<GetCameraSnapshot::Request>();
    fwd->want_color = true;
    fwd->want_depth = req->depth;
    fwd->want_camera_info = false;
    auto out = forward(it->second, fwd);
    if (!out) {
      res->status = 1;
      res->error_msg = "camera_server unreachable for " + req->camera + ".";
      return;
    }
    if (out->status != GetCameraSnapshot::Response::STATUS_OK) {
      res->status = 1;
      res->error_msg = "No camera data for " + req->camera + ".";
      return;
    }
    res->status = 0;
    res->error_msg = "";
    res->rgb_image = out->color;
    if (req->depth) res->depth_image = out->depth;
  }

  void handle_cloud(GetPointCloud::Request::ConstSharedPtr req,
                    GetPointCloud::Response::SharedPtr res) {
    auto it = cloud_clients_.find(req->camera);
    if (it == cloud_clients_.end()) {
      res->status = 1;
      res->error_msg = "Unsupported camera: " + req->camera + ".";
      return;
    }
    auto fwd = std::make_shared<GetCameraPointCloud::Request>();
    fwd->include_color = true;
    fwd->stride = 0;
    auto out = forward(it->second, fwd);
    if (!out) {
      res->status = 1;
      res->error_msg = "camera_server unreachable for " + req->camera + ".";
      return;
    }
    if (out->status != GetCameraPointCloud::Response::STATUS_OK) {
      res->status = 1;
      res->error_msg = "No camera data for " + req->camera + ".";
      return;
    }
    res->status = 0;
    res->error_msg = "";
    res->points = out->points;
  }

  void handle_orbbec_pc(GetOrbbecPC::Request::ConstSharedPtr req,
                        GetOrbbecPC::Response::SharedPtr res) {
    auto fwd = std::make_shared<GetCameraPointCloud::Request>();
    fwd->stride = req->stride;
    fwd->include_color = req->include_color;
    auto out = forward(cloud_clients_.at("orbbec"), fwd);
    if (!out) {
      res->status = 1;
      res->error_msg = "camera_server unreachable for orbbec.";
      return;
    }
    if (out->status != GetCameraPointCloud::Response::STATUS_OK) {
      res->status = 1;
      res->error_msg = out->error_msg;
      return;
    }
    res->status = 0;
    res->error_msg = "";
    res->points = out->points;
  }

  std::chrono::duration<double> forward_timeout_{5.0};
  std::map<std::string, rclcpp::Client<GetCameraSnapshot>::SharedPtr>
      snapshot_clients_;
  std::map<std::string, rclcpp::Client<GetCameraPointCloud>::SharedPtr>
      cloud_clients_;
  rclcpp::Service<GetImage>::SharedPtr image_srv_;
  rclcpp::Service<GetPointCloud>::SharedPtr cloud_srv_;
  rclcpp::Service<GetOrbbecPC>::SharedPtr orbbec_pc_srv_;
  rclcpp::CallbackGroup::SharedPtr client_group_, service_group_;
};

}  // namespace camera_server

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<camera_server::CompatBridgeNode>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),
                                                    4);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
```

- [ ] **Step 2: Add the executable to CMakeLists.txt**

```cmake
add_executable(camera_compat_bridge src/compat_bridge_node.cpp)
ament_target_dependencies(camera_compat_bridge
  rclcpp sensor_msgs tinker_vision_msgs_26)
install(TARGETS camera_compat_bridge DESTINATION lib/${PROJECT_NAME})
```

- [ ] **Step 3: Build and smoke-test the forwarding path end-to-end (no cameras)**

```bash
tkbuild tk26_vision --packages-select camera_server
source /home/tinker/tk25_ws/install/setup.zsh
ros2 run camera_server camera_server_node --ros-args -r __node:=head_camera_server &
ros2 run camera_server camera_compat_bridge &
sleep 3
# Full chain: bridge -> head server -> NO_DATA -> legacy error shape.
ros2 service call /get_image_service tinker_vision_msgs_26/srv/GetImage '{camera: orbbec, depth: true}'
ros2 service call /get_image_service tinker_vision_msgs_26/srv/GetImage '{camera: kinect}'
ros2 service call /get_orbbec_pc tinker_vision_msgs_26/srv/GetOrbbecPC '{stride: 2, include_color: false}'
kill %1 %2
```

Expected: first call `status: 1`, `error_msg: 'No camera data for orbbec.'`; second `status: 1`, `error_msg: 'Unsupported camera: kinect.'`; third `status: 1` (NO_DATA passthrough). No hang beyond `forward_timeout_sec`; both nodes stay alive.
IMPORTANT: run this with the real Python `get_image` node NOT running (check `ros2 service list | grep get_image` first) — duplicate service names would make the result ambiguous.

- [ ] **Step 4: Append changelog line to README.md**

```markdown
- 2026-07-13: camera_compat_bridge — legacy get_image_service/get_point_cloud_service/get_orbbec_pc forwarder (zero subscriptions, launch-gated OFF).
```

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git status --short
git add src/camera_server/src/compat_bridge_node.cpp \
        src/camera_server/CMakeLists.txt src/camera_server/README.md
git commit -m "feat(camera_server): legacy compat bridge (get_image/get_point_cloud/get_orbbec_pc forwarders)"
```

---

### Task 7: Launch files + vision_driver wiring

**Files:**
- Create: `src/tk26_vision/src/camera_server/launch/camera_server.launch.py`
- Modify: `src/tk26_vision/src/camera_server/CMakeLists.txt` (install launch/)
- Modify: `src/tk26_vision/src/vision_bringup/launch/vision_driver.launch.py`
- Modify: `src/tk26_vision/src/camera_server/README.md` (changelog)

**Interfaces:**
- Consumes: executables `camera_server_node`, `camera_compat_bridge` (Tasks 4–6).
- Produces: launchable instances `wrist_camera_server` / `head_camera_server`; `vision_driver.launch.py` arg `enable_camera_server` (default `'true'`) launching the head instance; `enable_legacy_services` (default `'false'`) gating the bridge in the package launch only.

Launch correction: `vision_driver.launch.py` must import
`launch_ros.actions.Node`, and `vision_bringup/package.xml` must declare
`<exec_depend>camera_server</exec_depend>`. Both head launch paths use
`/camera/color/camera_info` for registered Orbbec depth, matching the legacy
deprojection path. The standalone launch and `vision_driver` launch are
mutually exclusive when both enable the head server; the bridge must remain
disabled while legacy Python services own the root names.

- [ ] **Step 1: Write camera_server.launch.py**

```python
"""Standalone dev launch for the camera servers + optional compat bridge.

Production wiring: head instance is included by vision_bringup's
vision_driver.launch.py; the wrist instance is included by the manipulation
bringup that owns the RealSense (separate repo/commit).
Spec: docs/specs/2026-07-13-camera-server-design.md §9.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WRIST_PARAMS = {
    'color_topic': '/camera/xarm_camera/color/image_raw',
    'depth_topic': '/camera/xarm_camera/aligned_depth_to_color/image_raw',
    'color_info_topic': '/camera/xarm_camera/color/camera_info',
    'depth_info_topic': '/camera/xarm_camera/aligned_depth_to_color/camera_info',
}
HEAD_PARAMS = {
    'color_topic': '/camera/color/image_raw',
    'depth_topic': '/camera/depth/image_raw',
    'color_info_topic': '/camera/color/camera_info',
    'depth_info_topic': '/camera/color/camera_info',
}


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('enable_wrist', default_value='false',
                              description='Launch wrist_camera_server (RealSense).'),
        DeclareLaunchArgument('enable_head', default_value='true',
                              description='Launch head_camera_server (Orbbec).'),
        DeclareLaunchArgument(
            'enable_legacy_services', default_value='false',
            description='Serve legacy get_image_service/get_point_cloud_service/'
                        'get_orbbec_pc via the compat bridge. Keep FALSE while '
                        'the Python utility nodes still own those names.'),
        Node(
            package='camera_server', executable='camera_server_node',
            name='wrist_camera_server', output='screen',
            parameters=[WRIST_PARAMS],
            condition=IfCondition(LaunchConfiguration('enable_wrist')),
        ),
        Node(
            package='camera_server', executable='camera_server_node',
            name='head_camera_server', output='screen',
            parameters=[HEAD_PARAMS],
            condition=IfCondition(LaunchConfiguration('enable_head')),
        ),
        Node(
            package='camera_server', executable='camera_compat_bridge',
            name='camera_compat_bridge', output='screen',
            condition=IfCondition(LaunchConfiguration('enable_legacy_services')),
        ),
    ])
```

Add to `CMakeLists.txt` (after the executable installs):

```cmake
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})
```

- [ ] **Step 2: Wire the head instance into vision_driver.launch.py**

Read the file first. Two edits, following its existing gated-Node pattern (`enable_orbbec`/`enable_ffs`):

1. In the launch-arguments list, after the `'enable_orbbec'` declaration, add:

```python
        DeclareLaunchArgument(
            'enable_camera_server', default_value='true',
            description=(
                'Launch head_camera_server (camera_server pkg): on-demand '
                'snapshot/point-cloud/TF services over the Orbbec streams. '
                'Additive — new service names, no collisions with the '
                'Python utility nodes.'
            ),
        ),
```

2. In the same nodes list that contains the Orbbec and FFS `Node(...)` entries (so it inherits the launch-wide SHM profile environment), add after the Orbbec node block:

```python
    head_camera_server_node = Node(
        package='camera_server',
        executable='camera_server_node',
        name='head_camera_server',
        output='screen',
        parameters=[{
            'color_topic': '/camera/color/image_raw',
            'depth_topic': '/camera/depth/image_raw',
            'color_info_topic': '/camera/color/camera_info',
            'depth_info_topic': '/camera/color/camera_info',
        }],
        condition=IfCondition(LaunchConfiguration('enable_camera_server')),
    )
```

and append `head_camera_server_node` to the returned LaunchDescription in the same position group as the other driver-layer nodes. Match the file's exact list/variable style — read before editing; if the file builds nodes inline in a list, add the Node inline instead of via a variable.

NOTE: do NOT add the compat bridge to vision_driver — bridge stays dev-launch-only until cutover (spec §11).

- [ ] **Step 3: Build and verify launch descriptions parse**

```bash
tkbuild tk26_vision --packages-select camera_server vision_bringup
source /home/tinker/tk25_ws/install/setup.zsh
ros2 launch camera_server camera_server.launch.py enable_head:=true &
sleep 4
ros2 node list | grep head_camera_server
ros2 service list | grep -E "head_camera_server/(get_snapshot|get_point_cloud|get_transform)"
kill %1
python3 -c "
import launch, ament_index_python, importlib.util, sys
p = ament_index_python.get_package_share_directory('vision_bringup') + '/launch/vision_driver.launch.py'
spec = importlib.util.spec_from_file_location('vd', p); m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m); m.generate_launch_description(); print('vision_driver launch OK')
"
```

Expected: node + 3 services listed; `vision_driver launch OK` printed (parse check only — do NOT launch vision_driver, it would start the real Orbbec driver).

- [ ] **Step 4: Append changelog line to README.md**

```markdown
- 2026-07-13: launch — standalone camera_server.launch.py (wrist/head/bridge gated) + head instance wired into vision_driver.launch.py behind enable_camera_server (default true).
```

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git status --short   # careful: vision_driver.launch.py may have concurrent edits — stage only if your hunk is the only local change; otherwise coordinate
git add src/camera_server/launch/camera_server.launch.py \
        src/camera_server/CMakeLists.txt src/camera_server/README.md \
        src/vision_bringup/launch/vision_driver.launch.py
git commit -m "feat(camera_server): launch wiring — dev launch + head instance in vision_driver (gated, default on)"
```

---

### Task 8: T1/T2 test scripts + DEV_NOTES entry

**Files:**
- Create: `src/tk26_vision/scripts/tests/manual/camera_server_t1.sh`
- Create: `src/tk26_vision/scripts/tests/manual/camera_server_t2_parity.py`
- Modify: `src/tk26_vision/DEV_NOTES.md` (append run entry)

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: repeatable T1 (no-camera) check script and T2 (live-camera) parity/freshness harness for operator runs.

- [ ] **Step 1: Write camera_server_t1.sh**

```bash
#!/usr/bin/env bash
# T1 (startup, no cameras): camera_server must start, answer all three
# services with clean NO_DATA/UNAVAILABLE statuses, publish ~/status, and
# survive. Run from anywhere; no camera hardware needed.
set -uo pipefail
source /home/tinker/tk25_ws/install/setup.bash

FAIL=0
note() { echo "[camera_server_t1] $*"; }
check() { # name, expected_substring, actual
  if [[ "$3" == *"$2"* ]]; then note "PASS: $1"; else note "FAIL: $1 — wanted '$2' in: $3"; FAIL=1; fi
}

ros2 run camera_server camera_server_node --ros-args -r __node:=t1_camera_server &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null' EXIT
sleep 3

OUT=$(timeout 10 ros2 service call /t1_camera_server/get_snapshot \
  tinker_vision_msgs_26/srv/GetCameraSnapshot '{}' 2>&1)
check "snapshot NO_DATA" "status=1" "$OUT"

OUT=$(timeout 10 ros2 service call /t1_camera_server/get_point_cloud \
  tinker_vision_msgs_26/srv/GetCameraPointCloud '{include_color: false}' 2>&1)
check "point_cloud NO_DATA" "status=1" "$OUT"

OUT=$(timeout 10 ros2 service call /t1_camera_server/get_transform \
  tinker_vision_msgs_26/srv/GetTransform \
  '{target_frame: base_link, source_frame: map, timeout_sec: 0.2}' 2>&1)
check "transform UNAVAILABLE" "status=1" "$OUT"

OUT=$(timeout 10 ros2 topic echo --once /t1_camera_server/status 2>&1)
check "status publishes" "pair_seq" "$OUT"
check "status ages -1" "color_age_sec: -1.0" "$OUT"

# captured_after with a far-future stamp must time out within ~max_wait (2s).
T0=$SECONDS
OUT=$(timeout 15 ros2 service call /t1_camera_server/get_snapshot \
  tinker_vision_msgs_26/srv/GetCameraSnapshot \
  '{captured_after: {sec: 2000000000}}' 2>&1)
ELAPSED=$((SECONDS - T0))
check "captured_after NO_DATA (empty store)" "status=1" "$OUT"
[[ $ELAPSED -le 6 ]] && note "PASS: bounded wait (${ELAPSED}s)" || { note "FAIL: wait took ${ELAPSED}s"; FAIL=1; }

kill -0 $SRV_PID 2>/dev/null && note "PASS: node alive" || { note "FAIL: node died"; FAIL=1; }
exit $FAIL
```

Then: `chmod +x src/tk26_vision/scripts/tests/manual/camera_server_t1.sh`

Note: `ros2 service call` prints responses like `response:\n...status=1...` — if the actual formatting differs (`status: 1`), adjust the `check` expectations to match observed output on first run; the invariant being tested is the status value, not the formatting.

- [ ] **Step 2: Write camera_server_t2_parity.py**

```python
#!/usr/bin/env python3
"""T2 (live cameras): freshness, captured_after, TF-at-stamp, and point-cloud
parity for a running camera_server instance.

Usage (head camera, with vision_driver up):
    python3 camera_server_t2_parity.py --server /head_camera_server \
        --legacy-pc-service get_orbbec_pc --driver-cloud /camera/depth_registered/points

Usage (wrist camera, with the manip bringup up):
    python3 camera_server_t2_parity.py --server /wrist_camera_server
"""
import argparse
import math
import struct
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.time import Time

import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from tinker_vision_msgs_26.srv import (GetCameraPointCloud,
                                       GetCameraSnapshot, GetOrbbecPC,
                                       GetTransform)

RESULTS = []


def record(name, ok, detail=""):
    RESULTS.append((name, ok))
    print(f"[t2] {'PASS' if ok else 'FAIL'}: {name} {detail}")


def centroid(cloud: PointCloud2):
    n, sx, sy, sz = 0, 0.0, 0.0, 0.0
    for x, y, z in pc2.read_points(cloud, field_names=("x", "y", "z"),
                                   skip_nans=True):
        sx += x; sy += y; sz += z; n += 1
    return n, (sx / n, sy / n, sz / n) if n else (0, 0, 0)


class T2(Node):
    def __init__(self, server):
        super().__init__("camera_server_t2")
        self.snapshot = self.create_client(GetCameraSnapshot,
                                           server + "/get_snapshot")
        self.cloud = self.create_client(GetCameraPointCloud,
                                        server + "/get_point_cloud")
        self.transform = self.create_client(GetTransform,
                                            server + "/get_transform")

    def call(self, client, req, timeout=15.0):
        if not client.wait_for_service(timeout_sec=5.0):
            return None
        fut = client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=timeout)
        return fut.result()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--legacy-pc-service", default="")
    ap.add_argument("--driver-cloud", default="")
    ap.add_argument("--target-frame", default="base_link")
    args = ap.parse_args()

    rclpy.init()
    node = T2(args.server)

    # 1. Snapshot freshness + sync skew.
    res = node.call(node.snapshot, GetCameraSnapshot.Request())
    ok = res is not None and res.status == 0
    record("snapshot returns OK", ok, f"(status={getattr(res,'status','none')})")
    if ok:
        now = node.get_clock().now().nanoseconds
        age = (now - Time.from_msg(res.stamp).nanoseconds) * 1e-9
        record("pair age < 1s", 0 <= age < 1.0, f"(age={age:.3f}s)")
        skew = abs(Time.from_msg(res.color.header.stamp).nanoseconds -
                   Time.from_msg(res.depth.header.stamp).nanoseconds) * 1e-9
        record("color/depth skew <= 0.1s", skew <= 0.1, f"(skew={skew:.3f}s)")
        record("camera_info present", res.color_info.width > 0
               and res.depth_info.width > 0)

    # 2. captured_after returns a strictly newer pair.
    req = GetCameraSnapshot.Request()
    req.captured_after = node.get_clock().now().to_msg()
    req.wait_timeout_sec = 2.0
    res2 = node.call(node.snapshot, req)
    ok = (res2 is not None and res2.status == 0 and
          Time.from_msg(res2.stamp).nanoseconds >
          Time.from_msg(req.captured_after).nanoseconds)
    record("captured_after yields newer frame", ok,
           f"(status={getattr(res2,'status','none')})")

    # 3. TF at pair stamp: server transform vs the same lookup done locally.
    req = GetCameraSnapshot.Request()
    req.target_frames = [args.target_frame]
    res3 = node.call(node.snapshot, req)
    if res3 is not None and res3.status == 0 and res3.transforms_ok and \
            res3.transforms_ok[0]:
        t = res3.transforms[0].transform.translation
        record("snapshot transform ok", True,
               f"({args.target_frame}<-{res3.frame_id} t=({t.x:.3f},{t.y:.3f},{t.z:.3f}))")
        g = GetTransform.Request()
        g.target_frame, g.source_frame = args.target_frame, res3.frame_id
        g.lookup_time = res3.stamp
        g.timeout_sec = 0.5
        res4 = node.call(node.transform, g)
        same = (res4 is not None and res4.status == 0 and
                math.isclose(res4.transform.transform.translation.x, t.x,
                             abs_tol=1e-6))
        record("get_transform matches snapshot transform", same)
    else:
        record("snapshot transform ok", False,
               f"(transforms_ok={getattr(res3,'transforms_ok',None)}, "
               f"err={getattr(res3,'error_msg','')})")

    # 4. Point cloud sanity + optional parity vs legacy service.
    req = GetCameraPointCloud.Request()
    req.include_color = True
    res5 = node.call(node.cloud, req, timeout=30.0)
    ok = res5 is not None and res5.status == 0 and res5.points.width > 1000
    record("get_point_cloud returns cloud", ok,
           f"(n={getattr(res5.points,'width',0) if res5 else 0})")
    if ok and args.legacy_pc_service:
        legacy = node.create_client(GetOrbbecPC, args.legacy_pc_service)
        lres = node.call(legacy, GetOrbbecPC.Request(stride=1,
                                                     include_color=True),
                         timeout=30.0)
        if lres is not None and lres.status == 0:
            n_new, c_new = centroid(res5.points)
            n_old, c_old = centroid(lres.points)
            dn = abs(n_new - n_old) / max(n_old, 1)
            dc = math.dist(c_new, c_old)
            record("parity vs legacy: count within 15%", dn < 0.15,
                   f"(new={n_new} old={n_old})")
            record("parity vs legacy: centroid within 5cm", dc < 0.05,
                   f"(d={dc*100:.1f}cm)")
        else:
            record("legacy pc service reachable", False)

    node.destroy_node()
    rclpy.shutdown()
    failed = [n for n, ok in RESULTS if not ok]
    print(f"[t2] {len(RESULTS) - len(failed)}/{len(RESULTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run T1 (no hardware needed) and record**

```bash
source /home/tinker/tk25_ws/install/setup.bash
bash src/tk26_vision/scripts/tests/manual/camera_server_t1.sh
```

Expected: all PASS lines, exit 0. Fix `check` expectations against observed `ros2 service call` output formatting on the first run if needed (see Step 1 note).

T2 requires live cameras — do NOT attempt to start drivers; leave T2 for an operator run and note it as pending.

- [ ] **Step 4: Append DEV_NOTES.md entry**

Append at the end of `src/tk26_vision/DEV_NOTES.md`:

```markdown
## 2026-07-13 camera_server landing

- New pkg `camera_server` (spec docs/specs/2026-07-13-camera-server-design.md):
  wrist/head snapshot+point-cloud+TF servers, compat bridge (gated OFF).
- gtests: frame_store (5), deprojector (7) — PASS.
- T1 (`scripts/tests/manual/camera_server_t1.sh`): PASS <fill actual date/host>.
- T2 (`scripts/tests/manual/camera_server_t2_parity.py`): PENDING operator run
  with live cameras — head: parity vs get_orbbec_pc + driver cloud; wrist:
  freshness/TF only. Consumers NOT migrated (spec Appendix A).
```

(Replace `<fill actual date/host>` with the real values from the run.)

- [ ] **Step 5: Commit**

```bash
cd /home/tinker/tk25_ws/src/tk26_vision
git status --short
git add scripts/tests/manual/camera_server_t1.sh \
        scripts/tests/manual/camera_server_t2_parity.py DEV_NOTES.md
git commit -m "test(camera_server): T1 no-camera check script + T2 live parity harness; DEV_NOTES entry"
```

---

## Out of plan (explicitly deferred)

- Wrist-server launch inclusion in the manipulation bringup (`mobile_bringup` / `arm_bringup_cumotion`) — separate repo, separate commit, coordinate with its concurrent committer. Until then the wrist instance runs via `camera_server.launch.py enable_wrist:=true`.
- Enabling the compat bridge + disabling the Python utility nodes in `vision_bringup` (cutover, spec §11) — only after operator T2 passes.
- All consumer migrations (spec Appendix A / intake-refactor backend swap).

## Plan self-review (done at authoring)

- Spec coverage: §3 architecture → Tasks 4/7; §4 interfaces → Task 1; §5 internals → Tasks 2–5; §6 bridge → Task 6; §7 errors → Tasks 4–5 + T1; §8 observability → Task 4 (status+warn); §9 launch → Task 7; §10 testing → Tasks 2/3/8 (T2 = operator, harness provided); §11 cutover + §13/Appendix A → explicitly out of plan. Gap check: none.
- Placeholder scan: none ("not implemented yet" stub in Task 4 is replaced by Task 5 by design; DEV_NOTES `<fill actual>` is instructed to be replaced at run time).
- Type consistency: `FramePair.stamp_ns:int64`, `wait_for_pair_after(after_ns, timeout)` used identically in Tasks 2/4; `Deprojector::deproject(depth, info, color*, stride, optional<Isometry3f>, out, err)` identical in Tasks 3/5; srv field names (`lookup_time`, `transforms_ok`, `captured_after`) consistent across Tasks 1/4/5/6/8.
