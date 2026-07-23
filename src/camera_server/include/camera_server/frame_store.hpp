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
  int64_t stamp_ns = 0;  // Pair stamp (color header) in nanoseconds.
  uint64_t seq = 0;      // Monotonic valid synced-pair counter.
};

/// Thread-safe latest-frame store shared between the sync callback (writer)
/// and service handlers (readers). Stores ConstSharedPtrs -- no image copies.
class FrameStore {
 public:
  /// Stores a complete pair and wakes freshness waiters. Incomplete pairs are
  /// ignored: they do not replace the current pair, advance seq, or notify.
  void set_pair(sensor_msgs::msg::Image::ConstSharedPtr color,
                sensor_msgs::msg::Image::ConstSharedPtr depth);
  void set_color_info(sensor_msgs::msg::CameraInfo::ConstSharedPtr info);
  void set_depth_info(sensor_msgs::msg::CameraInfo::ConstSharedPtr info);

  FramePair latest_pair() const;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr color_info() const;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr depth_info() const;

  /// Blocks until a complete stored pair is stamped >= after_ns, or timeout
  /// elapses. Always returns the newest available pair; the caller checks
  /// both pointers and pair.stamp_ns >= after_ns to distinguish success from
  /// timeout/no data.
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
