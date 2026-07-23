#include "camera_server/frame_store.hpp"

#include <utility>

namespace camera_server {

namespace {
constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

int64_t stamp_ns_of(const sensor_msgs::msg::Image& image) {
  return static_cast<int64_t>(image.header.stamp.sec) *
             kNanosecondsPerSecond +
         static_cast<int64_t>(image.header.stamp.nanosec);
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
    pair_.stamp_ns = stamp_ns_of(*pair_.color);
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

FramePair FrameStore::wait_for_pair_after(
    int64_t after_ns, std::chrono::nanoseconds timeout) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait_for(lock, timeout, [this, after_ns] {
    return pair_.color != nullptr && pair_.depth != nullptr &&
           pair_.stamp_ns >= after_ns;
  });
  return pair_;
}

}  // namespace camera_server
