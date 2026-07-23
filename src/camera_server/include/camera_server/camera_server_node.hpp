#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <builtin_interfaces/msg/time.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <rclcpp/executors/single_threaded_executor.hpp>
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
  ~CameraServerNode() override;

 private:
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using GetCameraSnapshot = tinker_vision_msgs_26::srv::GetCameraSnapshot;
  using GetCameraPointCloud = tinker_vision_msgs_26::srv::GetCameraPointCloud;
  using GetTransform = tinker_vision_msgs_26::srv::GetTransform;
  using CameraServerStatus = tinker_vision_msgs_26::msg::CameraServerStatus;
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<Image, Image>;

  enum class AcquisitionStatus {
    kOk,
    kNoData,
    kStale,
    kWaitTimeout,
    kBadRequest,
  };

  void on_color(Image::ConstSharedPtr color);
  void on_depth(Image::ConstSharedPtr depth);
  void on_synced(Image::ConstSharedPtr color, Image::ConstSharedPtr depth);
  void on_color_info(CameraInfo::ConstSharedPtr info);
  void on_depth_info(CameraInfo::ConstSharedPtr info);
  void publish_status();

  AcquisitionStatus acquire_pair(
      float max_age_sec,
      const builtin_interfaces::msg::Time& captured_after,
      float wait_timeout_sec, FramePair& pair, std::string& error_msg);
  std::string stream_age_diagnostic();

  void handle_snapshot(GetCameraSnapshot::Request::ConstSharedPtr request,
                       GetCameraSnapshot::Response::SharedPtr response);
  void handle_point_cloud(
      GetCameraPointCloud::Request::ConstSharedPtr request,
      GetCameraPointCloud::Response::SharedPtr response);
  void handle_transform(GetTransform::Request::ConstSharedPtr request,
                        GetTransform::Response::SharedPtr response);

  FrameStore store_;
  Deprojector deprojector_;
  std::mutex deproject_mutex_;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  message_filters::Subscriber<Image> color_sub_;
  message_filters::Subscriber<Image> depth_sub_;
  std::unique_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Subscription<CameraInfo>::SharedPtr color_info_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr depth_info_sub_;

  rclcpp::Service<GetCameraSnapshot>::SharedPtr snapshot_service_;
  rclcpp::Service<GetCameraPointCloud>::SharedPtr point_cloud_service_;
  rclcpp::Service<GetTransform>::SharedPtr transform_service_;
  rclcpp::Publisher<CameraServerStatus>::SharedPtr status_publisher_;
  rclcpp::TimerBase::SharedPtr status_timer_;

  // auto_add=false is essential: this group is spun only by the node-owned
  // executor, so blocked services cannot starve camera ingestion in a
  // standalone executable or a component container.
  rclcpp::CallbackGroup::SharedPtr acquisition_group_;
  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::CallbackGroup::SharedPtr cloud_service_group_;
  rclcpp::executors::SingleThreadedExecutor acquisition_executor_;
  std::thread acquisition_thread_;

  std::atomic<int64_t> last_color_stamp_ns_{-1};
  std::atomic<int64_t> last_depth_stamp_ns_{-1};
  std::atomic<int64_t> last_color_arrival_steady_ns_{-1};
  std::atomic<int64_t> last_depth_arrival_steady_ns_{-1};

  uint64_t status_last_seq_ = 0;
  std::chrono::steady_clock::time_point status_last_time_;
  double sync_slop_sec_ = 0.1;
  double tf_lookup_timeout_sec_ = 0.1;
  double transform_timeout_cap_sec_ = 2.0;
  double max_wait_sec_ = 2.0;
  double status_period_sec_ = 1.0;
  double starvation_warn_sec_ = 2.0;
  size_t max_target_frames_ = 16;
};

}  // namespace camera_server
