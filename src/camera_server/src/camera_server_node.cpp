#include "camera_server/camera_server_node.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include <builtin_interfaces/msg/time.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <tf2/time.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/transform_listener.h>

namespace camera_server {
namespace {

constexpr int64_t kNanosecondsPerSecond = 1000000000LL;

int64_t steady_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool valid_time(const builtin_interfaces::msg::Time& stamp) {
  return stamp.sec >= 0 &&
         stamp.nanosec < static_cast<uint32_t>(kNanosecondsPerSecond);
}

int64_t to_ns(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<int64_t>(stamp.sec) * kNanosecondsPerSecond +
         static_cast<int64_t>(stamp.nanosec);
}

builtin_interfaces::msg::Time from_ns(int64_t nanoseconds) {
  builtin_interfaces::msg::Time stamp;
  if (nanoseconds <= 0) {
    return stamp;
  }
  stamp.sec =
      static_cast<int32_t>(nanoseconds / kNanosecondsPerSecond);
  stamp.nanosec =
      static_cast<uint32_t>(nanoseconds % kNanosecondsPerSecond);
  return stamp;
}

bool finite_nonnegative(float value) {
  return std::isfinite(value) && value >= 0.0F;
}

bool finite_positive(double value) {
  return std::isfinite(value) && value > 0.0;
}

bool blank(const std::string& value) {
  return value.find_first_not_of(" \t\r\n") == std::string::npos;
}

bool compatible_frame_ids(const std::string& lhs, const std::string& rhs) {
  return !lhs.empty() && !rhs.empty() && lhs == rhs;
}

void append_diagnostic(std::string& message, const std::string& detail) {
  if (detail.empty()) {
    return;
  }
  if (!message.empty()) {
    message += "; ";
  }
  message += detail;
}

std::string format_age(int64_t now_ns, int64_t stamp_ns) {
  if (stamp_ns < 0) {
    return "never";
  }
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(3)
         << static_cast<double>(now_ns - stamp_ns) * 1e-9 << "s";
  return stream.str();
}

std::string response_frame(const FramePair& pair) {
  if (pair.depth && !pair.depth->header.frame_id.empty()) {
    return pair.depth->header.frame_id;
  }
  return pair.color ? pair.color->header.frame_id : std::string{};
}

}  // namespace

CameraServerNode::CameraServerNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("camera_server", options),
      status_last_time_(std::chrono::steady_clock::now()) {
  const std::string color_topic =
      declare_parameter<std::string>("color_topic", "/camera/color/image_raw");
  const std::string depth_topic =
      declare_parameter<std::string>("depth_topic", "/camera/depth/image_raw");
  const std::string color_info_topic = declare_parameter<std::string>(
      "color_info_topic", "/camera/color/camera_info");
  const std::string depth_info_topic = declare_parameter<std::string>(
      "depth_info_topic", "/camera/color/camera_info");
  const int64_t sync_queue_size =
      declare_parameter<int64_t>("sync_queue_size", 10);
  sync_slop_sec_ = declare_parameter<double>("sync_slop_sec", 0.1);
  const double tf_cache_sec =
      declare_parameter<double>("tf_cache_sec", 180.0);
  tf_lookup_timeout_sec_ =
      declare_parameter<double>("tf_lookup_timeout_sec", 0.1);
  transform_timeout_cap_sec_ =
      declare_parameter<double>("transform_timeout_cap_sec", 2.0);
  max_wait_sec_ = declare_parameter<double>("max_wait_sec", 2.0);
  status_period_sec_ =
      declare_parameter<double>("status_period_sec", 1.0);
  starvation_warn_sec_ =
      declare_parameter<double>("starvation_warn_sec", 2.0);
  const int64_t max_target_frames =
      declare_parameter<int64_t>("max_target_frames", 16);
  const int64_t executor_threads =
      declare_parameter<int64_t>("num_executor_threads", 4);

  if (blank(color_topic) || blank(depth_topic) ||
      blank(color_info_topic) || blank(depth_info_topic)) {
    throw std::invalid_argument("camera topic parameters must be non-empty");
  }
  if (sync_queue_size < 2 ||
      sync_queue_size > std::numeric_limits<int32_t>::max()) {
    throw std::invalid_argument("sync_queue_size must be in [2, INT32_MAX]");
  }
  if (!std::isfinite(sync_slop_sec_) || sync_slop_sec_ < 0.0) {
    throw std::invalid_argument("sync_slop_sec must be finite and nonnegative");
  }
  if (!finite_positive(tf_cache_sec) ||
      !finite_positive(tf_lookup_timeout_sec_) ||
      !finite_positive(transform_timeout_cap_sec_) ||
      !finite_positive(max_wait_sec_) ||
      !finite_positive(status_period_sec_) ||
      !finite_positive(starvation_warn_sec_)) {
    throw std::invalid_argument(
        "TF cache/lookup/cap, max wait, status period, and starvation "
        "parameters must be finite and positive");
  }
  if (max_target_frames <= 0 || max_target_frames > 256) {
    throw std::invalid_argument("max_target_frames must be in [1, 256]");
  }
  if (executor_threads < 2) {
    throw std::invalid_argument("num_executor_threads must be at least 2");
  }
  max_target_frames_ = static_cast<size_t>(max_target_frames);

  acquisition_group_ = create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive,
      /*automatically_add_to_executor_with_node=*/false);
  service_group_ =
      create_callback_group(rclcpp::CallbackGroupType::Reentrant);
  cloud_service_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(
      get_clock(), tf2::durationFromSec(tf_cache_sec));
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(
      *tf_buffer_, this, /*spin_thread=*/true);

  rclcpp::SubscriptionOptions acquisition_options;
  acquisition_options.callback_group = acquisition_group_;
  rmw_qos_profile_t image_qos = rmw_qos_profile_sensor_data;
  image_qos.depth = 5;
  color_sub_.subscribe(this, color_topic, image_qos, acquisition_options);
  depth_sub_.subscribe(this, depth_topic, image_qos, acquisition_options);
  color_sub_.registerCallback(
      std::bind(&CameraServerNode::on_color, this, std::placeholders::_1));
  depth_sub_.registerCallback(
      std::bind(&CameraServerNode::on_depth, this, std::placeholders::_1));

  SyncPolicy policy(static_cast<uint32_t>(sync_queue_size));
  policy.setMaxIntervalDuration(
      rclcpp::Duration::from_seconds(sync_slop_sec_));
  sync_ =
      std::make_unique<message_filters::Synchronizer<SyncPolicy>>(policy);
  sync_->connectInput(color_sub_, depth_sub_);
  sync_->registerCallback(std::bind(&CameraServerNode::on_synced, this,
                                    std::placeholders::_1,
                                    std::placeholders::_2));

  const rclcpp::QoS info_qos =
      rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
  color_info_sub_ = create_subscription<CameraInfo>(
      color_info_topic, info_qos,
      std::bind(&CameraServerNode::on_color_info, this,
                std::placeholders::_1),
      acquisition_options);
  depth_info_sub_ = create_subscription<CameraInfo>(
      depth_info_topic, info_qos,
      std::bind(&CameraServerNode::on_depth_info, this,
                std::placeholders::_1),
      acquisition_options);

  snapshot_service_ = create_service<GetCameraSnapshot>(
      "~/get_snapshot",
      [this](GetCameraSnapshot::Request::ConstSharedPtr request,
             GetCameraSnapshot::Response::SharedPtr response) {
        try {
          handle_snapshot(std::move(request), response);
        } catch (const std::exception& exception) {
          response->status = GetCameraSnapshot::Response::STATUS_BAD_REQUEST;
          response->error_msg =
              "snapshot request failed safely: " +
              std::string(exception.what());
        }
      },
      rmw_qos_profile_services_default, service_group_);
  point_cloud_service_ = create_service<GetCameraPointCloud>(
      "~/get_point_cloud",
      [this](GetCameraPointCloud::Request::ConstSharedPtr request,
             GetCameraPointCloud::Response::SharedPtr response) {
        try {
          handle_point_cloud(std::move(request), response);
        } catch (const std::exception& exception) {
          response->status =
              GetCameraPointCloud::Response::STATUS_BAD_REQUEST;
          response->error_msg =
              "point-cloud request failed safely: " +
              std::string(exception.what());
        }
      },
      rmw_qos_profile_services_default, cloud_service_group_);
  transform_service_ = create_service<GetTransform>(
      "~/get_transform",
      [this](GetTransform::Request::ConstSharedPtr request,
             GetTransform::Response::SharedPtr response) {
        try {
          handle_transform(std::move(request), response);
        } catch (const std::exception& exception) {
          response->status = GetTransform::Response::STATUS_UNAVAILABLE;
          response->error_msg =
              "transform request failed safely: " +
              std::string(exception.what());
        }
      },
      rmw_qos_profile_services_default, service_group_);

  status_publisher_ =
      create_publisher<CameraServerStatus>("~/status", rclcpp::QoS(10));
  status_timer_ = create_wall_timer(
      std::chrono::duration<double>(status_period_sec_),
      std::bind(&CameraServerNode::publish_status, this));

  acquisition_executor_.add_callback_group(
      acquisition_group_, get_node_base_interface());
  acquisition_thread_ =
      std::thread([this]() { acquisition_executor_.spin(); });

  RCLCPP_INFO(
      get_logger(),
      "camera server ready: color=%s depth=%s sync(queue=%ld slop=%.3fs), "
      "TF cache=%.1fs, service executor threads=%ld, max targets=%zu",
      color_topic.c_str(), depth_topic.c_str(), sync_queue_size,
      sync_slop_sec_, tf_cache_sec, executor_threads, max_target_frames_);
}

CameraServerNode::~CameraServerNode() {
  acquisition_executor_.cancel();
  if (acquisition_thread_.joinable()) {
    acquisition_thread_.join();
  }
}

void CameraServerNode::on_color(Image::ConstSharedPtr color) {
  if (!color) {
    return;
  }
  if (!valid_time(color->header.stamp)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                         "discarding color image with invalid ROS stamp");
    return;
  }
  last_color_stamp_ns_.store(to_ns(color->header.stamp));
  last_color_arrival_steady_ns_.store(steady_now_ns());
}

void CameraServerNode::on_depth(Image::ConstSharedPtr depth) {
  if (!depth) {
    return;
  }
  if (!valid_time(depth->header.stamp)) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
                         "discarding depth image with invalid ROS stamp");
    return;
  }
  last_depth_stamp_ns_.store(to_ns(depth->header.stamp));
  last_depth_arrival_steady_ns_.store(steady_now_ns());
}

void CameraServerNode::on_synced(Image::ConstSharedPtr color,
                                 Image::ConstSharedPtr depth) {
  if (!color || !depth) {
    return;
  }
  if (!valid_time(color->header.stamp) ||
      !valid_time(depth->header.stamp)) {
    return;
  }
  if (!compatible_frame_ids(color->header.frame_id,
                            depth->header.frame_id) ||
      (color->header.frame_id.empty() &&
       depth->header.frame_id.empty())) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "rejecting registered pair with incompatible/empty frames: "
        "color='%s' depth='%s'",
        color->header.frame_id.c_str(), depth->header.frame_id.c_str());
    return;
  }
  store_.set_pair(std::move(color), std::move(depth));
  last_pair_arrival_steady_ns_.store(steady_now_ns());
}

void CameraServerNode::on_color_info(CameraInfo::ConstSharedPtr info) {
  store_.set_color_info(std::move(info));
}

void CameraServerNode::on_depth_info(CameraInfo::ConstSharedPtr info) {
  store_.set_depth_info(std::move(info));
}

std::string CameraServerNode::stream_age_diagnostic() {
  const int64_t now_ns = get_clock()->now().nanoseconds();
  return "color_age=" +
         format_age(now_ns, last_color_stamp_ns_.load()) +
         ", depth_age=" +
         format_age(now_ns, last_depth_stamp_ns_.load());
}

void CameraServerNode::publish_status() {
  const auto steady_now = std::chrono::steady_clock::now();
  const double elapsed =
      std::chrono::duration<double>(steady_now - status_last_time_).count();
  const int64_t now_ns = get_clock()->now().nanoseconds();
  const FramePair pair = store_.latest_pair();
  const int64_t color_ns = last_color_stamp_ns_.load();
  const int64_t depth_ns = last_depth_stamp_ns_.load();

  CameraServerStatus status;
  status.color_age_sec =
      color_ns < 0
          ? -1.0F
          : static_cast<float>((now_ns - color_ns) * 1e-9);
  status.depth_age_sec =
      depth_ns < 0
          ? -1.0F
          : static_cast<float>((now_ns - depth_ns) * 1e-9);
  status.pair_age_sec =
      pair.color
          ? static_cast<float>((now_ns - pair.stamp_ns) * 1e-9)
          : -1.0F;
  status.last_pair_stamp = from_ns(pair.color ? pair.stamp_ns : 0);
  status.pair_seq = pair.seq;
  status.sync_fps =
      elapsed > 0.0
          ? static_cast<float>(
                static_cast<double>(pair.seq - status_last_seq_) / elapsed)
          : 0.0F;
  status_publisher_->publish(status);
  status_last_seq_ = pair.seq;
  status_last_time_ = steady_now;

  const int64_t steady_ns = steady_now_ns();
  const double color_arrival_age =
      last_color_arrival_steady_ns_.load() < 0
          ? std::numeric_limits<double>::infinity()
          : (steady_ns - last_color_arrival_steady_ns_.load()) * 1e-9;
  const double depth_arrival_age =
      last_depth_arrival_steady_ns_.load() < 0
          ? std::numeric_limits<double>::infinity()
          : (steady_ns - last_depth_arrival_steady_ns_.load()) * 1e-9;
  const double pair_arrival_age =
      last_pair_arrival_steady_ns_.load() < 0
          ? std::numeric_limits<double>::infinity()
          : (steady_ns - last_pair_arrival_steady_ns_.load()) * 1e-9;
  const bool any_input_alive =
      color_arrival_age < starvation_warn_sec_ ||
      depth_arrival_age < starvation_warn_sec_;
  if (any_input_alive &&
      pair_arrival_age > starvation_warn_sec_) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "sync starved: color_age=%.3fs depth_age=%.3fs "
        "pair_arrival_age=%.3fs; check missing partner, QoS, frame IDs, "
        "and stamps",
        status.color_age_sec, status.depth_age_sec, pair_arrival_age);
  }
  if (color_arrival_age < starvation_warn_sec_ &&
      depth_arrival_age < starvation_warn_sec_ && color_ns >= 0 &&
      depth_ns >= 0 &&
      std::abs(static_cast<double>(color_ns - depth_ns) * 1e-9) >
          sync_slop_sec_) {
    RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 10000,
        "camera sync partners skew by %.3fs (configured slop %.3fs)",
        std::abs(static_cast<double>(color_ns - depth_ns) * 1e-9),
        sync_slop_sec_);
  }
}

CameraServerNode::AcquisitionStatus CameraServerNode::acquire_pair(
    float max_age_sec,
    const builtin_interfaces::msg::Time& captured_after,
    float wait_timeout_sec, FramePair& pair, std::string& error_msg) {
  if (!finite_nonnegative(max_age_sec) ||
      !finite_nonnegative(wait_timeout_sec)) {
    error_msg =
        "max_age_sec and wait_timeout_sec must be finite and nonnegative";
    return AcquisitionStatus::kBadRequest;
  }
  if (!valid_time(captured_after)) {
    error_msg =
        "captured_after must have sec >= 0 and nanosec < 1000000000";
    return AcquisitionStatus::kBadRequest;
  }

  const int64_t after_ns = to_ns(captured_after);
  if (after_ns > 0) {
    const double requested_wait =
        wait_timeout_sec > 0.0F
            ? static_cast<double>(wait_timeout_sec)
            : max_wait_sec_;
    const double wait_sec = std::min(requested_wait, max_wait_sec_);
    pair = store_.wait_for_pair_after(
        after_ns,
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::duration<double>(wait_sec)));
    if (!pair.color || !pair.depth) {
      error_msg = "no synchronized camera pair; " +
                  stream_age_diagnostic();
      return AcquisitionStatus::kNoData;
    }
    if (pair.stamp_ns < after_ns) {
      std::ostringstream message;
      message << std::fixed << std::setprecision(6)
              << "wait timed out; newest pair's older image is "
              << static_cast<double>(after_ns - pair.stamp_ns) * 1e-9
              << "s before captured_after";
      error_msg = message.str();
      return AcquisitionStatus::kWaitTimeout;
    }
  } else {
    pair = store_.latest_pair();
    if (!pair.color || !pair.depth) {
      error_msg = "no synchronized camera pair; " +
                  stream_age_diagnostic();
      return AcquisitionStatus::kNoData;
    }
  }

  if (max_age_sec > 0.0F) {
    const double age =
        static_cast<double>(get_clock()->now().nanoseconds() -
                            pair.stamp_ns) *
        1e-9;
    if (age > static_cast<double>(max_age_sec)) {
      std::ostringstream message;
      message << std::fixed << std::setprecision(6)
              << "cached pair's older image is " << age
              << "s old (max_age=" << max_age_sec << "s)";
      error_msg = message.str();
      return AcquisitionStatus::kStale;
    }
  }
  return AcquisitionStatus::kOk;
}

void CameraServerNode::handle_snapshot(
    GetCameraSnapshot::Request::ConstSharedPtr request,
    GetCameraSnapshot::Response::SharedPtr response) {
  if (request->target_frames.size() > max_target_frames_) {
    response->status = GetCameraSnapshot::Response::STATUS_BAD_REQUEST;
    response->error_msg =
        "target_frames exceeds configured max_target_frames=" +
        std::to_string(max_target_frames_);
    return;
  }
  if (!request->want_color && !request->want_depth &&
      !request->want_camera_info && request->target_frames.empty()) {
    response->status = GetCameraSnapshot::Response::STATUS_BAD_REQUEST;
    response->error_msg =
        "snapshot request must ask for payload, camera info, or transforms";
    return;
  }
  for (const auto& target : request->target_frames) {
    if (blank(target)) {
      response->status =
          GetCameraSnapshot::Response::STATUS_BAD_REQUEST;
      response->error_msg =
          "target_frames must not contain empty frame strings";
      return;
    }
  }

  FramePair pair;
  const AcquisitionStatus acquisition =
      acquire_pair(request->max_age_sec, request->captured_after,
                   request->wait_timeout_sec, pair,
                   response->error_msg);
  switch (acquisition) {
    case AcquisitionStatus::kOk:
      response->status = GetCameraSnapshot::Response::STATUS_OK;
      break;
    case AcquisitionStatus::kNoData:
      response->status = GetCameraSnapshot::Response::STATUS_NO_DATA;
      break;
    case AcquisitionStatus::kStale:
      response->status = GetCameraSnapshot::Response::STATUS_STALE;
      break;
    case AcquisitionStatus::kWaitTimeout:
      response->status =
          GetCameraSnapshot::Response::STATUS_WAIT_TIMEOUT;
      break;
    case AcquisitionStatus::kBadRequest:
      response->status =
          GetCameraSnapshot::Response::STATUS_BAD_REQUEST;
      break;
  }
  if (acquisition == AcquisitionStatus::kNoData ||
      acquisition == AcquisitionStatus::kBadRequest) {
    return;
  }

  response->stamp = from_ns(pair.stamp_ns);
  response->frame_id = response_frame(pair);
  if (request->want_color) {
    response->color = *pair.color;
  }
  if (request->want_depth) {
    response->depth = *pair.depth;
  }
  if (request->want_camera_info) {
    const auto color_info = store_.color_info();
    const auto depth_info = store_.depth_info();
    if (color_info &&
        compatible_frame_ids(response->frame_id,
                             color_info->header.frame_id)) {
      response->color_info = *color_info;
    } else if (color_info) {
      append_diagnostic(
          response->error_msg,
          "color camera_info frame is incompatible with aligned pair");
    } else {
      append_diagnostic(response->error_msg,
                        "color camera_info not received yet");
    }
    if (depth_info &&
        compatible_frame_ids(response->frame_id,
                             depth_info->header.frame_id)) {
      response->depth_info = *depth_info;
    } else if (depth_info) {
      append_diagnostic(
          response->error_msg,
          "depth camera_info frame is incompatible with aligned pair");
    } else {
      append_diagnostic(response->error_msg,
                        "depth camera_info not received yet");
    }
  }

  response->transforms.resize(request->target_frames.size());
  response->transforms_ok.assign(request->target_frames.size(), false);
  for (size_t index = 0; index < request->target_frames.size(); ++index) {
    const std::string& target = request->target_frames[index];
    try {
      response->transforms[index] = tf_buffer_->lookupTransform(
          target, response->frame_id,
          rclcpp::Time(response->stamp, RCL_ROS_TIME),
          rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_));
      response->transforms_ok[index] = true;
    } catch (const tf2::TransformException& exception) {
      append_diagnostic(
          response->error_msg,
          "TF " + target + "<-" + response->frame_id + ": " +
              exception.what());
    } catch (const std::exception& exception) {
      append_diagnostic(
          response->error_msg,
          "TF " + target + "<-" + response->frame_id +
              " failed safely: " + exception.what());
    }
  }
}

void CameraServerNode::handle_point_cloud(
    GetCameraPointCloud::Request::ConstSharedPtr request,
    GetCameraPointCloud::Response::SharedPtr response) {
  response->points = sensor_msgs::msg::PointCloud2{};
  FramePair pair;
  const AcquisitionStatus acquisition =
      acquire_pair(request->max_age_sec, request->captured_after,
                   request->wait_timeout_sec, pair,
                   response->error_msg);
  switch (acquisition) {
    case AcquisitionStatus::kOk:
      response->status = GetCameraPointCloud::Response::STATUS_OK;
      break;
    case AcquisitionStatus::kNoData:
      response->status = GetCameraPointCloud::Response::STATUS_NO_DATA;
      return;
    case AcquisitionStatus::kStale:
      response->status = GetCameraPointCloud::Response::STATUS_STALE;
      return;
    case AcquisitionStatus::kWaitTimeout:
      response->status = GetCameraPointCloud::Response::STATUS_WAIT_TIMEOUT;
      return;
    case AcquisitionStatus::kBadRequest:
      response->status = GetCameraPointCloud::Response::STATUS_BAD_REQUEST;
      return;
  }

  if (!pair.depth) {
    response->status = GetCameraPointCloud::Response::STATUS_NO_DATA;
    response->error_msg = "no depth frame in synchronized pair";
    return;
  }
  const auto depth_info = store_.depth_info();
  if (!depth_info) {
    response->status = GetCameraPointCloud::Response::STATUS_NO_DATA;
    response->error_msg = "depth camera_info not received yet";
    return;
  }
  if (request->include_color && !pair.color) {
    response->status = GetCameraPointCloud::Response::STATUS_NO_DATA;
    response->error_msg = "color requested but no color frame is available";
    return;
  }
  if (!request->target_frame.empty() && blank(request->target_frame)) {
    response->status = GetCameraPointCloud::Response::STATUS_BAD_REQUEST;
    response->error_msg = "target_frame must not be blank";
    return;
  }

  const std::string native_frame = pair.depth->header.frame_id;
  if (native_frame.empty()) {
    response->status = GetCameraPointCloud::Response::STATUS_NO_DATA;
    response->error_msg = "depth frame_id is empty";
    return;
  }
  const auto cloud_stamp = pair.depth->header.stamp;
  std::optional<Eigen::Isometry3f> transform;
  if (!request->target_frame.empty() && request->target_frame != native_frame) {
    if (to_ns(cloud_stamp) == 0) {
      response->status = GetCameraPointCloud::Response::STATUS_TF_FAIL;
      response->error_msg =
          "cannot perform time-correct target transform for zero depth stamp";
      return;
    }
    try {
      const auto tf_msg = tf_buffer_->lookupTransform(
          request->target_frame, native_frame,
          rclcpp::Time(cloud_stamp, RCL_ROS_TIME),
          rclcpp::Duration::from_seconds(tf_lookup_timeout_sec_));
      const Eigen::Isometry3f candidate =
          tf2::transformToEigen(tf_msg).cast<float>();
      if (!candidate.matrix().allFinite()) {
        throw std::runtime_error("TF transform contains non-finite values");
      }
      transform = candidate;
    } catch (const tf2::TransformException& exception) {
      response->status = GetCameraPointCloud::Response::STATUS_TF_FAIL;
      response->error_msg =
          "TF " + request->target_frame + "<-" + native_frame +
          " at depth stamp: " + exception.what();
      return;
    } catch (const std::exception& exception) {
      response->status = GetCameraPointCloud::Response::STATUS_TF_FAIL;
      response->error_msg =
          "TF conversion failed at depth stamp: " +
          std::string(exception.what());
      return;
    }
  }

  const sensor_msgs::msg::Image* color =
      request->include_color ? pair.color.get() : nullptr;
  std::string deprojection_error;
  {
    std::lock_guard<std::mutex> lock(deproject_mutex_);
    if (!deprojector_.deproject(*pair.depth, *depth_info, color,
                                request->stride, transform,
                                response->points, deprojection_error)) {
      response->status = GetCameraPointCloud::Response::STATUS_NO_DATA;
      response->error_msg = "camera data cannot be deprojected: " +
                            deprojection_error;
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 10000,
          "get_point_cloud rejected camera data: %s",
          deprojection_error.c_str());
      return;
    }
  }
  response->stamp = cloud_stamp;
  response->points.header.stamp = cloud_stamp;
  response->points.header.frame_id =
      request->target_frame.empty() ? native_frame : request->target_frame;
  response->status = GetCameraPointCloud::Response::STATUS_OK;
}

void CameraServerNode::handle_transform(
    GetTransform::Request::ConstSharedPtr request,
    GetTransform::Response::SharedPtr response) {
  if (blank(request->target_frame) || blank(request->source_frame)) {
    response->status = GetTransform::Response::STATUS_BAD_REQUEST;
    response->error_msg =
        "target_frame and source_frame must be non-empty";
    return;
  }
  if (!valid_time(request->lookup_time)) {
    response->status = GetTransform::Response::STATUS_BAD_REQUEST;
    response->error_msg =
        "lookup_time must have sec >= 0 and nanosec < 1000000000";
    return;
  }
  if (!finite_nonnegative(request->timeout_sec)) {
    response->status = GetTransform::Response::STATUS_BAD_REQUEST;
    response->error_msg =
        "timeout_sec must be finite and nonnegative";
    return;
  }
  const double timeout =
      request->timeout_sec > 0.0F
          ? std::min(static_cast<double>(request->timeout_sec),
                     transform_timeout_cap_sec_)
          : tf_lookup_timeout_sec_;
  try {
    if (to_ns(request->lookup_time) == 0) {
      response->transform = tf_buffer_->lookupTransform(
          request->target_frame, request->source_frame,
          tf2::TimePointZero, tf2::durationFromSec(timeout));
    } else {
      response->transform = tf_buffer_->lookupTransform(
          request->target_frame, request->source_frame,
          rclcpp::Time(request->lookup_time, RCL_ROS_TIME),
          rclcpp::Duration::from_seconds(timeout));
    }
    response->status = GetTransform::Response::STATUS_OK;
  } catch (const tf2::TransformException& exception) {
    response->status = GetTransform::Response::STATUS_UNAVAILABLE;
    response->error_msg = exception.what();
  } catch (const std::exception& exception) {
    response->status = GetTransform::Response::STATUS_UNAVAILABLE;
    response->error_msg =
        "transform lookup failed safely: " +
        std::string(exception.what());
  }
}

}  // namespace camera_server

RCLCPP_COMPONENTS_REGISTER_NODE(camera_server::CameraServerNode)
