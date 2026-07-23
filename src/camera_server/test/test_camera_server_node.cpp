#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <stdexcept>
#include <thread>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/static_transform_broadcaster.h>

#include "camera_server/camera_server_node.hpp"
#include "tinker_vision_msgs_26/msg/camera_server_status.hpp"
#include "tinker_vision_msgs_26/srv/get_camera_point_cloud.hpp"
#include "tinker_vision_msgs_26/srv/get_camera_snapshot.hpp"
#include "tinker_vision_msgs_26/srv/get_transform.hpp"

namespace {

using namespace std::chrono_literals;
using Snapshot = tinker_vision_msgs_26::srv::GetCameraSnapshot;
using PointCloud = tinker_vision_msgs_26::srv::GetCameraPointCloud;
using Transform = tinker_vision_msgs_26::srv::GetTransform;
using Status = tinker_vision_msgs_26::msg::CameraServerStatus;
using Image = sensor_msgs::msg::Image;
using CameraInfo = sensor_msgs::msg::CameraInfo;

float read_f32_le(const std::vector<uint8_t>& data, size_t offset) {
  uint32_t bits = static_cast<uint32_t>(data[offset]) |
                  (static_cast<uint32_t>(data[offset + 1]) << 8U) |
                  (static_cast<uint32_t>(data[offset + 2]) << 16U) |
                  (static_cast<uint32_t>(data[offset + 3]) << 24U);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

bool wait_until(const std::function<bool()>& predicate,
                std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(10ms);
  }
  return predicate();
}

template <typename Service>
typename Service::Response::SharedPtr call(
    const typename rclcpp::Client<Service>::SharedPtr& client,
    const typename Service::Request::SharedPtr& request,
    std::chrono::milliseconds timeout = 3s) {
  auto future = client->async_send_request(request);
  if (future.wait_for(timeout) != std::future_status::ready) {
    return nullptr;
  }
  return future.get();
}

class CameraServerNodeTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    setenv("ROS_LOG_DIR", "/tmp/camera_server_test_logs", 1);
    if (!rclcpp::ok()) {
      int argc = 0;
      rclcpp::init(argc, nullptr);
    }
  }

  static void TearDownTestSuite() {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }

  void SetUp() override {
    const int id = next_id_.fetch_add(1);
    prefix_ = "/camera_server_test_" + std::to_string(id);
    rclcpp::NodeOptions options;
    options.append_parameter_override("color_topic", prefix_ + "/color");
    options.append_parameter_override("depth_topic", prefix_ + "/depth");
    options.append_parameter_override("color_info_topic",
                                      prefix_ + "/color_info");
    options.append_parameter_override("depth_info_topic",
                                      prefix_ + "/depth_info");
    options.append_parameter_override("status_period_sec", 0.2);
    options.append_parameter_override("tf_lookup_timeout_sec", 0.05);
    options.append_parameter_override("max_wait_sec", 1.5);
    options.append_parameter_override("num_executor_threads", 2);
    server_ = std::make_shared<camera_server::CameraServerNode>(options);
    helper_ = std::make_shared<rclcpp::Node>(
        "camera_server_test_helper_" + std::to_string(id));

    color_publisher_ = helper_->create_publisher<Image>(
        prefix_ + "/color", rclcpp::SensorDataQoS());
    depth_publisher_ = helper_->create_publisher<Image>(
        prefix_ + "/depth", rclcpp::SensorDataQoS());
    const auto info_qos =
        rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();
    color_info_publisher_ =
        helper_->create_publisher<CameraInfo>(prefix_ + "/color_info",
                                              info_qos);
    depth_info_publisher_ =
        helper_->create_publisher<CameraInfo>(prefix_ + "/depth_info",
                                              info_qos);

    snapshot_client_ =
        helper_->create_client<Snapshot>("/camera_server/get_snapshot");
    cloud_client_ =
        helper_->create_client<PointCloud>("/camera_server/get_point_cloud");
    transform_client_ =
        helper_->create_client<Transform>("/camera_server/get_transform");
    status_subscription_ = helper_->create_subscription<Status>(
        "/camera_server/status", 10,
        [this](Status::ConstSharedPtr status) {
          std::lock_guard<std::mutex> lock(status_mutex_);
          last_status_ = *status;
          ++status_count_;
        });

    executor_ =
        std::make_unique<rclcpp::executors::MultiThreadedExecutor>(
            rclcpp::ExecutorOptions(), 2);
    executor_->add_node(server_);
    executor_->add_node(helper_);
    executor_thread_ = std::thread([this]() { executor_->spin(); });

    ASSERT_TRUE(snapshot_client_->wait_for_service(2s));
    ASSERT_TRUE(cloud_client_->wait_for_service(2s));
    ASSERT_TRUE(transform_client_->wait_for_service(2s));
    ASSERT_TRUE(wait_until([this]() {
      return color_publisher_->get_subscription_count() > 0 &&
             depth_publisher_->get_subscription_count() > 0;
    }));
  }

  void TearDown() override {
    executor_->cancel();
    if (executor_thread_.joinable()) {
      executor_thread_.join();
    }
    executor_->remove_node(helper_);
    executor_->remove_node(server_);
    helper_.reset();
    server_.reset();
    executor_.reset();
  }

  builtin_interfaces::msg::Time stamp_after(
      std::chrono::nanoseconds offset = 0ns) const {
    return helper_->now() + rclcpp::Duration(offset);
  }

  void publish_pair(const builtin_interfaces::msg::Time& color_stamp,
                    const builtin_interfaces::msg::Time& depth_stamp,
                    const std::string& frame = "camera_optical") {
    Image color;
    color.header.stamp = color_stamp;
    color.header.frame_id = frame;
    color.height = 1;
    color.width = 1;
    color.encoding = "rgb8";
    color.step = 3;
    color.data = {1, 2, 3};
    Image depth;
    depth.header.stamp = depth_stamp;
    depth.header.frame_id = frame;
    depth.height = 1;
    depth.width = 1;
    depth.encoding = "16UC1";
    depth.step = 2;
    depth.data = {0xe8, 0x03};
    color_publisher_->publish(color);
    depth_publisher_->publish(depth);
  }

  void publish_info(const std::string& frame = "camera_optical") {
    CameraInfo info;
    info.header.frame_id = frame;
    info.width = 1;
    info.height = 1;
    info.k = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    color_info_publisher_->publish(info);
    depth_info_publisher_->publish(info);
  }

  static int64_t nanoseconds(
      const builtin_interfaces::msg::Time& stamp) {
    return static_cast<int64_t>(stamp.sec) * 1000000000LL +
           static_cast<int64_t>(stamp.nanosec);
  }

  static std::atomic<int> next_id_;
  std::string prefix_;
  std::shared_ptr<camera_server::CameraServerNode> server_;
  rclcpp::Node::SharedPtr helper_;
  std::unique_ptr<rclcpp::executors::MultiThreadedExecutor> executor_;
  std::thread executor_thread_;
  rclcpp::Publisher<Image>::SharedPtr color_publisher_;
  rclcpp::Publisher<Image>::SharedPtr depth_publisher_;
  rclcpp::Publisher<CameraInfo>::SharedPtr color_info_publisher_;
  rclcpp::Publisher<CameraInfo>::SharedPtr depth_info_publisher_;
  rclcpp::Client<Snapshot>::SharedPtr snapshot_client_;
  rclcpp::Client<PointCloud>::SharedPtr cloud_client_;
  rclcpp::Client<Transform>::SharedPtr transform_client_;
  rclcpp::Subscription<Status>::SharedPtr status_subscription_;
  std::mutex status_mutex_;
  Status last_status_;
  size_t status_count_ = 0;
};

std::atomic<int> CameraServerNodeTest::next_id_{0};

TEST_F(CameraServerNodeTest, NoDataMalformedRequestsAndStubSurvive) {
  auto request = std::make_shared<Snapshot::Request>();
  request->want_color = true;
  request->want_depth = true;
  request->want_camera_info = true;
  const auto no_data = call<Snapshot>(snapshot_client_, request);
  ASSERT_NE(no_data, nullptr);
  EXPECT_EQ(no_data->status, Snapshot::Response::STATUS_NO_DATA);
  EXPECT_NE(no_data->error_msg.find("color_age=never"), std::string::npos);

  request->max_age_sec = std::numeric_limits<float>::quiet_NaN();
  const auto malformed = call<Snapshot>(snapshot_client_, request);
  ASSERT_NE(malformed, nullptr);
  EXPECT_EQ(malformed->status, Snapshot::Response::STATUS_BAD_REQUEST);

  request->max_age_sec = 0.0F;
  request->captured_after.nanosec = 1000000000U;
  const auto invalid_time = call<Snapshot>(snapshot_client_, request);
  ASSERT_NE(invalid_time, nullptr);
  EXPECT_EQ(invalid_time->status, Snapshot::Response::STATUS_BAD_REQUEST);

  request->captured_after.nanosec = 0U;
  request->target_frames.assign(17, "base");
  const auto too_many_targets = call<Snapshot>(snapshot_client_, request);
  ASSERT_NE(too_many_targets, nullptr);
  EXPECT_EQ(too_many_targets->status,
            Snapshot::Response::STATUS_BAD_REQUEST);

  request->target_frames = {""};
  const auto blank_target = call<Snapshot>(snapshot_client_, request);
  ASSERT_NE(blank_target, nullptr);
  EXPECT_EQ(blank_target->status,
            Snapshot::Response::STATUS_BAD_REQUEST);

  request = std::make_shared<Snapshot::Request>();
  request->want_color = false;
  request->want_depth = false;
  request->want_camera_info = false;
  const auto empty = call<Snapshot>(snapshot_client_, request);
  ASSERT_NE(empty, nullptr);
  EXPECT_EQ(empty->status, Snapshot::Response::STATUS_BAD_REQUEST);

  auto cloud_request = std::make_shared<PointCloud::Request>();
  const auto cloud = call<PointCloud>(cloud_client_, cloud_request);
  ASSERT_NE(cloud, nullptr);
  EXPECT_EQ(cloud->status, PointCloud::Response::STATUS_NO_DATA);

  request = std::make_shared<Snapshot::Request>();
  request->want_color = true;
  const auto still_alive = call<Snapshot>(snapshot_client_, request);
  ASSERT_NE(still_alive, nullptr);
  EXPECT_EQ(still_alive->status, Snapshot::Response::STATUS_NO_DATA);
}

TEST_F(CameraServerNodeTest,
       SnapshotFlagsMissingInfoStatusAndPartialTfStayAligned) {
  tf2_ros::StaticTransformBroadcaster broadcaster(helper_);
  geometry_msgs::msg::TransformStamped transform;
  transform.header.stamp = helper_->now();
  transform.header.frame_id = "base";
  transform.child_frame_id = "camera_optical";
  transform.transform.rotation.w = 1.0;
  transform.transform.translation.x = 0.25;
  broadcaster.sendTransform(transform);

  const auto older = stamp_after(10ms);
  publish_pair(older, older);

  auto request = std::make_shared<Snapshot::Request>();
  request->want_color = false;
  request->want_depth = true;
  request->want_camera_info = true;
  request->target_frames = {"base", "missing"};

  Snapshot::Response::SharedPtr response;
  ASSERT_TRUE(wait_until([&]() {
    response = call<Snapshot>(snapshot_client_, request, 500ms);
    return response &&
           response->status == Snapshot::Response::STATUS_OK;
  }));
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(nanoseconds(response->stamp), nanoseconds(older));
  EXPECT_EQ(response->frame_id, "camera_optical");
  EXPECT_EQ(response->color.width, 0U);
  EXPECT_EQ(response->depth.width, 1U);
  EXPECT_EQ(response->color_info.width, 0U);
  EXPECT_EQ(response->depth_info.width, 0U);
  EXPECT_NE(response->error_msg.find("not received"), std::string::npos);
  ASSERT_EQ(response->transforms.size(), 2U);
  ASSERT_EQ(response->transforms_ok.size(), 2U);
  EXPECT_TRUE(response->transforms_ok[0]);
  EXPECT_FALSE(response->transforms_ok[1]);
  EXPECT_DOUBLE_EQ(response->transforms[0].transform.translation.x, 0.25);

  ASSERT_TRUE(wait_until([this]() {
    std::lock_guard<std::mutex> lock(status_mutex_);
    return status_count_ > 0 && last_status_.pair_seq > 0;
  }));
  std::lock_guard<std::mutex> lock(status_mutex_);
  EXPECT_EQ(nanoseconds(last_status_.last_pair_stamp),
            nanoseconds(older));
  EXPECT_GT(last_status_.sync_fps, 0.0F);
  EXPECT_TRUE(std::isfinite(last_status_.pair_age_sec));
}

TEST_F(CameraServerNodeTest,
       PointCloudUsesDepthStampAndSupportsStrideColorAndTargetTf) {
  publish_info();
  const auto depth_stamp = stamp_after(10ms);
  publish_pair(depth_stamp, depth_stamp);

  auto request = std::make_shared<PointCloud::Request>();
  request->include_color = false;
  PointCloud::Response::SharedPtr native;
  ASSERT_TRUE(wait_until([&]() {
    native = call<PointCloud>(cloud_client_, request, 500ms);
    return native && native->status == PointCloud::Response::STATUS_OK;
  })) << (native ? native->error_msg : "no response");
  ASSERT_NE(native, nullptr);
  EXPECT_EQ(nanoseconds(native->stamp), nanoseconds(depth_stamp));
  EXPECT_EQ(nanoseconds(native->points.header.stamp),
            nanoseconds(depth_stamp));
  EXPECT_EQ(native->points.header.frame_id, "camera_optical");
  EXPECT_EQ(native->points.width, 1U);
  EXPECT_EQ(native->points.point_step, 12U);
  ASSERT_EQ(native->points.fields.size(), 3U);

  request->include_color = true;
  request->stride = 1;
  PointCloud::Response::SharedPtr colored;
  ASSERT_TRUE(wait_until([&]() {
    colored = call<PointCloud>(cloud_client_, request, 500ms);
    return colored && colored->status == PointCloud::Response::STATUS_OK;
  }));
  ASSERT_NE(colored, nullptr);
  EXPECT_EQ(colored->points.point_step, 16U);
  ASSERT_EQ(colored->points.fields.size(), 4U);
  EXPECT_EQ(colored->points.fields.back().name, "rgb");

  tf2_ros::StaticTransformBroadcaster broadcaster(helper_);
  geometry_msgs::msg::TransformStamped tf;
  tf.header.stamp = helper_->now();
  tf.header.frame_id = "base";
  tf.child_frame_id = "camera_optical";
  tf.transform.rotation.w = 1.0;
  tf.transform.translation.x = 0.5;
  broadcaster.sendTransform(tf);

  request->include_color = false;
  request->target_frame = "base";
  PointCloud::Response::SharedPtr transformed;
  ASSERT_TRUE(wait_until([&]() {
    transformed = call<PointCloud>(cloud_client_, request, 500ms);
    return transformed &&
           transformed->status == PointCloud::Response::STATUS_OK;
  }));
  ASSERT_NE(transformed, nullptr);
  EXPECT_EQ(transformed->points.header.frame_id, "base");
  EXPECT_NEAR(read_f32_le(transformed->points.data, 0), 0.5F, 1e-5F);
  EXPECT_EQ(nanoseconds(transformed->stamp), nanoseconds(depth_stamp));
}

TEST_F(CameraServerNodeTest, CloudFreshnessUsesOlderImageStamp) {
  publish_info();
  const auto boundary = stamp_after(100ms);
  const auto old_stamp = stamp_after(50ms);
  publish_pair(old_stamp, old_stamp);

  auto request = std::make_shared<PointCloud::Request>();
  request->captured_after = boundary;
  request->wait_timeout_sec = 0.05F;
  const auto response = call<PointCloud>(cloud_client_, request);
  ASSERT_NE(response, nullptr);
  EXPECT_EQ(response->status,
            PointCloud::Response::STATUS_WAIT_TIMEOUT) << response->error_msg;
  EXPECT_TRUE(response->points.data.empty());
}

TEST_F(CameraServerNodeTest, StaticTransformAndFreshnessFailuresAreStable) {
  tf2_ros::StaticTransformBroadcaster broadcaster(helper_);
  geometry_msgs::msg::TransformStamped sent;
  sent.header.stamp = helper_->now();
  sent.header.frame_id = "map";
  sent.child_frame_id = "sensor";
  sent.transform.rotation.w = 1.0;
  sent.transform.translation.z = 1.5;
  broadcaster.sendTransform(sent);

  auto transform_request = std::make_shared<Transform::Request>();
  transform_request->target_frame = "map";
  transform_request->source_frame = "sensor";
  Transform::Response::SharedPtr transform_response;
  ASSERT_TRUE(wait_until([&]() {
    transform_response =
        call<Transform>(transform_client_, transform_request, 500ms);
    return transform_response &&
           transform_response->status ==
               Transform::Response::STATUS_OK;
  }));
  EXPECT_DOUBLE_EQ(transform_response->transform.transform.translation.z,
                   1.5);

  transform_request->timeout_sec =
      std::numeric_limits<float>::infinity();
  transform_response =
      call<Transform>(transform_client_, transform_request);
  ASSERT_NE(transform_response, nullptr);
  EXPECT_EQ(transform_response->status,
            Transform::Response::STATUS_BAD_REQUEST);
  transform_request->timeout_sec = 0.0F;
  transform_request->lookup_time.sec = -1;
  transform_response =
      call<Transform>(transform_client_, transform_request);
  ASSERT_NE(transform_response, nullptr);
  EXPECT_EQ(transform_response->status,
            Transform::Response::STATUS_BAD_REQUEST);

  const builtin_interfaces::msg::Time old =
      helper_->now() - rclcpp::Duration(1s);
  publish_pair(old, old);
  auto snapshot_request = std::make_shared<Snapshot::Request>();
  snapshot_request->want_depth = true;
  snapshot_request->max_age_sec = 0.01F;
  Snapshot::Response::SharedPtr stale;
  ASSERT_TRUE(wait_until([&]() {
    stale = call<Snapshot>(snapshot_client_, snapshot_request, 500ms);
    return stale &&
           stale->status == Snapshot::Response::STATUS_STALE;
  }));
  EXPECT_EQ(stale->depth.width, 1U);

  snapshot_request->max_age_sec = 0.0F;
  snapshot_request->captured_after =
      stamp_after(std::chrono::seconds(1));
  snapshot_request->wait_timeout_sec = 0.05F;
  const auto timeout =
      call<Snapshot>(snapshot_client_, snapshot_request);
  ASSERT_NE(timeout, nullptr);
  EXPECT_EQ(timeout->status,
            Snapshot::Response::STATUS_WAIT_TIMEOUT);
  EXPECT_EQ(timeout->depth.width, 1U);
}

TEST_F(CameraServerNodeTest,
       ConcurrentWaitsCannotStarveDedicatedIngestionExecutor) {
  auto initial = stamp_after();
  publish_pair(initial, initial);
  auto probe = std::make_shared<Snapshot::Request>();
  probe->want_depth = true;
  ASSERT_TRUE(wait_until([&]() {
    const auto result =
        call<Snapshot>(snapshot_client_, probe, 500ms);
    return result && result->status == Snapshot::Response::STATUS_OK;
  }));

  const auto boundary = stamp_after(200ms);
  std::vector<rclcpp::Client<Snapshot>::SharedFuture> futures;
  for (size_t index = 0; index < 3; ++index) {
    auto request = std::make_shared<Snapshot::Request>();
    request->want_depth = true;
    request->captured_after = boundary;
    request->wait_timeout_sec = 1.2F;
    auto pending = snapshot_client_->async_send_request(request);
    futures.push_back(pending.future.share());
  }
  std::this_thread::sleep_for(100ms);
  const auto fresh =
      rclcpp::Time(boundary, RCL_ROS_TIME) + rclcpp::Duration(1ns);
  const builtin_interfaces::msg::Time fresh_stamp = fresh;
  publish_pair(fresh_stamp, fresh_stamp);

  for (auto& future : futures) {
    ASSERT_EQ(future.wait_for(2s), std::future_status::ready);
    const auto response = future.get();
    ASSERT_NE(response, nullptr);
    EXPECT_EQ(response->status, Snapshot::Response::STATUS_OK);
    EXPECT_GE(nanoseconds(response->stamp), nanoseconds(boundary));
  }
}

TEST_F(CameraServerNodeTest, ConstructorRejectsUnsafeRuntimeParameters) {
  rclcpp::NodeOptions options;
  options.append_parameter_override("sync_queue_size", 1);
  EXPECT_THROW(
      std::make_shared<camera_server::CameraServerNode>(options),
      std::invalid_argument);

  options = rclcpp::NodeOptions();
  options.append_parameter_override("num_executor_threads", 1);
  EXPECT_THROW(
      std::make_shared<camera_server::CameraServerNode>(options),
      std::invalid_argument);

  options = rclcpp::NodeOptions();
  options.append_parameter_override(
      "status_period_sec", std::numeric_limits<double>::infinity());
  EXPECT_THROW(
      std::make_shared<camera_server::CameraServerNode>(options),
      std::invalid_argument);
}

}  // namespace
