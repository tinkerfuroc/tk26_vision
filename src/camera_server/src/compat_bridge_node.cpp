// Legacy-name compatibility bridge. It owns no camera subscriptions; all
// payloads are forwarded to the per-camera C++ servers.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <rclcpp/executors/single_threaded_executor.hpp>
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
  explicit CompatBridgeNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
      : rclcpp::Node("camera_compat_bridge", options) {
    const std::string wrist =
        declare_parameter<std::string>("wrist_server", "/wrist_camera_server");
    const std::string head =
        declare_parameter<std::string>("head_server", "/head_camera_server");
    const double timeout_sec =
        declare_parameter<double>("forward_timeout_sec", 5.0);
    if (!std::isfinite(timeout_sec) || timeout_sec <= 0.0 ||
        timeout_sec > 60.0) {
      throw std::invalid_argument(
          "forward_timeout_sec must be finite and in (0, 60]");
    }
    forward_timeout_ = std::chrono::duration<double>(timeout_sec);

    client_group_ = create_callback_group(
        rclcpp::CallbackGroupType::MutuallyExclusive,
        /*automatically_add_to_executor_with_node=*/false);
    service_group_ =
        create_callback_group(rclcpp::CallbackGroupType::Reentrant);

    for (const auto& entry :
         std::map<std::string, std::string>{{"realsense", wrist},
                                            {"orbbec", head}}) {
      snapshot_clients_[entry.first] = create_client<GetCameraSnapshot>(
          entry.second + "/get_snapshot", rmw_qos_profile_services_default,
          client_group_);
      cloud_clients_[entry.first] = create_client<GetCameraPointCloud>(
          entry.second + "/get_point_cloud", rmw_qos_profile_services_default,
          client_group_);
    }

    image_service_ = create_service<GetImage>(
        "get_image_service",
        std::bind(&CompatBridgeNode::handle_image, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, service_group_);
    cloud_service_ = create_service<GetPointCloud>(
        "get_point_cloud_service",
        std::bind(&CompatBridgeNode::handle_cloud, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, service_group_);
    orbbec_pc_service_ = create_service<GetOrbbecPC>(
        "get_orbbec_pc",
        std::bind(&CompatBridgeNode::handle_orbbec_pc, this,
                  std::placeholders::_1, std::placeholders::_2),
        rmw_qos_profile_services_default, service_group_);

    client_executor_.add_callback_group(client_group_,
                                        get_node_base_interface());
    client_thread_ = std::thread([this]() { client_executor_.spin(); });
    RCLCPP_INFO(get_logger(),
                "compat bridge ready: wrist=%s head=%s timeout=%.3fs",
                wrist.c_str(), head.c_str(), timeout_sec);
  }

  ~CompatBridgeNode() override {
    client_executor_.cancel();
    if (client_thread_.joinable()) {
      client_thread_.join();
    }
  }

 private:
  template <typename ServiceT>
  std::shared_ptr<typename ServiceT::Response> forward(
      const typename rclcpp::Client<ServiceT>::SharedPtr& client,
      const typename ServiceT::Request::SharedPtr& request) {
    const auto deadline = std::chrono::steady_clock::now() + forward_timeout_;
    try {
      const auto now = std::chrono::steady_clock::now();
      if (now >= deadline ||
          !client->wait_for_service(std::min(
              std::chrono::milliseconds(200),
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  deadline - now)))) {
        return nullptr;
      }
      auto pending = client->async_send_request(request);
      const auto remaining = deadline - std::chrono::steady_clock::now();
      if (remaining <= std::chrono::steady_clock::duration::zero() ||
          pending.future.wait_for(remaining) != std::future_status::ready) {
        client->remove_pending_request(pending);
        return nullptr;
      }
      return pending.future.get();
    } catch (const std::exception& exception) {
      RCLCPP_WARN(get_logger(), "legacy forward failed: %s", exception.what());
      return nullptr;
    }
  }

  void handle_image(GetImage::Request::ConstSharedPtr request,
                    GetImage::Response::SharedPtr response) {
    const auto it = snapshot_clients_.find(request->camera);
    if (it == snapshot_clients_.end()) {
      response->status = 1;
      response->error_msg = "Unsupported camera: " + request->camera + ".";
      return;
    }
    auto forwarded = std::make_shared<GetCameraSnapshot::Request>();
    forwarded->want_color = true;
    forwarded->want_depth = request->depth;
    forwarded->want_camera_info = false;
    const auto result = forward<GetCameraSnapshot>(it->second, forwarded);
    if (!result || result->status != GetCameraSnapshot::Response::STATUS_OK) {
      response->status = 1;
      response->error_msg = "No camera data for " + request->camera + ".";
      return;
    }
    response->status = 0;
    response->error_msg.clear();
    response->rgb_image = result->color;
    if (request->depth) {
      response->depth_image = result->depth;
    }
  }

  void handle_cloud(GetPointCloud::Request::ConstSharedPtr request,
                    GetPointCloud::Response::SharedPtr response) {
    const auto it = cloud_clients_.find(request->camera);
    if (it == cloud_clients_.end()) {
      response->status = 1;
      response->error_msg = "Unsupported camera: " + request->camera + ".";
      return;
    }
    auto forwarded = std::make_shared<GetCameraPointCloud::Request>();
    forwarded->include_color = true;
    forwarded->stride = 0;
    const auto result = forward<GetCameraPointCloud>(it->second, forwarded);
    if (!result || result->status != GetCameraPointCloud::Response::STATUS_OK) {
      response->status = 1;
      response->error_msg = "No camera data for " + request->camera + ".";
      return;
    }
    response->status = 0;
    response->error_msg.clear();
    response->points = result->points;
  }

  void handle_orbbec_pc(GetOrbbecPC::Request::ConstSharedPtr request,
                        GetOrbbecPC::Response::SharedPtr response) {
    auto forwarded = std::make_shared<GetCameraPointCloud::Request>();
    forwarded->stride = request->stride;
    forwarded->include_color = request->include_color;
    const auto result =
        forward<GetCameraPointCloud>(cloud_clients_.at("orbbec"), forwarded);
    if (!result || result->status != GetCameraPointCloud::Response::STATUS_OK) {
      response->status = 1;
      response->error_msg = "No camera data for orbbec.";
      return;
    }
    response->status = 0;
    response->error_msg.clear();
    response->points = result->points;
  }

  std::chrono::duration<double> forward_timeout_{5.0};
  std::map<std::string, rclcpp::Client<GetCameraSnapshot>::SharedPtr>
      snapshot_clients_;
  std::map<std::string, rclcpp::Client<GetCameraPointCloud>::SharedPtr>
      cloud_clients_;
  rclcpp::Service<GetImage>::SharedPtr image_service_;
  rclcpp::Service<GetPointCloud>::SharedPtr cloud_service_;
  rclcpp::Service<GetOrbbecPC>::SharedPtr orbbec_pc_service_;
  rclcpp::CallbackGroup::SharedPtr client_group_;
  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::executors::SingleThreadedExecutor client_executor_;
  std::thread client_thread_;
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
