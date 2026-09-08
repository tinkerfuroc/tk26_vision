#include <cstddef>
#include <exception>
#include <memory>

#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>

#include "camera_server/camera_server_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node =
        std::make_shared<camera_server::CameraServerNode>();
    const auto threads = static_cast<size_t>(
        node->get_parameter("num_executor_threads").as_int());
    rclcpp::executors::MultiThreadedExecutor executor(
        rclcpp::ExecutorOptions(), threads);
    executor.add_node(node);
    executor.spin();
    executor.remove_node(node);
    node.reset();
  } catch (const std::exception& exception) {
    RCLCPP_FATAL(rclcpp::get_logger("camera_server_main"),
                 "camera server failed: %s", exception.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
