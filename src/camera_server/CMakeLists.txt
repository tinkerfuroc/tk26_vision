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

# Core logic (no rclcpp node): unit-testable pieces. PIC allows the static
# archive to be linked into the shared component added in a later task.
add_library(camera_server_core
  src/frame_store.cpp
  src/deprojector.cpp
)
set_target_properties(camera_server_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(camera_server_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${sensor_msgs_INCLUDE_DIRS})
target_link_libraries(camera_server_core PUBLIC Eigen3::Eigen ${OpenCV_LIBS})

add_library(camera_server_component SHARED
  src/camera_server_node.cpp
)
target_link_libraries(camera_server_component camera_server_core)
ament_target_dependencies(camera_server_component
  rclcpp
  rclcpp_components
  sensor_msgs
  geometry_msgs
  builtin_interfaces
  message_filters
  tf2
  tf2_ros
  tf2_eigen
  tinker_vision_msgs_26
)
rclcpp_components_register_nodes(camera_server_component
  "camera_server::CameraServerNode")

add_executable(camera_server_node src/camera_server_main.cpp)
target_link_libraries(camera_server_node camera_server_component)
ament_target_dependencies(camera_server_node rclcpp)

add_executable(camera_compat_bridge src/compat_bridge_node.cpp)
ament_target_dependencies(camera_compat_bridge
  rclcpp sensor_msgs tinker_vision_msgs_26)

install(TARGETS camera_server_core camera_server_component
  ARCHIVE DESTINATION lib LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS camera_server_node DESTINATION lib/${PROJECT_NAME})
install(TARGETS camera_compat_bridge DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_frame_store test/test_frame_store.cpp)
  target_link_libraries(test_frame_store camera_server_core)

  ament_add_gtest(test_deprojector test/test_deprojector.cpp)
  target_link_libraries(test_deprojector camera_server_core)

  ament_add_gtest(test_camera_server_node test/test_camera_server_node.cpp)
  target_link_libraries(test_camera_server_node camera_server_component)
  ament_target_dependencies(test_camera_server_node
    rclcpp sensor_msgs geometry_msgs tf2_ros tinker_vision_msgs_26)
endif()

ament_package()
