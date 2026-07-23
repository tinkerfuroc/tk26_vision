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
find_package(tinker_vision_msgs_26 REQUIRED)

# Core logic (no rclcpp node): unit-testable pieces. PIC allows the static
# archive to be linked into the shared component added in a later task.
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
