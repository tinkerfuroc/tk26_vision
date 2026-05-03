cmake_minimum_required(VERSION 3.8)
project(tinker_vision_msgs_26)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Generate interfaces
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox.msg"
  "msg/Face.msg"
  "msg/FaceResult.msg"
  "msg/PanTiltCommand.msg"
  "msg/Object.msg"
  "msg/Objects.msg"
  "msg/PanTiltCtrl.msg"
  "msg/PanTiltState.msg"
  "srv/DetectWaving.srv"
  "srv/DoorDetection.srv"
  "srv/FaceRegister.srv"
  "srv/FeatureExtraction.srv"
  "srv/FeatureMatching.srv"
  "srv/FollowHead.srv"
  "srv/GetImage.srv"
  "srv/GetPointCloud.srv"
  "srv/ObjectDetection.srv"
  "srv/ObjectDetectionGeneralist.srv"
  "srv/ObjectDetectionImage.srv"
  "srv/ObjectMatch.srv"
  "srv/PlacingLocation.srv"
  "srv/PointDirection.srv"
  "srv/SetTorque.srv"
  "srv/SetZero.srv"
  "srv/SeatRecommendation.srv"
  "srv/SeatRecommendBbox.srv"
  "action/Categorize.action"
  "action/FollowHeadAction.action"
  "action/HumanFollowing.action"
  "action/SpotOnShelf.action"
  "action/TrackPerson.action"
  DEPENDENCIES geometry_msgs std_msgs sensor_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
