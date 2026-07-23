#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace camera_server {

/// CPU depth-image deprojection with a cached per-intrinsics xy table.
///
/// Depth must already be registered to the color optical frame. Raw registered
/// images are supported: deprojection rays are distortion-aware for ROS
/// plumb_bob, rational_polynomial, and equidistant camera models. This class
/// does not perform cross-camera registration.
///
/// Handles 16UC1/mono16 depth in millimetres and 32FC1 depth in metres, for
/// either input byte order. Color may be rgb8 or bgr8. The output is always an
/// unorganized, little-endian XYZ or XYZRGB PointCloud2.
///
/// NOT thread-safe because the xy-table cache is mutable. Callers must
/// serialize access externally.
class Deprojector {
 public:
  /// stride 0 or 1 means full resolution. transform, when present, is
  /// target_frame <- optical. Invalid/non-positive depth samples are omitted.
  /// On failure, out is reset to an empty message and error_msg describes the
  /// validation or conversion error. On success, error_msg is empty. The
  /// caller sets out.header after this method returns.
  bool deproject(const sensor_msgs::msg::Image& depth,
                 const sensor_msgs::msg::CameraInfo& depth_info,
                 const sensor_msgs::msg::Image* color,
                 uint32_t stride,
                 const std::optional<Eigen::Isometry3f>& transform,
                 sensor_msgs::msg::PointCloud2& out,
                 std::string& error_msg);

 private:
  struct TableKey {
    uint32_t width = 0;
    uint32_t height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    std::string distortion_model;
    std::vector<double> distortion;

    bool operator==(const TableKey& other) const {
      return width == other.width && height == other.height &&
             fx == other.fx && fy == other.fy && cx == other.cx &&
             cy == other.cy &&
             distortion_model == other.distortion_model &&
             distortion == other.distortion;
    }
  };

  bool rebuild_table(const TableKey& key, std::string& error_msg);

  TableKey key_;
  std::vector<float> xy_table_;  // height*width*2: normalized x, normalized y
};

}  // namespace camera_server
