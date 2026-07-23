#include "camera_server/deprojector.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <sensor_msgs/msg/point_field.hpp>

namespace camera_server {
namespace {

constexpr size_t kXOffset = 0;
constexpr size_t kYOffset = 4;
constexpr size_t kZOffset = 8;
constexpr size_t kRgbOffset = 12;
constexpr uint32_t kXyzPointStep = 12;
constexpr uint32_t kXyzRgbPointStep = 16;

bool checked_multiply(size_t lhs, size_t rhs, size_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

uint16_t read_u16(const uint8_t* bytes, bool big_endian) {
  if (big_endian) {
    return static_cast<uint16_t>(
        (static_cast<uint16_t>(bytes[0]) << 8U) |
        static_cast<uint16_t>(bytes[1]));
  }
  return static_cast<uint16_t>(
      static_cast<uint16_t>(bytes[0]) |
      (static_cast<uint16_t>(bytes[1]) << 8U));
}

uint32_t read_u32(const uint8_t* bytes, bool big_endian) {
  if (big_endian) {
    return (static_cast<uint32_t>(bytes[0]) << 24U) |
           (static_cast<uint32_t>(bytes[1]) << 16U) |
           (static_cast<uint32_t>(bytes[2]) << 8U) |
           static_cast<uint32_t>(bytes[3]);
  }
  return static_cast<uint32_t>(bytes[0]) |
         (static_cast<uint32_t>(bytes[1]) << 8U) |
         (static_cast<uint32_t>(bytes[2]) << 16U) |
         (static_cast<uint32_t>(bytes[3]) << 24U);
}

float read_f32(const uint8_t* bytes, bool big_endian) {
  const uint32_t bits = read_u32(bytes, big_endian);
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void write_u32_le(uint8_t* bytes, uint32_t value) {
  bytes[0] = static_cast<uint8_t>(value & 0xffU);
  bytes[1] = static_cast<uint8_t>((value >> 8U) & 0xffU);
  bytes[2] = static_cast<uint8_t>((value >> 16U) & 0xffU);
  bytes[3] = static_cast<uint8_t>((value >> 24U) & 0xffU);
}

void write_f32_le(uint8_t* bytes, float value) {
  uint32_t bits = 0;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&bits, &value, sizeof(bits));
  write_u32_le(bytes, bits);
}

sensor_msgs::msg::PointField make_float_field(const std::string& name,
                                               uint32_t offset) {
  sensor_msgs::msg::PointField field;
  field.name = name;
  field.offset = offset;
  field.datatype = sensor_msgs::msg::PointField::FLOAT32;
  field.count = 1;
  return field;
}

bool validate_image_buffer(const sensor_msgs::msg::Image& image,
                           size_t bytes_per_pixel,
                           const std::string& label,
                           std::string& error_msg) {
  if (image.width == 0 || image.height == 0) {
    error_msg = label + " image dimensions must be nonzero";
    return false;
  }

  size_t minimum_step = 0;
  if (!checked_multiply(static_cast<size_t>(image.width), bytes_per_pixel,
                        minimum_step) ||
      static_cast<size_t>(image.step) < minimum_step) {
    error_msg = label + " image step is smaller than width * bytes_per_pixel";
    return false;
  }

  size_t required_data_size = 0;
  if (!checked_multiply(static_cast<size_t>(image.step),
                        static_cast<size_t>(image.height),
                        required_data_size)) {
    error_msg = label + " image step * height overflows";
    return false;
  }
  if (image.data.size() < required_data_size) {
    error_msg = label + " image data is shorter than step * height";
    return false;
  }
  return true;
}

bool validate_intrinsics(const sensor_msgs::msg::Image& depth,
                         const sensor_msgs::msg::CameraInfo& info,
                         std::string& error_msg) {
  if (info.width == 0 || info.height == 0) {
    error_msg = "depth camera_info dimensions must be nonzero";
    return false;
  }
  if (info.width != depth.width || info.height != depth.height) {
    error_msg = "depth image and camera_info dimensions do not match";
    return false;
  }

  const double fx = info.k[0];
  const double fy = info.k[4];
  const double cx = info.k[2];
  const double cy = info.k[5];
  if (!std::isfinite(fx) || !std::isfinite(fy) || fx <= 0.0 || fy <= 0.0) {
    error_msg = "camera_info focal lengths must be finite and positive";
    return false;
  }
  if (!std::isfinite(cx) || !std::isfinite(cy)) {
    error_msg = "camera_info principal point must be finite";
    return false;
  }
  return true;
}

}  // namespace

bool Deprojector::rebuild_table(const TableKey& key,
                                std::string& error_msg) {
  size_t pixel_count = 0;
  size_t table_size = 0;
  if (!checked_multiply(static_cast<size_t>(key.width),
                        static_cast<size_t>(key.height), pixel_count) ||
      !checked_multiply(pixel_count, size_t{2}, table_size) ||
      table_size > xy_table_.max_size()) {
    error_msg = "xy-table dimensions overflow";
    return false;
  }

  std::vector<float> table;
  try {
    table.resize(table_size);
  } catch (const std::length_error&) {
    error_msg = "xy-table dimensions exceed container capacity";
    return false;
  } catch (const std::bad_alloc&) {
    error_msg = "unable to allocate xy table";
    return false;
  }

  size_t index = 0;
  for (uint32_t v = 0; v < key.height; ++v) {
    for (uint32_t u = 0; u < key.width; ++u) {
      const double normalized_x =
          (static_cast<double>(u) - key.cx) / key.fx;
      const double normalized_y =
          (static_cast<double>(v) - key.cy) / key.fy;
      const float x = static_cast<float>(normalized_x);
      const float y = static_cast<float>(normalized_y);
      if (!std::isfinite(x) || !std::isfinite(y)) {
        error_msg = "camera intrinsics produce non-finite deprojection rays";
        return false;
      }
      table[index++] = x;
      table[index++] = y;
    }
  }

  key_ = key;
  xy_table_ = std::move(table);
  return true;
}

bool Deprojector::deproject(
    const sensor_msgs::msg::Image& depth,
    const sensor_msgs::msg::CameraInfo& depth_info,
    const sensor_msgs::msg::Image* color,
    uint32_t stride,
    const std::optional<Eigen::Isometry3f>& transform,
    sensor_msgs::msg::PointCloud2& out,
    std::string& error_msg) {
  out = sensor_msgs::msg::PointCloud2{};
  error_msg.clear();

  const bool is_u16 =
      depth.encoding == "16UC1" || depth.encoding == "mono16";
  const bool is_f32 = depth.encoding == "32FC1";
  if (!is_u16 && !is_f32) {
    error_msg = "unsupported depth encoding: " + depth.encoding;
    return false;
  }
  const size_t depth_bytes_per_pixel = is_u16 ? size_t{2} : size_t{4};
  if (!validate_image_buffer(depth, depth_bytes_per_pixel, "depth",
                             error_msg) ||
      !validate_intrinsics(depth, depth_info, error_msg)) {
    return false;
  }

  if (color != nullptr) {
    if (color->encoding != "rgb8" && color->encoding != "bgr8") {
      error_msg = "unsupported color encoding: " + color->encoding;
      return false;
    }
    if (color->width != depth.width || color->height != depth.height) {
      error_msg =
          "color and depth dimensions must match (depth must be registered)";
      return false;
    }
    if (!validate_image_buffer(*color, size_t{3}, "color", error_msg)) {
      return false;
    }
  }

  if (transform.has_value() && !transform->matrix().allFinite()) {
    error_msg = "transform contains non-finite values";
    return false;
  }

  const TableKey requested_key{depth.width, depth.height, depth_info.k[0],
                               depth_info.k[4], depth_info.k[2],
                               depth_info.k[5]};
  if (!(requested_key == key_) || xy_table_.empty()) {
    if (!rebuild_table(requested_key, error_msg)) {
      return false;
    }
  }

  const size_t sample_stride =
      stride <= 1 ? size_t{1} : static_cast<size_t>(stride);
  const size_t sampled_width =
      (static_cast<size_t>(depth.width) - 1) / sample_stride + 1;
  const size_t sampled_height =
      (static_cast<size_t>(depth.height) - 1) / sample_stride + 1;
  size_t max_points = 0;
  if (!checked_multiply(sampled_width, sampled_height, max_points) ||
      max_points > std::numeric_limits<uint32_t>::max()) {
    error_msg = "sampled point count exceeds PointCloud2 width capacity";
    return false;
  }

  const uint32_t point_step =
      color == nullptr ? kXyzPointStep : kXyzRgbPointStep;
  size_t maximum_data_size = 0;
  if (!checked_multiply(max_points, static_cast<size_t>(point_step),
                        maximum_data_size) ||
      maximum_data_size > std::numeric_limits<uint32_t>::max()) {
    error_msg = "point cloud data size exceeds PointCloud2 row_step capacity";
    return false;
  }

  std::vector<uint8_t> point_data;
  try {
    point_data.resize(maximum_data_size);
  } catch (const std::length_error&) {
    error_msg = "point cloud dimensions exceed container capacity";
    return false;
  } catch (const std::bad_alloc&) {
    error_msg = "unable to allocate point cloud";
    return false;
  }

  size_t point_count = 0;
  for (size_t v = 0; v < depth.height; v += sample_stride) {
    const uint8_t* depth_row =
        depth.data.data() + v * static_cast<size_t>(depth.step);
    for (size_t u = 0; u < depth.width; u += sample_stride) {
      const uint8_t* depth_pixel =
          depth_row + u * depth_bytes_per_pixel;
      const float z =
          is_u16 ? static_cast<float>(
                       read_u16(depth_pixel, depth.is_bigendian)) *
                       0.001F
                 : read_f32(depth_pixel, depth.is_bigendian);
      if (!std::isfinite(z) || z <= 0.0F) {
        continue;
      }

      const size_t table_index =
          (v * static_cast<size_t>(depth.width) + u) * 2;
      Eigen::Vector3f point(xy_table_[table_index] * z,
                            xy_table_[table_index + 1] * z, z);
      if (transform.has_value()) {
        point = *transform * point;
      }
      if (!point.allFinite()) {
        error_msg = "deprojection or transform produced a non-finite point";
        return false;
      }

      uint8_t* output =
          point_data.data() + point_count * static_cast<size_t>(point_step);
      write_f32_le(output + kXOffset, point.x());
      write_f32_le(output + kYOffset, point.y());
      write_f32_le(output + kZOffset, point.z());

      if (color != nullptr) {
        const uint8_t* color_pixel =
            color->data.data() + v * static_cast<size_t>(color->step) + u * 3;
        const bool bgr = color->encoding == "bgr8";
        const uint8_t red = color_pixel[bgr ? 2 : 0];
        const uint8_t green = color_pixel[1];
        const uint8_t blue = color_pixel[bgr ? 0 : 2];
        output[kRgbOffset] = blue;
        output[kRgbOffset + 1] = green;
        output[kRgbOffset + 2] = red;
        output[kRgbOffset + 3] = 0;
      }
      ++point_count;
    }
  }

  point_data.resize(point_count * static_cast<size_t>(point_step));

  sensor_msgs::msg::PointCloud2 result;
  result.height = 1;
  result.width = static_cast<uint32_t>(point_count);
  result.fields = {make_float_field("x", static_cast<uint32_t>(kXOffset)),
                   make_float_field("y", static_cast<uint32_t>(kYOffset)),
                   make_float_field("z", static_cast<uint32_t>(kZOffset))};
  if (color != nullptr) {
    result.fields.push_back(
        make_float_field("rgb", static_cast<uint32_t>(kRgbOffset)));
  }
  result.is_bigendian = false;
  result.point_step = point_step;
  result.row_step = static_cast<uint32_t>(
      point_count * static_cast<size_t>(point_step));
  result.data = std::move(point_data);
  result.is_dense = true;
  out = std::move(result);
  return true;
}

}  // namespace camera_server
