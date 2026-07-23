#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include "camera_server/deprojector.hpp"

using camera_server::Deprojector;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

namespace {

CameraInfo make_info(uint32_t width = 4, uint32_t height = 4,
                     double fx = 100.0, double fy = 100.0,
                     double cx = 2.0, double cy = 2.0) {
  CameraInfo info;
  info.header.frame_id = "camera_optical";
  info.width = width;
  info.height = height;
  info.k = {fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0};
  return info;
}

CameraInfo with_distortion(CameraInfo info, const std::string& model,
                           std::vector<double> coefficients) {
  info.distortion_model = model;
  info.d = std::move(coefficients);
  return info;
}

void write_u16(std::vector<uint8_t>& data, size_t offset, uint16_t value,
               bool big_endian) {
  if (big_endian) {
    data[offset] = static_cast<uint8_t>(value >> 8U);
    data[offset + 1] = static_cast<uint8_t>(value & 0xffU);
  } else {
    data[offset] = static_cast<uint8_t>(value & 0xffU);
    data[offset + 1] = static_cast<uint8_t>(value >> 8U);
  }
}

void write_f32(std::vector<uint8_t>& data, size_t offset, float value,
               bool big_endian) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  for (size_t i = 0; i < 4; ++i) {
    const size_t shift_index = big_endian ? 3 - i : i;
    data[offset + i] =
        static_cast<uint8_t>((bits >> (shift_index * 8U)) & 0xffU);
  }
}

float read_f32_le(const std::vector<uint8_t>& data, size_t offset) {
  const uint32_t bits = static_cast<uint32_t>(data[offset]) |
                        (static_cast<uint32_t>(data[offset + 1]) << 8U) |
                        (static_cast<uint32_t>(data[offset + 2]) << 16U) |
                        (static_cast<uint32_t>(data[offset + 3]) << 24U);
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

Image make_depth_u16(uint32_t width = 4, uint32_t height = 4,
                     uint16_t millimetres = 1000,
                     const std::string& encoding = "16UC1",
                     bool big_endian = false) {
  Image image;
  image.header.frame_id = "camera_optical";
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  image.is_bigendian = big_endian;
  image.step = width * 2;
  image.data.assign(static_cast<size_t>(image.step) * height, 0);
  for (uint32_t v = 0; v < height; ++v) {
    for (uint32_t u = 0; u < width; ++u) {
      write_u16(image.data, static_cast<size_t>(v) * image.step + u * 2,
                millimetres, big_endian);
    }
  }
  return image;
}

Image make_depth_f32(uint32_t width = 4, uint32_t height = 4,
                     float metres = 1.0F, bool big_endian = false) {
  Image image;
  image.header.frame_id = "camera_optical";
  image.width = width;
  image.height = height;
  image.encoding = "32FC1";
  image.is_bigendian = big_endian;
  image.step = width * 4;
  image.data.assign(static_cast<size_t>(image.step) * height, 0);
  for (uint32_t v = 0; v < height; ++v) {
    for (uint32_t u = 0; u < width; ++u) {
      write_f32(image.data, static_cast<size_t>(v) * image.step + u * 4,
                metres, big_endian);
    }
  }
  return image;
}

Image make_color(uint32_t width = 4, uint32_t height = 4,
                 const std::string& encoding = "rgb8") {
  Image image;
  image.header.frame_id = "camera_optical";
  image.width = width;
  image.height = height;
  image.encoding = encoding;
  image.step = width * 3;
  image.data.assign(static_cast<size_t>(image.step) * height, 0);
  return image;
}

bool deproject(Deprojector& deprojector, const Image& depth,
               const CameraInfo& info, PointCloud2& out, std::string& error,
               const Image* color = nullptr, uint32_t stride = 0,
               const std::optional<Eigen::Isometry3f>& transform =
                   std::nullopt) {
  return deprojector.deproject(depth, info, color, stride, transform, out,
                               error);
}

void expect_failure_and_empty(Deprojector& deprojector, const Image& depth,
                              const CameraInfo& info,
                              const Image* color = nullptr) {
  PointCloud2 out;
  out.width = 99;
  out.data = {1, 2, 3};
  std::string error = "old error";
  EXPECT_FALSE(deproject(deprojector, depth, info, out, error, color));
  EXPECT_FALSE(error.empty());
  EXPECT_EQ(out.width, 0U);
  EXPECT_TRUE(out.fields.empty());
  EXPECT_TRUE(out.data.empty());
}

}  // namespace

TEST(Deprojector, EmitsExactXyzLayoutAndCoordinates) {
  Deprojector deprojector;
  const Image depth = make_depth_u16();
  PointCloud2 out;
  std::string error = "stale";

  ASSERT_TRUE(
      deproject(deprojector, depth, make_info(), out, error, nullptr, 0))
      << error;
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(out.height, 1U);
  EXPECT_EQ(out.width, 16U);
  ASSERT_EQ(out.fields.size(), 3U);
  const std::array<std::string, 3> names{"x", "y", "z"};
  for (size_t i = 0; i < names.size(); ++i) {
    EXPECT_EQ(out.fields[i].name, names[i]);
    EXPECT_EQ(out.fields[i].offset, i * 4U);
    EXPECT_EQ(out.fields[i].datatype, PointField::FLOAT32);
    EXPECT_EQ(out.fields[i].count, 1U);
  }
  EXPECT_FALSE(out.is_bigendian);
  EXPECT_EQ(out.point_step, 12U);
  EXPECT_EQ(out.row_step, 16U * 12U);
  EXPECT_EQ(out.data.size(), out.row_step);
  EXPECT_TRUE(out.is_dense);
  EXPECT_NEAR(read_f32_le(out.data, 0), -0.02F, 1e-6F);
  EXPECT_NEAR(read_f32_le(out.data, 4), -0.02F, 1e-6F);
  EXPECT_NEAR(read_f32_le(out.data, 8), 1.0F, 1e-6F);
}

TEST(Deprojector, SupportsMono16And32FC1) {
  Deprojector deprojector;
  PointCloud2 out;
  std::string error;

  ASSERT_TRUE(deproject(deprojector,
                        make_depth_u16(1, 1, 2500, "mono16"),
                        make_info(1, 1, 1.0, 1.0, 0.0, 0.0), out, error))
      << error;
  EXPECT_NEAR(read_f32_le(out.data, 8), 2.5F, 1e-6F);

  ASSERT_TRUE(deproject(deprojector, make_depth_f32(1, 1, 3.25F),
                        make_info(1, 1, 1.0, 1.0, 0.0, 0.0), out, error))
      << error;
  EXPECT_NEAR(read_f32_le(out.data, 8), 3.25F, 1e-6F);
}

TEST(Deprojector, DecodesBigEndianDepthAndAlwaysEmitsLittleEndian) {
  Deprojector deprojector;
  PointCloud2 out;
  std::string error;
  const CameraInfo info = make_info(1, 1, 1.0, 1.0, 0.0, 0.0);

  ASSERT_TRUE(deproject(deprojector,
                        make_depth_u16(1, 1, 1250, "16UC1", true), info,
                        out, error))
      << error;
  EXPECT_FALSE(out.is_bigendian);
  EXPECT_EQ(out.data[8], 0x00U);
  EXPECT_EQ(out.data[9], 0x00U);
  EXPECT_EQ(out.data[10], 0xa0U);
  EXPECT_EQ(out.data[11], 0x3fU);
  EXPECT_NEAR(read_f32_le(out.data, 8), 1.25F, 1e-6F);

  ASSERT_TRUE(deproject(deprojector, make_depth_f32(1, 1, 2.5F, true), info,
                        out, error))
      << error;
  EXPECT_FALSE(out.is_bigendian);
  EXPECT_NEAR(read_f32_le(out.data, 8), 2.5F, 1e-6F);
}

TEST(Deprojector, EmitsPackedRgbForRgbAndBgrInput) {
  Deprojector deprojector;
  const Image depth = make_depth_u16(2, 1);
  const CameraInfo info = make_info(2, 1, 100.0, 100.0, 0.0, 0.0);
  PointCloud2 out;
  std::string error;

  Image rgb = make_color(2, 1, "rgb8");
  rgb.data = {10, 20, 30, 40, 50, 60};
  ASSERT_TRUE(deproject(deprojector, depth, info, out, error, &rgb)) << error;
  ASSERT_EQ(out.fields.size(), 4U);
  EXPECT_EQ(out.fields[3].name, "rgb");
  EXPECT_EQ(out.fields[3].offset, 12U);
  EXPECT_EQ(out.fields[3].datatype, PointField::FLOAT32);
  EXPECT_EQ(out.point_step, 16U);
  EXPECT_EQ(out.row_step, 32U);
  EXPECT_EQ(out.data.size(), 32U);
  EXPECT_EQ(out.data[12], 30U);
  EXPECT_EQ(out.data[13], 20U);
  EXPECT_EQ(out.data[14], 10U);
  EXPECT_EQ(out.data[15], 0U);
  sensor_msgs::PointCloud2ConstIterator<uint8_t> red(out, "r");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> green(out, "g");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> blue(out, "b");
  EXPECT_EQ(*red, 10U);
  EXPECT_EQ(*green, 20U);
  EXPECT_EQ(*blue, 30U);

  Image bgr = make_color(2, 1, "bgr8");
  bgr.data = {30, 20, 10, 60, 50, 40};
  ASSERT_TRUE(deproject(deprojector, depth, info, out, error, &bgr)) << error;
  EXPECT_EQ(out.data[12], 30U);
  EXPECT_EQ(out.data[13], 20U);
  EXPECT_EQ(out.data[14], 10U);
}

TEST(Deprojector, AppliesRotationAndTranslation) {
  Deprojector deprojector;
  PointCloud2 out;
  std::string error;
  Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
  transform.linear() =
      Eigen::AngleAxisf(static_cast<float>(M_PI_2), Eigen::Vector3f::UnitY())
          .toRotationMatrix();
  transform.translation() = Eigen::Vector3f(1.0F, 2.0F, 3.0F);

  ASSERT_TRUE(deproject(
      deprojector, make_depth_u16(1, 1),
      make_info(1, 1, 1.0, 1.0, 0.0, 0.0), out, error, nullptr, 1,
      transform))
      << error;
  EXPECT_NEAR(read_f32_le(out.data, 0), 2.0F, 1e-5F);
  EXPECT_NEAR(read_f32_le(out.data, 4), 2.0F, 1e-5F);
  EXPECT_NEAR(read_f32_le(out.data, 8), 3.0F, 1e-5F);
}

TEST(Deprojector, InvalidatesXyCacheWhenIntrinsicsOrDimensionsChange) {
  Deprojector deprojector;
  const Image depth = make_depth_u16(2, 1);
  PointCloud2 out;
  std::string error;

  ASSERT_TRUE(deproject(deprojector, depth,
                        make_info(2, 1, 100.0, 100.0, 0.0, 0.0), out,
                        error))
      << error;
  EXPECT_NEAR(read_f32_le(out.data, 12), 0.01F, 1e-6F);

  ASSERT_TRUE(deproject(deprojector, depth,
                        make_info(2, 1, 50.0, 100.0, 1.0, 0.0), out,
                        error))
      << error;
  EXPECT_NEAR(read_f32_le(out.data, 0), -0.02F, 1e-6F);
  EXPECT_NEAR(read_f32_le(out.data, 12), 0.0F, 1e-6F);

  ASSERT_TRUE(deproject(deprojector, make_depth_u16(1, 2),
                        make_info(1, 2, 100.0, 100.0, 0.0, 1.0), out,
                        error))
      << error;
  EXPECT_NEAR(read_f32_le(out.data, 4), -0.01F, 1e-6F);
}

TEST(Deprojector, UndistortsRawPlumbBobAndInvalidatesCacheOnDChange) {
  Deprojector deprojector;
  const Image depth = make_depth_u16(3, 1);
  PointCloud2 out;
  std::string error;
  const CameraInfo positive = with_distortion(
      make_info(3, 1, 1.0, 1.0, 0.0, 0.0), "plumb_bob",
      {0.1, 0.0, 0.0, 0.0, 0.0});
  const CameraInfo negative = with_distortion(
      make_info(3, 1, 1.0, 1.0, 0.0, 0.0), "plumb_bob",
      {-0.1, 0.0, 0.0, 0.0, 0.0});

  ASSERT_TRUE(deproject(deprojector, depth, positive, out, error)) << error;
  const float positive_ray = read_f32_le(out.data, 2U * 12U);
  EXPECT_LT(positive_ray, 2.0F);

  ASSERT_TRUE(deproject(deprojector, depth, negative, out, error)) << error;
  const float negative_ray = read_f32_le(out.data, 2U * 12U);
  EXPECT_GT(negative_ray, positive_ray);
}

TEST(Deprojector, SupportsRationalPolynomialAndEquidistantModels) {
  Deprojector deprojector;
  const Image depth = make_depth_u16(2, 1);
  PointCloud2 out;
  std::string error;

  ASSERT_TRUE(deproject(
      deprojector, depth,
      with_distortion(make_info(2, 1, 2.0, 2.0, 0.0, 0.0),
                      "rational_polynomial",
                      {0.01, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0}),
      out, error))
      << error;
  EXPECT_EQ(out.width, 2U);

  ASSERT_TRUE(deproject(
      deprojector, depth,
      with_distortion(make_info(2, 1, 2.0, 2.0, 0.0, 0.0),
                      "equidistant", {0.01, 0.0, 0.0, 0.0}),
      out, error))
      << error;
  EXPECT_EQ(out.width, 2U);
}

TEST(Deprojector, TreatsUnnamedAllZeroDistortionAsRectifiedPinhole) {
  Deprojector deprojector;
  CameraInfo info = make_info(2, 1, 100.0, 100.0, 0.0, 0.0);
  info.d = {0.0};
  PointCloud2 out;
  std::string error;
  ASSERT_TRUE(
      deproject(deprojector, make_depth_u16(2, 1), info, out, error))
      << error;
  EXPECT_NEAR(read_f32_le(out.data, 12U), 0.01F, 1e-6F);
}

TEST(Deprojector, AcceptsValidPaddedRows) {
  Deprojector deprojector;
  Image depth = make_depth_u16(2, 2);
  depth.step = 8;
  depth.data.assign(16, 0xaa);
  write_u16(depth.data, 0, 1000, false);
  write_u16(depth.data, 2, 2000, false);
  write_u16(depth.data, 8, 3000, false);
  write_u16(depth.data, 10, 4000, false);

  Image color = make_color(2, 2);
  color.step = 8;
  color.data.assign(16, 0xee);
  color.data[0] = 1;
  color.data[1] = 2;
  color.data[2] = 3;
  color.data[3] = 4;
  color.data[4] = 5;
  color.data[5] = 6;
  color.data[8] = 7;
  color.data[9] = 8;
  color.data[10] = 9;
  color.data[11] = 10;
  color.data[12] = 11;
  color.data[13] = 12;

  PointCloud2 out;
  std::string error;
  ASSERT_TRUE(deproject(
      deprojector, depth,
      make_info(2, 2, 10.0, 10.0, 0.0, 0.0), out, error, &color))
      << error;
  ASSERT_EQ(out.width, 4U);
  EXPECT_NEAR(read_f32_le(out.data, 2U * 16U + 8U), 3.0F, 1e-6F);
  EXPECT_EQ(out.data[2U * 16U + 12U], 9U);
  EXPECT_EQ(out.data[2U * 16U + 14U], 7U);
}

TEST(Deprojector, DropsNonfiniteNonpositiveAndAllInvalidDepth) {
  Deprojector deprojector;
  Image depth = make_depth_f32(5, 1);
  const std::array<float, 5> samples{
      std::numeric_limits<float>::quiet_NaN(),
      std::numeric_limits<float>::infinity(), -1.0F, 0.0F, 2.0F};
  for (size_t i = 0; i < samples.size(); ++i) {
    write_f32(depth.data, i * 4, samples[i], false);
  }
  PointCloud2 out;
  std::string error;
  ASSERT_TRUE(deproject(
      deprojector, depth,
      make_info(5, 1, 100.0, 100.0, 0.0, 0.0), out, error))
      << error;
  EXPECT_EQ(out.width, 1U);
  EXPECT_EQ(out.data.size(), 12U);
  EXPECT_NEAR(read_f32_le(out.data, 8), 2.0F, 1e-6F);

  ASSERT_TRUE(deproject(
      deprojector, make_depth_u16(3, 2, 0),
      make_info(3, 2, 100.0, 100.0, 0.0, 0.0), out, error))
      << error;
  EXPECT_EQ(out.height, 1U);
  EXPECT_EQ(out.width, 0U);
  EXPECT_EQ(out.row_step, 0U);
  EXPECT_TRUE(out.data.empty());
  EXPECT_TRUE(out.is_dense);
}

TEST(Deprojector, HandlesNormalAndExtremeStrideWithoutWraparound) {
  Deprojector deprojector;
  const Image depth = make_depth_u16(4, 3);
  const CameraInfo info = make_info(4, 3);
  PointCloud2 out;
  std::string error;

  ASSERT_TRUE(deproject(deprojector, depth, info, out, error, nullptr, 2))
      << error;
  EXPECT_EQ(out.width, 4U);

  ASSERT_TRUE(deproject(deprojector, depth, info, out, error, nullptr,
                        std::numeric_limits<uint32_t>::max()))
      << error;
  EXPECT_EQ(out.width, 1U);
  EXPECT_EQ(out.data.size(), 12U);
}

TEST(Deprojector, RejectsMalformedDepthDimensionsAndBuffers) {
  Deprojector deprojector;

  Image depth = make_depth_u16();
  depth.encoding = "mono8";
  expect_failure_and_empty(deprojector, depth, make_info());

  depth = make_depth_u16();
  depth.width = 0;
  expect_failure_and_empty(deprojector, depth, make_info(0, 4));

  depth = make_depth_u16();
  depth.step = 7;
  expect_failure_and_empty(deprojector, depth, make_info());

  depth = make_depth_u16();
  depth.data.pop_back();
  expect_failure_and_empty(deprojector, depth, make_info());

  depth = make_depth_f32();
  depth.width = std::numeric_limits<uint32_t>::max();
  depth.step = std::numeric_limits<uint32_t>::max();
  expect_failure_and_empty(
      deprojector, depth,
      make_info(std::numeric_limits<uint32_t>::max(), depth.height));
}

TEST(Deprojector, RejectsMalformedCameraInfoAndIntrinsics) {
  Deprojector deprojector;
  const Image depth = make_depth_u16();

  expect_failure_and_empty(deprojector, depth, make_info(3, 4));
  expect_failure_and_empty(deprojector, depth, make_info(4, 0));
  expect_failure_and_empty(
      deprojector, depth, make_info(4, 4, 0.0, 100.0, 2.0, 2.0));
  expect_failure_and_empty(
      deprojector, depth,
      make_info(4, 4, std::numeric_limits<double>::infinity(), 100.0, 2.0,
                2.0));
  expect_failure_and_empty(
      deprojector, depth,
      make_info(4, 4, 100.0, 100.0,
                std::numeric_limits<double>::quiet_NaN(), 2.0));

  expect_failure_and_empty(
      deprojector, depth,
      with_distortion(make_info(), "", {0.1}));
  expect_failure_and_empty(
      deprojector, depth,
      with_distortion(make_info(), "plumb_bob", {0.1, 0.0}));
  expect_failure_and_empty(
      deprojector, depth,
      with_distortion(make_info(), "unsupported_model", {}));
  expect_failure_and_empty(
      deprojector, depth,
      with_distortion(
          make_info(), "plumb_bob",
          {std::numeric_limits<double>::quiet_NaN(), 0.0, 0.0, 0.0, 0.0}));
}

TEST(Deprojector, RejectsMalformedOrUnregisteredColor) {
  Deprojector deprojector;
  const Image depth = make_depth_u16();
  const CameraInfo info = make_info();

  Image color = make_color();
  color.encoding = "rgba8";
  expect_failure_and_empty(deprojector, depth, info, &color);

  color = make_color(2, 2);
  expect_failure_and_empty(deprojector, depth, info, &color);

  color = make_color();
  color.step = 11;
  expect_failure_and_empty(deprojector, depth, info, &color);

  color = make_color();
  color.data.pop_back();
  expect_failure_and_empty(deprojector, depth, info, &color);
}

TEST(Deprojector, RejectsIncompatibleNonemptyFrameIds) {
  Deprojector deprojector;
  Image depth = make_depth_u16();
  depth.header.frame_id = "depth_optical";
  CameraInfo info = make_info();
  info.header.frame_id = "other_optical";
  expect_failure_and_empty(deprojector, depth, info);

  info.header.frame_id = "depth_optical";
  Image color = make_color();
  color.header.frame_id = "color_optical";
  expect_failure_and_empty(deprojector, depth, info, &color);

  color.header.frame_id.clear();
  expect_failure_and_empty(deprojector, depth, info, &color);
}

TEST(Deprojector, RejectsNonfiniteTransformAndClearsPriorOutput) {
  Deprojector deprojector;
  PointCloud2 out;
  out.width = 7;
  out.data = {9};
  std::string error = "old error";
  Eigen::Isometry3f transform = Eigen::Isometry3f::Identity();
  transform.translation().x() = std::numeric_limits<float>::infinity();

  EXPECT_FALSE(deproject(
      deprojector, make_depth_u16(1, 1),
      make_info(1, 1, 1.0, 1.0, 0.0, 0.0), out, error, nullptr, 0,
      transform));
  EXPECT_NE(error.find("transform"), std::string::npos);
  EXPECT_EQ(out.width, 0U);
  EXPECT_TRUE(out.data.empty());
}
