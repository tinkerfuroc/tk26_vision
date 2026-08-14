#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <utility>

#include "camera_server/frame_store.hpp"

using camera_server::FramePair;
using camera_server::FrameStore;
using sensor_msgs::msg::CameraInfo;
using sensor_msgs::msg::Image;

namespace {
Image::ConstSharedPtr make_image(int64_t stamp_ns) {
  auto image = std::make_shared<Image>();
  image->header.stamp.sec =
      static_cast<int32_t>(stamp_ns / 1000000000LL);
  image->header.stamp.nanosec =
      static_cast<uint32_t>(stamp_ns % 1000000000LL);
  image->header.frame_id = "camera_color_optical_frame";
  return image;
}
}  // namespace

TEST(FrameStore, EmptyStoreReturnsNullPair) {
  FrameStore store;
  const FramePair pair = store.latest_pair();
  EXPECT_EQ(pair.color, nullptr);
  EXPECT_EQ(pair.depth, nullptr);
  EXPECT_EQ(pair.color_stamp_ns, 0);
  EXPECT_EQ(pair.depth_stamp_ns, 0);
  EXPECT_EQ(pair.stamp_ns, 0);
  EXPECT_EQ(pair.received_at_ns, 0);
  EXPECT_EQ(pair.seq, 0u);
}

TEST(FrameStore, SetPairStoresPointersAndBumpsSeq) {
  FrameStore store;
  const auto color = make_image(100);
  const auto depth = make_image(99);
  store.set_pair(color, depth, 1234);

  const FramePair first = store.latest_pair();
  EXPECT_EQ(first.color, color);
  EXPECT_EQ(first.depth, depth);
  EXPECT_EQ(first.color_stamp_ns, 100);
  EXPECT_EQ(first.depth_stamp_ns, 99);
  EXPECT_EQ(first.stamp_ns, 99);
  EXPECT_EQ(first.received_at_ns, 1234);
  EXPECT_EQ(first.seq, 1u);

  store.set_pair(make_image(200), make_image(199), 5678);
  const FramePair second = store.latest_pair();
  EXPECT_EQ(second.color_stamp_ns, 200);
  EXPECT_EQ(second.depth_stamp_ns, 199);
  EXPECT_EQ(second.stamp_ns, 199);
  EXPECT_EQ(second.received_at_ns, 5678);
  EXPECT_EQ(second.seq, 2u);
}

TEST(FrameStore, CameraInfosStoredIndependentlyWithPointerIdentity) {
  FrameStore store;
  EXPECT_EQ(store.color_info(), nullptr);
  EXPECT_EQ(store.depth_info(), nullptr);

  auto color_info_mutable = std::make_shared<CameraInfo>();
  color_info_mutable->width = 1280;
  CameraInfo::ConstSharedPtr color_info = color_info_mutable;
  auto depth_info_mutable = std::make_shared<CameraInfo>();
  depth_info_mutable->width = 640;
  CameraInfo::ConstSharedPtr depth_info = depth_info_mutable;

  store.set_color_info(color_info);
  EXPECT_EQ(store.color_info(), color_info);
  EXPECT_EQ(store.color_info()->width, 1280u);
  EXPECT_EQ(store.depth_info(), nullptr);

  store.set_depth_info(depth_info);
  EXPECT_EQ(store.color_info(), color_info);
  EXPECT_EQ(store.depth_info(), depth_info);
  EXPECT_EQ(store.depth_info()->width, 640u);
}

TEST(FrameStore, AlreadyFreshPairReturnsImmediately) {
  FrameStore store;
  store.set_pair(make_image(600), make_image(600));

  const auto start = std::chrono::steady_clock::now();
  const FramePair pair =
      store.wait_for_pair_after(500, std::chrono::seconds(1));
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_EQ(pair.stamp_ns, 600);
  EXPECT_LT(elapsed, std::chrono::milliseconds(100));
}

TEST(FrameStore, ExactFreshnessBoundaryIsAccepted) {
  FrameStore store;
  store.set_pair(make_image(500), make_image(500));

  const auto start = std::chrono::steady_clock::now();
  const FramePair pair =
      store.wait_for_pair_after(500, std::chrono::seconds(1));
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_EQ(pair.stamp_ns, 500);
  EXPECT_LT(elapsed, std::chrono::milliseconds(100));
}

TEST(FrameStore, FreshnessBoundaryRequiresBothImages) {
  FrameStore store;
  store.set_pair(make_image(600), make_image(499));

  const auto start = std::chrono::steady_clock::now();
  const FramePair pair =
      store.wait_for_pair_after(500, std::chrono::milliseconds(60));
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_GE(elapsed, std::chrono::milliseconds(45));
  EXPECT_EQ(pair.color_stamp_ns, 600);
  EXPECT_EQ(pair.depth_stamp_ns, 499);
  EXPECT_EQ(pair.stamp_ns, 499);
}

TEST(FrameStore, EmptyStoreWaitTimesOutReturningNullPair) {
  FrameStore store;

  const auto start = std::chrono::steady_clock::now();
  const FramePair pair =
      store.wait_for_pair_after(0, std::chrono::milliseconds(60));
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_GE(elapsed, std::chrono::milliseconds(45));
  EXPECT_EQ(pair.color, nullptr);
  EXPECT_EQ(pair.depth, nullptr);
  EXPECT_EQ(pair.seq, 0u);
}

TEST(FrameStore, WaitTimesOutReturningNewestPair) {
  FrameStore store;
  const auto color = make_image(100);
  const auto depth = make_image(100);
  store.set_pair(color, depth);

  const auto start = std::chrono::steady_clock::now();
  const FramePair pair =
      store.wait_for_pair_after(500, std::chrono::milliseconds(100));
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_GE(elapsed, std::chrono::milliseconds(85));
  EXPECT_EQ(pair.color, color);
  EXPECT_EQ(pair.depth, depth);
  EXPECT_EQ(pair.stamp_ns, 100);
}

TEST(FrameStore, WaitUnblocksOnFreshPair) {
  FrameStore store;
  store.set_pair(make_image(100), make_image(100));
  std::thread feeder([&store] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    store.set_pair(make_image(600), make_image(600));
  });

  const FramePair pair =
      store.wait_for_pair_after(500, std::chrono::seconds(2));
  feeder.join();

  EXPECT_EQ(pair.stamp_ns, 600);
  EXPECT_EQ(pair.seq, 2u);
}

TEST(FrameStore, NullPairInputsAreIgnoredWithoutAdvancingSequence) {
  FrameStore store;
  const auto original_color = make_image(100);
  const auto original_depth = make_image(100);
  store.set_pair(original_color, original_depth);

  store.set_pair(nullptr, make_image(200));
  store.set_pair(make_image(300), nullptr);

  const FramePair pair = store.latest_pair();
  EXPECT_EQ(pair.color, original_color);
  EXPECT_EQ(pair.depth, original_depth);
  EXPECT_EQ(pair.stamp_ns, 100);
  EXPECT_EQ(pair.seq, 1u);
}

TEST(FrameStore, IncompletePairDoesNotSatisfyOrWakeFreshnessWait) {
  FrameStore store;
  std::thread invalid_feeder([&store] {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    store.set_pair(make_image(600), nullptr);
  });

  const auto start = std::chrono::steady_clock::now();
  const FramePair pair =
      store.wait_for_pair_after(500, std::chrono::milliseconds(80));
  const auto elapsed = std::chrono::steady_clock::now() - start;
  invalid_feeder.join();

  EXPECT_GE(elapsed, std::chrono::milliseconds(65));
  EXPECT_EQ(pair.color, nullptr);
  EXPECT_EQ(pair.depth, nullptr);
  EXPECT_EQ(pair.seq, 0u);
}
