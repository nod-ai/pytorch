#pragma once

#include <c10/zoom/ZoomStream.h>
#include <iostream>
#include <utility>

// CUDA Graphs utils used by c10 and aten.
// aten/cuda/CUDAGraphsUtils.cuh adds utils used by aten only.

namespace c10::zoom {

using CaptureId_t = unsigned long long;

// first is set if the instance is created by CUDAGraph::capture_begin.
// second is set if the instance is created by at::zoom::graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

// RAII guard for "hipStreamCaptureMode", a thread-local value
// that controls the error-checking strictness of a capture.
struct ZoomStreamCaptureModeGuard {
  ZoomStreamCaptureModeGuard(hipStreamCaptureMode desired)
      : strictness_(desired) {
    C10_ZOOM_CHECK(hipThreadExchangeStreamCaptureMode(&strictness_));
  }
  ~ZoomStreamCaptureModeGuard() {
    C10_ZOOM_CHECK_WARN(hipThreadExchangeStreamCaptureMode(&strictness_));
  }

 private:
  hipStreamCaptureMode strictness_;
};

// Protects against enum hipStreamCaptureStatus implementation changes.
// Some compilers seem not to like static_assert without the messages.
static_assert(
    int(hipStreamCaptureStatus::hipStreamCaptureStatusNone) == 0,
    "unexpected int(hipStreamCaptureStatusNone) value");
static_assert(
    int(hipStreamCaptureStatus::hipStreamCaptureStatusActive) == 1,
    "unexpected int(hipStreamCaptureStatusActive) value");
static_assert(
    int(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated) == 2,
    "unexpected int(hipStreamCaptureStatusInvalidated) value");

enum class CaptureStatus : int {
  None = int(hipStreamCaptureStatus::hipStreamCaptureStatusNone),
  Active = int(hipStreamCaptureStatus::hipStreamCaptureStatusActive),
  Invalidated = int(hipStreamCaptureStatus::hipStreamCaptureStatusInvalidated)
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::None:
      os << "hipStreamCaptureStatusNone";
      break;
    case CaptureStatus::Active:
      os << "hipStreamCaptureStatusActive";
      break;
    case CaptureStatus::Invalidated:
      os << "hipStreamCaptureStatusInvalidated";
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown HIP graph CaptureStatus", int(status));
  }
  return os;
}

// Use this version where you're sure a HIP context exists already.
inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  hipStreamCaptureStatus is_capturing{hipStreamCaptureStatusNone};
  C10_ZOOM_CHECK(
      hipStreamIsCapturing(c10::zoom::getCurrentZoomStream(), &is_capturing));
  return CaptureStatus(is_capturing);
}

} // namespace c10::zoom