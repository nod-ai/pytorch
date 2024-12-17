#pragma once

#include "ZoomGeneratorImpl.h"
#include "ZoomEvent.h"
#include "PhiloxUtils.hpp"
// #include <ATen/cuda/detail/CUDAHooks.h>"
#include "ZoomHooks.h"
#include "ZoomHooksInterface.h"
#include <c10/core/StreamGuard.h>
#include "HIPGraphsC10Utils.h"
#include "ZoomGuard.h"

// c10/cuda/CUDAGraphsC10Utils.h has utils used by both c10 and aten.
// This file adds utils used by aten only.

namespace at::zoom {

using CaptureId_t = c10::zoom::CaptureId_t;
using CaptureStatus = c10::zoom::CaptureStatus;

// Use this version where you don't want to create a CUDA context if none exists.
inline CaptureStatus currentStreamCaptureStatus() {
  // don't create a context if we don't have to
  if (c10::zoom::hasPrimaryContext(c10::zoom::current_device())) {
    return c10::zoom::currentStreamCaptureStatusMayInitCtx();
  } else {
    return CaptureStatus::None;
  }
}

inline void assertNotCapturing(std::string attempt) {
  auto status = currentStreamCaptureStatus();
  TORCH_CHECK(status == CaptureStatus::None,
              attempt,
              " during HIP graph capture. If you need this call to be captured, "
              "please file an issue. "
              "Current hipStreamCaptureStatus: ",
              status);
}

} // namespace at::zoom