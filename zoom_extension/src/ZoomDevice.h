#pragma once

#include "ZoomException.h"

#include <hip/hip_runtime.h>

namespace at::zoom {

inline Device getDeviceFromPtr(void* ptr) {
  hipPointerAttribute_t attr{};

  C10_ZOOM_CHECK(hipPointerGetAttributes(&attr, ptr));

  return {c10::DeviceType::PrivateUse1, static_cast<DeviceIndex>(attr.device)};
}

} // namespace at::zoom