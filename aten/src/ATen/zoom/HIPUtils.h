#pragma once

#include <ATen/zoom/ZoomContext.h>

namespace at::zoom {

// Check if every tensor in a list of tensors matches the current
// device.
inline bool check_device(ArrayRef<Tensor> ts) {
  if (ts.empty()) {
    return true;
  }
  Device curDevice = Device(kPrivateUse1, c10::zoom::current_device());
  for (const Tensor& t : ts) {
    if (t.device() != curDevice) return false;
  }
  return true;
}

} // namespace at::zoom