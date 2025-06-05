#include <c10/zoom/ZoomMiscFunctions.h>
#include <cstdlib>

namespace c10::zoom {

const char* get_hip_check_suffix() noexcept {
  static char* device_blocking_flag = getenv("HIP_LAUNCH_BLOCKING");
  static bool blocking_enabled =
      (device_blocking_flag && atoi(device_blocking_flag));
  if (blocking_enabled) {
    return "";
  } else {
    return "\nHIP kernel errors might be asynchronously reported at some"
           " other API call, so the stacktrace below might be incorrect."
           "\nFor debugging consider passing HIP_LAUNCH_BLOCKING=1";
  }
}
std::mutex* getFreeMutex() {
  static std::mutex hip_free_mutex;
  return &hip_free_mutex;
}

} // namespace c10::zoom