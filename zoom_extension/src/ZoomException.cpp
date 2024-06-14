#include "ZoomException.h"
#include <c10/util/Exception.h>

#include <string>

namespace c10::zoom {

void c10_zoom_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions) {
  const auto hip_error = static_cast<hipError_t>(err);
  // TODO(Arham): for now zoom ignores this, kernel registry to be implemented
  const auto hip_kernel_failure = false;
//   const auto cuda_kernel_failure = include_device_assertions
//       ? c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref().has_failed()
//       : false;

  if (C10_LIKELY(hip_error == hipSuccess && !hip_kernel_failure)) {
    return;
  }

  auto error_unused C10_UNUSED = hipGetLastError();
  (void)error_unused;

  std::string check_message;
#ifndef STRIP_ERROR_MESSAGES
  check_message.append("ZOOM error: ");
  check_message.append(hipGetErrorString(hip_error));
  // checks if CUDA_LAUNCH_BLOCKING in CUDA, unimplemented here for now
//   check_message.append(c10::cuda::get_cuda_check_suffix());
  check_message.append("\n");
  // TODO: similarly here, no device side checks because no kernel registry
//   if (include_device_assertions) {
//     check_message.append(c10_retrieve_device_side_assertion_info());
//   } else {
//     check_message.append(
//         "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
//   }
  check_message.append(
        "Device-side assertions were explicitly omitted for this error check; the error probably arose while initializing the DSA handlers.");
#endif

  TORCH_CHECK(false, check_message);
}

} // namespace c10::zoom
