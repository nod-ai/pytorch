#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include "ZoomDefines.h"
#include <hipblas/hipblas.h>

// Note [CHECK macro]
// ~~~~~~~~~~~~~~~~~~
// This is a macro so that AT_ERROR can get accurate __LINE__
// and __FILE__ information.  We could split this into a short
// macro and a function implementation if we pass along __LINE__
// and __FILE__, but no one has found this worth doing.

// Used to denote errors from CUDA framework.
// This needs to be declared here instead util/Exception.h for proper conversion
// during hipify.
namespace c10 {
class ZoomError : public c10::Error {
  using Error::Error;
};
} // namespace c10

#define C10_ZOOM_CHECK(EXPR)                                        \
  do {                                                              \
    const hipError_t __err = EXPR;                                 \
    c10::zoom::c10_zoom_check_implementation(                       \
        static_cast<int32_t>(__err),                                \
        __FILE__,                                                   \
        __func__, /* Line number data type not well-defined between \
                      compilers, so we perform an explicit cast */  \
        static_cast<uint32_t>(__LINE__),                            \
        true);                                                      \
  } while (0)

#define C10_ZOOM_CHECK_WARN(EXPR)                              \
  do {                                                         \
    const hipError_t __err = EXPR;                            \
    if (C10_UNLIKELY(__err != hipSuccess)) {                  \
      auto error_unused C10_UNUSED = hipGetLastError();       \
      (void)error_unused;                                      \
      TORCH_WARN("ZOOM warning: ", hipGetErrorString(__err)); \
    }                                                          \
  } while (0)

// Indicates that a CUDA error is handled in a non-standard way
#define C10_ZOOM_ERROR_HANDLED(EXPR) EXPR

// Intentionally ignore a CUDA error
#define C10_ZOOM_IGNORE_ERROR(EXPR)                             \
  do {                                                          \
    const hipError_t __err = EXPR;                             \
    if (C10_UNLIKELY(__err != hipSuccess)) {                   \
      hipError_t error_unused C10_UNUSED = hipGetLastError(); \
      (void)error_unused;                                       \
    }                                                           \
  } while (0)

// Clear the last CUDA error
#define C10_ZOOM_CLEAR_ERROR()                                \
  do {                                                        \
    hipError_t error_unused C10_UNUSED = hipGetLastError(); \
    (void)error_unused;                                       \
  } while (0)

// This should be used directly after every kernel launch to ensure
// the launch happened correctly and provide an early, close-to-source
// diagnostic if it didn't.
#define C10_ZOOM_KERNEL_LAUNCH_CHECK() C10_ZOOM_CHECK(hipGetLastError())

/// Launches a CUDA kernel appending to it all the information need to handle
/// device-side assertion failures. Checks that the launch was successful.
// #define TORCH_DSA_KERNEL_LAUNCH(                                      \
//     kernel, blocks, threads, shared_mem, stream, ...)                 \
//   do {                                                                \
//     auto& launch_registry =                                           \
//         c10::cuda::CUDAKernelLaunchRegistry::get_singleton_ref();     \
//     kernel<<<blocks, threads, shared_mem, stream>>>(                  \
//         __VA_ARGS__,                                                  \
//         launch_registry.get_uvm_assertions_ptr_for_current_device(),  \
//         launch_registry.insert(                                       \
//             __FILE__, __FUNCTION__, __LINE__, #kernel, stream.id())); \
//     C10_CUDA_KERNEL_LAUNCH_CHECK();                                   \
//   } while (0)

#define HIP_DRIVER_CHECK(EXPR)                                                \
  do {                                                                            \
    hipError_t __err = EXPR;                                                        \
    if (__err != hipSuccess) {                                                  \
      AT_ERROR("HIP driver error: ", static_cast<int>(__err));                   \
    }                                                                             \
  } while (0)

#define ZOOM_HIPRTC_CHECK(EXPR)                                       \
  do {                                                                                              \
    hiprtcResult __err = EXPR;                                                                       \
    if (__err != HIPRTC_SUCCESS) {                                                                   \
      if (static_cast<int>(__err) != 7) {                                                           \
        AT_ERROR("HIPRTC error: ", hiprtcGetErrorString(__err));  \
      } else {                                                                                      \
        AT_ERROR("HIPRTC error: HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE");                        \
      }                                                                                             \
    }                                                                                               \
  } while (0)


#define ZOOM_KERNEL_ASSERT(cond)                                         \
  if (C10_UNLIKELY(!(cond))) {                                           \
    __assert_fail(                                                       \
        #cond, __FILE__, static_cast<unsigned int>(__LINE__), __func__); \
  }


namespace at::zoom::blas {
  const char* _hipblasGetErrorEnum(hipblasStatus_t error);
}


#define TORCH_HIPBLAS_CHECK(EXPR)                              \
do {                                                          \
  hipblasStatus_t __err = EXPR;                                \
  TORCH_CHECK(__err == HIPBLAS_STATUS_SUCCESS,                 \
              "HIP error: ",                                 \
              at::zoom::blas::_hipblasGetErrorEnum(__err),     \
              " when calling `" #EXPR "`");                   \
} while (0)

#define TORCH_WARN_DISABLE_HIPBLASLT TORCH_WARN_ONCE("hipblasLt temporarily disabled in Zoom backend, using hipblas instead")
#define TORCH_CHECK_DISABLE_HIPBLAS_LT TORCH_CHECK(false, "Error: hipblasLt routine called, but hipblasLt is disabled in the Zoom backend")

namespace c10::zoom {

/// In the event of a CUDA failure, formats a nice error message about that
/// failure and also checks for device-side assertion failures
void c10_zoom_check_implementation(
    const int32_t err,
    const char* filename,
    const char* function_name,
    const int line_number,
    const bool include_device_assertions);

} // namespace c10::zoom
