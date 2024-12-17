#pragma once

#include <ATen/zoom/ATenZoomGeneral.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

namespace at { namespace zoom {


// NOTE [ USE OF NVRTC AND DRIVER API ]
//
// ATen does not directly link to either libnvrtc or libcuda because they
// require libcuda to be installed, yet we want our GPU build to work on CPU
// machines as long as CUDA is not initialized.
//
// Normal CUDA code in torch uses the cuda runtime libraries which can be
// installed even if the driver is not installed, but sometimes we specifically
// need to use the driver API (e.g., to load JIT compiled code).
// To accomplish this, we lazily link libcaffe2_nvrtc which provides a struct
// at::zoom::HIPRTC that contains function pointers to all of the apis we need.
//
// IT IS AN ERROR TO TRY TO CALL ANY nvrtc* or cu* FUNCTION DIRECTLY.
// INSTEAD USE, e.g.
//   detail::getZoomHooks().nvrtc().cuLoadModule(...)
// or
//   globalContext().getNVRTC().cuLoadModule(...)
//
// If a function is missing add it to the list in ATen/cuda/nvrtc_stub/ATenNVRTC.h
// and edit ATen/cuda/detail/LazyNVRTC.cpp accordingly (e.g., via one of the stub
// macros).


// NOTE [ ATen NVRTC Stub and HIP ]
//
// ATen's NVRTC stub library, caffe2_nvrtc, provides dynamic loading of both
// NVRTC and driver APIs. While the former is not yet supported for HIP, the
// later is supported and needed (e.g., in CUDAHooks::getDeviceWithPrimaryContext()
// used by tensor.pin_memory()).
//
// The macro below strips out certain unsupported operations on HIP from the full
// list above.
//
// HIP doesn't have
//   cuGetErrorString  (maps to non-functional hipGetErrorString___)
//
// HIP from ROCm 3.5 on renamed hipOccupancyMaxActiveBlocksPerMultiprocessor
// to hipModuleOccupancyMaxActiveBlocksPerMultiprocessor.
// #if TORCH_HIP_VERSION < 305
// #define HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR hipOccupancyMaxActiveBlocksPerMultiprocessor
// #else
// #define HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR hipModuleOccupancyMaxActiveBlocksPerMultiprocessor
// #endif

#define HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR hipModuleOccupancyMaxActiveBlocksPerMultiprocessor

#define AT_FORALL_HIPRTC(_)                        \
  _(hiprtcVersion)                                 \
  _(hiprtcCreateProgram)                           \
  _(hiprtcAddNameExpression)                       \
  _(hiprtcDestroyProgram)                          \
  _(hiprtcGetCodeSize)                              \
  _(hiprtcGetCode)                                  \
  _(hipModuleLoadData)                             \
  _(hipModuleGetFunction)                          \
  _(HIPOCCUPANCYMAXACTIVEBLOCKSPERMULTIPROCESSOR) \
  _(hiprtcGetErrorString)                          \
  _(hiprtcGetProgramLogSize)                       \
  _(hiprtcGetProgramLog)                           \
  _(hipModuleLaunchKernel)                               \
  _(hiprtcCompileProgram)                          \
  _(hipCtxGetCurrent)                              \
  _(hiprtcGetLoweredName)                          \
  _(hipModuleUnload)                               \
  _(hipDevicePrimaryCtxGetState)



extern "C" typedef struct HIPRTC {
#define CREATE_MEMBER(name) decltype(&name) name;
  AT_FORALL_HIPRTC(CREATE_MEMBER)
#undef CREATE_MEMBER
} HIPRTC;

extern "C" TORCH_ZOOM_API HIPRTC* load_hiprtc();
}} // at::zoom
