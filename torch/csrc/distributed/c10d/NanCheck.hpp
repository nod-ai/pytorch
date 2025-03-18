#pragma once

#ifdef USE_C10D_NCCL

#include <ATen/ATen.h>
#ifdef USE_ZOOM
#include <c10/zoom/ZoomStream.h>
using GPUStream = c10::zoom::ZoomStream;
#else
#include <c10/cuda/CUDAStream.h>
using GPUStream = at::cuda::CUDAStream;
#endif

namespace c10d {

// Check for NaNs in a tensor on a given stream. If any are found, throw a
// device-side error.
void checkForNan(const at::Tensor& tensor, GPUStream& stream);

} // namespace c10d

#endif // USE_C10D_NCCL
