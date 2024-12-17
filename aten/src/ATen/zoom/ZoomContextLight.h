#pragma once
// Light-weight version of ZoomContext.h with fewer transitive includes
#define DISABLE_HIPBLASLT

#include <cstdint>

#include <hip/hip_runtime.h>
#include <c10/core/Allocator.h>
#include <c10/zoom/ZoomFunctions.h>

#include <hipsparse/hipsparse.h>
#include <hipsolver/hipsolver.h>
#include <hipblas/hipblas.h>
#ifndef DISABLE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#endif

namespace c10 {
struct Allocator;
}

namespace at::zoom {

/*
A common CUDA interface for ATen.

This interface is distinct from CUDAHooks, which defines an interface that links
to both CPU-only and CUDA builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
CUDA builds.

CUDAContext, on the other hand, should be preferred by files only included in
CUDA builds. It is intended to expose CUDA functionality in a consistent
manner.

This means there is some overlap between the CUDAContext and CUDAHooks, but
the choice of which to use is simple: use CUDAContext when in a CUDA-only file,
use CUDAHooks otherwise.

Note that CUDAContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single CUDA context/state.
*/

/**
 * DEPRECATED: use device_count() instead
 */
inline int64_t getNumGPUs() {
    return c10::zoom::device_count();
}

/**
 * CUDA is available if we compiled with CUDA, and there are one or more
 * devices.  If we compiled with CUDA but there is a driver problem, etc.,
 * this function will report CUDA is not available (rather than raise an error.)
 */
inline bool is_available() {
    return c10::zoom::device_count() > 0;
}

TORCH_ZOOM_API hipDeviceProp_t* getCurrentDeviceProperties();

TORCH_ZOOM_API int warp_size();

TORCH_ZOOM_API hipDeviceProp_t* getDeviceProperties(c10::DeviceIndex device);

TORCH_ZOOM_API bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device);

TORCH_ZOOM_API c10::Allocator* getZoomDeviceAllocator();

// TORCH_ZOOM_API hipsparseHandle_t getCurrentHIPSparseHandle();
// TORCH_ZOOM_API hipblasHandle_t getCurrentHIPBlasHandle();
// #ifndef DISABLE_HIPBLASLT
// TORCH_ZOOM_API hipblasLtHandle_t getCurrentHIPBlasLtHandle();
// #endif
//
//
// #if defined(hipsolverVersionMajor)
// TORCH_ZOOM_API hipsolverDnHandle_t getCurrentHIPSolverDnHandle();
// #endif

} // namespace at::zoom