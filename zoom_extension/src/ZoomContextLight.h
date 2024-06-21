#pragma once
// Light-weight version of CUDAContext.h with fewer transitive includes

#include <cstdint>

#include <hip/hip_runtime.h>

#ifdef HIPSOLVER_VERSION
#include <hipsolver/hipsolver.h>
#endif

#include <c10/core/Allocator.h>
#include "ZoomFunctions.h"

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

hipDeviceProp_t* getCurrentDeviceProperties();

int warp_size();

hipDeviceProp_t* getDeviceProperties(c10::DeviceIndex device);

bool canDeviceAccessPeer(
    c10::DeviceIndex device,
    c10::DeviceIndex peer_device);

c10::Allocator* getZoomDeviceAllocator();


// TODO (Arham), optionally integrate hipsolver libs
#if defined(HIPSOLVER_VERSION)
hipsolverDnHandle_t getCurrentHIPSolverDnHandle();
#endif

} // namespace at::zoom