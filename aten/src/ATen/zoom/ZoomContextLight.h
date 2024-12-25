#pragma once
// Light-weight version of ZoomContext.h with fewer transitive includes
#include <cstdint>
#include <hip/hip_runtime.h>
#include <c10/core/Allocator.h>
#include <c10/zoom/ZoomFunctions.h>

namespace c10 {
struct Allocator;
}

namespace at::zoom {

/*
A common Zoom interface for ATen.

This interface is distinct from ZoomHooks, which defines an interface that links
to both CPU-only and Zoom builds. That interface is intended for runtime
dispatch and should be used from files that are included in both CPU-only and
Zoom builds.

ZoomContext, on the other hand, should be preferred by files only included in
Zoom builds. It is intended to expose Zoom functionality in a consistent
manner.

This means there is some overlap between the ZoomContext and ZoomHooks, but
the choice of which to use is simple: use ZoomContext when in a Zoom-only file,
use ZoomHooks otherwise.

Note that ZoomContext simply defines an interface with no associated class.
It is expected that the modules whose functions compose this interface will
manage their own state. There is only a single Zoom context/state.
*/

/**
 * DEPRECATED: use device_count() instead
 */
inline int64_t getNumGPUs() {
    return c10::zoom::device_count();
}

/**
 * Zoom is available if we compiled with Zoom, and there are one or more
 * devices.  If we compiled with Zoom but there is a driver problem, etc.,
 * this function will report Zoom is not available (rather than raise an error.)
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

} // namespace at::zoom