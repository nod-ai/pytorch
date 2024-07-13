#pragma once

#include <ATen/core/CachingHostAllocator.h>
#include <c10/core/Allocator.h>
#include <ATen/zoom/detail/ZoomHooks.h>
#include <ATen/detail/ZoomHooksInterface.h>
#include <c10/zoom/ZoomStream.h>

namespace at::zoom {

//
// A caching allocator for CUDA host allocations (pinned memory).
//
// This provides a drop-in replacement for THCudaHostAllocator, which re-uses
// freed pinned (page-locked) memory allocations. This avoids device
// synchronizations due to cudaFreeHost calls.
//
// To ensure correct behavior, THCCachingHostAllocator_recordEvent must be
// called anytime a pointer from this allocator is used in a cudaMemcpyAsync
// call between host and device, and passed the corresponding context from the
// allocation. This is currently invoked by at::native::copy_kernel_cuda.
//
c10::Allocator* getCachingHostAllocator();

// Records an event in the specified stream. The allocation corresponding to the
// input `ptr`/`ctx` will not be re-used until the event has occurred.
bool CachingHostAllocator_recordEvent(
    void* ptr,
    void* ctx,
    c10::zoom::ZoomStream stream);

// Releases cached pinned memory allocations via cudaHostFree
void CachingHostAllocator_emptyCache();

inline at::DataPtr HostAlloc(size_t size) {
  return getCachingHostAllocator()->allocate(size);
}

} // namespace at::zoom