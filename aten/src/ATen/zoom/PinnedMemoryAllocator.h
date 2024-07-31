#pragma once

#include <c10/core/Allocator.h>
#include <ATen/zoom/CachingHostAllocator.h>

namespace at::zoom {

inline TORCH_ZOOM_API at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}
} // namespace at::zoom