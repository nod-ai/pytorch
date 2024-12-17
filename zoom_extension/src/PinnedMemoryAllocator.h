#pragma once

#include <c10/core/Allocator.h>
#include "CachingHostAllocator.h"

namespace at::zoom {

inline at::Allocator* getPinnedMemoryAllocator() {
  return getCachingHostAllocator();
}
} // namespace at::zoom