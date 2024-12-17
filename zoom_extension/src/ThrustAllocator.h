#pragma once

#include <cstddef>
#include "ZoomCachingAllocator.h"

namespace at::zoom {

/// Allocator for Thrust to re-route its internal device allocations
/// to the THC allocator
class ThrustAllocator {
public:
  typedef char value_type;

  char* allocate(std::ptrdiff_t size) {
    return static_cast<char*>(c10::zoom::ZoomCachingAllocator::raw_alloc(size));
  }

  void deallocate(char* p, size_t size) {
    c10::zoom::ZoomCachingAllocator::raw_delete(p);
  }
};

} // namespace at::zoom