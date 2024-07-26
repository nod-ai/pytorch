// !!! This is a file automatically generated by hipify!!!
#pragma once
#include <ATen/zoom/ZoomException.h>
#include <ATen/zoom/ZoomContext.h>
// #include <ATen/hip/HIPConfig.h>
#include <ATen/zoom/PinnedMemoryAllocator.h>

namespace at {
namespace native {

static inline int zoom_int_cast(int64_t value, const char* varname) {
  auto result = static_cast<int>(value);
  TORCH_CHECK(static_cast<int64_t>(result) == value,
              "zoom_int_cast: The value of ", varname, "(", (long long)value,
              ") is too large to fit into a int (", sizeof(int), " bytes)");
  return result;
}

// Creates an array of size elements of type T, backed by pinned memory
// wrapped in a Storage
template<class T>
static inline Storage pin_memory(int64_t size) {
  auto* allocator = zoom::getPinnedMemoryAllocator();
  int64_t adjusted_size = size * sizeof(T);
  return Storage(
      Storage::use_byte_size_t(),
      adjusted_size,
      allocator,
      /*resizable=*/false);
}

} // namespace native
} // namespace at
