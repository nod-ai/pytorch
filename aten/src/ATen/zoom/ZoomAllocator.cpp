#include <ATen/zoom/ZoomAllocator.h>

namespace c10::zoom {

    static ZoomAllocator zoom_alloc;
    REGISTER_ALLOCATOR(c10::DeviceType::PrivateUse1, &zoom_alloc);

}