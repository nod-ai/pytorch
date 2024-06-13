#pragma once

#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <aten/src/ATen/zoom/ZoomDefines.h>
#include <iostream>

namespace c10::zoom {

    struct ZoomAllocator final : at::Allocator {
        ZoomAllocator() = default;
        at::DataPtr allocate(size_t nbytes) override {
            std::cout << "ZOOM ALLOCATOR called" << std::endl;
            void* devicePtr;
            HIP_ASSERT(hipMalloc((void**)&devicePtr, nbytes));

            return {devicePtr, devicePtr, &ReportAndDelete, at::Device(at::DeviceType::PrivateUse1)};
        }

        static void ReportAndDelete(void* ptr) {
            if (!ptr) {
            return;
            }
            std::cout << "Custom Zoom allocator's delete() called!" << std::endl;

            HIP_ASSERT(hipFree(ptr));
        }

        at::DeleterFnPtr raw_deleter() const override {
            return &ReportAndDelete;
        }

        void copy_data(void* dest, const void* src, std::size_t count) const override {
            HIP_ASSERT(hipMemcpy(dest, src, count, hipMemcpyDeviceToDevice));
        }
    };

}