#pragma once
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include "ZoomDefines.h"
#include "ZoomFunctions.h"
#include <iostream>

namespace c10::zoom {

    struct ZoomAllocator final : at::Allocator {
        ZoomAllocator() = default;
        at::DataPtr allocate(size_t nbytes) override {
            void* devicePtr = nullptr;

            if(nbytes > 0)
                HIP_ASSERT(hipMalloc((void**)&devicePtr, nbytes));

            return {devicePtr, devicePtr, &local_delete, at::Device(at::DeviceType::PrivateUse1)};
        }

        static void local_delete(void* ptr) {
            if (!ptr) {
            return;
            }

            HIP_ASSERT(hipFree(ptr));
        }

        at::DeleterFnPtr raw_deleter() const override {
            return &local_delete;
        }

        void copy_data(void* dest, const void* src, std::size_t count) const override {
            HIP_ASSERT(hipMemcpy(dest, src, count, hipMemcpyDeviceToDevice));
        }
    };

}