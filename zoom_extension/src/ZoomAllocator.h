#pragma once
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <c10/util/CallOnce.h>
#include "ZoomDefines.h"
#include "ZoomException.h"
#include "ZoomFunctions.h"
#include <iostream>
#include <atomic>
#include <mutex>

namespace c10::zoom {

    struct ZoomAllocator final : at::Allocator {
        std::atomic<ZoomAllocator*> allocator;

        ZoomAllocator() = default;

        at::DataPtr allocate(size_t nbytes) override {
            void* devicePtr = nullptr;
            DeviceIndex device = 0;
            C10_ZOOM_CHECK(c10::zoom::GetDevice(&device));

            if(nbytes > 0)
                HIP_ASSERT(hipMalloc((void**)&devicePtr, nbytes));

            return {devicePtr, devicePtr, &local_delete, at::Device(at::DeviceType::PrivateUse1, device)};
        }

        static void local_delete(void* ptr) {
            const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
            if (C10_UNLIKELY(interp)) {
                (*interp)->trace_gpu_memory_deallocation(
                    c10::DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(ptr));
            }
            C10_ZOOM_CHECK(hipFree(ptr));
        }

        at::DeleterFnPtr raw_deleter() const override {
            return &local_delete;
        }

        void copy_data(void* dest, const void* src, std::size_t count) const override {
            C10_ZOOM_CHECK(hipMemcpy(dest, src, count, hipMemcpyDeviceToDevice));
        }

    };

}