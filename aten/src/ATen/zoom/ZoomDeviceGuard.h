#pragma once

#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <aten/src/ATen/zoom/ZoomDefines.h>

#define CHECK_ZOOM_DEVICE(d) \
TORCH_INTERNAL_ASSERT(d.type() == at::DeviceType::PrivateUse1); \
TORCH_INTERNAL_ASSERT(d.index() < deviceCount(), "Error: device index ", d.index(), " does not exist.");

namespace c10::zoom {

    struct ZoomDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
        ZoomDeviceGuardImpl() {}
        explicit ZoomDeviceGuardImpl(c10::DeviceType t) {
            TORCH_INTERNAL_ASSERT(t == c10::DeviceType::PrivateUse1);
        }
        at::DeviceType type() const override {
            return at::DeviceType::PrivateUse1;
        }
        at::Device exchangeDevice(at::Device d) const override {
            CHECK_ZOOM_DEVICE(d);
            at::Device prev_device = getDevice();
            if(prev_device.index() != d.index()) {
                setDevice(d);
            }
            return prev_device;
        }
        at::Device getDevice() const override {
            // writes the default device id for the calling thread to curr_device
            int curr_device;
            HIP_ASSERT(hipGetDevice(&curr_device));
            return at::Device(at::DeviceType::PrivateUse1, curr_device);
        }
        void setDevice(at::Device d) const override {
            CHECK_ZOOM_DEVICE(d);
            // sets the default device for the calling thread to deviceId
            HIP_ASSERT(hipSetDevice(d.index()));
        }
        void uncheckedSetDevice(at::Device d) const noexcept override {
            hipSetDevice(d.index());
        }
        c10::Stream getStream(at::Device d) const noexcept override{
            // TODO
            return Stream(Stream::DEFAULT, Device(c10::DeviceType::PrivateUse1, -1));
        } 
        Stream getNewStream(Device, int priority = 0) const override {
            // TODO 
            (void)priority;
            return Stream(Stream::DEFAULT, Device(c10::DeviceType::PrivateUse1, -1));
        }
        Stream exchangeStream(Stream) const noexcept override {
            // TODO
            return Stream(Stream::DEFAULT, Device(c10::DeviceType::PrivateUse1, -1));
        }
        c10::DeviceIndex deviceCount() const noexcept override {
            int deviceCount;
            hipError_t result = hipGetDeviceCount(&deviceCount);
            // we can not raise an exception here, so just report zero devices
            if(result != hipSuccess) {
                return 0;
            }
            return deviceCount;
        }

    };

}
