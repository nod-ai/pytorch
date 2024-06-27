#include "Copy.h"

namespace at::detail {

    at::Tensor zoom_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
        const at::OptionalDeviceGuard device_guard(at::device_of(self));
        hipMemcpyKind memcpyKind;
        // host to device
        if(self.is_cpu() && dst.device().type() == c10::DeviceType::PrivateUse1) {
            memcpyKind = hipMemcpyHostToDevice;
        }
        // device to host
        else if(dst.is_cpu() && self.device().type() == c10::DeviceType::PrivateUse1) {
            memcpyKind = hipMemcpyDeviceToHost;
        }
        // device to device
        else if(self.device().type() == c10::DeviceType::PrivateUse1 && dst.device().type() == c10::DeviceType::PrivateUse1) {
            memcpyKind = hipMemcpyDeviceToDevice;
        }
        else {
            TORCH_CHECK(false, "Zoom only allows copy between HIP Devices or between CPU and HIP Devices");
        }

        // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
        TORCH_CHECK(self.sizes() == dst.sizes());
        TORCH_CHECK(self.scalar_type() == dst.scalar_type());
        TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

        

        hipMemcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes(), memcpyKind);
        return dst;
    }


    // Copy kernels only support overriding the registered DispatchStub for a hardcoded set of in-tree devices,
    // in principle we can do that rather than using TORCH_LIBRARY_IMPL if we add zoom as a 'supported device'
    // this is a TODO for once we're in-tree
    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("_copy_from", &zoom_copy_from);
    }

}