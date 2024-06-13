#include "ZoomFunctions.h"

namespace c10::zoom {

    hipError_t GetDevice(DeviceIndex* device) {
        int tmp_device =- 1;
        auto err = hipGetDevice(&tmp_device);
        if(err == hipSuccess) {
            TORCH_INTERNAL_ASSERT(
                tmp_device >= 0 &&
                    tmp_device <= std::numeric_limits<DeviceIndex>::max(),
                "hipGetDevice returns invalid device ",
                tmp_device);
            *device = static_cast<DeviceIndex>(tmp_device);
        }

        return err;
    }

}