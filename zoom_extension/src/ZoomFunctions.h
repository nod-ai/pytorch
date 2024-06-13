#pragma once
#include "ZoomDefines.h"
#include <c10/core/Device.h>
#include <limits>

namespace c10::zoom {

    hipError_t GetDevice(DeviceIndex* deviceIndex);

}