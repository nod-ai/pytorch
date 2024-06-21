#include "extension.h"


c10::Device get_custom_device(int idx) {
  return c10::Device(c10::DeviceType::PrivateUse1, idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_device", &get_custom_device, "get custom device object");
}