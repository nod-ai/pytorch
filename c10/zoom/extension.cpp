#include <c10/zoom/extension.h>


c10::Device get_custom_device(int idx) {
  return c10::Device(c10::DeviceType::PrivateUse1, idx);
}

void init_zoom() {
  at::zoom::detail::register_zoom_hooks();
}

PYBIND11_MODULE(torch_zoom, m) {
    m.def("init_zoom", &init_zoom, "Register module under PrivateUse1 Dispatch");
    m.def("custom_device", &get_custom_device, "get custom device object");
}