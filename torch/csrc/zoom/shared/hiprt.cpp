#include <torch/csrc/utils/pybind.h>
#include <hip/hip_runtime_api.h>
#include <c10/zoom/ZoomException.h>
#include <c10/zoom/ZoomGuard.h>

namespace torch::zoom::shared {

namespace {
hipError_t hipReturnSuccess() {
  return hipSuccess;
}
} // namespace

void initHiprtBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto hiprt = m.def_submodule("_hiprt", "hip runtime bindings");

  py::enum_<hipError_t>(
      hiprt,
      "hip"
      "Error")
      .value("success", hipSuccess);

  hiprt.def(
      "hip"
      "GetErrorString",
      hipGetErrorString);
  hiprt.def(
      "hip"
      "ProfilerStart",
      hipReturnSuccess
  );
  hiprt.def(
      "hip"
      "ProfilerStop",
      hipReturnSuccess
  );
  hiprt.def(
      "hip"
      "HostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> hipError_t {
        return C10_ZOOM_ERROR_HANDLED(
            hipHostRegister((void*)ptr, size, flags));
      });
  hiprt.def(
      "hip"
      "HostUnregister",
      [](uintptr_t ptr) -> hipError_t {
        return C10_ZOOM_ERROR_HANDLED(hipHostUnregister((void*)ptr));
      });
  hiprt.def(
      "hip"
      "StreamCreate",
      [](uintptr_t ptr) -> hipError_t {
        return C10_ZOOM_ERROR_HANDLED(hipStreamCreate((hipStream_t*)ptr));
      });
  hiprt.def(
      "hip"
      "StreamDestroy",
      [](uintptr_t ptr) -> hipError_t {
        return C10_ZOOM_ERROR_HANDLED(hipStreamDestroy((hipStream_t)ptr));
      });
  hiprt.def(
      "hip"
      "MemGetInfo",
      [](c10::DeviceIndex device) -> std::pair<size_t, size_t> {
        c10::zoom::ZoomGuard guard(device);
        size_t device_free = 0;
        size_t device_total = 0;
        C10_ZOOM_CHECK(hipMemGetInfo(&device_free, &device_total));
        return {device_free, device_total};
      });
}

} // namespace torch::zoom::shared
