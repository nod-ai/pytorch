#include <torch/csrc/python_headers.h>

#include <pybind11/chrono.h>

#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>

#include <ATen/zoom/HIPGraph.h>
#include <c10/zoom/HIPGraphsC10Utils.h>

// Cargo culted partially from csrc/distributed/c10d/init.cpp
// and partially from csrc/zoom/Stream.cpp.
// THCPStream_init is also declared at global scope.

// Because THCPGraph_init is forward declared in the only consumer
// (csrc/Module.cpp) I don't think we need a Graph.h.

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraph_init(PyObject* module) {
  // Pybind11 patch notes say "py::module_" is more up-to-date syntax,
  // but CI linter and some builds prefer "module".
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_graph_pool_handle", &::at::zoom::graph_pool_handle);

  shared_ptr_class_<::at::zoom::HIPGraph>(torch_C_m, "_HIPGraph")
      .def(py::init<>())
      .def(
          "capture_begin",
          [](::at::zoom::HIPGraph& self,
             std::optional<c10::zoom::MempoolId_t> pool_opt,
             std::string capture_error_mode) {
            hipStreamCaptureMode capture_mode;
            c10::zoom::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::zoom::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = hipStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = hipStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = hipStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::zoom::HIPGraph::capture_end))
      .def(
          "register_generator_state",
          [](::at::zoom::HIPGraph& self, py::handle raw_generator) {
            auto generator = THPGenerator_Unwrap(raw_generator.ptr());
            // We've unwrapped Python object to C++ object,
            // so we could release GIL before calling into C++
            py::gil_scoped_release release;
            return self.register_generator_state(generator);
          },
          py::arg("generator"))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::zoom::HIPGraph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::zoom::HIPGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::zoom::HIPGraph::pool))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::zoom::HIPGraph::debug_dump))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::zoom::HIPGraph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::zoom::HIPGraph::debug_dump),
          py::arg("debug_path"));
}
