#include <pybind11/pybind11.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/zoom/Event.h>
#include <torch/csrc/zoom/Module.h>
#include <torch/csrc/zoom/Stream.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include <c10/zoom/ZoomGuard.h>

#include <hip/hip_runtime_api.h>
#include <structmember.h>

PyObject* THCPEventClass = nullptr;

static PyObject* THCPEvent_pynew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  unsigned char enable_timing = 0;
  unsigned char blocking = 0;
  unsigned char interprocess = 0;

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "enable_timing", "blocking", "interprocess", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|bbb",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &enable_timing,
          &blocking,
          &interprocess)) {
    return nullptr;
  }

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }

  THCPEvent* self = (THCPEvent*)ptr.get();
  unsigned int flags = (blocking ? hipEventBlockingSync : hipEventDefault) |
      (enable_timing ? hipEventDefault : hipEventDisableTiming) |
      (interprocess ? hipEventInterprocess : hipEventDefault);

  new (&self->zoom_event) at::zoom::ZoomEvent(flags);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPEvent_from_ipc_handle(
    PyObject* _type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto type = (PyTypeObject*)_type;

  static torch::PythonArgParser parser({
      "from_ipc_handle(Device device, std::string ipc_handle)",
  });
  torch::ParsedArgs<2> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);

  at::Device device = r.device(0);
  std::string handle_string = r.string(1);

  TORCH_CHECK(
      handle_string.size() == sizeof(hipIpcEventHandle_t),
      "hipIpcEventHandle_t expects byte-like object of size ",
      sizeof(hipIpcEventHandle_t),
      ", but got ",
      handle_string.size());
  TORCH_CHECK(
      device.type() == at::kPrivateUse1,
      "Event can only be created on "
      "Zoom devices, but got device type ",
      device.type())

  THPObjectPtr ptr(type->tp_alloc(type, 0));
  if (!ptr) {
    return nullptr;
  }
  THCPEvent* self = (THCPEvent*)ptr.get();

  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  hipIpcEventHandle_t handle;
  std::memcpy(&handle, handle_string.c_str(), handle_string.size());
  new (&self->zoom_event) at::zoom::ZoomEvent(device.index(), &handle);

  return (PyObject*)ptr.release();
  END_HANDLE_TH_ERRORS
}

static void THCPEvent_dealloc(THCPEvent* self) {
  {
    pybind11::gil_scoped_release no_gil{};
    self->zoom_event.~ZoomEvent();
  }
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject* THCPEvent_get_zoom_event(THCPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  return PyLong_FromVoidPtr(self->zoom_event.event());
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPEvent_get_device(THCPEvent* self, void* unused) {
  HANDLE_TH_ERRORS
  at::optional<at::Device> device = self->zoom_event.device();
  if (!device) {
    Py_RETURN_NONE;
  }
  return THPDevice_New(device.value());
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPEvent_record(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS
  auto self = (THCPEvent*)_self;
  auto stream = (THCPStream*)_stream;
  self->zoom_event.record(stream->zoom_stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPEvent_wait(PyObject* _self, PyObject* _stream) {
  HANDLE_TH_ERRORS {
    auto self = (THCPEvent*)_self;
    auto stream = (THCPStream*)_stream;
    pybind11::gil_scoped_release no_gil{};
    self->zoom_event.block(stream->zoom_stream);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPEvent_query(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THCPEvent*)_self;
  return PyBool_FromLong(self->zoom_event.query());
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPEvent_elapsed_time(PyObject* _self, PyObject* _other) {
  HANDLE_TH_ERRORS
  auto self = (THCPEvent*)_self;
  auto other = (THCPEvent*)_other;
  return PyFloat_FromDouble(self->zoom_event.elapsed_time(other->zoom_event));
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPEvent_synchronize(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    auto self = (THCPEvent*)_self;
    pybind11::gil_scoped_release no_gil{};
    self->zoom_event.synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPEvent_ipc_handle(PyObject* _self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto self = (THCPEvent*)_self;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  hipIpcEventHandle_t handle;
  self->zoom_event.ipc_handle(&handle);
  return PyBytes_FromStringAndSize((const char*)&handle, sizeof(handle));
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static struct PyGetSetDef THCPEvent_properties[] = {
    {"device", (getter)THCPEvent_get_device, nullptr, nullptr, nullptr},
    {"zoom_event", (getter)THCPEvent_get_zoom_event, nullptr, nullptr, nullptr},
    {nullptr}};

// NOLINTNEXTLINE(*c-arrays*, *global-variables)
static PyMethodDef THCPEvent_methods[] = {
    {(char*)"from_ipc_handle",
     castPyCFunctionWithKeywords(THCPEvent_from_ipc_handle),
     METH_CLASS | METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {(char*)"record", THCPEvent_record, METH_O, nullptr},
    {(char*)"wait", THCPEvent_wait, METH_O, nullptr},
    {(char*)"query", THCPEvent_query, METH_NOARGS, nullptr},
    {(char*)"elapsed_time", THCPEvent_elapsed_time, METH_O, nullptr},
    {(char*)"synchronize", THCPEvent_synchronize, METH_NOARGS, nullptr},
    {(char*)"ipc_handle", THCPEvent_ipc_handle, METH_NOARGS, nullptr},
    {nullptr}};

PyTypeObject THCPEventType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch._C._ZoomEventBase", /* tp_name */
    sizeof(THCPEvent), /* tp_basicsize */
    0, /* tp_itemsize */
    (destructor)THCPEvent_dealloc, /* tp_dealloc */
    0, /* tp_vectorcall_offset */
    nullptr, /* tp_getattr */
    nullptr, /* tp_setattr */
    nullptr, /* tp_reserved */
    nullptr, /* tp_repr */
    nullptr, /* tp_as_number */
    nullptr, /* tp_as_sequence */
    nullptr, /* tp_as_mapping */
    nullptr, /* tp_hash  */
    nullptr, /* tp_call */
    nullptr, /* tp_str */
    nullptr, /* tp_getattro */
    nullptr, /* tp_setattro */
    nullptr, /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE, /* tp_flags */
    nullptr, /* tp_doc */
    nullptr, /* tp_traverse */
    nullptr, /* tp_clear */
    nullptr, /* tp_richcompare */
    0, /* tp_weaklistoffset */
    nullptr, /* tp_iter */
    nullptr, /* tp_iternext */
    THCPEvent_methods, /* tp_methods */
    nullptr, /* tp_members */
    THCPEvent_properties, /* tp_getset */
    nullptr, /* tp_base */
    nullptr, /* tp_dict */
    nullptr, /* tp_descr_get */
    nullptr, /* tp_descr_set */
    0, /* tp_dictoffset */
    nullptr, /* tp_init */
    nullptr, /* tp_alloc */
    THCPEvent_pynew, /* tp_new */
};

void THCPEvent_init(PyObject* module) {
  THCPEventClass = (PyObject*)&THCPEventType;
  if (PyType_Ready(&THCPEventType) < 0) {
    throw python_error();
  }
  Py_INCREF(&THCPEventType);
  if (PyModule_AddObject(module, "_ZoomEventBase", (PyObject*)&THCPEventType) <
      0) {
    throw python_error();
  }
}
