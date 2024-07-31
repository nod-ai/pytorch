#include <ATen/ATen.h>
#include <ATen/CachedTensorUtils.h>
#include <ATen/core/TensorBody.h>
#include <ATen/zoom/HIPConfig.h>
#include <ATen/native/ConvUtils.h>
#include <c10/core/Device.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/UniqueVoidPtr.h>
#include <pybind11/pytypes.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <unordered_set>

#include <ATen/zoom/ZoomContext.h>
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/zoom/CachingHostAllocator.h>
// #include <ATen/zoom/Sleep.h>
#include <ATen/zoom/detail/ZoomHooks.h>
// #include <ATen/zoom/jiterator.h>
#include <c10/core/StorageImpl.h>
#include <c10/zoom/ZoomAllocatorConfig.h>
#include <c10/zoom/ZoomCachingAllocator.h>
#include <c10/zoom/ZoomFunctions.h>
#include <c10/zoom/ZoomMiscFunctions.h>
#include <ATen/zoom/HIPGraphsUtils.hpp>

#include <c10/util/CallOnce.h>
#include <c10/util/irange.h>

// #include <torch/csrc/HIPIPCTypes.h>
// #include <torch/csrc/Generator.h>
// #include <torch/csrc/zoom/ZoomPluggableAllocator.h>
// #include <torch/csrc/zoom/THCP.h>
// #include <torch/csrc/zoom/memory_snapshot.h>
// #include <torch/csrc/zoom/python_comm.h>
#include <torch/csrc/profiler/python/combined_traceback.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils.h>
#include <torch/csrc/copy_utils.h>
#include <array>
#include <chrono>
#include <iostream>
#include <sstream>
#include <thread>
#include <unordered_map>
#ifndef WIN32
#include <pthread.h>
#endif

using namespace torch;

static bool in_bad_fork = false; // True for children forked after zoom init

#ifndef WIN32
// Called in the forked child if zoom has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_device_init(at::kPrivateUse1, true);
}
#endif

// Should be called before the first zoom call.
// Note: This is distinct from initExtension because a stub zoom implementation
// has some working functions (e.g. device_count) but cannot fully initialize.
static void poison_fork() {
#ifndef WIN32
  static c10::once_flag flag;
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
#endif
}

////////////////////////////////////////////////////////////////////////////////
// Zoom management methods
////////////////////////////////////////////////////////////////////////////////

PyObject* THCPModule_setDevice_wrap(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to setDevice");
  auto device = THPUtils_unpackLong(arg);

  torch::utils::device_lazy_init(at::kPrivateUse1);
  c10::zoom::set_device(static_cast<c10::DeviceIndex>(device));

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_exchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = c10::zoom::ExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_maybeExchangeDevice(PyObject* self, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (device_index < 0) {
    return THPUtils_packInt32(-1);
  }

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto current_device = c10::zoom::MaybeExchangeDevice(device_index);

  return THPUtils_packDeviceIndex(current_device);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getDevice_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  torch::utils::device_lazy_init(at::kPrivateUse1);
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int32_t>(c10::zoom::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_canDeviceAccessPeer_wrap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "can_device_peer_access",
        1,
        "(int device, int peer_device);");
    return nullptr;
  }
  TORCH_CHECK(
      THPUtils_checkLong(arg1), "invalid argument to canDeviceAccessPeer");
  TORCH_CHECK(
      THPUtils_checkLong(arg2), "invalid argument to canDeviceAccessPeer");
  int64_t device = THPUtils_unpackLong(arg1);
  int64_t peer_device = THPUtils_unpackLong(arg2);

  torch::utils::device_lazy_init(at::kPrivateUse1);
  auto can_access = at::zoom::canDeviceAccessPeer(device, peer_device);
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getDeviceCount_wrap(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  poison_fork();
  return THPUtils_packUInt64(c10::zoom::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getArchFlags(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  poison_fork();
#ifdef ROCM_ARCH_FLAGS
  static const char* flags = C10_STRINGIZE(ROCM_ARCH_FLAGS);
  return THPUtils_packString(flags);
#else
  Py_RETURN_NONE;
#endif
  END_HANDLE_TH_ERRORS
}

static PyObject* THCPModule_isInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getCurrentStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  auto stream = c10::zoom::getCurrentZoomStream(c10_device_index);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple, 1, THPUtils_packDeviceIndex(stream.device_index()));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getCurrentStream_raw(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  return PyLong_FromVoidPtr(
      c10::zoom::getCurrentZoomStream(c10_device_index).stream());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getDefaultStream_wrap(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  auto c10_device_index = THPUtils_unpackDeviceIndex(device_index);
  auto stream = c10::zoom::getDefaultZoomStream(c10_device_index);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple, 1, THPUtils_packDeviceIndex(stream.device_index()));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_setStream_wrap(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = c10::zoom::ZoomStream::unpack3(
      stream_id,
      static_cast<c10::DeviceIndex>(device_index),
      static_cast<c10::DeviceType>(device_type));

  auto device = c10::zoom::current_device();
  if (device != stream.device_index()) {
    c10::zoom::set_device(stream.device_index());
  }
  c10::zoom::setCurrentZoomStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getCompiledVersion(PyObject* self, PyObject* noargs) {
  return THPUtils_packInt64((int64_t)ROCM_VERSION);
}

PyObject* THCPModule_zoomHostAllocator(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::Allocator* allocator = at::zoom::getCachingHostAllocator();
  return PyLong_FromVoidPtr(allocator);
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_zoomCachingAllocator_raw_alloc(
    PyObject* _unused,
    PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* size_o = nullptr;
  PyObject* stream_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &size_o, &stream_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "caching_allocator_alloc",
        1,
        "(ssize_t size, intptr_t stream);");
    return nullptr;
  }
  auto size = PyLong_AsSsize_t(size_o);
  hipStream_t stream = static_cast<hipStream_t>(PyLong_AsVoidPtr(stream_o));
  void* mem = nullptr;
  {
    pybind11::gil_scoped_release no_gil;
    mem = c10::zoom::ZoomCachingAllocator::raw_alloc_with_stream(size, stream);
  }
  return PyLong_FromVoidPtr(mem);
  END_HANDLE_TH_ERRORS
}

// Unpack a PyObject to at::Scalar, throw an exception if it fails
at::Scalar as_scalar(PyObject* arg) {
  // Zero-dim tensors are converted to Scalars as-is. Note this doesn't
  // currently handle most NumPy scalar types except np.float64.
  if (THPVariable_Check(arg)) {
    return THPVariable_Unpack(arg).item();
  }

  if (THPUtils_checkLong(arg)) {
    return at::Scalar(static_cast<int64_t>(THPUtils_unpackLong(arg)));
  }

  if (PyBool_Check(arg)) {
    return at::Scalar(THPUtils_unpackBool(arg));
  }

  if (PyComplex_Check(arg)) {
    return at::Scalar(THPUtils_unpackComplexDouble(arg));
  }
  return at::Scalar(THPUtils_unpackDouble(arg));
}

// Entrypoint for the callable created by torch.zoom.jiterator
// See jiterator.py for more details
// PyObject* THCPModule_zoomJiteratorCompileAndLaunchKernel(
//     PyObject* _unused,
//     PyObject* args) {
//   HANDLE_TH_ERRORS

//   PyObject* code_string_o = nullptr;
//   PyObject* kernel_name_o = nullptr;
//   PyObject* return_by_ref_o = nullptr;
//   PyObject* num_outputs_o = nullptr;
//   PyObject* tensors_o = nullptr;
//   PyObject* kwargs_o = nullptr;
//   if (!PyArg_ParseTuple(
//           args,
//           "OOOOO|O",
//           &code_string_o,
//           &kernel_name_o,
//           &return_by_ref_o,
//           &num_outputs_o,
//           &tensors_o,
//           &kwargs_o)) {
//     return nullptr;
//   }

//   const std::string code_string = THPUtils_unpackString(code_string_o);
//   const std::string kernel_name = THPUtils_unpackString(kernel_name_o);
//   const bool return_by_ref = THPUtils_unpackBool(return_by_ref_o);
//   const int num_outputs = static_cast<int>(THPUtils_unpackLong(num_outputs_o));

//   TORCH_CHECK(
//       PyTuple_Check(tensors_o),
//       "tensors argument is expected to "
//       "be a tuple, but got ",
//       THPUtils_typename(tensors_o));
//   Py_ssize_t num_tensors = PyTuple_GET_SIZE(tensors_o);

//   c10::SmallVector<at::Tensor> tensors;
//   for (const auto i : c10::irange(num_tensors)) {
//     PyObject* _tensor = PyTuple_GET_ITEM(tensors_o, i);
//     TORCH_CHECK(
//         THPVariable_Check(_tensor),
//         i,
//         " of input tensors tuple is not a Tensor");

//     tensors.emplace_back(THPVariable_Unpack(_tensor));
//   }

//   c10::SmallVector<at::Scalar> extra_args;
//   PyObject* key = nullptr;
//   PyObject* value = nullptr;
//   Py_ssize_t pos = 0;
//   while (PyDict_Next(kwargs_o, &pos, &key, &value)) {
//     extra_args.emplace_back(as_scalar(value));
//   }

//   c10::SmallVector<at::Tensor> outputs = at::zoom::CompileAndLaunchKernel(
//       code_string,
//       kernel_name,
//       num_outputs,
//       tensors,
//       extra_args,
//       return_by_ref);

//   if (num_outputs == 1) {
//     return THPVariable_Wrap(outputs[0]);
//   } else {
//     PyObject* output_tuple = PyTuple_New(num_outputs);
//     for (int i = 0; i < num_outputs; ++i) {
//       PyTuple_SetItem(output_tuple, i, THPVariable_Wrap(outputs[i]));
//     }
//     return output_tuple;
//   }

//   END_HANDLE_TH_ERRORS
// }

PyObject* THCPModule_zoomCachingAllocator_raw_delete(
    PyObject* _unused,
    PyObject* obj) {
  HANDLE_TH_ERRORS
  void* mem_ptr = PyLong_AsVoidPtr(obj);
  {
    pybind11::gil_scoped_release no_gil;
    c10::zoom::ZoomCachingAllocator::raw_delete(mem_ptr);
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_zoomCachingAllocator_set_allocator_settings(
    PyObject* _unused,
    PyObject* env) {
  HANDLE_TH_ERRORS
  c10::zoom::ZoomCachingAllocator::setAllocatorSettings(
      THPUtils_unpackString(env));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getAllocatorBackend(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return THPUtils_packString(c10::zoom::ZoomCachingAllocator::name());
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_zoomSynchronize(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS {
    pybind11::gil_scoped_release no_gil;
    c10::zoom::device_synchronize();
  }
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

// PyObject* THCPModule_zoomIPCCollect(PyObject* _unused, PyObject* noargs) {
//   HANDLE_TH_ERRORS
//   torch::CudaIPCCollect();
//   Py_RETURN_NONE;
//   END_HANDLE_TH_ERRORS
// }

// PyObject* THCPModule_zoomSleep(PyObject* _unused, PyObject* cycles) {
//   HANDLE_TH_ERRORS
//   TORCH_CHECK(
//       THPUtils_checkLong(cycles), "torch.zoom._sleep(): expected 'int'");
//   int64_t unpacked_cycles = THPUtils_unpackLong(cycles);
//   {
//     pybind11::gil_scoped_release no_gil;
//     at::zoom::sleep(unpacked_cycles);
//   }
//   Py_RETURN_NONE;
//   END_HANDLE_TH_ERRORS
// }

// We need to ensure that as long as a thread will NEVER loose the GIL as long
// as it holds the CUDA mutex. Otherwise another thread might be scheduled and
// try to e.g. allocate a new tensor which will cause a deadlock. It's enough to
// have a single global, because it can be only set once (zoomMutex is not
// recursive) by the thread that owns the mutex (obviously there can be only one
// such thread).
static PyGILState_STATE zoomMutexGILState;

PyObject* THCPModule_zoomLockMutex(PyObject* module, PyObject* noargs) {
  auto mutex = c10::zoom::getFreeMutex();
  // This has to be a busy loop because we **absolutely need to** hold the GIL
  // or it's a recipe for a deadlock otherwise (if we let other Python threads
  // run while we have the zoomMutex, but not the GIL, they might try to e.g.
  // free a CUDA tensor and acquire the zoomMutex without giving up the GIL,
  // because it happens deep within THC).
  while (true) {
    if (mutex->try_lock())
      break;
    {
      pybind11::gil_scoped_release no_gil;
      std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
  }

  zoomMutexGILState = PyGILState_Ensure();
  Py_RETURN_NONE;
}

PyObject* THCPModule_zoomUnlockMutex(PyObject* module, PyObject* noargs) {
  auto mutex = c10::zoom::getFreeMutex();
  PyGILState_Release(zoomMutexGILState);
  mutex->unlock();
  Py_RETURN_NONE;
}

PyObject* THCPModule_hasPrimaryContext(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to has_primary_context");
  auto device_index = THPUtils_unpackDeviceIndex(arg);
  if (c10::zoom::hasPrimaryContext(device_index)) {
    Py_RETURN_TRUE;
  } else {
    Py_RETURN_FALSE;
  }
  END_HANDLE_TH_ERRORS
}

// PyObject* THCPModule_setMemoryFraction(PyObject* _unused, PyObject* args) {
//   HANDLE_TH_ERRORS
//   PyObject* fraction_o = nullptr;
//   PyObject* device_o = nullptr;
//   if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
//     THPUtils_invalidArguments(
//         args,
//         nullptr,
//         "set_memory_fraction",
//         1,
//         "(double fraction, int device);");
//     return nullptr;
//   }
//   double fraction = PyFloat_AsDouble(fraction_o);
//   auto device_index = THPUtils_unpackDeviceIndex(device_o);

//   c10::zoom::ZoomCachingAllocator::setMemoryFraction(fraction, device_index);
//   END_HANDLE_TH_ERRORS
//   Py_RETURN_NONE;
// }

PyObject* THCPModule_emptyCache(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  c10::zoom::ZoomCachingAllocator::emptyCache();
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THCPModule_memoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);

  using c10::zoom::ZoomCachingAllocator::DeviceStats;
  using c10::zoom::ZoomCachingAllocator::Stat;
  using c10::zoom::ZoomCachingAllocator::StatArray;
  using c10::zoom::ZoomCachingAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (const auto i : c10::irange(statTypeNames.size())) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      c10::zoom::ZoomCachingAllocator::getDeviceStats(device_index);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["num_sync_all_streams"] = stats.num_sync_all_streams;
  result["num_device_alloc"] = stats.num_device_alloc;
  result["num_device_free"] = stats.num_device_free;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["requested_bytes"] = statArrayToDict(stats.requested_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_resetAccumulatedMemoryStats(
    PyObject* _unused,
    PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg),
      "invalid argument to reset_accumulated_memory_stats");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);
  c10::zoom::ZoomCachingAllocator::resetAccumulatedStats(device_index);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* THCPModule_resetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const auto device_index = THPUtils_unpackDeviceIndex(arg);
  c10::zoom::ZoomCachingAllocator::resetPeakStats(device_index);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

CapturedTraceback* getFromContext(
    const std::shared_ptr<c10::GatheredContext>& x) {
  if (CapturedTraceback* sc = dynamic_cast<CapturedTraceback*>(x.get())) {
    return sc;
  }
  TORCH_CHECK(
      false,
      "attempting to gather stack context from the wrong StackContext type.");
}

PyObject* THCPModule_memorySnapshot(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS

  using c10::zoom::ZoomCachingAllocator::BlockInfo;
  using c10::zoom::ZoomCachingAllocator::SegmentInfo;

  py::str device_s = "device";
  py::str address_s = "address";
  py::str total_size_s = "total_size";
  py::str allocated_size_s = "allocated_size";
  py::str active_size_s = "active_size";
  py::str requested_size_s = "requested_size";
  py::str stream_s = "stream";
  py::str segment_type_s = "segment_type";
  py::str segment_pool_id = "segment_pool_id";
  py::str large_s = "large";
  py::str small_s = "small";
  py::str size_s = "size";
  py::str state_s = "state";
  py::str active_allocated_s = "active_allocated";
  py::str active_pending_free_s = "active_pending_free";
  py::str inactive_s = "inactive";
  py::str addr_s = "addr";
  py::str cpp_frames_s = "cpp_frames";
  py::str blocks_s = "blocks";
  py::str is_expandable_s = "is_expandable";
  py::str frames_s = "frames";
  py::str time_us_s = "time_us";

  py::list empty_frames;
  std::vector<CapturedTraceback*> to_gather_frames;
  std::vector<py::dict> to_gather_dest;

  auto add_frame_key = [&](const py::dict& d,
                           const std::shared_ptr<c10::GatheredContext>& ctx) {
    if (ctx) {
      auto sc = getFromContext(ctx);
      to_gather_frames.emplace_back(sc);
      to_gather_dest.emplace_back(d);
    } else {
      d[frames_s] = empty_frames;
    }
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict[device_s] = segmentInfo.device;
    segmentDict[address_s] = segmentInfo.address;
    segmentDict[total_size_s] = segmentInfo.total_size;
    segmentDict[allocated_size_s] = segmentInfo.allocated_size;
    segmentDict[active_size_s] = segmentInfo.active_size;
    segmentDict[requested_size_s] = segmentInfo.requested_size;
    // we want the python objects to pickle easily so use an int to
    // represent the stream rather than a torch.zoom.stream object
    segmentDict[stream_s] = int64_t(segmentInfo.stream);
    segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);
    segmentDict[segment_pool_id] = segmentInfo.owner_private_pool_id;
    segmentDict[is_expandable_s] = segmentInfo.is_expandable;
    add_frame_key(segmentDict, segmentInfo.context_when_allocated);

    auto address = segmentInfo.address;
    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict[address_s] = address;
      blockDict[size_s] = blockInfo.size;
      blockDict[requested_size_s] = blockInfo.requested_size;
      blockDict[state_s] =
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s));
      add_frame_key(blockDict, blockInfo.context_when_allocated);
      blocks.append(blockDict);
      address += blockInfo.size;
    }
    segmentDict[blocks_s] = blocks;

    return segmentDict;
  };

  auto snapshot = c10::zoom::ZoomCachingAllocator::snapshot();

  py::list segments;

  for (const auto& segmentInfo : snapshot.segments) {
    segments.append(segmentInfoToDict(segmentInfo));
  }

  py::list traces;
  py::str action_s = "action";
  py::str alloc_s = "alloc";
  py::str free_requested_s = "free_requested";
  py::str free_completed_s = "free_completed";
  py::str segment_alloc_s = "segment_alloc";
  py::str segment_free_s = "segment_free";
  py::str segment_map_s = "segment_map";
  py::str segment_unmap_s = "segment_unmap";

  py::str snapshot_s = "snapshot";
  py::str oom_s = "oom";
  py::str device_free_s = "device_free";

  using namespace c10::zoom::ZoomCachingAllocator;

  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
      case TraceEntry::SEGMENT_UNMAP:
        return segment_unmap_s;
      case TraceEntry::SEGMENT_MAP:
        return segment_map_s;
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    py::list trace;
    for (const auto& te : traceInfo) {
      py::dict trace_entry;
      if (te.context_) {
        // without further compression frames can get really large on dump
        auto sc = getFromContext(te.context_);
        to_gather_frames.emplace_back(sc);
        to_gather_dest.emplace_back(trace_entry);
      }
      trace_entry[action_s] = action_to_str(te.action_);
      trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] =
          te.addr_;
      trace_entry[size_s] = te.size_;
      trace_entry[stream_s] = int64_t(te.stream_);
      trace_entry[time_us_s] = te.time_.t_;
      trace.append(trace_entry);
    }
    traces.append(trace);
  }

  py::dict allocator_settings;
  py::str last_allocator_settings_s = "PYTORCH_ZOOM_ALLOC_CONF";
  py::str max_split_size_s = "max_split_size";
  py::str garbage_collection_threshold_s = "garbage_collection_threshold";
  py::str expandable_segments_s = "expandable_segments";
  py::str pinned_num_register_threads_s = "pinned_num_register_threads";
  py::str release_lock_on_malloc_s = "release_lock_on_hipMalloc";
  py::str pinned_use_host_register_s = "pinned_use_zoom_host_register";
  py::str roundup_power2_divisions_s = "roundup_power2_divisions";

  allocator_settings[last_allocator_settings_s] =
      snapshot.config_metadata.last_allocator_settings;
  allocator_settings[max_split_size_s] =
      int64_t(snapshot.config_metadata.max_split_size);
  allocator_settings[garbage_collection_threshold_s] =
      snapshot.config_metadata.garbage_collection_threshold;
  allocator_settings[expandable_segments_s] =
      snapshot.config_metadata.expandable_segments;
  allocator_settings[pinned_num_register_threads_s] =
      int64_t(snapshot.config_metadata.pinned_num_register_threads);
  allocator_settings[release_lock_on_malloc_s] =
      snapshot.config_metadata.release_lock_on_malloc;
  allocator_settings[pinned_use_host_register_s] =
      snapshot.config_metadata.pinned_use_host_register;
  unsigned int roundup_key = 1;
  py::dict roundup_settings;
  for (const auto& v : snapshot.config_metadata.roundup_power2_divisions) {
    py::str roundup_key_s = std::to_string(roundup_key);
    roundup_settings[roundup_key_s] = int64_t(v);
    roundup_key *= 2;
  }
  allocator_settings[roundup_power2_divisions_s] = roundup_settings;

  py::dict result;
  result["segments"] = segments;
  result["device_traces"] = traces;
  result["allocator_settings"] = allocator_settings;

  auto frames = py_symbolize(to_gather_frames);
  for (auto i : c10::irange(frames.size())) {
    to_gather_dest.at(i)[frames_s] = frames.at(i);
  }

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_attachOutOfMemoryObserver(
    PyObject* _unused,
    PyObject* observer) {
  HANDLE_TH_ERRORS
  Py_XINCREF(observer);
  auto obs = [observer](
                 int64_t device,
                 int64_t alloc,
                 int64_t device_allocated,
                 int64_t device_free) {
    py::gil_scoped_acquire g;
    PyObject* result = PyObject_CallFunction(
        observer, "LLLL", device, alloc, device_allocated, device_free);
    if (!result) {
      throw py::error_already_set();
    }
    Py_XDECREF(result);
  };
  at::globalContext().lazyInitPrivateUse1();
  c10::zoom::ZoomCachingAllocator::attachOutOfMemoryObserver(std::move(obs));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_zoomSetSyncDebugMode(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  TORCH_WARN_ONCE(
      "Synchronization debug mode is a prototype feature and does not yet detect all "
      "synchronizing operations");
  TORCH_CHECK(
      THPUtils_checkLong(arg), "invalid argument to set_sync_debug_mode");
  int64_t debug_mode = THPUtils_unpackLong(arg);
  TORCH_CHECK(
      debug_mode >= 0 && debug_mode <= 2,
      "invalid value of debug_mode, expected one of 0,1,2");
  c10::zoom::SyncDebugMode l = c10::zoom::SyncDebugMode::L_DISABLED;
  switch (debug_mode) {
    case 0:
      l = c10::zoom::SyncDebugMode::L_DISABLED;
      break;
    case 1:
      l = c10::zoom::SyncDebugMode::L_WARN;
      break;
    case 2:
      l = c10::zoom::SyncDebugMode::L_ERROR;
      break;
    default:
      break; // can't happen
  }
  c10::zoom::warning_state().set_sync_debug_mode(l);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_zoomGetSyncDebugMode(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  auto debug_mode = c10::zoom::warning_state().get_sync_debug_mode();
  switch (debug_mode) {
    case c10::zoom::SyncDebugMode::L_DISABLED:
      return THPUtils_packInt32(0);
    case c10::zoom::SyncDebugMode::L_WARN:
      return THPUtils_packInt32(1);
    case c10::zoom::SyncDebugMode::L_ERROR:
      return THPUtils_packInt32(2);
    default:
      return THPUtils_packInt32(-1); // can't happen
  }
  END_HANDLE_TH_ERRORS
}

////////////////////////////////////////////////////////////////////////////////
// Zoom module initialization
////////////////////////////////////////////////////////////////////////////////

static void registerZoomDeviceProperties(PyObject* module) {
  // Add _hipDeviceProp_tertires class to torch._C
  auto m = py::handle(module).cast<py::module>();
  py::class_<hipDeviceProp_t>(m, "_ZoomDeviceProperties")
      .def_readonly("name", &hipDeviceProp_t::name)
      .def_readonly("major", &hipDeviceProp_t::major)
      .def_readonly("minor", &hipDeviceProp_t::minor)
      .def_readonly("is_multi_gpu_board", &hipDeviceProp_t::isMultiGpuBoard)
      .def_readonly("is_integrated", &hipDeviceProp_t::integrated)
      .def_readonly(
          "multi_processor_count", &hipDeviceProp_t::multiProcessorCount)
      .def_readonly("total_memory", &hipDeviceProp_t::totalGlobalMem)
      .def_readonly(
          "max_threads_per_multi_processor",
          &hipDeviceProp_t::maxThreadsPerMultiProcessor)
      .def_readonly(
          "gcnArchName",
          &hipDeviceProp_t::gcnArchName
          )
      .def("__repr__", [](const hipDeviceProp_t& prop) {
        std::ostringstream stream;
        stream << "_ZoomDeviceProperties(name='" << prop.name
               << "', major=" << prop.major << ", minor=" << prop.minor
               << ", gcnArchName='" << prop.gcnArchName << "'"
               << ", total_memory=" << prop.totalGlobalMem / (1024ull * 1024)
               << "MB, multi_processor_count=" << prop.multiProcessorCount
               << ")";
        return stream.str();
      });

  // m.def(
  //     "_zoom_record_memory_history_legacy",
  //     static_cast<void (*)(bool, bool, int64_t, bool, bool)>(
  //         torch::zoom::_record_memory_history));

  // m.def(
  //     "_zoom_record_memory_history",
  //     static_cast<void (*)(
  //         std::optional<std::string>,
  //         std::optional<std::string>,
  //         const std::string&,
  //         size_t)>(torch::zoom::_record_memory_history));

  m.def("_zoom_isHistoryEnabled", []() {
    return c10::zoom::ZoomCachingAllocator::isHistoryEnabled();
  });

  // m.def("_zoom_get_conv_benchmark_empty_cache", []() {
  //   return at::native::_cudnn_get_conv_benchmark_empty_cache();
  // });

  // m.def("_cudnn_set_conv_benchmark_empty_cache", [](bool enable) {
  //   return at::native::_cudnn_set_conv_benchmark_empty_cache(enable);
  // });
}

// We choose to ignore certain blocks that are currently allocated
// when we set the pool to its checkpoint. For those blocks, we need
// to swap out the deleter function of their corresponding blocks
// so that a deallocation is not triggered when they die.
void removeStorageDeleterFns(
    const std::vector<c10::StorageImpl*>& stale_live_storages,
    std::unordered_set<void*> definitely_stale_pointers) {
  for (c10::StorageImpl* stale_storage : stale_live_storages) {
    auto ptr = stale_storage->data_ptr().get();
    auto allocated_pointer = definitely_stale_pointers.find(ptr);
    TORCH_CHECK(allocated_pointer != definitely_stale_pointers.end());
    auto t = c10::zoom::ZoomCachingAllocator::get();
    bool succeeded = stale_storage->mutable_data_ptr().compare_exchange_deleter(
        t->raw_deleter(), &c10::detail::deleteNothing);

    TORCH_CHECK(
        succeeded,
        "Unexpected deleter function on storage, could not swap function");
  }
}

void addStorageDeleterFns(
    std::vector<c10::StorageImpl*>& storages_to_add_deleters_to,
    c10::zoom::ZoomCachingAllocator::CheckpointDelta& delta) {
  std::unordered_map<void*, c10::StorageImpl*> storages;
  for (auto& storage : storages_to_add_deleters_to) {
    storages[storage->data_ptr().get()] = storage;
  }

  for (auto& data_ptr : delta.dataptrs_allocd) {
    auto storage_pair = storages.find(data_ptr.get());
    if (storage_pair != storages.end()) {
      auto ctx = storage_pair->second->data_ptr().get_context();
      TORCH_CHECK(ctx == nullptr, " Not expecting deleter function");
      storage_pair->second->set_data_ptr_noswap(std::move(data_ptr));
    } else {
      data_ptr.release_context();
    }
  }
}

// static void registerCudaPluggableAllocator(PyObject* module) {
//   auto m = py::handle(module).cast<py::module>();

//   // NOLINTNEXTLINE(bugprone-unused-raii)
//   py::class_<
//       c10::zoom::ZoomCachingAllocator::ZoomAllocator,
//       std::shared_ptr<c10::zoom::ZoomCachingAllocator::ZoomAllocator>>(
//       m, "_zoom_ZoomAllocator");
//   m.def("_zoom_getAllocator", []() {
//     return py::cast(torch::zoom::ZoomPluggableAllocator::getCurrentAllocator());
//   });

//   m.def(
//       "_zoom_changeCurrentAllocator",
//       [](const std::shared_ptr<c10::zoom::ZoomCachingAllocator::ZoomAllocator>&
//              allocator) {
//         torch::zoom::ZoomPluggableAllocator::changeCurrentAllocator(allocator);
//       });
//   py::class_<
//       torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator,
//       c10::zoom::ZoomCachingAllocator::ZoomAllocator,
//       std::shared_ptr<
//           torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator>>(
//       m, "_ZoomPluggableAllocator")
//       .def(
//           "set_init_fn",
//           [](torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator& self,
//              uint64_t func_ptr) {
//             using FuncType = void(int);
//             std::function<FuncType> func =
//                 // NOLINTNEXTLINE(performance-no-int-to-ptr)
//                 reinterpret_cast<FuncType*>(func_ptr);
//             self.set_init_fn(func);
//           })
//       .def(
//           "set_reset_fn",
//           [](torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator& self,
//              uint64_t func_ptr) {
//             using FuncType = void();
//             std::function<FuncType> func =
//                 // NOLINTNEXTLINE(performance-no-int-to-ptr)
//                 reinterpret_cast<FuncType*>(func_ptr);
//             self.set_reset_fn(func);
//           })
//       .def(
//           "set_memory_fraction_fn",
//           [](torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator& self,
//              uint64_t func_ptr) {
//             using FuncType = void(double, int);
//             std::function<FuncType> func =
//                 // NOLINTNEXTLINE(performance-no-int-to-ptr)
//                 reinterpret_cast<FuncType*>(func_ptr);
//             self.set_memory_fraction_fn(func);
//           })
//       .def(
//           "set_base_alloc_fn",
//           [](torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator& self,
//              uint64_t func_ptr) {
//             using FuncType = void*(void*, size_t*);
//             std::function<FuncType> func =
//                 // NOLINTNEXTLINE(performance-no-int-to-ptr)
//                 reinterpret_cast<FuncType*>(func_ptr);
//             self.set_base_alloc_fn(func);
//           })
//       .def(
//           "set_record_stream_fn",
//           [](torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator& self,
//              uint64_t func_ptr) {
//             using FuncType = void(void*, hipStream_t);
//             std::function<FuncType> func =
//                 // NOLINTNEXTLINE(performance-no-int-to-ptr)
//                 reinterpret_cast<FuncType*>(func_ptr);
//             self.set_record_stream_fn(func);
//           })
//       .def(
//           "set_begin_allocate_to_pool",
//           [](torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator& self,
//              uint64_t func_ptr) {
//             using FuncType = void(
//                 int, c10::zoom::MempoolId_t, std::function<bool(hipStream_t)>);
//             std::function<FuncType> func =
//                 // NOLINTNEXTLINE(performance-no-int-to-ptr)
//                 reinterpret_cast<FuncType*>(func_ptr);
//             self.set_begin_allocate_to_pool(func);
//           })
//       .def(
//           "set_end_allocate_to_pool_fn",
//           [](torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator& self,
//              uint64_t func_ptr) {
//             using FuncType = void(int, c10::zoom::MempoolId_t);
//             std::function<FuncType> func =
//                 // NOLINTNEXTLINE(performance-no-int-to-ptr)
//                 reinterpret_cast<FuncType*>(func_ptr);
//             self.set_end_allocate_to_pool_fn(func);
//           })
//       .def(
//           "set_release_pool",
//           [](torch::zoom::ZoomPluggableAllocator::ZoomPluggableAllocator& self,
//              uint64_t func_ptr) {
//             using FuncType = void(int, c10::zoom::MempoolId_t);
//             std::function<FuncType> func =
//                 // NOLINTNEXTLINE(performance-no-int-to-ptr)
//                 reinterpret_cast<FuncType*>(func_ptr);
//             self.set_release_pool(func);
//           });
//   m.def("_zoom_customAllocator", [](uint64_t malloc_ptr, uint64_t free_ptr) {
//     using MallocFuncType = void*(size_t, int, hipStream_t);
//     using FreeFuncType = void(void*, size_t, int, hipStream_t);
//     std::function<MallocFuncType> malloc_fn =
//         // NOLINTNEXTLINE(performance-no-int-to-ptr)
//         reinterpret_cast<MallocFuncType*>(malloc_ptr);
//     std::function<FreeFuncType> free_fn =
//         // NOLINTNEXTLINE(performance-no-int-to-ptr)
//         reinterpret_cast<FreeFuncType*>(free_ptr);
//     return torch::zoom::ZoomPluggableAllocator::createCustomAllocator(
//         malloc_fn, free_fn);
//   });

//   // NOLINTNEXTLINE(bugprone-unused-raii)
//   py::class_<
//       c10::zoom::ZoomCachingAllocator::AllocatorState,
//       std::shared_ptr<c10::zoom::ZoomCachingAllocator::AllocatorState>>(
//       m, "_zoom_ZoomAllocator_AllocatorState");

//   m.def(
//       "_zoom_getCheckpointState",
//       [](c10::DeviceIndex device, c10::zoom::MempoolId_t id) {
//         return c10::zoom::ZoomCachingAllocator::getCheckpointState(device, id);
//       });

//   m.def("_free_And_Remove_DeleterFn", [](size_t storage_impl_ptr) {
//     // NOLINTNEXTLINE(performance-no-int-to-ptr)
//     c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
//     auto alloc = c10::zoom::ZoomCachingAllocator::get();
//     auto data_ptr = storage_impl->data_ptr().get();
//     bool succeeded = storage_impl->mutable_data_ptr().compare_exchange_deleter(
//         alloc->raw_deleter(), c10::detail::deleteNothing);
//     TORCH_CHECK(succeeded, "Expected standard deleter");
//     c10::zoom::ZoomCachingAllocator::raw_delete(data_ptr);
//   });

//   m.def(
//       "_set_storage_access_error_msg", [](const at::Tensor& t, std::string s) {
//         t.unsafeGetTensorImpl()
//             ->release_storage_and_set_meta_custom_data_ptr_error_msg_(s);
//       });

//   m.def("_has_Standard_Deleter", [](size_t storage_impl_ptr) {
//     // NOLINTNEXTLINE(performance-no-int-to-ptr)
//     c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
//     auto alloc = c10::zoom::ZoomCachingAllocator::get();
//     return (storage_impl->data_ptr().get_deleter() == alloc->raw_deleter());
//   });

//   m.def("_set_cached_tensors_enabled", [](bool enabled) {
//     at::caching::set_cached_tensors_enabled(enabled);
//   });

//   m.def("_add_cached_tensor", [](const at::Tensor& t) {
//     at::caching::add_cached_tensor(t);
//   });

//   m.def("_remove_cached_tensor", [](const at::Tensor& t) {
//     at::caching::remove_cached_tensor(t);
//   });

//   m.def("_is_cached_tensor", [](const at::Tensor& t) {
//     return at::caching::is_cached_tensor(t);
//   });

//   m.def("_storage_Use_Count", [](size_t storage_impl_ptr) {
//     // NOLINTNEXTLINE(performance-no-int-to-ptr)
//     c10::StorageImpl* storage_impl = (c10::StorageImpl*)storage_impl_ptr;
//     return c10::raw::weak_intrusive_ptr::use_count(storage_impl);
//   });

//   m.def(
//       "_tensors_data_ptrs_at_indices_equal",
//       [](py::list& tensors, py::list& data_ptrs, py::list& indices) {
//         for (size_t i = 0, end = indices.size(); i < end; ++i) {
//           auto index = indices[i].cast<int64_t>();
//           auto t = tensors[index].cast<at::Tensor>();
//           auto data_ptr = data_ptrs[index].cast<int64_t>();
//           if (reinterpret_cast<int64_t>(t.data_ptr()) != data_ptr) {
//             return false;
//           }
//         }
//         return true;
//       });

//   m.def(
//       "_construct_Zoom_Tensor_From_Storage_And_Metadata",
//       [](py::dict& metadata, c10::Storage s) {
//         auto dtype_arg = metadata["dtype"].ptr();
//         auto meta = scalarTypeToTypeMeta(toScalarType(dtype_arg));

//         constexpr c10::DispatchKeySet zoom_dks(c10::DispatchKey::PrivateUse1);
//         at::Tensor tensor = at::detail::make_tensor_base<c10::TensorImpl>(
//             std::move(s), zoom_dks, meta);

//         tensor.unsafeGetTensorImpl()->set_sizes_and_strides(
//             metadata["size"].cast<std::vector<int64_t>>(),
//             metadata["stride"].cast<std::vector<int64_t>>());
//         tensor.unsafeGetTensorImpl()->set_storage_offset(
//             metadata["storage_offset"].cast<int64_t>());
//         return tensor;
//       });

//   m.def(
//       "_zoom_beginAllocateCurrentStreamToPool",
//       [](c10::DeviceIndex device, at::zoom::MempoolId_t mempool_id) {
//         auto stream = c10::zoom::getCurrentZoomStream(device);
//         TORCH_CHECK(stream, "Expected stream capture to be under way");
//         c10::zoom::ZoomCachingAllocator::beginAllocateToPool(
//             device, mempool_id, [stream](hipStream_t target) {
//               return target == stream;
//             });
//       });

//   m.def(
//       "_zoom_endAllocateCurrentStreamToPool",
//       [](c10::DeviceIndex device, at::zoom::MempoolId_t mempool_id) {
//         c10::zoom::ZoomCachingAllocator::endAllocateToPool(device, mempool_id);
//       });

//   m.def(
//       "_zoom_releasePool",
//       [](c10::DeviceIndex device, at::zoom::MempoolId_t mempool_id) {
//         c10::zoom::ZoomCachingAllocator::releasePool(device, mempool_id);
//       });

//   m.def(
//       "_zoom_checkPoolLiveAllocations",
//       [](c10::DeviceIndex device,
//          at::zoom::MempoolId_t mempool_id,
//          const py::set& expected_live_allocations) {
//         std::unordered_set<void*> allocations;
//         allocations.reserve(expected_live_allocations.size());
//         for (auto& elem : expected_live_allocations) {
//           // NOLINTNEXTLINE(performance-no-int-to-ptr)
//           allocations.insert(reinterpret_cast<void*>(py::cast<size_t>(elem)));
//         }
//         return c10::zoom::ZoomCachingAllocator::checkPoolLiveAllocations(
//             device, mempool_id, allocations);
//       });

//   m.def(
//       "_zoom_setCheckpointPoolState",
//       [](c10::DeviceIndex device,
//          std::shared_ptr<c10::zoom::ZoomCachingAllocator::AllocatorState> pps,
//          const std::vector<size_t>& stale_storages_ptr,
//          const std::vector<size_t>& storages_to_add_deleters_to_ptr = {}) {
//         std::unordered_set<c10::StorageImpl*> ptr_set;
//         // iterate on std::vector for determinism
//         std::vector<c10::StorageImpl*> ptrs;
//         for (size_t ptr_int : stale_storages_ptr) {
//           // NOLINTNEXTLINE(performance-no-int-to-ptr)
//           c10::StorageImpl* ptr = (c10::StorageImpl*)ptr_int;
//           if (!ptr_set.count(ptr)) {
//             ptrs.push_back(ptr);
//             ptr_set.insert(ptr);
//           }
//         }
//         auto delta = c10::zoom::ZoomCachingAllocator::setCheckpointPoolState(
//             device, std::move(pps));
//         auto& freed_pointers = delta.ptrs_freed;

//         std::unordered_set<void*> allocd_set;
//         for (auto& data_ptr : delta.dataptrs_allocd) {
//           allocd_set.insert(data_ptr.get());
//         }
//         std::unordered_set<void*> freed_pointer_set;
//         size_t definite_freed_count = 0;
//         for (void* ptr : freed_pointers) {
//           if (!allocd_set.count(ptr)) {
//             definite_freed_count += 1;
//           }
//           freed_pointer_set.insert((ptr));
//         }
//         // that block has already been freed,
//         // so even those this will error, so too will the allocator
//         // when the corresponding tensor dies because there is no
//         // live tensor corresponding to it
//         TORCH_CHECK(
//             ptr_set.size() >= definite_freed_count,
//             "Any stale tensors which are being manually freed"
//             " must be passed to set checkpoint");

//         removeStorageDeleterFns(ptrs, freed_pointer_set);
//         std::vector<c10::StorageImpl*> storages_to_add_deleters_to;
//         storages_to_add_deleters_to.reserve(
//             storages_to_add_deleters_to_ptr.size());
//         for (size_t ptr_int : storages_to_add_deleters_to_ptr) {
//           // NOLINTNEXTLINE(performance-no-int-to-ptr)
//           storages_to_add_deleters_to.push_back((c10::StorageImpl*)ptr_int);
//         }

//         addStorageDeleterFns(storages_to_add_deleters_to, delta);
//       });
// }

static void bindGetDeviceProperties(PyObject* module) {
  // Add method to torch.zoom
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_get_device_properties",
      [](c10::DeviceIndex device) -> hipDeviceProp_t* {
        return at::zoom::getDeviceProperties(device);
      },
      py::return_value_policy::reference);
}

// Callback for python part. Used for additional initialization of python
// classes
static PyObject* THCPModule_initExtension(PyObject* self, PyObject* noargs) {
#if C10_ASAN_ENABLED
  TORCH_WARN(
      "torch.zoom: your pytorch binary has address sanitizer (asan) built in, "
      "asan is currently not compatible with torch.zoom module, "
      "you might get unexpected behavior (eg. out of memory, crash, etc.), "
      "please rebuild pytorch without asan if you need to use this module");
#endif
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  poison_fork();
  at::globalContext().lazyInitPrivateUse1();

  auto m = THPObjectPtr(PyImport_ImportModule("torch.zoom"));
  if (!m)
    throw python_error();

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  auto num_gpus = c10::zoom::device_count();
  auto default_zoom_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  for (const auto i : c10::irange(num_gpus)) {
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(
        at::zoom::detail::getDefaultZoomGenerator(i));
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_zoom_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_zoom_generators);
  bindGetDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THCPModule_getCurrentBlasHandle_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  hipblasHandle_t handle = at::zoom::getCurrentHIPBlasHandle();
  return PyLong_FromVoidPtr(handle);
  END_HANDLE_TH_ERRORS
}


// PyObject* THCPModule_rocm_is_backward_pass(
//     PyObject* _unused,
//     PyObject* noargs) {
//   HANDLE_TH_ERRORS
// #if USE_ROCM
//   if (at::ROCmBackwardPassGuard::is_backward_pass()) {
//     Py_RETURN_TRUE;
//   } else {
//     Py_RETURN_FALSE;
//   }
// #else
//   Py_RETURN_FALSE;
// #endif
//   END_HANDLE_TH_ERRORS
// }

static PyObject* THCPModule_isCurrentStreamCapturing_wrap(
    PyObject* self,
    PyObject* noargs) {
  HANDLE_TH_ERRORS
  // If there's no zoom context, at::zoom::currentStreamCaptureStatus returns
  // CaptureStatus::None without initializing a context.
  if (at::zoom::currentStreamCaptureStatus() == at::zoom::CaptureStatus::None) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(*-c-arrays*, *-global-variables)
static struct PyMethodDef _THCPModule_methods[] = {
    {"_zoom_init", THCPModule_initExtension, METH_NOARGS, nullptr},
    {"_zoom_setDevice", THCPModule_setDevice_wrap, METH_O, nullptr},
    {"_zoom_exchangeDevice", THCPModule_exchangeDevice, METH_O, nullptr},
    {"_zoom_maybeExchangeDevice",
     THCPModule_maybeExchangeDevice,
     METH_O,
     nullptr},
    {"_zoom_getDevice", THCPModule_getDevice_wrap, METH_NOARGS, nullptr},
    {"_zoom_getDeviceCount",
     THCPModule_getDeviceCount_wrap,
     METH_NOARGS,
     nullptr},
    {"_zoom_canDeviceAccessPeer",
     THCPModule_canDeviceAccessPeer_wrap,
     METH_VARARGS,
     nullptr},
    {"_zoom_getArchFlags", THCPModule_getArchFlags, METH_NOARGS, nullptr},
    {"_zoom_isInBadFork", THCPModule_isInBadFork, METH_NOARGS, nullptr},
    {"_zoom_getCurrentStream",
     THCPModule_getCurrentStream_wrap,
     METH_O,
     nullptr},
    {"_zoom_getCurrentRawStream",
     THCPModule_getCurrentStream_raw,
     METH_O,
     nullptr},
    {"_zoom_getDefaultStream",
     THCPModule_getDefaultStream_wrap,
     METH_O,
     nullptr},
    {"_zoom_getCurrentBlasHandle",
     THCPModule_getCurrentBlasHandle_wrap,
     METH_NOARGS,
     nullptr},
    {"_zoom_isCurrentStreamCapturing",
     THCPModule_isCurrentStreamCapturing_wrap,
     METH_NOARGS,
     nullptr},
    {"_zoom_setStream",
     castPyCFunctionWithKeywords(THCPModule_setStream_wrap),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_zoom_getCompiledVersion",
     THCPModule_getCompiledVersion,
     METH_NOARGS,
     nullptr},
    {"_zoom_hasPrimaryContext", THCPModule_hasPrimaryContext, METH_O, nullptr},
    // {"_zoom_setMemoryFraction",
    //  THCPModule_setMemoryFraction,
    //  METH_VARARGS,
    //  nullptr},
    {"_zoom_emptyCache", THCPModule_emptyCache, METH_NOARGS, nullptr},
    {"_zoom_memoryStats", THCPModule_memoryStats, METH_O, nullptr},
    {"_zoom_resetAccumulatedMemoryStats",
     THCPModule_resetAccumulatedMemoryStats,
     METH_O,
     nullptr},
    {"_zoom_resetPeakMemoryStats",
     THCPModule_resetPeakMemoryStats,
     METH_O,
     nullptr},
    {"_zoom_memorySnapshot", THCPModule_memorySnapshot, METH_NOARGS, nullptr},
    {"_zoom_attach_out_of_memory_observer",
     THCPModule_attachOutOfMemoryObserver,
     METH_O,
     nullptr},
    {"_zoom_zoomHostAllocator",
     THCPModule_zoomHostAllocator,
     METH_NOARGS,
     nullptr},
    {"_zoom_zoomCachingAllocator_raw_alloc",
     THCPModule_zoomCachingAllocator_raw_alloc,
     METH_VARARGS,
     nullptr},
    {"_zoom_zoomCachingAllocator_raw_delete",
     THCPModule_zoomCachingAllocator_raw_delete,
     METH_O,
     nullptr},
    {"_zoom_zoomCachingAllocator_set_allocator_settings",
     THCPModule_zoomCachingAllocator_set_allocator_settings,
     METH_O,
     nullptr},
    {"_zoom_getAllocatorBackend",
     THCPModule_getAllocatorBackend,
     METH_NOARGS,
     nullptr},
    {"_zoom_synchronize", THCPModule_zoomSynchronize, METH_NOARGS, nullptr},
    // {"_zoom_ipc_collect", THCPModule_zoomIPCCollect, METH_NOARGS, nullptr},
    // {"_zoom_sleep", THCPModule_zoomSleep, METH_O, nullptr},
    {"_zoom_lock_mutex", THCPModule_zoomLockMutex, METH_NOARGS, nullptr},
    {"_zoom_unlock_mutex", THCPModule_zoomUnlockMutex, METH_NOARGS, nullptr},
    {"_zoom_set_sync_debug_mode",
     THCPModule_zoomSetSyncDebugMode,
     METH_O,
     nullptr},
    {"_zoom_get_sync_debug_mode",
     THCPModule_zoomGetSyncDebugMode,
     METH_NOARGS,
     nullptr},
    // {"_zoom_jiterator_compile_and_launch_kernel",
    //  THCPModule_zoomJiteratorCompileAndLaunchKernel,
    //  METH_VARARGS,
    //  nullptr},
    // {"_rocm_is_backward_pass",
    //  THCPModule_rocm_is_backward_pass,
    //  METH_NOARGS,
    //  nullptr},
    {nullptr}};

PyMethodDef* THCPModule_methods() {
  return _THCPModule_methods;
}

namespace torch::zoom {

namespace shared {

// void initCudartBindings(PyObject* module);
// void initNvtxBindings(PyObject* module);
// #if defined(USE_CUDNN) || defined(USE_ROCM)
// void initCudnnBindings(PyObject* module);
// #endif

} // namespace shared

void initModule(PyObject* module) {
//   python::initCommMethods(module);
//   // As weird as it seems, this file is also compiled for ROCm,
//   // so this condition might not always be true...
//   shared::initCudartBindings(module);
//   shared::initNvtxBindings(module);
// #if defined(USE_CUDNN) || defined(USE_ROCM)
//   shared::initCudnnBindings(module);
// #endif
  registerZoomDeviceProperties(module);
  // registerCudaPluggableAllocator(module);
}

} // namespace torch::zoom
