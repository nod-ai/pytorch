#include <c10/zoom/ZoomFunctions.h>
#include <c10/macros/Macros.h>

#include <limits>

namespace c10::zoom {

namespace {
// returns -1 on failure
int32_t driver_version() {
  int driver_version = -1;
  C10_ZOOM_IGNORE_ERROR(hipDriverGetVersion(&driver_version));
  return driver_version;
}

int device_count_impl(bool fail_if_no_driver) {
  int count = 0;
  auto err = C10_ZOOM_ERROR_HANDLED(c10::zoom::GetDeviceCount(&count));
  if (err == hipSuccess) {
    return count;
  }
  // Clear out the error state, so we don't spuriously trigger someone else.
  // (This shouldn't really matter, since we won't be running very much CUDA
  // code in this regime.)
  hipError_t last_err C10_UNUSED = hipGetLastError();
  switch (err) {
    case hipErrorNoDevice:
      // Zero devices is ok here
      count = 0;
      break;
    case hipErrorInsufficientDriver: {
      auto version = driver_version();
      if (version <= 0) {
        if (!fail_if_no_driver) {
          // No hip driver means no devices
          count = 0;
          break;
        }
        TORCH_CHECK(
            false,
            "Found no ROCm driver on your system. Please check that you "
            "have an AMD GPU and installed a driver from "
            "https://rocm.docs.amd.com/projects/install-on-linux/en/develop/tutorial/quick-start.html#rocm-install-quick");
      } else {
        TORCH_CHECK(
            false,
            "The ROCm driver on your system is too old (found version ",
            version,
            "). Please update your GPU driver by downloading and installing "
            "a new version from the URL: "
            "https://rocm.docs.amd.com/projects/install-on-linux/en/develop/tutorial/quick-start.html#rocm-install-quick");
      }
    } break;
    case hipErrorInitializationError:
      TORCH_CHECK(
          false,
          "ROCm driver initialization failed, you might not "
          "have a ROCm gpu.");
      break;
    case hipErrorUnknown:
      TORCH_CHECK(
          false,
          "ZOOM unknown error - this may be due to an "
          "incorrectly set up environment, e.g. changing env "
          "variable ZOOM_VISIBLE_DEVICES after program start. "
          "Setting the available devices to be zero.");
      break;
#if C10_ASAN_ENABLED
    case hipErrorMemoryAllocation:
      // In ASAN mode, we know that a hipErrorMemoryAllocation error will
      // pop up if compiled with hipcc (clang-hip is fine)
      TORCH_CHECK(
          false,
          "Got 'out of memory' error while trying to initialize ZOOM. "
          "ZOOM with hipcc does not work well with ASAN and it's probably "
          "the reason. We will simply shut down HIP support. If you "
          "would like to use GPUs, turn off ASAN.");
      break;
#endif // C10_ASAN_ENABLED
    default:
      TORCH_CHECK(
          false,
          "Unexpected error from hipGetDeviceCount(). Did you run "
          "some hip functions before calling NumZoomDevices() "
          "that might have already set an error? Error ",
          err,
          ": ",
          hipGetErrorString(err));
  }
  return count;
}
} // namespace

DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      auto result = device_count_impl(/*fail_if_no_driver=*/false);
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many ROCm devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("ZOOM initialization: ", ex.msg());
      return 0;
    }
  }();
  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count_ensure_non_zero() {
  // Call the implementation every time to throw the exception
  int count = device_count_impl(/*fail_if_no_driver=*/true);
  // Zero gpus doesn't produce a warning in `device_count` but we fail here
  TORCH_CHECK(count, "No ROCm GPUs are available");
  TORCH_INTERNAL_ASSERT(
      count <= std::numeric_limits<DeviceIndex>::max(),
      "Too many ROCm devices, DeviceIndex overflowed");
  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  C10_ZOOM_CHECK(c10::zoom::GetDevice(&cur_device));
  return cur_device;
}

void set_device(DeviceIndex device) {
  C10_ZOOM_CHECK(c10::zoom::SetDevice(device));
}

void device_synchronize() {
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::DeviceType::PrivateUse1);
  }
  C10_ZOOM_CHECK(hipDeviceSynchronize());
}

// this function has to be called from callers performing cuda synchronizing
// operations, to raise proper error or warning
void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing HIP operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_WARN("called a synchronizing HIP operation");
  }
}

std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext() {
  // check current device first
  auto current_device_index = current_device();
  if (current_device_index >= 0) {
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (const auto device_index : c10::irange(c10::zoom::device_count())) {
    if (device_index == current_device_index)
      continue;
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return c10::nullopt;
}

namespace _internal {
bool dummyHasPrimaryContext(C10_UNUSED DeviceIndex device_index) {
  TORCH_CHECK(false, "Should never been called - did you remember to lazyInitPrivateUse1()?");
}
bool (*hasPrimaryContext)(DeviceIndex) = dummyHasPrimaryContext;

// Private api to be called from CUDAHooks.cpp
void setHasPrimaryContext(bool (*func)(DeviceIndex)) {
  hasPrimaryContext = func ? func : dummyHasPrimaryContext;
}
} // namespace _internal

bool hasPrimaryContext(DeviceIndex device_index) {
  return _internal::hasPrimaryContext(device_index);
}

// Wrappers for raw CUDA device management functions
hipError_t GetDeviceCount(int* dev_count) {
  return hipGetDeviceCount(dev_count);
}

// This is a codepath for CUDA 12 that comes with a critical change in behavior
// of `cudaSetDevice`. Unlike to previous CUDA versions that allocate context
// lazily CUDA 12.x eagerly allocates primary context the moment `cudaSetDevice`
// is called. This can lead to dramatic consequences and pollute the device
// memory in distributed runs. To avoid unnecessary context creation a new
// function called `MaybeSetDevice` was introduced. This function is to be
// called in device guard destructor and at the exit of torch.cuda.device
// context manager. The behavior of `MaybeSetDevice` is quite simple, it calls
// to `cudaSetDevice` if context already exist or if context was not allocated
// on targeted device it simply saves the device index. This way we can keep
// PyTorch backward compatible for applications like this:
//
// ```
// import torch
// x = torch.empty(1, device=“cuda:1”) # no CUDA context on cuda:0 after this
// call y = torch.empty(1, device=“cuda”) # CUDA context is created on cuda:0
// ```

thread_local DeviceIndex targetDeviceIndex = -1;

hipError_t GetDevice(DeviceIndex* device) {
  if (targetDeviceIndex >= 0) {
    *device = targetDeviceIndex;
    return hipSuccess;
  }
  int tmp_device = -1;
  auto err = hipGetDevice(&tmp_device);
  if (err == hipSuccess) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<DeviceIndex>::max(),
        "hipGetDevice returns invalid device ",
        tmp_device);
    *device = static_cast<DeviceIndex>(tmp_device);
  }
  return err;
}

hipError_t SetDevice(DeviceIndex device) {
  TORCH_CHECK(device >= 0, "device id must be positive!", device);
  targetDeviceIndex = -1;
  int cur_device = -1;
  C10_ZOOM_CHECK(hipGetDevice(&cur_device));
  if (device == cur_device) {
    return hipSuccess;
  }
  return hipSetDevice(device);
}

hipError_t MaybeSetDevice(DeviceIndex device) {
  if (hasPrimaryContext(device)) {
    return c10::zoom::SetDevice(device);
  }
  targetDeviceIndex = device;
  return hipSuccess;
}

// This function always initializes the CUDA context
// on to_device
DeviceIndex ExchangeDevice(DeviceIndex to_device) {
  auto cur_device = targetDeviceIndex;
  targetDeviceIndex = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    C10_ZOOM_CHECK(hipGetDevice(&tmp_device));
    cur_device = static_cast<DeviceIndex>(tmp_device);
    if (to_device == cur_device) {
      return cur_device;
    }
  }
  C10_ZOOM_CHECK(hipSetDevice(to_device));
  return cur_device;
}

// This function does not initialize the CUDA context
// on to_device if it does not already exist
DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
  int tmp_cur_device = -1;
  C10_ZOOM_CHECK(hipGetDevice(&tmp_cur_device));
  TORCH_INTERNAL_ASSERT(
      tmp_cur_device >= 0 &&
          tmp_cur_device <= std::numeric_limits<DeviceIndex>::max(),
      "hipGetDevice returns invalid device ",
      tmp_cur_device);
  auto cur_device = static_cast<DeviceIndex>(tmp_cur_device);
  if (to_device == tmp_cur_device) {
    return cur_device;
  }
  if (hasPrimaryContext(to_device)) {
    C10_ZOOM_CHECK(hipSetDevice(to_device));
  } else {
    targetDeviceIndex = to_device;
  }
  return cur_device;
}

void SetTargetDevice() {
  if (targetDeviceIndex >= 0) {
    C10_ZOOM_CHECK(c10::zoom::SetDevice(targetDeviceIndex));
  }
}


} // namespace c10::zoom