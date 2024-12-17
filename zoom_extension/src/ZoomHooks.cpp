
// #include <ATen/cuda/CUDAGeneratorImpl.h>
#include "ZoomGeneratorImpl.h"
#include <ATen/Context.h>
#include <ATen/DeviceGuard.h>
#include <ATen/DynamicLibrary.h>
#include <ATen/core/Vitals.h>
// #include <ATen/cuda/CUDAConfig.h>
// #include <ATen/cuda/CUDADevice.h>
#include "ZoomDevice.h"
// #include <ATen/cuda/Exceptions.h>
#include "ZoomException.h"
#include "PeerToPeerAccess.h"
// #include <ATen/cuda/PinnedMemoryAllocator.h>
#include "PinnedMemoryAllocator.h"
// #include <ATen/cuda/nvrtc_stub/ATenNVRTC.h>
#include "ZoomHooks.h"
// #include <ATen/native/cuda/CuFFTPlanCache.h>
#include <c10/util/Exception.h>
#include "ZoomCachingAllocator.h"
#include "ZoomFunctions.h"
#include <c10/util/irange.h>

// #if AT_CUDNN_ENABLED()
// #include <ATen/cudnn/cudnn-wrapper.h>
// #endif

// #if AT_MAGMA_ENABLED()
// #include <magma_v2.h>
// #endif

// #if defined(USE_ROCM)
// #include <miopen/version.h>
// #endif

// #ifndef USE_ROCM
// #include <ATen/cuda/detail/LazyNVRTC.h>
// #endif

// #include <cuda.h>

#include <sstream>
#include <cstddef>
#include <functional>
#include <memory>
#include <iostream>
#include <string>

namespace c10::zoom::_internal {
void setHasPrimaryContext(bool (*func)(DeviceIndex));
}

namespace at::zoom::detail {

// const at::zoom::NVRTC& nvrtc();
DeviceIndex current_device();

// static void (*magma_init_fn)() = nullptr;

// void set_magma_init_fn(void (*fn)()) {
//   magma_init_fn = fn;
// }

namespace {
bool _hasPrimaryContext(DeviceIndex device_index) {
  TORCH_CHECK(device_index >= 0 && device_index < c10::zoom::device_count(),
              "hasPrimaryContext expects a valid device index, but got device_index=", device_index);
  unsigned int ctx_flags;
  // In standalone tests of cuDevicePrimaryCtxGetState, I've seen the "active" argument end up with weird
  // (garbage-looking nonzero) values when the context is not active, unless I initialize it to zero.
  int ctx_is_active = 0;
//   AT_CUDA_DRIVER_CHECK(nvrtc().cuDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
    hipDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active);
  return ctx_is_active == 1;
}

// Register hasPrimaryContext back to c10::zoom
struct _Initializer {
  _Initializer() {
      c10::zoom::_internal::setHasPrimaryContext(_hasPrimaryContext);
  }
  ~_Initializer() {
      c10::zoom::_internal::setHasPrimaryContext(nullptr);
  }
} initializer;
} // anonymous namespace

// Sets the CUDA_MODULE_LOADING environment variable
// if it's not set by the user.
void maybe_set_zoom_module_loading(const std::string &def_value) {
  auto value = std::getenv("ZOOM_MODULE_LOADING");
  if (!value) {
#ifdef _WIN32
    auto env_var = "ZOOM_MODULE_LOADING=" + def_value;
    _putenv(env_var.c_str());
#else
    setenv("ZOOM_MODULE_LOADING", def_value.c_str(), 1);
#endif
  }
}

// NB: deleter is dynamic, because we need it to live in a separate
// compilation unit (alt is to have another method in hooks, but
// let's not if we don't need to!)
void ZoomHooks::initZoom() const {
  std::cout << "INITZOOM" << std::endl;
  C10_LOG_API_USAGE_ONCE("aten.init.zoom");
  // Force the update to enable unit testing. This code get executed before unit tests
  // have a chance to enable vitals.
  at::vitals::VitalsAPI.setVital("ZOOM", "used", "true", /* force = */ true);

  maybe_set_zoom_module_loading("LAZY");
  const auto num_devices = c10::zoom::device_count_ensure_non_zero();
  std::cout << "NUMDEVICES: " << std::to_string(num_devices) << std::endl;
  c10::zoom::ZoomCachingAllocator::init(num_devices);
  at::zoom::detail::init_p2p_access_cache(num_devices);
}

const Generator& ZoomHooks::getDefaultZoomGenerator(DeviceIndex device_index) const {
  return at::zoom::detail::getDefaultZoomGenerator(device_index);
}

Device ZoomHooks::getDeviceFromPtr(void* data) const {
  return at::zoom::getDeviceFromPtr(data);
}

bool ZoomHooks::isPinnedPtr(const void* data) const {
  // First check if driver is broken/missing, in which case PyTorch CPU
  // functionalities should still work, we should report `false` here.
  if (!at::zoom::is_available()) {
    return false;
  }
  // cudaPointerGetAttributes grabs context on the current device, so we set
  // device to one that already has context, if exists.
  at::OptionalDeviceGuard device_guard;
  auto primary_ctx_device_index = c10::zoom::getDeviceIndexWithPrimaryContext();
  if (primary_ctx_device_index.has_value()) {
    device_guard.reset_device(at::Device(at::DeviceType::PrivateUse1, *primary_ctx_device_index));
  }
  hipPointerAttribute_t attr;
  // We do not believe that CUDA needs mutable access to the data
  // here.
  hipError_t err = hipPointerGetAttributes(&attr, data);
  // HIP throws hipErrorUnknown here
  if (err != hipSuccess) {
    (void)hipGetLastError(); // clear HIP error
    return false;
  }
  return attr.type == hipMemoryTypeHost;
}

bool ZoomHooks::hasROCM() const {
  return at::zoom::is_available();
}

DeviceIndex current_device() {
  c10::DeviceIndex device = 0;
  hipError_t err = c10::zoom::GetDevice(&device);
  if (err == hipSuccess) {
    return device;
  }
  return -1;
}

DeviceIndex ZoomHooks::current_device() const {
  return at::zoom::detail::current_device();
}

bool ZoomHooks::hasPrimaryContext(DeviceIndex device_index) const {
  return _hasPrimaryContext(device_index);
}

Allocator* ZoomHooks::getPinnedMemoryAllocator() const {
  return at::zoom::getPinnedMemoryAllocator();
}

Allocator* ZoomHooks::getZoomDeviceAllocator() const {
  return at::zoom::getZoomDeviceAllocator();
}

std::string ZoomHooks::showConfig() const {
  std::ostringstream oss;

  int runtimeVersion;
  hipRuntimeGetVersion(&runtimeVersion);

  auto printHIPStyleVersion = [&](int v) {

    // HIP_VERSION value format was changed after ROCm v4.2 to include the patch number
    if(v < 500) {
      // If major=xx, minor=yy then format -> xxyy
      oss << (v / 100) << "." << (v % 10);
    }
    else {
      // If major=xx, minor=yy & patch=zzzzz then format -> xxyyzzzzz
      oss << (v / 10000000) << "." << (v / 100000 % 100) << "." << (v % 100000);
    }

  };


  oss << "  - HIP Runtime ";

  printHIPStyleVersion(runtimeVersion);
  oss << "\n";

  return oss.str();
}

int ZoomHooks::getNumGPUs() const {
  auto cnt = c10::zoom::device_count();
  std::cout << "numgpu: " << cnt << std::endl;
  return cnt;
}

void ZoomHooks::deviceSynchronize(DeviceIndex device_index) const {
  at::DeviceGuard device_guard(at::Device(at::DeviceType::PrivateUse1, device_index));
  c10::zoom::device_synchronize();
}

// // Sigh, the registry doesn't support namespaces :(
// using at::zoomHooksRegistry;
// using at::RegistererCUDAHooksRegistry;

// REGISTER_CUDA_HOOKS(ZoomHooks);

using at::PrivateUse1HooksRegistry;
using at::RegistererPrivateUse1HooksRegistry;
REGISTER_PRIVATEUSE1_HOOKS(ZoomHooks);

static ZoomHooks* zoom_hooks_impl = nullptr;
void register_zoom_hooks() {
  if(zoom_hooks_impl == nullptr){
    zoom_hooks_impl = new ZoomHooks({});
    RegisterPrivateUse1HooksInterface(zoom_hooks_impl);
  }
}


} // namespace at::zoom::detail