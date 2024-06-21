#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <ATen/detail/PrivateUse1HooksInterface.h>

// Forward-declares at::Generator and at::cuda::NVRTC
// namespace at {
// struct Generator;
// namespace cuda {
// struct NVRTC;
// } // namespace cuda
// } // namespace at

// NB: Class must live in `at` due to limitations of Registry.h.
namespace at {

// #ifdef _MSC_VER
// constexpr const char* ZOOM_HELP =
//   "PyTorch splits its backend into two shared libraries: a CPU library "
//   "and a CUDA library; this error has occurred because you are trying "
//   "to use some CUDA functionality, but the CUDA library has not been "
//   "loaded by the dynamic linker for some reason.  The CUDA library MUST "
//   "be loaded, EVEN IF you don't directly use any symbols from the CUDA library! "
//   "One common culprit is a lack of -INCLUDE:?warp_size@cuda@at@@YAHXZ "
//   "in your link arguments; many dynamic linkers will delete dynamic library "
//   "dependencies if you don't depend on any of their symbols.  You can check "
//   "if this has occurred by using link on your binary to see if there is a "
//   "dependency on *_cuda.dll library.";
// #else
constexpr const char* ZOOM_HELP =
  "PyTorch splits its backend into two shared libraries: a CPU library "
  "and a CUDA library; this error has occurred because you are trying "
  "to use some CUDA functionality, but the CUDA library has not been "
  "loaded by the dynamic linker for some reason.  The CUDA library MUST "
  "be loaded, EVEN IF you don't directly use any symbols from the CUDA library! "
  "One common culprit is a lack of -Wl,--no-as-needed in your link arguments; many "
  "dynamic linkers will delete dynamic library dependencies if you don't "
  "depend on any of their symbols.  You can check if this has occurred by "
  "using ldd on your binary to see if there is a dependency on *_cuda.so "
  "library.";
// #endif

// The ZoomHooksInterface is an omnibus interface for any CUDA functionality
// which we may want to call into from CPU code (and thus must be dynamically
// dispatched, to allow for separate compilation of CUDA code).  How do I
// decide if a function should live in this class?  There are two tests:
//
//  1. Does the *implementation* of this function require linking against
//     CUDA libraries?
//
//  2. Is this function *called* from non-CUDA ATen code?
//
// (2) should filter out many ostensible use-cases, since many times a CUDA
// function provided by ATen is only really ever used by actual CUDA code.
//
// TODO: Consider putting the stub definitions in another class, so that one
// never forgets to implement each virtual function in the real implementation
// in CUDAHooks.  This probably doesn't buy us much though.
struct TORCH_API ZoomHooksInterface : PrivateUse1HooksInterface {
  // This should never actually be implemented, but it is used to
  // squelch -Werror=non-virtual-dtor
  virtual ~ZoomHooksInterface() override = default;

  // Initialize THCState and, transitively, the CUDA state
  virtual void initZoom() const {
    TORCH_CHECK(false, "Cannot initialize CUDA without ATen_cuda library. ", ZOOM_HELP);
  }

  virtual const Generator& getDefaultZoomGenerator(C10_UNUSED DeviceIndex device_index = -1) const {
    TORCH_CHECK(false, "Cannot get default CUDA generator without ATen_cuda library. ", ZOOM_HELP);
  }

  virtual const Generator& GetDefaultGenerator(DeviceIndex device_index);

  virtual Device getDeviceFromPtr(void* /*data*/) const {
    TORCH_CHECK(false, "Cannot get device of pointer on CUDA without ATen_cuda library. ", ZOOM_HELP);
  }

  virtual bool isPinnedPtr(const void* /*data*/) const {
    return false;
  }

  virtual bool hasROCM() const {
    return false;
  }

  virtual bool hasPrimaryContext(DeviceIndex device_index) const override {
    TORCH_CHECK(false, "Cannot call hasPrimaryContext(", device_index, ") without ATen_cuda library. ", ZOOM_HELP);
  }

  virtual DeviceIndex current_device() const {
    return -1;
  }

  virtual Allocator* getPinnedMemoryAllocator() const {
    TORCH_CHECK(false, "Pinned memory requires CUDA. ", ZOOM_HELP);
  }

  virtual Allocator* getZoomDeviceAllocator() const {
    TORCH_CHECK(false, "CUDADeviceAllocator requires CUDA. ", ZOOM_HELP);
  }

  virtual std::string showConfig() const {
    TORCH_CHECK(false, "Cannot query detailed CUDA version without ATen_cuda library. ", ZOOM_HELP);
  }

  virtual int getNumGPUs() const {
    return 0;
  }

  virtual void deviceSynchronize(DeviceIndex /*device_index*/) const {
    TORCH_CHECK(false, "Cannot synchronize CUDA device without ATen_cuda library. ", ZOOM_HELP);
  }
};

// NB: dummy argument to suppress "ISO C++11 requires at least one argument
// for the "..." in a variadic macro"
// struct TORCH_API CUDAHooksArgs {};

// TORCH_DECLARE_REGISTRY(CUDAHooksRegistry, ZoomHooksInterface, CUDAHooksArgs);
// #define REGISTER_CUDA_HOOKS(clsname) \
//   C10_REGISTER_CLASS(CUDAHooksRegistry, clsname, clsname)

struct TORCH_API ZoomHooksArgs {};

namespace detail {
TORCH_API const ZoomHooksInterface& getZoomHooks();
} // namespace detail
} // namespace at