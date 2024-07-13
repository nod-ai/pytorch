#include <ATen/detail/ZoomHooksInterface.h>

#include <c10/util/CallOnce.h>

#include <memory>

namespace at {
namespace detail {

// NB: We purposely leak the CUDA hooks object.  This is because under some
// situations, we may need to reference the CUDA hooks while running destructors
// of objects which were constructed *prior* to the first invocation of
// getZoomHooks.  The example which precipitated this change was the fused
// kernel cache in the JIT.  The kernel cache is a global variable which caches
// both CPU and CUDA kernels; CUDA kernels must interact with CUDA hooks on
// destruction.  Because the kernel cache handles CPU kernels too, it can be
// constructed before we initialize CUDA; if it contains CUDA kernels at program
// destruction time, you will destruct the CUDA kernels after CUDA hooks has
// been unloaded.  In principle, we could have also fixed the kernel cache store
// CUDA kernels in a separate global variable, but this solution is much
// simpler.
//
// CUDAHooks doesn't actually contain any data, so leaking it is very benign;
// you're probably losing only a word (the vptr in the allocated object.)
static ZoomHooksInterface* zoom_hooks = nullptr;

// init and register extension hooks
void initZoomHooks() {
  if (zoom_hooks == nullptr) {
    zoom_hooks = new ZoomHooksInterface();
    RegisterPrivateUse1HooksInterface(zoom_hooks);
  }
}

const ZoomHooksInterface& getZoomHooks() {
  initZoomHooks();
  return *zoom_hooks;
}

} // namespace detail

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, ZoomHooksInterface, ZoomHooksArgs)

} // namespace at