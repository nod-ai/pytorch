#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <c10/zoom/HIPMathCompat.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

void copysign_kernel_zoom(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.common_dtype(), "copysign_zoom", [&]() {
    gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
      return c10::hip::compat::copysign(a, b);
    });
  });
}

REGISTER_PRIVATEUSE1_DISPATCH(copysign_stub, &copysign_kernel_zoom);

} // namespace at::native
