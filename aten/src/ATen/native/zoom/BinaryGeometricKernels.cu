#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

void atan2_kernel_zoom(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.common_dtype(), "atan2_zoom",
      [&]() {
        gpu_kernel_with_scalars(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return ::atan2(a, b);
        });
      });
}

void hypot_kernel_zoom(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      iter.common_dtype(), "hypot_zoom",
      [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
            iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
          return ::hypot(a, b);
        });
      });
}

REGISTER_DISPATCH(atan2_stub, &atan2_kernel_zoom);
REGISTER_DISPATCH(hypot_stub, &hypot_kernel_zoom);

} // namespace at::native
