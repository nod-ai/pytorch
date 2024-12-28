#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/TensorIterator.h>
#include <ATen/native/zoom/Reduce.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/Dispatch.h>

namespace at::native {

void and_kernel_zoom(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "and_zoom", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
              return (static_cast<bool>(a) && static_cast<bool>(b));
            }),
            true);
      });
}

void or_kernel_zoom(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      kHalf, kBFloat16, kBool, iter.common_dtype(), "or_zoom", [&]() {
        gpu_reduce_kernel<scalar_t, bool>(
            iter,
            func_wrapper<bool>([] GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
              return (static_cast<bool>(a) || static_cast<bool>(b));
            }),
            false);
      });
}

REGISTER_PRIVATEUSE1_DISPATCH(and_stub, &and_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(or_stub, &or_kernel_zoom);

} // namespace at::native