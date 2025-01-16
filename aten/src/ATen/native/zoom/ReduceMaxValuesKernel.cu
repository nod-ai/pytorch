#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/zoom/ReduceOps.h>
#include <ATen/zoom/NumericLimits.cuh>
#include <ATen/native/zoom/Reduce.cuh>

#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/zoom/NumericLimits.cuh>

namespace at::native {

template <typename acc_t>
struct MaxNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    return (at::_isnan(a) || a > b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
void max_values_kernel_zoom_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      func_wrapper<acc_t>(MaxNanFunctor<acc_t>()),
      at::numeric_limits<acc_t>::lower_bound());
}

void max_values_kernel_zoom(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(), "max_values_zoom", [&]() {
        max_values_kernel_zoom_impl<scalar_t>(iter);
      });
}

void max_launch_kernel(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.input_dtype(), "max_zoom", [&]() {
        gpu_reduce_kernel<scalar_t, scalar_t>(
            iter,
            MaxOps<scalar_t>{},
            thrust::pair<scalar_t, int64_t>(
                at::numeric_limits<scalar_t>::lower_bound(), 0));
      });
}

void max_all_launch_kernel(TensorIterator &iter) {
  AT_DISPATCH_ALL_TYPES_AND3(kBFloat16, kHalf, kBool, iter.input_dtype(), "max_all_zoom", [&] {
    max_values_kernel_zoom_impl<scalar_t>(iter);
  });
}

REGISTER_PRIVATEUSE1_DISPATCH(max_values_stub, &max_values_kernel_zoom);

} // namespace at::native
