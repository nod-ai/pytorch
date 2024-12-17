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

template <typename scalar_t, typename acc_t = scalar_t>
void argmax_kernel_zoom_impl(TensorIterator& iter) {
  gpu_reduce_kernel<scalar_t, int64_t>(
      iter,
      ArgMaxOps<acc_t>{},
      thrust::pair<acc_t, int64_t>(
          at::numeric_limits<acc_t>::lower_bound(), 0));
};

void argmax_kernel_zoom(TensorIterator& iter) {
  // For float16 & bfloat16, instead of implementing is_nan and warp_shfl_down,
  // we can convert float16 & bfloat16 to float and do all the operations in
  // float.
  if (iter.dtype(1) == kHalf) {
    argmax_kernel_zoom_impl<at::Half, float>(iter);
  } else if (iter.dtype(1) == kBFloat16) {
    argmax_kernel_zoom_impl<at::BFloat16, float>(iter);
  } else {
    AT_DISPATCH_ALL_TYPES(iter.dtype(1), "argmax_zoom", [&]() {
      argmax_kernel_zoom_impl<scalar_t>(iter);
    });
  }
}

REGISTER_PRIVATEUSE1_DISPATCH(argmax_stub, &argmax_kernel_zoom);

} // namespace at::native
