#define TORCH_ASSERT_NO_OPERATORS
#define _USE_MATH_DEFINES

#include <ATen/native/Activation.h>

#include <cmath>

#include <thrust/tuple.h>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <c10/core/Scalar.h>
#include <c10/zoom/HIPMathCompat.h>
#include <ATen/zoom/ApplyGridUtils.cuh>
#include <ATen/zoom/jit/OffsetCalculator.cuh>
#include <ATen/zoom/jit/Loops.cuh>

namespace at::native {

// -----------------------------------
// prelu
// -----------------------------------
void prelu_kernel(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_zoom", [&] {
    gpu_kernel(iter,
      [] GPU_LAMBDA (scalar_t input, scalar_t weight) -> scalar_t {
        return (input > 0) ? input : weight * input;
      });
  });
}

void prelu_backward_kernel(TensorIterator &iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, kHalf, iter.dtype(), "prelu_backward_zoom", [&] {
    gpu_kernel_multiple_outputs(iter,
      [] GPU_LAMBDA (scalar_t input, scalar_t weight, scalar_t grad) -> thrust::tuple<scalar_t, scalar_t> {
        auto mask = input > 0;
        auto grad_input = mask ? grad : weight * grad;
        auto grad_weight = mask ? scalar_t{0} : input * grad;
        return {grad_input, grad_weight};
      });
  });
}

REGISTER_PRIVATEUSE1_DISPATCH(prelu_stub, &prelu_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(prelu_backward_stub, &prelu_backward_kernel);

} // namespace at::native
