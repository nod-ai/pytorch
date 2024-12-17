#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/zoom/jit/Loops.cuh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

void maximum_kernel_zoom(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(
        iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a || b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "max_elementwise_zoom", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::max(a, b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "max_elementwise_zoom", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
          iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        if (a != a) {
          return a;
        } else if (b != b) {
          return b;
        } else {
          return ::max(a, b);
        }
      });
    });
  }
}

void minimum_kernel_zoom(TensorIteratorBase& iter) {
  if (iter.dtype() == ScalarType::Bool) {
    opmath_symmetric_gpu_kernel_with_scalars<bool>(iter, []GPU_LAMBDA(bool a, bool b) -> bool {
      return a && b;
    });
  } else if (isIntegralType(iter.dtype(), /*includeBool=*/ false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "minimum_zoom", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::min(a, b);
      });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.dtype(), "min_elementwise_zoom", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        if (a != a) {
          return a;
        } else if (b != b) {
          return b;
        } else {
          return ::min(a, b);
        }
      });
    });
  }
}

void fmax_kernel_zoom(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "fmax_zoom", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::fmax(a, b);
      });
    });
  } else {
    maximum_kernel_zoom(iter);
  }
}

void fmin_kernel_zoom(TensorIteratorBase& iter) {
  if (isFloatingType(iter.common_dtype())) {
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, iter.common_dtype(), "fmin_zoom", [&]() {
      opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        return ::fmin(a, b);
      });
    });
  } else {
    minimum_kernel_zoom(iter);
  }
}

REGISTER_PRIVATEUSE1_DISPATCH(maximum_stub, &maximum_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(minimum_stub, &minimum_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(fmax_stub, &fmax_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(fmin_stub, &fmin_kernel_zoom);

} // namespace at::native