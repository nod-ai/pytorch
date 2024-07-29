// !!! This is a file automatically generated by hipify!!!
#define TORCH_ASSERT_NO_OPERATORS
#include <limits>
#include <ATen/native/UnaryOps.h>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/zoom/jit/jit_utils.h>
#include <ATen/zoom/jit/JitLoops.cuh>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/zoom/Math.cuh>

namespace at::native {

#if AT_USE_JITERATOR()
CONSTEXPR_EXCEPT_WIN_CUDA char log_name[] = "log_kernel";
#endif

void log_kernel_zoom(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR()
    static const auto log_string = jiterator_stringify(
        template <typename T> T log_kernel(T x) { return ::log(x); });
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, common_dtype, "log_zoom", [&]() {
      jitted_gpu_kernel<
          /*name=*/log_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, log_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, iter.common_dtype(), "log_zoom", [&]() {
      gpu_kernel(
          iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return ::log(static_cast<opmath_t>(a));
          });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "log_zoom", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::log(a);
      });
    });
  }
}

CONSTEXPR_EXCEPT_WIN_CUDA char log10_name[] = "log10_kernel";
void log10_kernel_zoom(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR()
    static const auto log10_string = jiterator_stringify(
        template <typename T> T log10_kernel(T x) { return std::log10(x); });
    AT_DISPATCH_COMPLEX_TYPES(common_dtype, "log10_zoom", [&]() {
      jitted_gpu_kernel<
          /*name=*/log10_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, log10_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "log10_zoom", [&]() {
      gpu_kernel(
          iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return ::log10(a); });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "log10_zoom", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::log10(a);
      });
    });
  }
}

void log1p_kernel_zoom(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "log1p_zoom", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return ::log1p(a);
    });
  });
}

CONSTEXPR_EXCEPT_WIN_CUDA char log2_name[] = "log2_kernel";
void log2_kernel_zoom(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
#if AT_USE_JITERATOR()
    static const auto log2_string = jiterator_stringify(
        template <typename T> T log2_kernel(T x) { return std::log2(x); });
    AT_DISPATCH_COMPLEX_TYPES(common_dtype, "log2_zoom", [&]() {
      jitted_gpu_kernel<
          /*name=*/log2_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, log2_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES(iter.common_dtype(), "log2_zoom", [&]() {
      gpu_kernel(
          iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return ::log2(a); });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.common_dtype(), "log2_zoom", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
        return ::log2(a);
      });
    });
  }
}

REGISTER_PRIVATEUSE1_DISPATCH(log_stub, &log_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(log10_stub, &log10_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(log2_stub, &log2_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(log1p_stub, &log1p_kernel_zoom);

} // namespace at::native
