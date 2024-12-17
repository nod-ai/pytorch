#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/zoom/jit/JitLoops.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/zoom/Math.cuh>
#include <c10/util/TypeSafeSignMath.h>
#include <ATen/OpMathType.h>

#include <type_traits>

namespace at::native {

void logical_not_kernel_zoom(TensorIteratorBase& iter) {
  // error check -- this is just ensuring we don't dispatch on types that aren't in ALL_TYPES_AND_COMPLEX_AND3(...)
  // so we don't have to maintain a separate list or to do double dispatch.
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(0), "logical_not_zoom", [&]() {});

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(kBool, kHalf, kBFloat16, iter.dtype(1), "logical_not_zoom", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return !a; });
  });
}

// NB: Ignores the negative bit on tensors
CONSTEXPR_EXCEPT_WIN_CUDA char neg_name[] = "neg_kernel";
void neg_kernel_zoom(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  if (at::isComplexType(dtype)) {
  static const auto neg_string = jiterator_stringify(
      template <typename T>
      T neg_kernel(T a) {
        return -a;
      }
  ); // neg_string
  AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "neg_zoom", [&]() {
      jitted_gpu_kernel<
        /*name=*/ neg_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, neg_string);
  });

  } else {
  AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, dtype, "neg_zoom", [&]() {
    gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
      return -a;
    });
  });
  }
}

void sign_kernel_zoom(TensorIteratorBase& iter){
  if (iter.dtype() == ScalarType::Bool) {
    gpu_kernel(iter, []GPU_LAMBDA(bool a){
      return a;
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND2(ScalarType::Half, ScalarType::BFloat16, iter.dtype(), "sign_zoom", [&]() {
        gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> scalar_t {
            return c10::signum(a);
        });
    });
  }
}

void signbit_kernel_zoom(TensorIteratorBase& iter){
  // NOTE: signbit does not always support integral arguments.
  if (at::isIntegralType(iter.input_dtype(), /*includeBool=*/false)) {
    AT_DISPATCH_INTEGRAL_TYPES(iter.input_dtype(), "signbit_zoom", [&]() {
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return is_negative(a); });
    });
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(kBFloat16, ScalarType::Half, iter.input_dtype(), "signbit_zoom", [&]() {
      using opmath_t = at::opmath_type<scalar_t>;
      gpu_kernel(iter, []GPU_LAMBDA(scalar_t a) -> bool { return signbit(opmath_t{a}); });
    });
  }
}

template<typename T>
C10_HOST_DEVICE static inline c10::complex<T> sgn_wrapper(c10::complex<T> z) {
  if (z == c10::complex<T>(0, 0)) {
    return c10::complex<T>(0, 0);
  } else {
    return z / std::abs(z);
  }
}

CONSTEXPR_EXCEPT_WIN_CUDA char sgn_name[] = "sgn_kernel";
void sgn_kernel_zoom(TensorIteratorBase& iter){
  auto dtype = iter.dtype();
    static const auto sgn_string = jiterator_stringify(
        template <typename T>
        T sgn_kernel(T z) {
          const T zero = T(0);
          if (z == zero) {
            return zero;
          } else {
            return z / std::abs(z);
          }
        }
      ); // sgn_string
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "sgn_zoom", [&]() {
      jitted_gpu_kernel<
        /*name=*/ sgn_name,
        /*return_dtype=*/ scalar_t,
        /*common_dtype=*/ scalar_t,
        /*arity=*/ 1>(iter, sgn_string);
      });
}

REGISTER_PRIVATEUSE1_DISPATCH(logical_not_stub, &logical_not_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(neg_stub, &neg_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(sign_stub, &sign_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(signbit_stub, &signbit_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(sgn_stub, &sgn_kernel_zoom);

} // namespace at::native