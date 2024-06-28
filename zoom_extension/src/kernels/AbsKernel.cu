// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include "../jit/Loops.cuh"
#include "../jit/JitLoops.cuh"
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include "../extension.h"
// #include <torch/library.h>

namespace at::native {


/*CONSTEXPR_EXCEPT_WIN_CUDA*/ constexpr char abs_name[] = "abs_kernel";
void abs_kernel_zoom(TensorIteratorBase& iter) {
  auto dtype = iter.dtype();
  static const auto abs_string = jiterator_stringify(
        template <typename T> T abs_kernel(T x) { return std::abs(x); });
  if (at::isComplexType(dtype)) {
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "abs_zoom", [&]() {
      jitted_gpu_kernel<
          /*name=*/abs_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, abs_string);
    });
  } else {
    AT_DISPATCH_ALL_TYPES_AND3(ScalarType::Half,
        ScalarType::BFloat16,
        ScalarType::Bool,
        iter.dtype(),
        "abs_zoom", [&]() {
      jitted_gpu_kernel<
          /*name=*/abs_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, abs_string);
    });
  }
}

REGISTER_PRIVATEUSE1_DISPATCH(abs_stub, &abs_kernel_zoom);

// Tensor& abs_out(const Tensor& self, Tensor& result) {
//   return unary_op_impl_with_complex_to_float_out(result, self, abs_stub, /*promotes_integer_to_float=*/false);
// }

// Tensor abs(const Tensor & self) {
//   return unary_op_impl_with_complex_to_float(self, at::zoom::native::abs_out);
// }

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("abs", &abs);
  m.impl("abs.out", &abs_out);
}


} // namespace at::native