#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/native/UnaryOps.h>

#include <limits>

#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/zoom/jit/JitLoops.cuh>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/native/zoom/Math.cuh>
#include <ATen/zoom/jit/jit_utils.h>
#include <ATen/NumericUtils.h>
#include <c10/core/Scalar.h>
#include <c10/zoom/HIPMathCompat.h>
#include <c10/util/complex.h>

namespace at::native {
namespace {
CONSTEXPR_EXCEPT_WIN_CUDA char bessel_j0_name[] = "bessel_j0_forward";

void bessel_j0_kernel_zoom(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j0_zoom", [&]() {
        jitted_gpu_kernel<bessel_j0_name, scalar_t, scalar_t, 1>(iterator, bessel_j0_string);
    });
#else
    AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "bessel_j0_zoom", [&]() {
        gpu_kernel(iterator, []GPU_LAMBDA(scalar_t a) -> scalar_t {
            return bessel_j0_forward(a);
        });
    });
#endif // AT_USE_JITERATOR()
}

} // anonymous namespace

REGISTER_PRIVATEUSE1_DISPATCH(special_bessel_j0_stub, &bessel_j0_kernel_zoom);
} // namespace at::native
