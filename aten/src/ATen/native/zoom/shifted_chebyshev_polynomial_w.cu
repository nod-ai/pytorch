// !!! This is a file automatically generated by hipify!!!
#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/zoom/jit/JitLoops.cuh>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/zoom/Math.cuh>
#include <ATen/zoom/jit/jit_utils.h>

namespace at::native {
        namespace {
            CONSTEXPR_EXCEPT_WIN_CUDA char shifted_chebyshev_polynomial_w_name[] = "shifted_chebyshev_polynomial_w_forward";

            void shifted_chebyshev_polynomial_w_kernel_zoom(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_zoom", [&]() {
                    opmath_jitted_gpu_kernel_with_scalars<shifted_chebyshev_polynomial_w_name, scalar_t, scalar_t>(iterator, shifted_chebyshev_polynomial_w_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "shifted_chebyshev_polynomial_w_zoom", [&]() {
                    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
                        return shifted_chebyshev_polynomial_w_forward<scalar_t, true>(x, n);
                    });
                });
#endif
            } // shifted_chebyshev_polynomial_w_kernel_zoom
        } // namespace (anonymous)

        REGISTER_PRIVATEUSE1_DISPATCH(shifted_chebyshev_polynomial_w_stub, &shifted_chebyshev_polynomial_w_kernel_zoom);
} // namespace at::native
