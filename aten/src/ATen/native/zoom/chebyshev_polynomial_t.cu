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
            CONSTEXPR_EXCEPT_WIN_CUDA char chebyshev_polynomial_t_name[] = "chebyshev_polynomial_t_forward";

            void chebyshev_polynomial_t_kernel_zoom(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_t_zoom", [&]() {
                    opmath_jitted_gpu_kernel_with_scalars<chebyshev_polynomial_t_name, scalar_t, scalar_t>(iterator, chebyshev_polynomial_t_string);
                });
#else
                AT_DISPATCH_FLOATING_TYPES(iterator.common_dtype(), "chebyshev_polynomial_t_zoom", [&]() {
                    gpu_kernel_with_scalars(iterator, []GPU_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
                        return chebyshev_polynomial_t_forward<scalar_t, true>(x, n);
                    });
                });
#endif
            } // chebyshev_polynomial_t_kernel_zoom
        } // namespace (anonymous)

        REGISTER_PRIVATEUSE1_DISPATCH(chebyshev_polynomial_t_stub, &chebyshev_polynomial_t_kernel_zoom);
} // namespace at::native
