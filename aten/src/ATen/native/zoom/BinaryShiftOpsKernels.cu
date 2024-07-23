#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {


void lshift_kernel_zoom(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "lshift_zoom", [&]() {
    gpu_kernel_with_scalars(iter,
      []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        constexpr scalar_t max_shift = sizeof(scalar_t) * CHAR_BIT;
        if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) || (b >= max_shift)) {
          return 0;
        }
        return static_cast<std::make_unsigned_t<scalar_t>>(a) << b;
    });
  });
}

void rshift_kernel_zoom(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES(iter.dtype(), "rshift_zoom", [&]() {
    gpu_kernel_with_scalars(iter,
      []GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
        // right shift value to retain sign bit for signed and no bits for unsigned
        constexpr scalar_t max_shift = sizeof(scalar_t) * CHAR_BIT - std::is_signed_v<scalar_t>;
        if ((static_cast<std::make_signed_t<scalar_t>>(b) < 0) || (b >= max_shift)) {
          return a >> max_shift;
        }
        return a >> b;
    });
  });
}

REGISTER_PRIVATEUSE1_DISPATCH(lshift_stub, &lshift_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(rshift_stub, &rshift_kernel_zoom);

} // namespace at::native
