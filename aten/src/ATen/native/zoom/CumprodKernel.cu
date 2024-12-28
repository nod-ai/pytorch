#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/core/TensorBase.h>
#include <ATen/Dispatch.h>

#include <ATen/native/zoom/ScanKernels.h>
#include <ATen/native/zoom/ScanUtils.cuh>

namespace at::native {

void launch_cumprod_zoom_kernel(const TensorBase& result, const TensorBase& self, int64_t dim) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half, ScalarType::BFloat16, self.scalar_type(), "cumprod_zoom", [&]() {
        scalar_t init = 1;
        scan_dim<scalar_t>(
            self,
            result,
            dim,
            init,
            std::multiplies<scalar_t>());
      });
}

} // namespace at::native
