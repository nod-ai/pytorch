#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/zoom/DistributionTemplates.h>

namespace at::native {

void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::normal_kernel(self, mean, std, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(normal_stub, &normal_kernel);

} // namespace at::native
