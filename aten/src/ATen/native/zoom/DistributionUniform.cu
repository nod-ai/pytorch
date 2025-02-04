#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/zoom/DistributionTemplates.h>

namespace at::native {

void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  templates::zoom::uniform_kernel(iter, from, to, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(uniform_stub, &uniform_kernel);

} // namespace at::native