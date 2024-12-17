#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/zoom/DistributionTemplates.h>

namespace at::native {

void geometric_kernel(TensorIteratorBase& iter, double p_, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::geometric_kernel(iter, p_, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(geometric_stub, &geometric_kernel);

} // namespace at::native
