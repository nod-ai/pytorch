#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/zoom/DistributionTemplates.h>


namespace at::native {

void exponential_kernel(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::exponential_kernel(iter, lambda, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(exponential_stub, &exponential_kernel);

} // namespace at::native
