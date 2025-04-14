#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/zoom/DistributionTemplates.h>

namespace at::native {

void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::log_normal_kernel(iter, mean, std, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(log_normal_stub, &log_normal_kernel);

} // namespace at::native
