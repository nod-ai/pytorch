// #define TORCH_ASSERT_NO_OPERATORS
#include <torch/library.h>
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/zoom/DistributionTemplates.h>

namespace at::native {

// See note: [Declarations for Random Stubs]
Tensor & uniform_(Tensor & self, double from, double to, ::std::optional<Generator> generator);

void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  templates::zoom::uniform_kernel(iter, from, to, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(uniform_stub, &uniform_kernel);

Tensor & zoom_uniform_(Tensor & self, double from, double to, ::std::optional<Generator> generator) {
  return uniform_(self, from, to, std::move(generator));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("uniform_", &zoom_uniform_);
}

} // namespace at::native