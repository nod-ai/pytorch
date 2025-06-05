#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/zoom/DistributionTemplates.h>

namespace at::native {

void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<ZoomGeneratorImpl>(gen_, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::random_from_to_kernel(iter, range, base, gen);
}

void random_full_64_bits_range_kernel(TensorIteratorBase& iter, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<ZoomGeneratorImpl>(gen_, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::random_full_64_bits_range_kernel(iter, gen);
}

void random_kernel(TensorIteratorBase& iter, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<ZoomGeneratorImpl>(gen_, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::random_kernel(iter, gen);
}

REGISTER_PRIVATEUSE1_DISPATCH(random_from_to_stub, &random_from_to_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(random_stub, &random_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(random_full_64_bits_range_stub, &random_full_64_bits_range_kernel);

} // namespace at::native
