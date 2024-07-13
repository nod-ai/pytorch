// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/zoom/DistributionTemplates.h>
#include <ATen/native/DistributionTemplates.h>
#include <torch/library.h>

namespace at::native {

// Note: [Declarations for Random Stubs]
// these are defined in ATen/native/Distributions.cpp but not exposed, in order to avoid
// duplicating logic we override the appropriate dispatch stubs and then route the 
// dispatcher for PU1 through these generic random functions, this is also something that
// can be refactored once we're in-tree. This is necessary because by default, the PU1
// backend does not search for dispatch stubs but instead uses the new dispatch key mechanism
// to enable C++ extensions - however, using dispatch stubs enables the least code repetition

Tensor & random_(Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<Generator> generator); 
Tensor & random_(Tensor & self, int64_t to, ::std::optional<Generator> generator);
Tensor & random_(Tensor & self, ::std::optional<Generator> generator);



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

Tensor & zoom_random_from(Tensor & self, int64_t from, ::std::optional<int64_t> to, ::std::optional<Generator> generator) {
  return random_(self, from, to, std::move(generator));
}
Tensor & zoom_random_to(Tensor & self, int64_t to, ::std::optional<Generator> generator) {
  return random_(self, to, std::move(generator));
}
Tensor & zoom_random_(Tensor & self, ::std::optional<Generator> generator) {
  return random_(self, std::move(generator));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("random_.from", &zoom_random_from);
  m.impl("random_.to", &zoom_random_to);
  m.impl("random_", &zoom_random_);
}



} // namespace at::native
