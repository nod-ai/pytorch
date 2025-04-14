#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/zoom/ZoomApplyUtils.cuh>
#include <ATen/AccumulateType.h>
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/zoom/DistributionTemplates.h>

#include <hiprand.h>
#include <hiprand_kernel.h>
#include <utility>
#include <functional>

#include <ATen/native/Distributions.h>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/native/TensorIterator.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

namespace at::native {

void bernoulli_tensor_kernel(const TensorBase &self, const TensorBase &p_, std::optional<Generator> gen_) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen_, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::bernoulli_kernel(self, p_, generator);
}

void bernoulli_scalar_kernel(const TensorBase &self, double p, std::optional<Generator> gen) {
  auto iter = TensorIterator::borrowing_nullary_op(self);
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::bernoulli_kernel(iter, p, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(bernoulli_tensor_stub, &bernoulli_tensor_kernel);
REGISTER_PRIVATEUSE1_DISPATCH(bernoulli_scalar_stub, &bernoulli_scalar_kernel);


} // namespace at::native
