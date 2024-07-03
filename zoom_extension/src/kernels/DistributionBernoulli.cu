// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include "../ZoomApplyUtils.cuh"
#include <ATen/AccumulateType.h>
#include "../ZoomGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include "DistributionTemplates.h"
#include <ATen/native/TensorFactories.h>
#include <ATen/native/Resize.h>

#include <hiprand.h>
#include <hiprand_kernel.h>
//#include <curand_philox4x32_x.h>
#include <utility>
#include <functional>

#include <ATen/native/Distributions.h>
#include "../jit/Loops.cuh"
#include <ATen/native/TensorIterator.h>

#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

#include <torch/library.h>

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

// See note: [Declarations for Random Stubs]


Tensor bernoulli(const Tensor & self, ::std::optional<Generator> generator); // {"schema": "aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor", "dispatch": "True", "default": "True"}
Tensor & bernoulli_out(const Tensor & self, ::std::optional<Generator> generator, Tensor & out); // {"schema": "aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor & bernoulli_(Tensor & self, const Tensor & p, ::std::optional<Generator> generator); // {"schema": "aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor & bernoulli_(Tensor & self, double p, ::std::optional<Generator> generator); // {"schema": "aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor bernoulli(const Tensor & self, double p, ::std::optional<Generator> generator); // {"schema": "aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor", "dispatch": "True", "default": "True"}


Tensor zoom_bernoulli(const Tensor & self, ::std::optional<Generator> generator) {
    return at::native::bernoulli(self, std::move(generator));
}
Tensor & zoom_bernoulli_out1(const Tensor & self, ::std::optional<Generator> generator, Tensor & out) {
    return bernoulli_out(self, std::move(generator), out);
}
Tensor & zoom_bernoulli_(Tensor & self, const Tensor & p, ::std::optional<Generator> generator) {
    return bernoulli_(self, p, std::move(generator));
}
Tensor & zoom_bernoulli_p(Tensor & self, double p, ::std::optional<Generator> generator) {
    return bernoulli_(self, p, std::move(generator));
}
Tensor zoom_bernoulli_p2(const Tensor & self, double p, ::std::optional<Generator> generator) {
    return at::native::bernoulli(self, p, std::move(generator));
}

Tensor & zoom_bernoulli_out2(const Tensor & self, const Tensor & p, ::std::optional<Generator> generator, Tensor & out) {
  // Check if the output tensor has the correct shape and dtype
  at::native::resize_output(out, self.sizes());
  
  // If the output tensor doesn't match the input tensor's dtype, we need to cast it
  if (out.scalar_type() != self.scalar_type()) {
      out = out.to(self.scalar_type());
  }
  bernoulli_tensor_kernel(out, p, std::move(generator));
  return out;
}
Tensor zoom_bernoulli_p3(const Tensor & self, const Tensor & p, ::std::optional<Generator> generator) {
    // Create a new tensor with the same size and dtype as the input tensor
    Tensor out = at::empty_like(self);
    zoom_bernoulli_out2(self, p, generator, out);
    return out;
}
Tensor & zoom_bernoulli_out3(const Tensor & self, double p, ::std::optional<Generator> generator, Tensor & out) {
  // Check if the output tensor has the correct shape and dtype
  at::native::resize_output(out, self.sizes());
  
  // If the output tensor doesn't match the input tensor's dtype, we need to cast it
  if (out.scalar_type() != self.scalar_type()) {
      out = out.to(self.scalar_type());
  }
  bernoulli_scalar_kernel(out, p, std::move(generator));
  return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("bernoulli", &zoom_bernoulli);
    m.impl("bernoulli.out", &zoom_bernoulli_out1);
    m.impl("bernoulli_.Tensor", &zoom_bernoulli_);
    m.impl("bernoulli_.float", &zoom_bernoulli_p);
    m.impl("bernoulli.p", &zoom_bernoulli_p2);
    m.impl("bernoulli.Tensor_out", &zoom_bernoulli_out2);
    m.impl("bernoulli.Tensor", &zoom_bernoulli_p3);
    m.impl("bernoulli.float_out", &zoom_bernoulli_out3);
}


} // namespace at::native
