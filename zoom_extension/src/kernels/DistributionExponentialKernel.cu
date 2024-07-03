// #define TORCH_ASSERT_NO_OPERATORS
#include "../ZoomGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include "DistributionTemplates.h"
#include <ATen/native/TensorFactories.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>


namespace at::native {

void exponential_kernel(TensorIteratorBase& iter, double lambda, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::exponential_kernel(iter, lambda, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(exponential_stub, &exponential_kernel);

// See note: [Declarations for Random Stubs]


// Tensor & exponential_out(const Tensor & self, double lambd, ::std::optional<Generator> generator, Tensor & out); // {"schema": "aten::exponential.out(Tensor self, float lambd=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "True"}
// Tensor exponential(const Tensor & self, double lambd, ::std::optional<Generator> generator); // {"schema": "aten::exponential(Tensor self, float lambd=1, *, Generator? generator=None) -> Tensor", "dispatch": "True", "default": "True"}
Tensor & exponential_(Tensor & self, double lambd, ::std::optional<Generator> generator); // {"schema": "aten::exponential_(Tensor(a!) self, float lambd=1, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}

Tensor & zoom_exponential_out(const Tensor & self, double lambd, ::std::optional<Generator> generator, Tensor & out) {
    // Check if the output tensor has the correct shape and dtype
    at::native::resize_output(out, self.sizes());
    
    // If the output tensor doesn't match the input tensor's dtype, we need to cast it
    if (out.scalar_type() != self.scalar_type()) {
        out = out.to(self.scalar_type());
    }

    auto iter = TensorIteratorConfig()
        .add_output(out)
        .add_input(self)
        .check_all_same_dtype(false)
        .build();

    exponential_kernel(iter, lambd, std::move(generator));

    return out;
}
Tensor zoom_exponential(const Tensor & self, double lambd, ::std::optional<Generator> generator) {
    // Create a new tensor with the same size and dtype as the input tensor
    Tensor out = at::empty_like(self);
    zoom_exponential_out(self, lambd, generator, out);
    return out;
}
Tensor & zoom_exponential_(Tensor & self, double lambd, ::std::optional<Generator> generator) {
    return exponential_(self, lambd, std::move(generator));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("exponential.out", &zoom_exponential_out);
    m.impl("exponential", &zoom_exponential);
    m.impl("exponential_", &zoom_exponential_);
}


} // namespace at::native
