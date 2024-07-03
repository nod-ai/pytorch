// #define TORCH_ASSERT_NO_OPERATORS
#include "../ZoomGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include "DistributionTemplates.h"
#include <ATen/native/TensorFactories.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

namespace at::native {

void cauchy_kernel(TensorIteratorBase& iter, double median, double sigma, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::cauchy_kernel(iter, median, sigma, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(cauchy_stub, &cauchy_kernel);

// See note: [Declarations for Random Stubs]

Tensor & cauchy_(Tensor & self, double median, double sigma, ::std::optional<Generator> generator); // {"schema": "aten::cauchy_(Tensor(a!) self, float median=0, float sigma=1, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}


Tensor & zoom_cauchy_(Tensor & self, double median, double sigma, ::std::optional<Generator> generator) {
    return cauchy_(self, median, sigma, std::move(generator));
}
Tensor & zoom_cauchy_out(const Tensor & self, double median, double sigma, ::std::optional<Generator> generator, Tensor & out) {
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

    cauchy_kernel(iter, median, sigma, std::move(generator));

    return out;
}

Tensor zoom_cauchy(const Tensor & self, double median, double sigma, ::std::optional<Generator> generator) {
    // Create a new tensor with the same size and dtype as the input tensor
    Tensor out = at::empty_like(self);
    zoom_cauchy_out(self, median, sigma, generator, out);
    return out;
}



TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("cauchy_", &zoom_cauchy_);
    m.impl("cauchy.out", &zoom_cauchy_out);
    m.impl("cauchy", &zoom_cauchy);
}


} // namespace at::native
