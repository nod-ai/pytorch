// #define TORCH_ASSERT_NO_OPERATORS
#include "../ZoomGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include "DistributionTemplates.h"
#include <ATen/native/TensorFactories.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

namespace at::native {

void log_normal_kernel(TensorIteratorBase& iter, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::log_normal_kernel(iter, mean, std, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(log_normal_stub, &log_normal_kernel);

// See note: [Declarations for Random Stubs]


Tensor & log_normal_(Tensor & self, double mean, double std, ::std::optional<Generator> generator); // {"schema": "aten::log_normal_(Tensor(a!) self, float mean=1, float std=2, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}


Tensor & zoom_log_normal_(Tensor & self, double mean, double std, ::std::optional<Generator> generator) {
    return log_normal_(self, mean, std, std::move(generator));
}
Tensor & zoom_log_normal_out(const Tensor & self, double mean, double std, ::std::optional<Generator> generator, Tensor & out) {
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

    log_normal_kernel(iter, mean, std, std::move(generator));

    return out;
}
Tensor zoom_log_normal(const Tensor & self, double mean, double std, ::std::optional<Generator> generator) {
    // Create a new tensor with the same size and dtype as the input tensor
    Tensor out = at::empty_like(self);
    zoom_log_normal_out(self, mean, std, generator, out);
    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("log_normal_", &zoom_log_normal_);
    m.impl("log_normal.out", &zoom_log_normal_out);
    m.impl("log_normal", &zoom_log_normal);
}


} // namespace at::native
