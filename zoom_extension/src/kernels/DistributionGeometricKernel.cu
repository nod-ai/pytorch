// #define TORCH_ASSERT_NO_OPERATORS
#include "../ZoomGeneratorImpl.h"
#include <ATen/native/UnaryOps.h>
#include "DistributionTemplates.h"
#include <ATen/native/TensorFactories.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

namespace at::native {

void geometric_kernel(TensorIteratorBase& iter, double p_, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::geometric_kernel(iter, p_, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(geometric_stub, &geometric_kernel);

// See note: [Declarations for Random Stubs]


Tensor & geometric_(Tensor & self, double p, ::std::optional<Generator> generator); // {"schema": "aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)", "dispatch": "True", "default": "False"}
// Tensor & geometric_out(const Tensor & self, double p, ::std::optional<Generator> generator, Tensor & out); // {"schema": "aten::geometric.out(Tensor self, float p, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "True"}
// Tensor geometric(const Tensor & self, double p, ::std::optional<Generator> generator); // {"schema": "aten::geometric(Tensor self, float p, *, Generator? generator=None) -> Tensor", "dispatch": "True", "default": "True"}

Tensor & zoom_geometric_(Tensor & self, double p, ::std::optional<Generator> generator) {
    return geometric_(self, p, std::move(generator));
}
Tensor & zoom_geometric_out(const Tensor & self, double p, ::std::optional<Generator> generator, Tensor & out) {
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

    geometric_kernel(iter, p, std::move(generator));

    return out;
}
Tensor zoom_geometric(const Tensor & self, double p, ::std::optional<Generator> generator) {
    // Create a new tensor with the same size and dtype as the input tensor
    Tensor out = at::empty_like(self);
    zoom_geometric_out(self, p, generator, out);
    return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("geometric_", &zoom_geometric_);
    m.impl("geometric.out", &zoom_geometric_out);
    m.impl("geometric", &zoom_geometric);
}


} // namespace at::native
