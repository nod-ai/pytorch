// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/native/UnaryOps.h>
#include <ATen/zoom/ZoomGeneratorImpl.h>
#include <ATen/native/zoom/DistributionTemplates.h>
#include <torch/library.h>

namespace at::native {

void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  auto generator = get_generator_or_default<ZoomGeneratorImpl>(gen, zoom::detail::getDefaultZoomGenerator());
  at::native::templates::zoom::normal_kernel(self, mean, std, generator);
}

REGISTER_PRIVATEUSE1_DISPATCH(normal_stub, &normal_kernel);

// See note: [Declarations for Random Stubs]


Tensor & normal_(Tensor & self, double mean, double std, ::std::optional<Generator> generator);
Tensor normal_functional(const Tensor & self, double mean, double std, ::std::optional<Generator> generator); // {"schema": "aten::normal_functional(Tensor self, float mean=0, float std=1, *, Generator? generator=None) -> Tensor", "dispatch": "True", "default": "True"}
Tensor & normal_out(const Tensor & mean, double std, ::std::optional<Generator> generator, Tensor & out); // {"schema": "aten::normal.Tensor_float_out(Tensor mean, float std=1, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor normal(const Tensor & mean, double std, ::std::optional<Generator> generator); // {"schema": "aten::normal.Tensor_float(Tensor mean, float std=1, *, Generator? generator=None) -> Tensor", "dispatch": "True", "default": "False"}
Tensor & normal_out(double mean, const Tensor & std, ::std::optional<Generator> generator, Tensor & out); // {"schema": "aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor normal(double mean, const Tensor & std, ::std::optional<Generator> generator); // {"schema": "aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor", "dispatch": "True", "default": "False"}
Tensor & normal_out(const Tensor & mean, const Tensor & std, ::std::optional<Generator> generator, Tensor & out); // {"schema": "aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
Tensor normal(const Tensor & mean, const Tensor & std, ::std::optional<Generator> generator); // {"schema": "aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor", "dispatch": "True", "default": "False"}

// Tensor normal(double mean, double std, c10::IntArrayRef size, ::std::optional<Generator> generator, ::std::optional<ScalarType> dtype, ::std::optional<Layout> layout, ::std::optional<Device> device, ::std::optional<bool> pin_memory); // {"schema": "aten::normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", "dispatch": "True", "default": "True"}
// Tensor & normal_out(double mean, double std, c10::IntArrayRef size, ::std::optional<Generator> generator, Tensor & out); // {"schema": "aten::normal.float_float_out(float mean, float std, SymInt[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "True"}

Tensor & zoom_normal_(Tensor & self, double mean, double std, ::std::optional<Generator> generator) {
  return normal_(self, mean, std, std::move(generator));
}
Tensor zoom_normal_functional(const Tensor & self, double mean, double std, ::std::optional<Generator> generator) {
    return normal_functional(self, mean, std, std::move(generator));
}
Tensor & zoom_normal_out1(const Tensor & mean, double std, ::std::optional<Generator> generator, Tensor & out) {
    return normal_out(mean, std, std::move(generator), out);
}
Tensor zoom_normal1(const Tensor & mean, double std, ::std::optional<Generator> generator) {
    return normal(mean, std, std::move(generator));
}
Tensor & zoom_normal_out2(double mean, const Tensor & std, ::std::optional<Generator> generator, Tensor & out) {
    return normal_out(mean, std, std::move(generator), out);
}
Tensor zoom_normal2(double mean, const Tensor & std, ::std::optional<Generator> generator) {
    return normal(mean, std, std::move(generator));
}
Tensor & zoom_normal_out3(const Tensor & mean, const Tensor & std, ::std::optional<Generator> generator, Tensor & out) {
    return normal_out(mean, std, std::move(generator), out);
}
Tensor zoom_normal3(const Tensor & mean, const Tensor & std, ::std::optional<Generator> generator) {
    return normal(mean, std, std::move(generator));
}

// Tensor zoom_normal4(double mean, double std, c10::IntArrayRef size, ::std::optional<Generator> generator, ::std::optional<ScalarType> dtype, ::std::optional<Layout> layout, ::std::optional<Device> device, ::std::optional<bool> pin_memory) {
//     return normal(mean, std, size, std::move(generator), dtype, layout, device, pin_memory);
// }
// Tensor & zoom_normal_out4(double mean, double std, c10::IntArrayRef size, ::std::optional<Generator> generator, Tensor & out) {
//     return normal_out(mean, std, size, std::move(generator), out);
// }

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("normal_", &zoom_normal_);
    m.impl("normal_functional", &zoom_normal_functional);
    m.impl("normal.Tensor_float_out", &zoom_normal_out1);
    m.impl("normal.Tensor_float", &zoom_normal1);
    m.impl("normal.float_Tensor_out", &zoom_normal_out2);
    m.impl("normal.float_Tensor", &zoom_normal2);
    m.impl("normal.Tensor_Tensor_out", &zoom_normal_out3);
    m.impl("normal.Tensor_Tensor", &zoom_normal3);
    // m.impl("normal.float_float", &zoom_normal4);
    // m.impl("normal.float_float_out", &zoom_normal_out4);
}


} // namespace at::native
