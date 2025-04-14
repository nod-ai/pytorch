#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/zoom/Distributions.h>
#include <ATen/TensorIterator.h>
#include <ATen/zoom/ZoomGeneratorImpl.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_dirichlet_grad_native.h>
#include <ATen/ops/_sample_dirichlet_native.h>
#include <ATen/ops/_standard_gamma_grad_native.h>
#include <ATen/ops/_standard_gamma_native.h>
#include <ATen/ops/binomial_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/poisson_native.h>
#endif

namespace at::native {

Tensor _s_poisson_zoom(const Tensor& lambda, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<ZoomGeneratorImpl>(gen_, zoom::detail::getDefaultZoomGenerator());
  Tensor ret = at::empty(lambda.sizes(), lambda.options());
  launch_poisson_zoom_kernel(ret, lambda, gen);
  return ret;
}

Tensor _s_binomial_zoom(const Tensor& count, const Tensor& prob, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<ZoomGeneratorImpl>(gen_, zoom::detail::getDefaultZoomGenerator());
  Tensor ret = at::empty(count.sizes(), count.options());
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(count)
      .add_input(prob)
      .build();
  launch_binomial_zoom_kernel(iter, gen);
  return ret;
}

Tensor _s_gamma_zoom(const Tensor& alpha, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<ZoomGeneratorImpl>(gen_, zoom::detail::getDefaultZoomGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  launch_gamma_kernel(ret, alpha, gen);
  return ret;
}

Tensor _s_dirichlet_zoom(const Tensor& alpha, std::optional<Generator> gen_) {
  auto gen = get_generator_or_default<ZoomGeneratorImpl>(gen_, zoom::detail::getDefaultZoomGenerator());
  Tensor ret = at::empty(alpha.sizes(), alpha.options());
  launch_gamma_kernel(ret, alpha, gen);
  auto gamma_sum = ret.sum(/*dim=*/-1, /*keepdim=*/true);
  at::TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(ret)
      .add_input(gamma_sum)
      .build();
  launch_dirichlet_kernel(iter);
  return ret;
}

Tensor _standard_gamma_grad_zoom(const Tensor& self, const Tensor& output) {
  Tensor ret = at::empty(self.sizes(), self.options());
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(self)
      .add_input(output)
      .build();
  launch_standard_gamma_grad_kernel(iter);
  return ret;
}

Tensor _dirichlet_grad_zoom(const Tensor& x, const Tensor& alpha, const Tensor& total) {
  Tensor ret = at::empty(x.sizes(), x.options());
  TensorIterator iter = at::TensorIteratorConfig()
      .add_output(ret)
      .add_input(x)
      .add_input(alpha)
      .add_input(total)
      .build();
  launch_dirichlet_grad_kernel(iter);
  return ret;
}

} // namespace at::native