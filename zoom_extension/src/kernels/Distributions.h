#pragma once

namespace at {
struct ZoomGeneratorImpl;
struct TensorIteratorBase;
class TensorBase;

namespace native {

void launch_poisson_zoom_kernel(
    const TensorBase &ret, const TensorBase &lambda, ZoomGeneratorImpl *gen);

void launch_gamma_kernel(
    const TensorBase &ret, const TensorBase &alpha, ZoomGeneratorImpl *gen);

void launch_binomial_zoom_kernel(
    TensorIteratorBase &iter, ZoomGeneratorImpl *gen);

void launch_dirichlet_kernel(TensorIteratorBase &iter);

void launch_standard_gamma_grad_kernel(TensorIteratorBase &iter);

void launch_dirichlet_grad_kernel(TensorIteratorBase &iter);

}}  // namespace at::native