#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/sparse/SparseStubs.h>
#include <ATen/native/sparse/FlattenIndicesCommon.h>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/native/zoom/KernelUtils.cuh>
#include <ATen/zoom/jit/OffsetCalculator.cuh>
#include <ATen/AccumulateType.h>

namespace at::native {

namespace {

template <typename func_t>
struct HIPKernelLauncher {
  static void launch(TensorIteratorBase& iter, const func_t& f) {
    gpu_kernel(iter, f);
  }
};

Tensor flatten_indices_zoom_kernel(const Tensor& indices, IntArrayRef size) {
  return _flatten_indices<HIPKernelLauncher>(indices, size);
}

}

REGISTER_PRIVATEUSE1_DISPATCH(flatten_indices_stub, &flatten_indices_zoom_kernel);

} // namespace at::native
