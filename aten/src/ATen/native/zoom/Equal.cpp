// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/NamedTensorUtils.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <c10/zoom/ZoomFunctions.h>
#include <ATen/ops/eq.h>
#else
#include <ATen/ops/eq.h>
#include <ATen/ops/equal_native.h>
#endif

namespace at::native {

bool zoom_equal(const Tensor& self, const Tensor &src) {
  if (!at::namedinference::are_names_equal(
          self.unsafeGetTensorImpl(), src.unsafeGetTensorImpl())) {
    return false;
  }
  at::NoNamesGuard guard;
  TORCH_CHECK(self.device() == src.device(), "Cannot compare two tensors on "
              "different devices. Got: ", self.device(), " and ", src.device());
  if (self.sizes() != src.sizes()) {
    return false;
  }
  if (self.numel() == 0) {
    return true;
  }

  // This is the same optimization done in the cpu_equal. Since the flags like neg/conj should be already handled outside the
  // cuda_equal, it should be safe to have the following fast path by
  // ensuring the storage and strides exactly the same.
  if (self.is_alias_of(src)
      && self.storage_offset() == src.storage_offset()
      && self.dtype() == src.dtype()
      && self.is_contiguous() == src.is_contiguous()
      && self.strides().equals(src.strides())
      // Extra checks to ensure the safety in case cuda_equal is directly called in C++.
      && self.layout() == src.layout()
      && self.is_neg() == src.is_neg()
      && self.is_conj() == src.is_conj()) {
    return true;
  }

  return at::eq(self, src).all().item().to<bool>();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("equal", &zoom_equal);
}

} // namespace at::native