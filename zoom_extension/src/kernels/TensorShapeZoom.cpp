// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include "../ZoomContext.h"
#include <ATen/native/Resize.h>
#include "Resize.h"
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/set_native.h>
#endif

namespace at::native {

Tensor & zoom_set_source_Storage(Tensor & result, Storage source) {
    caffe2::TypeMeta dtype = result.dtype();
    result.set_(source, 0, {0}, {});
    TORCH_INTERNAL_ASSERT(dtype == result.dtype());
    return result;
}

Tensor& set_zoom_(Tensor& result) {
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      at::zoom::getZoomDeviceAllocator(),
      true);
  return zoom_set_source_Storage(result, storage);
}

Tensor& set_storage_zoom_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  checkSetStorage(result, storage, storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ?
                                          at::OptionalIntArrayRef(stride) : c10::nullopt;
  at::native::resize_impl_zoom_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

Tensor & zoom_set_source_Tensor_storage_offset(Tensor & self, const Tensor & source, int64_t storage_offset, c10::IntArrayRef size, c10::IntArrayRef stride) {
    auto storage = source.storage();
    return set_storage_zoom_(self, storage, storage_offset, size, stride);
}
Tensor & zoom_set_source_Tensor(Tensor & self, const Tensor & source) {
    auto storage = source.storage();
    return zoom_set_source_Storage(self, storage);
}

Tensor _reshape_from_tensor(const Tensor& self, const Tensor& shape_tensor);
Tensor _shape_as_tensor(const Tensor & self);

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("set_", &set_zoom_);
    m.impl("set_.source_Storage", &zoom_set_source_Storage);
    m.impl("set_.source_Storage_storage_offset", &set_storage_zoom_);
    m.impl("set_.source_Tensor_storage_offset", &zoom_set_source_Tensor_storage_offset);
    m.impl("set_.source_Tensor", &zoom_set_source_Tensor);

    m.impl("_reshape_from_tensor", &_reshape_from_tensor);
    m.impl("_shape_as_tensor", &_shape_as_tensor);

}

} // namespace at::native