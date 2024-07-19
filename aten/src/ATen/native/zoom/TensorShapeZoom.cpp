#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/zoom/ZoomContext.h>
#include <ATen/native/Resize.h>
#include <ATen/native/zoom/Resize.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/set_native.h>
#endif

namespace at::native {

Tensor& set_zoom_(Tensor& result) {
  caffe2::TypeMeta dtype = result.dtype();
  Storage storage(
      Storage::use_byte_size_t(),
      0,
      at::zoom::getZoomDeviceAllocator(),
      true);
    result.set_(storage, 0, {0}, {});
    TORCH_INTERNAL_ASSERT(dtype == result.dtype());
  return result;
}

Tensor& set_storage_zoom_(Tensor& result, Storage storage, int64_t storage_offset, IntArrayRef size, IntArrayRef stride) {
  checkSetStorage(result, storage, storage_offset, size, stride);

  result.unsafeGetTensorImpl()->set_storage_offset(storage_offset);
  at::OptionalIntArrayRef stride_opt = stride.data() != nullptr ?
                                          at::OptionalIntArrayRef(stride) : c10::nullopt;
  at::native::resize_impl_zoom_(result.unsafeGetTensorImpl(), size, stride_opt);
  return result;
}

} // namespace at::native