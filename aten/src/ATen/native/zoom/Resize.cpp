// #define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/zoom/Resize.h>
#include <ATen/core/Tensor.h>
#include <ATen/zoom/ZoomContext.h>
#include <ATen/zoom/PeerToPeerAccess.h>
#include <ATen/native/ResizeCommon.h>
#include <c10/zoom/ZoomGuard.h>
#include <c10/zoom/extension.h>

namespace at::native {

void resize_bytes_zoom(StorageImpl* storage, size_t size_bytes) {
  TORCH_CHECK(storage->resizable(), "Trying to resize storage that is not resizable");
  auto allocator = storage->allocator();
  TORCH_CHECK(allocator != nullptr, "Trying to resize storage without an allocator");

  c10::Device device = storage->device();

  if (size_bytes == 0) {
    storage->set_data_ptr_noswap(at::DataPtr(nullptr, device));
    storage->set_nbytes(0);
    return;
  }

  c10::zoom::ZoomGuard guard(device.index());
  at::DataPtr data = allocator->allocate(size_bytes);
  if (storage->data_ptr()) {
    at::globalContext().lazyInitPrivateUse1();

    C10_ZOOM_CHECK(
        hipMemcpyAsync(
            data.get(),
            storage->data(),
            std::min(storage->nbytes(), size_bytes),
            hipMemcpyDeviceToDevice,
            c10::zoom::getCurrentZoomStream()));
  }

  // Destructively overwrite data_ptr
  storage->set_data_ptr_noswap(std::move(data));
  storage->set_nbytes(size_bytes);
}

const Tensor& resize_zoom_(
    const Tensor& self,
    IntArrayRef size,
    std::optional<MemoryFormat> optional_memory_format) {
  if (self.has_names()) {
    return resize_named_tensor_(self, size, optional_memory_format);
  }
  auto* self_ = self.unsafeGetTensorImpl();
  int64_t old_storage_nbytes = self_->unsafe_storage() ? self_->unsafe_storage().nbytes() : 0;
  resize_impl_zoom_(self_, size, /*strides=*/c10::nullopt);
  if (optional_memory_format.has_value()) {
    auto memory_format =
        optional_memory_format.value();
    TORCH_CHECK(
        memory_format != MemoryFormat::Preserve,
        "Unsupported memory format",
        memory_format);
    self_->empty_tensor_restride(memory_format);
  }
  // See Note [Enabling Deterministic Operations]
  if (C10_UNLIKELY(at::globalContext().deterministicAlgorithms() && at::globalContext().deterministicFillUninitializedMemory())) {
    at::native::fill_resize_deterministic_(self, old_storage_nbytes);
  }
  return self;
}


TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("resize_", &resize_zoom_);
}

} // namespace at::native