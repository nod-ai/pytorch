#include <torch/library.h>
#include <ATen/DeviceGuard.h>
#include <ATen/zoom/ZoomAllocator.h>
#include <ATen/native/cpu/Loops.h>
#include <c10/core/TensorOptions.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorFactories.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ops/empty.h>
#include <iostream>


at::Tensor zoom_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  std::cout << "zoom aten::add.Tensor() called!" << std::endl;
  // Since this zoom device is just for testing, not bothering to implement kernels.
  return at::empty(self.sizes(), self.options());
}

at::Tensor zoom_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format) {
  const at::OptionalDeviceGuard device_guard(device);
  std::cout << "zoom aten::empty.memory_format() called!" << std::endl;
  constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
  auto device_ = device_or_default(device);
  auto allocator = at::native::ZeroTensorAllocator(device_);
  return at::detail::empty_generic(size, &allocator, private_use_ks, c10::dtype_or_default(dtype), memory_format);
}

at::Tensor & zoom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  // Not bothering to implement.
  // Should fill the tensor's data with "value".
  return self;
}

// basic dummy copy_() function, so we can copy from the zoom device to/from CPU
at::Tensor zoom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  std::cout << "zoom aten::_copy_from() called!" << std::endl;
  TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
  TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");

  // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
  TORCH_CHECK(self.sizes() == dst.sizes());
  TORCH_CHECK(self.scalar_type() == dst.scalar_type());
  TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

  std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
  return dst;
}





TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("add.Tensor", &zoom_add_Tensor);
  m.impl("empty.memory_format", &zoom_empty_memory_format);
  m.impl("fill_.Scalar", &zoom_fill__scalar);
  m.impl("_copy_from", &zoom__copy_from);
}