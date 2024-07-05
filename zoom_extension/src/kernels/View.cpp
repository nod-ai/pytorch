#include <torch/library.h>

namespace at::native {
    // aten::view is purely metadata manipulation, so we can just use the implementation from native PyTorch
    // we just need to declare this here to route the dispatcher appropriately, torch doesn't expose this declaration elsewhere.
    // implemented in ATen/native/TensorShape.cpp
    Tensor view(const Tensor & self, at::IntArrayRef size);
    // similarly for as_strided
    Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_);

    const Tensor & zoom_as_strided_(const Tensor & self, c10::IntArrayRef size, c10::IntArrayRef stride, ::std::optional<int64_t> storage_offset) {
        Tensor result = as_strided_tensorimpl(self, size, stride, storage_offset);
        self.set_(result);
        return self;
    } // {"schema": "aten::as_strided_(Tensor(a!) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a!)", "dispatch": "True", "default": "True"}

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("view", &view);
        m.impl("as_strided", &as_strided_tensorimpl);
        m.impl("as_strided_", &zoom_as_strided_);
    }

}