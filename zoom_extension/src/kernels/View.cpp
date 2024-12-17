#include <torch/library.h>
#include <ATen/ops/empty_like.h>

namespace at::native {
    // aten::view is purely metadata manipulation, so we can just use the implementation from native PyTorch
    // we just need to declare this here to route the dispatcher appropriately, torch doesn't expose this declaration elsewhere.
    // implemented in ATen/native/TensorShape.cpp
    Tensor view(const Tensor & self, at::IntArrayRef size);
    Tensor & zoom_view_copy_out(const Tensor & self, c10::IntArrayRef size, Tensor & out) {
        out.copy_(self);
        view(out, size);
        return out;
    }
    Tensor & zoom_view_copy_out_dtype(const Tensor & self, ScalarType dtype, Tensor & out) {
        out.copy_(self);
        out.to(dtype);
        return out;
    }
    Tensor zoom_view_copy(const Tensor & self, c10::IntArrayRef size) {
        Tensor result = at::empty_like(self, self.options());
        zoom_view_copy_out(self, size, result);
        return result;
    }
    Tensor zoom_view_copy_dtype(const Tensor & self, ScalarType dtype) {
        Tensor result = at::empty_like(self, self.options());
        result.copy_(self);
        result.to(dtype);
        return result;
    }

    Tensor view_as_real(const Tensor& self);
    Tensor & zoom_view_as_real_copy_out(const Tensor & self, Tensor & out) {
        out.copy_(self);
        view_as_real(out);
        return out;
    }
    Tensor zoom_view_as_real_copy(const Tensor & self) {
        Tensor result = at::empty_like(self, self.options());
        zoom_view_as_real_copy_out(self, result);
        return result;
    }

    Tensor view_as_complex(const Tensor& self);
    Tensor & zoom_view_as_complex_copy_out(const Tensor & self, Tensor & out) {
        out.copy_(self);
        view_as_complex(out);
        return out;
    }
    Tensor zoom_view_as_complex_copy(const Tensor & self) {
        Tensor result = at::empty_like(self, self.options());
        zoom_view_as_complex_copy_out(self, result);
        return result;
    }

    // similarly for as_strided
    Tensor as_strided_tensorimpl(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_);

    const Tensor & zoom_as_strided_(const Tensor & self, c10::IntArrayRef size, c10::IntArrayRef stride, ::std::optional<int64_t> storage_offset) {
        Tensor result = as_strided_tensorimpl(self, size, stride, storage_offset);
        self.set_(result);
        return self;
    } 

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("view", &view);
        m.impl("view_copy", &zoom_view_copy);
        m.impl("view_copy.dtype", &zoom_view_copy_dtype);
        m.impl("view_copy.out", &zoom_view_copy_out);
        m.impl("view_copy.dtype_out", &zoom_view_copy_out_dtype);
        m.impl("view_as_real", &view_as_real);
        m.impl("view_as_real_copy", &zoom_view_as_real_copy);
        m.impl("view_as_real_copy.out", &zoom_view_as_real_copy_out);
        m.impl("view_as_complex", &view_as_complex);
        m.impl("view_as_complex_copy", &zoom_view_as_complex_copy);
        m.impl("view_as_complex_copy.out", &zoom_view_as_complex_copy_out);
        m.impl("as_strided", &as_strided_tensorimpl);
        m.impl("as_strided_", &zoom_as_strided_);
    }

}