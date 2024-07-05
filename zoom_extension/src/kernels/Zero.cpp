#include <torch/library.h>
#include <ATen/ops/fill.h>

namespace at::native {

    Tensor zoom_zero(const Tensor & self) {
        return at::fill(self, 0);
    }

    Tensor & zoom_zero_(Tensor & self) {
        return at::fill_(self, 0);
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("zero", &zoom_zero);
        m.impl("zero_", &zoom_zero_);
    }

}