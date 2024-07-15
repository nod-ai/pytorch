#pragma once
#include <torch/library.h>
#include <ATen/zoom/ZoomContext.h>

namespace at{

    namespace native {

        void direct_copy_kernel_zoom(TensorIteratorBase &iter);
        Tensor& zoom_copy_(Tensor & self, const Tensor & dst, bool non_blocking); // {"schema": "aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor", "dispatch": "True", "default": "False"}
        Tensor zoom_copy_from(const Tensor& self, const Tensor& dst, bool non_blocking);
    
    }
}