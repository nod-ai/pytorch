#pragma once
#include "../extension.h"
#include "../ZoomContext.h"

namespace at{

    namespace native {

        Tensor zoom_copy_from(Tensor & self, const Tensor & src, bool non_blocking); // {"schema": "aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor", "dispatch": "True", "default": "False"}
    
    }
}