#pragma once

namespace at {
struct TensorIteratorBase;

    namespace native {

        void direct_copy_kernel_zoom(TensorIteratorBase &iter);
    
    }
}