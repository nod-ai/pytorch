// #define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include "../jit/Loops.cuh"
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/TensorOperators.h>
#include <c10/core/Scalar.h>
#include <torch/library.h>

namespace at::native {

template<typename scalar_t>
struct FillFunctor {
  FillFunctor(scalar_t v): value(v) {}
  __device__ __forceinline__ scalar_t operator() () const {
    return value;
  }
  private:
    scalar_t value;
};

void fill_kernel_zoom(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_V2(iter.dtype(), "fill_zoom", AT_WRAP([&]() {
    gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
  }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kBool, kHalf, kBFloat16, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));
}

Tensor & zoom_fill_Scalar_(Tensor & self, const Scalar & value) {
     auto iter = TensorIteratorConfig()
        .set_check_mem_overlap(false)  // Fill is idempotent, so overlap is okay
        .check_all_same_dtype(false)
        .add_output(self)
        .resize_outputs(false)
        .build();
    fill_kernel_zoom(iter, value);
    return self;
}

Tensor& zoom_fill_Tensor_(Tensor& self, const Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
  if (self.device() != value.device()){
    return zoom_fill_Scalar_(self, value.item());
  }
  // Check if value is a view of self and if it is we clone
  // it to avoid overwriting self prematurely
  if(self.is_alias_of(value)) {
    self.copy_(value.clone());
  } else{
    self.copy_(value);
  }
  return self;
}

Tensor zoom_fill_Scalar(const Tensor & self, const Scalar & value) {
    return at::empty_like(self).fill_(value);
}

Tensor zoom_fill_Tensor(const Tensor& self, const Tensor& value) {
  return at::empty_like(self).fill_(value);
}



TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("fill_.Scalar", &zoom_fill_Scalar_);
  m.impl("fill.Scalar", &zoom_fill_Scalar);
  m.impl("fill_.Tensor", &zoom_fill_Tensor_);
  m.impl("fill.Tensor", &zoom_fill_Tensor);
}

} // namespace at::native