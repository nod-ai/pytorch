#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/zoom/jit/Loops.cuh>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/BinaryOps.h>

namespace at::native {

template<typename scalar_t>
struct BitwiseAndFunctor {
  __device__ __forceinline__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a & b;
  }
};

template<>
struct BitwiseAndFunctor<bool> {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a && b;
  }
};

void bitwise_and_kernel_zoom(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_and_zoom", [&]() {
    BitwiseAndFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

template<typename scalar_t>
struct BitwiseOrFunctor {
  __device__ __forceinline__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a | b;
  }
};

template<>
struct BitwiseOrFunctor<bool> {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a || b;
  }
};

void bitwise_or_kernel_zoom(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_or_zoom", [&]() {
    BitwiseOrFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

template<typename scalar_t>
struct BitwiseXorFunctor {
  __device__ __forceinline__ scalar_t operator()(scalar_t a, scalar_t b) const {
    return a ^ b;
  }
};

template<>
struct BitwiseXorFunctor<bool> {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a != b;
  }
};

void bitwise_xor_kernel_zoom(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_xor_zoom", [&]() {
    BitwiseXorFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

REGISTER_PRIVATEUSE1_DISPATCH(bitwise_and_stub, &bitwise_and_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(bitwise_or_stub, &bitwise_or_kernel_zoom);
REGISTER_PRIVATEUSE1_DISPATCH(bitwise_xor_stub, &bitwise_xor_kernel_zoom);


} // namespace at::native
