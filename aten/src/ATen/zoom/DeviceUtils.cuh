#pragma once

#include <hip/hip_runtime.h>
#include <c10/util/complex.h>
#include <c10/util/Half.h>

__device__ __forceinline__ unsigned int ACTIVE_MASK()
{
// will be ignored anyway
    return 0xffffffff;
}

__device__ __forceinline__ void WARP_SYNC(unsigned mask = 0xffffffff) {

}


__device__ __forceinline__ unsigned long long int WARP_BALLOT(int predicate)
{
return __ballot(predicate);
}


template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR(T value, int laneMask, int width = warpSize, unsigned int mask = 0xffffffff)
{
    return __shfl_xor(value, laneMask, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL(T value, int srcLane, int width = warpSize, unsigned int mask = 0xffffffff)
{
    return __shfl(value, srcLane, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_UP(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
    return __shfl_up(value, delta, width);
}

template <typename T>
__device__ __forceinline__ T WARP_SHFL_DOWN(T value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
    return __shfl_down(value, delta, width);
}

template<>
__device__ __forceinline__ int64_t WARP_SHFL_DOWN<int64_t>(int64_t value, unsigned int delta, int width , unsigned int mask)
{
  //(HIP doesn't support int64_t). Trick from https://devblogs.nvidia.com/faster-parallel-reductions-kepler/
  int2 a = *reinterpret_cast<int2*>(&value);
  a.x = __shfl_down(a.x, delta);
  a.y = __shfl_down(a.y, delta);
  return *reinterpret_cast<int64_t*>(&a);
}

template<>
__device__ __forceinline__ c10::Half WARP_SHFL_DOWN<c10::Half>(c10::Half value, unsigned int delta, int width, unsigned int mask)
{
  return c10::Half(WARP_SHFL_DOWN<unsigned short>(value.x, delta, width, mask), c10::Half::from_bits_t{});
}

template <typename T>
__device__ __forceinline__ c10::complex<T> WARP_SHFL_DOWN(c10::complex<T> value, unsigned int delta, int width = warpSize, unsigned int mask = 0xffffffff)
{
    return c10::complex<T>(
        __shfl_down(value.real_, delta, width),
        __shfl_down(value.imag_, delta, width));
}

template <typename T>
__device__ __forceinline__ T doLdg(const T* p) {
  return *p;
}