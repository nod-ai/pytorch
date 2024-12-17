#pragma once
#include <cstdint>

// Collection of direct PTX functions

namespace at::zoom {

template <typename T>
struct Bitfield {};

template <>
struct Bitfield<unsigned int> {
  static __device__ __host__ __forceinline__
  unsigned int getBitfield(unsigned int val, int pos, int len) {
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    return (val >> pos) & m;
  }

  static __device__ __host__ __forceinline__
  unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
    pos &= 0xff;
    len &= 0xff;

    unsigned int m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
  }
};

template <>
struct Bitfield<uint64_t> {
  static __device__ __host__ __forceinline__
  uint64_t getBitfield(uint64_t val, int pos, int len) {
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    return (val >> pos) & m;
  }

  static __device__ __host__ __forceinline__
  uint64_t setBitfield(uint64_t val, uint64_t toInsert, int pos, int len) {
    pos &= 0xff;
    len &= 0xff;

    uint64_t m = (1u << len) - 1u;
    toInsert &= m;
    toInsert <<= pos;
    m <<= pos;

    return (val & ~m) | toInsert;
  }
};

__device__ __forceinline__ int getLaneId() {
  return __lane_id();
}

__device__ __forceinline__ unsigned long long int getLaneMaskLt() {
  const std::uint64_t m = (1ull << getLaneId()) - 1ull;
  return m;
}

__device__ __forceinline__ unsigned long long int getLaneMaskLe() {
  std::uint64_t m = UINT64_MAX >> (sizeof(std::uint64_t) * CHAR_BIT - (getLaneId() + 1));
  return m;
}

__device__ __forceinline__ unsigned long long int getLaneMaskGt() {
  const std::uint64_t m = getLaneMaskLe();
  return m ? ~m : m;
}

__device__ __forceinline__ unsigned long long int getLaneMaskGe() {
  const std::uint64_t m = getLaneMaskLt();
  return ~m;
}

} // namespace at::zoom