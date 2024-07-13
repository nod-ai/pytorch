#pragma once

#include <c10/core/ScalarType.h>

#include <hip/hip_runtime.h>
#include <hip/library_types.h>
#include <hip/hip_complex.h>

namespace at::zoom {

template <typename scalar_t>
hipDataType getHIPDataType() {
  TORCH_INTERNAL_ASSERT(false, "Cannot convert type ", typeid(scalar_t).name(), " to hipDataType.")
}

template<> inline hipDataType getHIPDataType<at::Half>() {
  return HIP_R_16F;
}
template<> inline hipDataType getHIPDataType<float>() {
  return HIP_R_32F;
}
template<> inline hipDataType getHIPDataType<double>() {
  return HIP_R_64F;
}
template<> inline hipDataType getHIPDataType<c10::complex<c10::Half>>() {
  return HIP_C_16F;
}
template<> inline hipDataType getHIPDataType<c10::complex<float>>() {
  return HIP_C_32F;
}
template<> inline hipDataType getHIPDataType<c10::complex<double>>() {
  return HIP_C_64F;
}

template<> inline hipDataType getHIPDataType<uint8_t>() {
  return HIP_R_8U;
}
template<> inline hipDataType getHIPDataType<int8_t>() {
  return HIP_R_8I;
}
template<> inline hipDataType getHIPDataType<int>() {
  return HIP_R_32I;
}

template<> inline hipDataType getHIPDataType<int16_t>() {
  return HIP_R_16I;
}
template<> inline hipDataType getHIPDataType<int64_t>() {
  return HIP_R_64I;
}
template<> inline hipDataType getHIPDataType<at::BFloat16>() {
  return HIP_R_16BF;
}

inline hipDataType ScalarTypeToHIPDataType(const c10::ScalarType& scalar_type) {
  switch (scalar_type) {
    case c10::ScalarType::Byte:
      return HIP_R_8U;
    case c10::ScalarType::Char:
      return HIP_R_8I;
    case c10::ScalarType::Int:
      return HIP_R_32I;
    case c10::ScalarType::Half:
      return HIP_R_16F;
    case c10::ScalarType::Float:
      return HIP_R_32F;
    case c10::ScalarType::Double:
      return HIP_R_64F;
    case c10::ScalarType::ComplexHalf:
      return HIP_C_16F;
    case c10::ScalarType::ComplexFloat:
      return HIP_C_32F;
    case c10::ScalarType::ComplexDouble:
      return HIP_C_64F;
    case c10::ScalarType::Short:
      return HIP_R_16I;
    case c10::ScalarType::Long:
      return HIP_R_64I;
    case c10::ScalarType::BFloat16:
      return HIP_R_16BF;
#if defined(HIP_NEW_TYPE_ENUMS)
    case c10::ScalarType::Float8_e4m3fnuz:
      return HIP_R_8F_E4M3_FNUZ;
    case c10::ScalarType::Float8_e5m2fnuz:
      return HIP_R_8F_E5M2_FNUZ;
#else
    case c10::ScalarType::Float8_e4m3fnuz:
      return static_cast<hipDataType>(1000);
    case c10::ScalarType::Float8_e5m2fnuz:
      return static_cast<hipDataType>(1001);
#endif
    default:
      TORCH_INTERNAL_ASSERT(false, "Cannot convert ScalarType ", scalar_type, " to hipDataType.")
  }
}

} // namespace at::zoom