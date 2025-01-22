#include <ATen/AccumulateType.h>

namespace at {

// TODO(Arham): exchange keys
c10::ScalarType toAccumulateType(c10::ScalarType type, c10::DeviceType device) {
  switch (type) {
#define DEFINE_CASE(scalar_t, TypeNum)                                                                    \
    case ScalarType::TypeNum:                                                                             \
      switch (device) {                                                                                   \
        case DeviceType::CUDA:                                                                            \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::CUDA>>::value;        \
        case DeviceType::PrivateUse1:                                                                     \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::PrivateUse1>>::value; \
        case DeviceType::MPS:                                                                             \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::MPS>>::value;         \
        default:                                                                                          \
          return CppTypeToScalarType<at::acc_type_device<scalar_t, c10::DeviceType::CPU>>::value;         \
      }

    AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ(DEFINE_CASE)
#undef DEFINE_CASE

    default: TORCH_INTERNAL_ASSERT(false, "Unrecognized ScalarType: ", type);
  }
}

c10::ScalarType toAccumulateType(c10::ScalarType type, bool is_cuda) {
  #ifndef USE_ZOOM
    return is_cuda ? toAccumulateType(type, c10::DeviceType::CUDA) : toAccumulateType(type, c10::DeviceType::CPU);
  #else
    // TODO(Arham): exchange keys
    return is_cuda ? toAccumulateType(type, c10::DeviceType::PrivateUse1) : toAccumulateType(type, c10::DeviceType::CPU);
  #endif
}

}
