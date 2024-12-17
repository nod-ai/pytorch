#pragma once

#include <string>
#include <c10/macros/Export.h>

namespace at::zoom {

TORCH_ZOOM_API const std::string &get_traits_string();
TORCH_ZOOM_API const std::string &get_cmath_string();
TORCH_ZOOM_API const std::string &get_complex_body_string();
TORCH_ZOOM_API const std::string &get_complex_half_body_string();
TORCH_ZOOM_API const std::string &get_complex_math_string();

} // namespace at::zoom