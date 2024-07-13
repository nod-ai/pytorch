#pragma once

#include <string>
// #include <c10/macros/Export.h>

namespace at::zoom {

const std::string &get_traits_string();
const std::string &get_cmath_string();
const std::string &get_complex_body_string();
const std::string &get_complex_half_body_string();
const std::string &get_complex_math_string();

} // namespace at::zoom