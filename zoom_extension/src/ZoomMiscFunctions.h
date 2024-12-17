#pragma once

#include <mutex>

namespace c10::zoom {
const char* get_zoom_check_suffix() noexcept;
std::mutex* getFreeMutex();
} // namespace c10::zoom