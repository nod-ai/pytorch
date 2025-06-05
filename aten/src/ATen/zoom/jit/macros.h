#include <string>

#define AT_USE_JITERATOR() true
#define jiterator_stringify(...) std::string(#__VA_ARGS__);