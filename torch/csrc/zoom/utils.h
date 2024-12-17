#include <c10/zoom/ZoomStream.h>

std::vector<std::optional<c10::zoom::ZoomStream>>
THPUtils_PySequence_to_ZoomStreamList(PyObject* obj);