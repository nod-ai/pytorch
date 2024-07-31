#include <c10/macros/Macros.h>
#include <cstdint>
#include <ATen/zoom/ZoomContext.h>

namespace at::zoom {
namespace detail {
void init_p2p_access_cache(int64_t num_devices);
}

TORCH_ZOOM_API bool get_p2p_access(int source_dev, int dest_dev);

}  // namespace at::zoom