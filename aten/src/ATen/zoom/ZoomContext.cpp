#include <ATen/zoom/ZoomContext.h>
#include <c10/zoom/ZoomCachingAllocator.h>
#include <c10/util/CallOnce.h>

#include <mutex>
#include <deque>
#include <vector>

namespace at::zoom {

namespace {

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<hipDeviceProp_t> device_properties;

void initZoomContextVectors() {
  num_gpus = c10::zoom::device_count();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  hipDeviceProp_t device_prop;
  C10_ZOOM_CHECK(hipGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

// We need this function to force the linking against torch_cuda(_cpp) on Windows.
// If you need to modify this function, please specify a new function and apply
// the changes according to https://github.com/pytorch/pytorch/pull/34288.
// Related issue: https://github.com/pytorch/pytorch/issues/31611.
/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

hipDeviceProp_t* getCurrentDeviceProperties() {
  auto device = c10::zoom::current_device();
  return getDeviceProperties(device);
}

hipDeviceProp_t* getDeviceProperties(c10::DeviceIndex device) {
  c10::call_once(init_flag, initZoomContextVectors);
  if (device == -1) device = c10::zoom::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", device, ", num_gpus=", num_gpus);
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

bool canDeviceAccessPeer(c10::DeviceIndex device, c10::DeviceIndex peer_device) {
  c10::call_once(init_flag, initZoomContextVectors);
  if (device == -1) device = c10::zoom::current_device();
  AT_ASSERT(device >= 0 && device < num_gpus, "device=", device, ", num_gpus=", num_gpus);
  AT_ASSERT(peer_device >= 0 && peer_device < num_gpus, "peer_device=", peer_device, ", num_gpus=", num_gpus);
  int can_access = 0;
  C10_ZOOM_CHECK(hipDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

Allocator* getZoomDeviceAllocator() {
  return c10::zoom::ZoomCachingAllocator::get();
}

} // namespace at::zoom