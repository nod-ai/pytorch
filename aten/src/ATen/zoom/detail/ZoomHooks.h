#pragma once

#include <ATen/detail/ZoomHooksInterface.h>

#include <ATen/Generator.h>
#include <c10/util/Optional.h>

// TODO: No need to have this whole header, we can just put it all in
// the cpp file

namespace at::zoom::detail {


// The real implementation of ZoomHooksInterface
struct ZoomHooks : public ZoomHooksInterface {
  ZoomHooks(ZoomHooksArgs) {}
  void initZoom() const override;
  void initPrivateUse1() const override;
  Device getDeviceFromPtr(void* data) const override;
  bool isPinnedPtr(const void* data) const override;
  const Generator& getDefaultZoomGenerator(DeviceIndex device_index = -1) const override;
  bool hasROCM() const override;
  bool checkHIPBlasDeterministic() const override;
  const at::zoom::HIPRTC& hiprtc() const override;
  DeviceIndex current_device() const override;
  bool hasPrimaryContext(DeviceIndex device_index) const override;
  Allocator* getZoomDeviceAllocator() const override;
  Allocator* getPinnedMemoryAllocator() const override;
  std::string showConfig() const override;
  int getNumGPUs() const override;
  void deviceSynchronize(DeviceIndex device_index) const override;
};

void register_zoom_hooks();

} // at::zoom::detail