#pragma once

#include <c10/core/Device.h>
#include <c10/core/impl/GPUTrace.h>
#include "ZoomDefines.h"
#include "ZoomException.h"

namespace c10::zoom {

// NB: In the past, we were inconsistent about whether or not this reported
// an error if there were driver problems are not.  Based on experience
// interacting with users, it seems that people basically ~never want this
// function to fail; it should just return zero if things are not working.
// Oblige them.
// It still might log a warning for user first time it's invoked
DeviceIndex device_count() noexcept;

// Version of device_count that throws is no devices are detected
DeviceIndex device_count_ensure_non_zero();

DeviceIndex current_device();

void set_device(DeviceIndex device);

void device_synchronize();

void warn_or_error_on_sync();

// Raw CUDA device management functions
hipError_t GetDeviceCount(int* dev_count);

hipError_t GetDevice(DeviceIndex* device);

hipError_t SetDevice(DeviceIndex device);

hipError_t MaybeSetDevice(DeviceIndex device);

DeviceIndex ExchangeDevice(DeviceIndex device);

DeviceIndex MaybeExchangeDevice(DeviceIndex device);

void SetTargetDevice();

enum class SyncDebugMode { L_DISABLED = 0, L_WARN, L_ERROR };

// this is a holder for c10 global state (similar to at GlobalContext)
// currently it's used to store cuda synchronization warning state,
// but can be expanded to hold other related global state, e.g. to
// record stream usage
class WarningState {
 public:
  void set_sync_debug_mode(SyncDebugMode l) {
    sync_debug_mode = l;
  }

  SyncDebugMode get_sync_debug_mode() {
    return sync_debug_mode;
  }

 private:
  SyncDebugMode sync_debug_mode = SyncDebugMode::L_DISABLED;
};

__inline__ WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}
// the subsequent functions are defined in the header because for performance
// reasons we want them to be inline
void __inline__ memcpy_and_sync(
    void* dst,
    const void* src,
    int64_t nbytes,
    hipMemcpyKind kind,
    hipStream_t stream) {
  if (C10_UNLIKELY(
          warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_synchronization(
        c10::DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(stream));
  }

  HIP_ASSERT(hipMemcpyWithStream(dst, src, nbytes, kind, stream));

}

void __inline__ stream_synchronize(hipStream_t stream) {
  if (C10_UNLIKELY(
          warning_state().get_sync_debug_mode() != SyncDebugMode::L_DISABLED)) {
    warn_or_error_on_sync();
  }
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_stream_synchronization(
        c10::DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(stream));
  }
  C10_ZOOM_CHECK(hipStreamSynchronize(stream));
}

bool hasPrimaryContext(DeviceIndex device_index);
std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext();

} // namespace c10::zoom