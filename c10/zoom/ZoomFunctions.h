#pragma once

#include <c10/core/Device.h>
#include <c10/core/impl/GPUTrace.h>
#include <hip/hip_runtime.h>
#include <c10/zoom/ZoomException.h>
#include <c10/zoom/ZoomMacros.h>

namespace c10::zoom {

// NB: In the past, we were inconsistent about whether or not this reported
// an error if there were driver problems are not.  Based on experience
// interacting with users, it seems that people basically ~never want this
// function to fail; it should just return zero if things are not working.
// Oblige them.
// It still might log a warning for user first time it's invoked
C10_ZOOM_API DeviceIndex device_count() noexcept;

// Version of device_count that throws is no devices are detected
C10_ZOOM_API DeviceIndex device_count_ensure_non_zero();

C10_ZOOM_API DeviceIndex current_device();

C10_ZOOM_API void set_device(DeviceIndex device);

C10_ZOOM_API void device_synchronize();

C10_ZOOM_API void warn_or_error_on_sync();

// Raw CUDA device management functions
C10_ZOOM_API hipError_t GetDeviceCount(int* dev_count);

C10_ZOOM_API hipError_t GetDevice(DeviceIndex* device);

C10_ZOOM_API hipError_t SetDevice(DeviceIndex device);

C10_ZOOM_API hipError_t MaybeSetDevice(DeviceIndex device);

C10_ZOOM_API DeviceIndex ExchangeDevice(DeviceIndex device);

C10_ZOOM_API DeviceIndex MaybeExchangeDevice(DeviceIndex device);

C10_ZOOM_API void SetTargetDevice();

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

C10_ZOOM_API __inline__ WarningState& warning_state() {
  static WarningState warning_state_;
  return warning_state_;
}
// the subsequent functions are defined in the header because for performance
// reasons we want them to be inline
C10_ZOOM_API void __inline__ memcpy_and_sync(
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

  #if defined(TORCH_HIP_VERSION) && (TORCH_HIP_VERSION >= 301)
    C10_ZOOM_CHECK(hipMemcpyWithStream(dst, src, nbytes, kind, stream));
  #else
    C10_ZOOM_CHECK(hipMemcpyAsync(dst, src, nbytes, kind, stream));
    C10_ZOOM_CHECK(hipStreamSynchronize(stream));
  #endif

}

C10_ZOOM_API void __inline__ stream_synchronize(hipStream_t stream) {
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

C10_ZOOM_API bool hasPrimaryContext(DeviceIndex device_index);
C10_ZOOM_API std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext();

} // namespace c10::zoom