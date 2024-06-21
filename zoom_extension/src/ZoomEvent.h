#pragma once

// #include <ATen/cuda/ATenCUDAGeneral.h>
#include <hip/hip_runtime.h>
#include <c10/macros/Export.h>

#include "ZoomContext.h"
#include <c10/core/impl/GPUTrace.h>
#include "ZoomStream.h"
#include "ZoomGuard.h"
#include "ZoomException.h"
#include <c10/util/Exception.h>

#include <cstdint>
#include <utility>

namespace at::zoom {

/*
* CUDAEvents are movable not copyable wrappers around CUDA's events.
*
* CUDAEvents are constructed lazily when first recorded unless it is
* reconstructed from a cudaIpcEventHandle_t. The event has a device, and this
* device is acquired from the first recording stream. However, if reconstructed
* from a handle, the device should be explicitly specified; or if ipc_handle() is
* called before the event is ever recorded, it will use the current device.
* Later streams that record the event must match this device.
*/
struct ZoomEvent {
  // Constructors
  // Default value for `flags` is specified below - it's cudaEventDisableTiming
  ZoomEvent() noexcept = default;
  ZoomEvent(unsigned int flags) noexcept : flags_{flags} {}

  ZoomEvent(
      DeviceIndex device_index, const hipIpcEventHandle_t* handle) {
      device_index_ = device_index;
      c10::zoom::ZoomGuard guard(device_index_);

      C10_ZOOM_CHECK(hipIpcOpenEventHandle(&event_, *handle));
      is_created_ = true;
  }

  // Note: event destruction done on creating device to avoid creating a
  // CUDA context on other devices.
  ~ZoomEvent() {
    try {
      if (is_created_) {
        c10::zoom::ZoomGuard guard(device_index_);
        const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
        if (C10_UNLIKELY(interp)) {
          (*interp)->trace_gpu_event_deletion(DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(event_));
        }
        C10_ZOOM_CHECK(hipEventDestroy(event_));
      }
    } catch (...) { /* No throw */ }
  }

  ZoomEvent(const ZoomEvent&) = delete;
  ZoomEvent& operator=(const ZoomEvent&) = delete;

  ZoomEvent(ZoomEvent&& other) noexcept { moveHelper(std::move(other)); }
  ZoomEvent& operator=(ZoomEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator hipEvent_t() const { return event(); }

  // Less than operator (to allow use in sets)
  friend bool operator<(const ZoomEvent& left, const ZoomEvent& right) {
    return left.event_ < right.event_;
  }

  optional<at::Device> device() const {
    if (is_created_) {
      return at::Device(DeviceType::PrivateUse1, device_index_);
    } else {
      return {};
    }
  }

  bool isCreated() const { return is_created_; }
  DeviceIndex device_index() const {return device_index_;}
  hipEvent_t event() const { return event_; }

  // Note: hipEventQuery can be safely called from any device
  bool query() const {
    if (!is_created_) {
      return true;
    }

    hipError_t err = hipEventQuery(event_);
    if (err == hipSuccess) {
      return true;
    } else if (err != hipErrorNotReady) {
      C10_ZOOM_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)hipGetLastError();
    }

    return false;
  }

  void record() { record(c10::zoom::getCurrentZoomStream()); }

  void recordOnce(const c10::zoom::ZoomStream& stream) {
    if (!was_recorded_) record(stream);
  }

  // Note: hipEventRecord must be called on the same device as the event.
  void record(const c10::zoom::ZoomStream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(device_index_ == stream.device_index(), "Event device ", device_index_,
      " does not match recording stream's device ", stream.device_index(), ".");
    c10::zoom::ZoomGuard guard(device_index_);
    C10_ZOOM_CHECK(hipEventRecord(event_, stream));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(DeviceType::PrivateUse1,
          reinterpret_cast<uintptr_t>(event_),
          reinterpret_cast<uintptr_t>(stream.stream())
      );
    }
    was_recorded_ = true;
  }

  // Note: hipStreamWaitEvent must be called on the same device as the stream.
  // The event has no actual GPU resources associated with it.
  void block(const c10::zoom::ZoomStream& stream) {
    if (is_created_) {
      c10::zoom::ZoomGuard guard(stream.device_index());
      C10_ZOOM_CHECK(hipStreamWaitEvent(stream, event_, 0));
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_wait(DeviceType::PrivateUse1,
            reinterpret_cast<uintptr_t>(event_),
            reinterpret_cast<uintptr_t>(stream.stream())
        );
      }
    }
  }

  // Note: hipEventElapsedTime can be safely called from any device
  float elapsed_time(const ZoomEvent& other) const {
    TORCH_CHECK(is_created_ && other.isCreated(),
      "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    // We do not strictly have to set the device index to the same as our event,
    // but if we don't and the current device is not initialized, it will
    // create a new hip context, which will consume a lot of memory.
    c10::zoom::ZoomGuard guard(device_index_);
    // raise hipErrorNotReady if either event is recorded but not yet completed
    C10_ZOOM_CHECK(hipEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  // Note: hipEventSynchronize can be safely called from any device
  void synchronize() const {
    if (is_created_) {
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
          (*interp)->trace_gpu_event_synchronization(DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(event_));
      }
      C10_ZOOM_CHECK(hipEventSynchronize(event_));
    }
  }

  // Note: hipIpcGetEventHandle must be called on the same device as the event
  void ipc_handle(hipIpcEventHandle_t * handle) {
      if (!is_created_) {
        // this ZoomEvent object was initially constructed from flags but event_
        // is not created yet.
        createEvent(c10::zoom::getCurrentZoomStream().device_index());
      }
      c10::zoom::ZoomGuard guard(device_index_);
      C10_ZOOM_CHECK(hipIpcGetEventHandle(handle, event_));
  }

private:
  unsigned int flags_ = hipEventDisableTiming;
  bool is_created_ = false;
  bool was_recorded_ = false;
  DeviceIndex device_index_ = -1;
  hipEvent_t event_{};

  void createEvent(DeviceIndex device_index) {
    device_index_ = device_index;
    c10::zoom::ZoomGuard guard(device_index_);
    C10_ZOOM_CHECK(hipEventCreateWithFlags(&event_, flags_));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(event_));
    }
    is_created_ = true;
  }

  void moveHelper(ZoomEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace at::zoom