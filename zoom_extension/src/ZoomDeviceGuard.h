#pragma once

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/GPUTrace.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include "ZoomAllocator.h"
#include "ZoomAllocator.h"
#include "ZoomFunctions.h"
#include "ZoomStream.h"
#include "ZoomDefines.h"

#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/PyInterpreter.h>
#include <c10/util/Optional.h>
#include <cstdint>

namespace c10::zoom {

struct ZoomDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = DeviceType::PrivateUse1;

  ZoomDeviceGuardImpl() = default;
  explicit ZoomDeviceGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == DeviceType::PrivateUse1);
  }
  DeviceType type() const override {
    return DeviceType::PrivateUse1;
  }
  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    auto old_device_index = c10::zoom::ExchangeDevice(d.index());
    return Device(DeviceType::PrivateUse1, old_device_index);
  }
  Device getDevice() const override {
    DeviceIndex device = 0;
    C10_ZOOM_CHECK(c10::zoom::GetDevice(&device));
    return Device(DeviceType::PrivateUse1, device);
  }
  std::optional<Device> uncheckedGetDevice() const noexcept {
    DeviceIndex device{-1};
    const auto err = C10_ZOOM_ERROR_HANDLED(c10::zoom::GetDevice(&device));
    C10_ZOOM_CHECK_WARN(err);
    if (err != hipSuccess) {
      return c10::nullopt;
    }
    return Device(DeviceType::PrivateUse1, device);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.is_privateuseone());
    C10_ZOOM_CHECK(c10::zoom::SetDevice(d.index()));
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    C10_ZOOM_CHECK_WARN(c10::zoom::MaybeSetDevice(d.index()));
  }
  Stream getStream(Device d) const noexcept override {
    return getCurrentZoomStream(d.index()).unwrap();
  }
  Stream getDefaultStream(Device d) const override {
    return getDefaultZoomStream(d.index());
  }
  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
  }
  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
  }
  // NB: These do NOT set the current device
  Stream exchangeStream(Stream s) const noexcept override {
    ZoomStream cs(s);
    auto old_stream = getCurrentZoomStream(s.device().index());
    setCurrentZoomStream(cs);
    return old_stream.unwrap();
  }
  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  // Event-related functions
  void createEvent(hipEvent_t* zoom_event, const EventFlag flag) const {
    // Maps PyTorch's Event::Flag to CUDA flag
    auto hip_flag = hipEventDefault;
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
        hip_flag = hipEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
        hip_flag = hipEventDefault;
        break;
      default:
        TORCH_CHECK(false, "HIP event received unknown flag");
    }

    C10_ZOOM_CHECK(hipEventCreateWithFlags(zoom_event, hip_flag));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(
          c10::DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(zoom_event));
    }
  }

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    auto zoom_event = static_cast<hipEvent_t>(event);
    DeviceIndex orig_device{-1};
    C10_ZOOM_CHECK_WARN(c10::zoom::GetDevice(&orig_device));
    C10_ZOOM_CHECK_WARN(c10::zoom::SetDevice(device_index));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_deletion(
          c10::DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(zoom_event));
    }
    C10_ZOOM_CHECK_WARN(hipEventDestroy(zoom_event));
    C10_ZOOM_CHECK_WARN(c10::zoom::SetDevice(orig_device));
  }

  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");

    hipEvent_t zoom_event = static_cast<hipEvent_t>(*event);
    ZoomStream zoom_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!zoom_event)
      createEvent(&zoom_event, flag);
    C10_ZOOM_CHECK(hipEventRecord(zoom_event, zoom_stream));
    // Makes the void* point to the (possibly just allocated) CUDA event
    *event = zoom_event;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          c10::DeviceType::PrivateUse1,
          reinterpret_cast<uintptr_t>(zoom_event),
          reinterpret_cast<uintptr_t>(zoom_stream.stream()));
    }

    // Resets device
    setDevice(orig_device);
  }

  void block(void* event, const Stream& stream) const override {
    if (!event)
      return;
    hipEvent_t zoom_event = static_cast<hipEvent_t>(event);
    ZoomStream zoom_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    C10_ZOOM_CHECK(hipStreamWaitEvent(
        zoom_stream,
        zoom_event,
        /*flags (must be zero)=*/0));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_wait(
          c10::DeviceType::PrivateUse1,
          reinterpret_cast<uintptr_t>(zoom_event),
          reinterpret_cast<uintptr_t>(zoom_stream.stream()));
    }
    setDevice(orig_device);
  }

  // May be called from any device
  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    hipEvent_t zoom_event = static_cast<hipEvent_t>(event);
    // Note: hipEventQuery can be safely called from any device
    const hipError_t err = C10_ZOOM_ERROR_HANDLED(hipEventQuery(zoom_event));
    if (err != hipErrorNotReady) {
      C10_ZOOM_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)hipGetLastError();
    }
    return (err == hipSuccess);
  }

  // Stream-related functions
  bool queryStream(const Stream& stream) const override {
    ZoomStream zoom_stream{stream};
    return zoom_stream.query();
  }

  void synchronizeStream(const Stream& stream) const override {
    ZoomStream zoom_stream{stream};
    zoom_stream.synchronize();
  }

  void synchronizeEvent(void* event) const override {
    if (!event)
      return;
    hipEvent_t zoom_event = static_cast<hipEvent_t>(event);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_synchronization(
          c10::DeviceType::PrivateUse1, reinterpret_cast<uintptr_t>(zoom_event));
    }
    // Note: hipEventSynchronize can be safely called from any device
    C10_ZOOM_CHECK(hipEventSynchronize(zoom_event));
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream)
      const override {
    ZoomStream zoom_stream{stream};
    ZoomAllocator::recordStream(data_ptr, zoom_stream);
  }

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");
    // Even though hipEventElapsedTime can be safely called from any device, if
    // the current device is not initialized, it will create a new cuda context,
    // which will consume a lot of memory.
    DeviceIndex orig_device{-1};
    C10_ZOOM_CHECK(c10::zoom::GetDevice(&orig_device));
    C10_ZOOM_CHECK(c10::zoom::SetDevice(device_index));
    hipEvent_t zoom_event1 = static_cast<hipEvent_t>(event1);
    hipEvent_t zoom_event2 = static_cast<hipEvent_t>(event2);
    float time_ms = 0;
    // raise hipErrorNotReady if either event is recorded but not yet completed
    C10_ZOOM_CHECK(hipEventElapsedTime(&time_ms, zoom_event1, zoom_event2));
    C10_ZOOM_CHECK(c10::zoom::SetDevice(orig_device));
    return static_cast<double>(time_ms);
  }
};

} // namespace c10::zoom