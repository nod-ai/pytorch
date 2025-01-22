#pragma once

#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>
// #include <c10/cuda/CUDAMacros.h>
#include <c10/zoom/impl/ZoomGuardImpl.h>

namespace c10::zoom {

// This code is kind of boilerplatey.  See Note [Whither the DeviceGuard
// boilerplate]

/// A variant of DeviceGuard that is specialized for HIP.  It accepts
/// integer indices (interpreting them as HIP devices) and is a little
/// more efficient than DeviceGuard (it compiles to straight line
/// hipSetDevice/hipGetDevice calls); however, it can only be used
/// from code that links against HIP directly.
struct ZoomGuard {
  /// No default constructor; see Note [Omitted default constructor from RAII]
  explicit ZoomGuard() = delete;

  /// Set the current HIP device to the passed device index.
  explicit ZoomGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current HIP device to the passed device.  Errors if the passed
  /// device is not a HIP device.
  explicit ZoomGuard(Device device) : guard_(device) {}

  // Copy is not allowed
  ZoomGuard(const ZoomGuard&) = delete;
  ZoomGuard& operator=(const ZoomGuard&) = delete;

  // Move is not allowed (there is no uninitialized state)
  ZoomGuard(ZoomGuard&& other) = delete;
  ZoomGuard& operator=(ZoomGuard&& other) = delete;

  /// Sets the HIP device to the given device.  Errors if the given device
  /// is not a HIP device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the HIP device to the given device.  Errors if the given device
  /// is not a HIP device.  (This method is provided for uniformity with
  /// DeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the HIP device to the given device index.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::ZoomGuardImpl> guard_;
};

/// A variant of OptionalDeviceGuard that is specialized for HIP.  See
/// ZoomGuard for when you can use this.
struct OptionalZoomGuard {
  /// Create an uninitialized OptionalZoomGuard.
  explicit OptionalZoomGuard() : guard_() {}

  /// Set the current HIP device to the passed Device, if it is not nullopt.
  explicit OptionalZoomGuard(optional<Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current HIP device to the passed device index, if it is not
  /// nullopt
  explicit OptionalZoomGuard(optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy is not allowed
  OptionalZoomGuard(const OptionalZoomGuard&) = delete;
  OptionalZoomGuard& operator=(const OptionalZoomGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalZoomGuard(OptionalZoomGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalZoomGuard& operator=(OptionalZoomGuard&& other) = delete;

  /// Sets the HIP device to the given device, initializing the guard if it
  /// is not already initialized.  Errors if the given device is not a HIP
  /// device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the HIP device to the given device, initializing the guard if it is
  /// not already initialized.  Errors if the given device is not a HIP device.
  /// (This method is provided for uniformity with OptionalDeviceGuard).
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the HIP device to the given device index, initializing the guard if
  /// it is not already initialized.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set immediately prior to initialization of the
  /// guard, or nullopt if the guard is uninitialized.
  optional<Device> original_device() const {
    return guard_.original_device();
  }

  /// Returns the most recent device that was set using this device guard,
  /// either from construction, or via set_device, if the guard is initialized,
  /// or nullopt if the guard is uninitialized.
  optional<Device> current_device() const {
    return guard_.current_device();
  }

  /// Restore the original HIP device, resetting this guard to uninitialized
  /// state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<impl::ZoomGuardImpl> guard_;
};

/// A variant of StreamGuard that is specialized for HIP.  See ZoomGuard
/// for when you can use this.
struct ZoomStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit ZoomStreamGuard() = delete;

  /// Set the current HIP device to the device associated with the passed
  /// stream, and set the current HIP stream on that device to the passed
  /// stream. Errors if the Stream is not a HIP stream.
  explicit ZoomStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  ZoomStreamGuard(const ZoomStreamGuard&) = delete;
  ZoomStreamGuard& operator=(const ZoomStreamGuard&) = delete;

  /// Move is disallowed, as ZoomStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  ZoomStreamGuard(ZoomStreamGuard&& other) = delete;
  ZoomStreamGuard& operator=(ZoomStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a HIP stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on HIP, use ZoomMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the HIP stream that was set at the time the guard was
  /// constructed.
  ZoomStream original_stream() const {
    return ZoomStream(ZoomStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent HIP stream that was set using this device guard,
  /// either from construction, or via set_stream.
  ZoomStream current_stream() const {
    return ZoomStream(ZoomStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent HIP device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the HIP device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::ZoomGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for HIP.  See
/// ZoomGuard for when you can use this.
struct OptionalZoomStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalZoomStreamGuard() : guard_() {}

  /// Set the current HIP device to the device associated with the passed
  /// stream, and set the current HIP stream on that device to the passed
  /// stream. Errors if the Stream is not a HIP stream.
  explicit OptionalZoomStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalZoomStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalZoomStreamGuard(const OptionalZoomStreamGuard&) = delete;
  OptionalZoomStreamGuard& operator=(const OptionalZoomStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalZoomStreamGuard(OptionalZoomStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalZoomStreamGuard& operator=(OptionalZoomStreamGuard&& other) = delete;

  /// Resets the currently set HIP stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the HIP stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  optional<ZoomStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return make_optional(ZoomStream(ZoomStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Returns the most recent HIP stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  optional<ZoomStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return make_optional(ZoomStream(ZoomStream::UNCHECKED, r.value()));
    } else {
      return nullopt;
    }
  }

  /// Restore the original HIP device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::ZoomGuardImpl> guard_;
};

/// A variant of MultiStreamGuard that is specialized for HIP.
struct ZoomMultiStreamGuard {
  explicit ZoomMultiStreamGuard(ArrayRef<ZoomStream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  ZoomMultiStreamGuard(const ZoomMultiStreamGuard&) = delete;
  ZoomMultiStreamGuard& operator=(const ZoomMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  ZoomMultiStreamGuard(ZoomMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  ZoomMultiStreamGuard& operator=(ZoomMultiStreamGuard&& other) = delete;

 private:
  c10::impl::InlineMultiStreamGuard<impl::ZoomGuardImpl> guard_;

  static std::vector<Stream> unwrapStreams(ArrayRef<ZoomStream> zoomStreams) {
    std::vector<Stream> streams;
    streams.reserve(zoomStreams.size());
    for (const ZoomStream& zoomStream : zoomStreams) {
      streams.push_back(zoomStream);
    }
    return streams;
  }
};

} // namespace c10::zoom