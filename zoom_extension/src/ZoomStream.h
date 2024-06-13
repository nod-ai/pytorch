// #pragma once

// #include <c10/core/Device.h>
// #include <c10/core/DeviceGuard.h>
// #include <c10/core/Stream.h>
// #include <aten/src/ATen/zoom/ZoomDefines.h>

// namespace c10 {
//     namespace zoom {
//         class ZoomStream {
//             public:
//                 explicit ZoomStream(c10::Stream stream) : stream_(stream) {
//                     TORCH_CHECK(stream_.device_type() == at::DeviceType::PrivateUse1);
//                 }

//                 bool operator==(const ZoomStream& other) const noexcept {
//                     return unwrap() == other.unwrap();
//                 }
//                 bool operator!=(const CUDAStream& other) const noexcept {
//                     return unwrap() != other.unwrap();
//                 }

//                 /// Implicit conversion to hipStream_t.
//                 operator hipStream_t() const {
//                     return stream();
//                 }

//                 /// Implicit conversion to Stream
//                 operator Stream() const {
//                     return unwrap();
//                 }

//                 /// Used to avoid baking in device type explicitly to Python-side API.
//                 DeviceType device_type() const {
//                     return at::DeviceType::PrivateUse1;
//                 }

//                 DeviceIndex device_index() const {
//                     return stream_.device_index();
//                 }

//                 /// Get the full Device that this stream is associated with.  The Device
//                 /// is guaranteed to be a CUDA device.
//                 Device device() const {
//                     return Device(at::DeviceType::PrivateUse1, device_index());
//                 }

//                 /// Return the stream ID corresponding to this particular stream.
//                 StreamId id() const {
//                     return stream_.id();
//                 }

//                 // get underlying hipStream_t 
//                 hipStream_t stream() const {
                    
//                 }


//         }
//     }
// }