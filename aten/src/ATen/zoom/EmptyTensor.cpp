#include <ATen/zoom/EmptyTensor.h>
#include <ATen/EmptyTensor.h>
#include <ATen/zoom/ZoomContext.h>
#include <c10/zoom/ZoomCachingAllocator.h>
#include <ATen/detail/ZoomHooksInterface.h>
#include <iostream>
#include <ATen/DeviceGuard.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/view_native.h>
#include <torch/library.h>


namespace at::detail {

    TensorBase zoom_empty_generic(IntArrayRef size, ScalarType dtype, std::optional<Device> device_opt, std::optional<c10::MemoryFormat> memory_format_opt) {
        at::globalContext().lazyInitPrivateUse1();
        const auto device = device_or_default(device_opt);
        TORCH_INTERNAL_ASSERT(device.is_privateuseone());
        const DeviceGuard device_guard(device);
        auto* allocator = at::zoom::getZoomDeviceAllocator();
        constexpr c10::DispatchKeySet zoom_dks(c10::DispatchKey::PrivateUse1);
        return at::detail::empty_generic(
            size, allocator, zoom_dks, dtype, memory_format_opt);
    }

    TensorBase zoom_empty_memory_format(IntArrayRef size, ::std::optional<ScalarType> dtype_opt, ::std::optional<Layout> layout_opt, ::std::optional<Device> device_opt, ::std::optional<bool> pin_memory_opt, ::std::optional<MemoryFormat> memory_format_opt) {
        TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

        const auto dtype = dtype_or_default(dtype_opt);
        return zoom_empty_generic(size, dtype, device_opt, memory_format_opt);
    }

    TensorBase empty_zoom(IntArrayRef size, const TensorOptions &options) {
        return zoom_empty_memory_format(size, 
            optTypeMetaToScalarType(options.dtype_opt()),
            options.layout_opt(),
            options.device_opt(),
            options.pinned_memory_opt(),
            options.memory_format_opt());
    }
    

    TensorBase zoom_empty_strided_generic(IntArrayRef size, IntArrayRef stride, ScalarType dtype, ::std::optional<Device> device_opt) {
        at::globalContext().lazyInitPrivateUse1();
        const auto device = device_or_default(device_opt);
        TORCH_INTERNAL_ASSERT(device.is_privateuseone());
        const DeviceGuard device_guard(device);
        auto* allocator = at::zoom::getZoomDeviceAllocator();
        constexpr c10::DispatchKeySet zoom_dks(c10::DispatchKey::PrivateUse1);
        return at::detail::empty_strided_generic(
            size, stride, allocator, zoom_dks, dtype);
    }
    
    TensorBase zoom_empty_strided(IntArrayRef size, IntArrayRef stride, ::std::optional<ScalarType> dtype_opt, ::std::optional<Layout> layout_opt, ::std::optional<Device> device_opt, ::std::optional<bool> pin_memory_opt){
        TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

        const auto dtype = dtype_or_default(dtype_opt);
        return zoom_empty_strided_generic(size, stride, dtype, device_opt);
    }

    TensorBase empty_strided_zoom(
                IntArrayRef size,
                IntArrayRef stride,
                const TensorOptions &options) {
        return zoom_empty_strided(
            size,
            stride,
            optTypeMetaToScalarType(options.dtype_opt()),
            options.layout_opt(),
            options.device_opt(),
            options.pinned_memory_opt());
    }

}

namespace {
    at::Tensor wrapper_PrivateUse1_memory_format_empty(c10::SymIntArrayRef size, ::std::optional<at::ScalarType> dtype,
                                                       ::std::optional<at::Layout> layout,
                                                       ::std::optional<at::Device> device,
                                                       ::std::optional<bool> pin_memory,
                                                       ::std::optional<at::MemoryFormat> memory_format) {
        std::optional<c10::Device> common_device = std::nullopt;
        (void) common_device; // Suppress unused variable warning
        at::globalContext().lazyInitPrivateUse1();
        const c10::DeviceGuard device_guard(device_or_default(device));
        c10::TensorOptions opts{};
        return at::detail::empty_zoom(
            C10_AS_INTARRAYREF_SLOW(size),
            opts.dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory).memory_format(memory_format));
    }

    at::Tensor wrapper_PrivateUse1__view(const at::Tensor &self, c10::SymIntArrayRef size) {
        // No device check
        // DeviceGuard omitted
        return at::native::view(self, C10_AS_INTARRAYREF_SLOW(size));
    }

    at::Tensor &wrapper_PrivateUse1_out_abs_out(const at::Tensor &self, at::Tensor &out) {
        // No device check
        const c10::OptionalDeviceGuard device_guard(device_of(self));
        return at::native::abs_out(self, out);
    }
} // anonymous namespace

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format",
           TORCH_FN(wrapper_PrivateUse1_memory_format_empty));
    m.impl("view",
           TORCH_FN(wrapper_PrivateUse1__view));
    m.impl("abs.out",
           TORCH_FN(wrapper_PrivateUse1_out_abs_out));
};
