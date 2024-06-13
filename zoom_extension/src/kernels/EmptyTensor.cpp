#include "EmptyTensor.h"

namespace at::detail {

    Tensor empty_generic_zoom(IntArrayRef size, ScalarType dtype, std::optional<Device> device_opt, std::optional<c10::MemoryFormat> memory_format_opt) {
        const auto device = device_or_default(device_opt);
        TORCH_INTERNAL_ASSERT(device.is_privateuseone());
        const DeviceGuard device_guard(device);
        auto allocator = c10::zoom::ZoomAllocator();
        constexpr c10::DispatchKeySet zoom_dks(c10::DispatchKey::PrivateUse1);
        return at::detail::empty_generic(
            size, &allocator, zoom_dks, dtype, memory_format_opt);
    }

    Tensor empty_memory_format_zoom(IntArrayRef size, ::std::optional<ScalarType> dtype_opt, ::std::optional<Layout> layout_opt, ::std::optional<Device> device_opt, ::std::optional<bool> pin_memory_opt, ::std::optional<MemoryFormat> memory_format_opt) {
        TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

        const auto dtype = dtype_or_default(dtype_opt);
        return at::detail::empty_generic_zoom(size, dtype, device_opt, memory_format_opt);
    }
    

    Tensor empty_strided_generic_zoom(IntArrayRef size, IntArrayRef stride, ScalarType dtype, ::std::optional<Device> device_opt) {
        const auto device = device_or_default(device_opt);
        TORCH_INTERNAL_ASSERT(device.is_privateuseone());
        const DeviceGuard device_guard(device);
        auto allocator = at::zoom::ZoomAllocator();
        constexpr c10::DispatchKeySet zoom_dks(c10::DispatchKey::PrivateUse1);
        return at::detail::empty_strided_generic(
            size, stride, &allocator, zoom_dks, dtype);
    }
    
    Tensor empty_strided_zoom(IntArrayRef size, IntArrayRef stride, ::std::optional<ScalarType> dtype_opt, ::std::optional<Layout> layout_opt, ::std::optional<Device> device_opt, ::std::optional<bool> pin_memory_opt){
        TORCH_CHECK(!pin_memory_opt.has_value() || !*pin_memory_opt, "Only dense CPU tensors can be pinned");
        TORCH_INTERNAL_ASSERT_DEBUG_ONLY(layout_or_default(layout_opt) == Layout::Strided);

        const auto dtype = dtype_or_default(dtype_opt);
        return at::detail::empty_strided_generic_zoom(size, stride, dtype, device_opt);
    }

    TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
        m.impl("empty.memory_format", &empty_memory_format_zoom);
        m.impl("empty_strided", &empty_strided_zoom);
    }

}