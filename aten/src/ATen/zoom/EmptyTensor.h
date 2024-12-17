#pragma once
#include <ATen/core/TensorBase.h>

namespace at::detail {

    TensorBase zoom_empty_generic(IntArrayRef size, ScalarType dtype, std::optional<Device> device, std::optional<c10::MemoryFormat> memory_format);
    TensorBase zoom_empty_memory_format(IntArrayRef size, ::std::optional<ScalarType> dtype, ::std::optional<Layout> layout, ::std::optional<Device> device, ::std::optional<bool> pin_memory, ::std::optional<MemoryFormat> memory_format); // {"schema": "aten::empty.memory_format(SymInt[] size, *, ScalarTy
    TORCH_ZOOM_API TensorBase empty_zoom(IntArrayRef size, const TensorOptions &options);

    TensorBase zoom_empty_strided_generic(IntArrayRef size, IntArrayRef stride, ScalarType dtype, ::std::optional<Device> device_opt);
    TensorBase zoom_empty_strided(IntArrayRef size, IntArrayRef stride, ::std::optional<ScalarType> dtype_opt, ::std::optional<Layout> layout_opt, ::std::optional<Device> device_opt, ::std::optional<bool> pin_memory_opt); // {"schema": "aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> TensorBase", "dispatch": "True", "default": "False"}
    TORCH_ZOOM_API TensorBase empty_strided_zoom(IntArrayRef size, IntArrayRef stride, const TensorOptions &options);

}