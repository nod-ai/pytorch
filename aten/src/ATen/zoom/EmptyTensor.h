#pragma once
#include <c10/zoom/extension.h>
#include <ATen/zoom/ZoomContext.h>

namespace at::detail {

    Tensor zoom_empty_generic(IntArrayRef size, ScalarType dtype, std::optional<Device> device, std::optional<c10::MemoryFormat> memory_format);
    Tensor zoom_empty_memory_format(IntArrayRef size, ::std::optional<ScalarType> dtype, ::std::optional<Layout> layout, ::std::optional<Device> device, ::std::optional<bool> pin_memory, ::std::optional<MemoryFormat> memory_format); // {"schema": "aten::empty.memory_format(SymInt[] size, *, ScalarTy
    
    Tensor zoom_empty_strided_generic(IntArrayRef size, IntArrayRef stride, ScalarType dtype, ::std::optional<Device> device_opt); 
    Tensor zoom_empty_strided(IntArrayRef size, IntArrayRef stride, ::std::optional<ScalarType> dtype_opt, ::std::optional<Layout> layout_opt, ::std::optional<Device> device_opt, ::std::optional<bool> pin_memory_opt); // {"schema": "aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor", "dispatch": "True", "default": "False"}

}